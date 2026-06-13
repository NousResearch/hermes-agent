"""Tests for agent.turn_finalizer._capture_safe_workspace_status.

The workspace diagnostic is captured on non-cooperative Kanban worker exits
(iteration exhaustion) and flows into kanban_comment bodies and
``_record_task_failure`` event payloads — i.e. it is persisted and surfaced
to other profiles. The redaction and capping behavior is therefore
security-relevant and must be pinned.

Real ``git`` is used for the happy paths (local binary, no network);
failure and parsing variants mock ``subprocess.run``.
"""

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agent.turn_finalizer import _capture_safe_workspace_status

HAS_GIT = shutil.which("git") is not None


def _git(workdir, *args):
    subprocess.run(
        ["git", "-C", str(workdir), "-c", "user.email=t@test", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "ws"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "hermes-test-branch", str(repo)],
        check=True,
        capture_output=True,
    )
    (repo / "tracked.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


class TestCaptureInvalidWorkspaces:
    def test_empty_path_returns_none(self):
        assert _capture_safe_workspace_status("") is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert _capture_safe_workspace_status(str(tmp_path / "nope")) is None

    @pytest.mark.skipif(not HAS_GIT, reason="git binary required")
    def test_non_git_dir_returns_none(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert _capture_safe_workspace_status(str(plain)) is None

    def test_subprocess_oserror_returns_none(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise OSError("git not runnable")

        monkeypatch.setattr("agent.turn_finalizer.subprocess.run", boom)
        assert _capture_safe_workspace_status(str(tmp_path)) is None

    def test_subprocess_timeout_returns_none(self, tmp_path, monkeypatch):
        def hang(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="git", timeout=5)

        monkeypatch.setattr("agent.turn_finalizer.subprocess.run", hang)
        assert _capture_safe_workspace_status(str(tmp_path)) is None


@pytest.mark.skipif(not HAS_GIT, reason="git binary required")
class TestCaptureRealRepo:
    def test_clean_repo_reports_dirty_false_with_branch(self, git_repo):
        # `git status --short --branch` always emits the `## <branch>` header,
        # so dirtiness is decided by the porcelain change lines only. A clean
        # repo must report dirty=False while still carrying branch info and
        # the header-only status.
        result = _capture_safe_workspace_status(str(git_repo))
        assert result is not None
        assert result["git_repo"] is True
        assert result["dirty"] is False
        assert result["branch"] == "hermes-test-branch"
        assert result["git_status_raw"] == "## hermes-test-branch"

    def test_dirty_repo_lists_untracked_and_extracts_branch(self, git_repo):
        (git_repo / "new-file.py").write_text("x = 1\n", encoding="utf-8")
        result = _capture_safe_workspace_status(str(git_repo))
        assert result["dirty"] is True
        assert result["branch"] == "hermes-test-branch"
        assert "new-file.py" in result["git_status_raw"]

    def test_secret_like_filename_is_redacted(self, git_repo):
        # Untracked filenames are attacker/agent-controlled content that
        # flows into persisted kanban comments — key=value shapes must not
        # survive verbatim.
        (git_repo / "token=abc123secretvalue").write_text("", encoding="utf-8")
        result = _capture_safe_workspace_status(str(git_repo))
        raw = result["git_status_raw"]
        assert "abc123secretvalue" not in raw
        assert "[REDACTED]" in raw

    def test_line_cap_keeps_first_50_lines(self, git_repo):
        for i in range(60):
            (git_repo / f"f{i:03d}.txt").write_text("", encoding="utf-8")
        result = _capture_safe_workspace_status(str(git_repo))
        raw = result["git_status_raw"]
        # 1 header line + 49 file lines == 50 lines kept
        assert raw.count("\n") == 49
        assert "f059.txt" not in raw
        assert "… [truncated]" not in raw

    def test_byte_cap_appends_truncation_marker(self, git_repo):
        # Hyphenated names avoid the base64-like redaction pattern so the
        # byte cap itself is what is exercised here.
        for i in range(45):
            (git_repo / f"{i:02d}-{'segment-' * 13}tail.txt").write_text(
                "", encoding="utf-8"
            )
        result = _capture_safe_workspace_status(str(git_repo))
        raw = result["git_status_raw"]
        assert raw.endswith("… [truncated]")
        assert len(raw) <= 4096 + len("\n… [truncated]")


class TestCaptureParsingAndRedaction:
    """Parsing variants pinned against crafted `git status` output."""

    def _fake_run(self, monkeypatch, stdout, returncode=0):
        def fake(*a, **kw):
            return SimpleNamespace(returncode=returncode, stdout=stdout)

        monkeypatch.setattr("agent.turn_finalizer.subprocess.run", fake)

    def test_branch_with_upstream_marker_is_split(self, tmp_path, monkeypatch):
        self._fake_run(
            monkeypatch, "## main...origin/main [ahead 2]\n M run_agent.py\n"
        )
        result = _capture_safe_workspace_status(str(tmp_path))
        assert result["branch"] == "main"
        assert result["dirty"] is True
        assert " M run_agent.py" in result["git_status_raw"]

    def test_nonzero_exit_returns_none(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, "", returncode=128)
        assert _capture_safe_workspace_status(str(tmp_path)) is None

    def test_empty_output_reports_clean(self, tmp_path, monkeypatch):
        # Real git always emits the `## <branch>` header, so empty output only
        # occurs with degenerate/mocked git. Pinned so the early-return
        # contract for that case does not silently change.
        self._fake_run(monkeypatch, "\n")
        result = _capture_safe_workspace_status(str(tmp_path))
        assert result == {"git_repo": True, "dirty": False}

    def test_header_only_output_reports_clean_with_branch(self, tmp_path, monkeypatch):
        # Clean repo through the real-git output shape: header line only,
        # no porcelain change lines → dirty=False, branch still extracted.
        self._fake_run(monkeypatch, "## main...origin/main\n")
        result = _capture_safe_workspace_status(str(tmp_path))
        assert result["dirty"] is False
        assert result["branch"] == "main"
        assert result["git_status_raw"] == "## main...origin/main"

    @pytest.mark.parametrize(
        "leak",
        [
            "?? api_key=sk-live-very-secret-value",
            "?? sk-" + "a" * 40,
            "?? ghp_" + "b" * 40,
            "?? " + "Q" * 48 + "==",  # base64-like long run
        ],
    )
    def test_secret_patterns_redacted(self, tmp_path, monkeypatch, leak):
        self._fake_run(monkeypatch, f"## main\n{leak}\n")
        result = _capture_safe_workspace_status(str(tmp_path))
        raw = result["git_status_raw"]
        assert "[REDACTED]" in raw
        for token in ("sk-live", "a" * 40, "b" * 40, "Q" * 48):
            assert token not in raw
