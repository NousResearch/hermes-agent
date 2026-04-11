"""Tests for runtime self-mutation detection and hard block in tools.approval.

These tests verify that commands which would mutate the agent's own
Python runtime are detected by ``detect_runtime_mutation`` and hard-
blocked by ``_hard_block_runtime_mutation`` / ``check_all_command_guards``.

Coverage: venv creation (uv venv, python -m venv, virtualenv), recursive
delete (rm -rf), move (mv), and uv pip uninstall without --python.
"""

import os
import pytest

from tools.approval import (
    detect_dangerous_command,
    detect_runtime_mutation,
    _hard_block_runtime_mutation,
    check_all_command_guards,
)


@pytest.fixture
def fake_runtime(tmp_path):
    """A real directory that stands in for the agent's own venv."""
    fake = tmp_path / "fake-venv"
    (fake / "bin").mkdir(parents=True)
    return str(fake.resolve())


@pytest.fixture
def runtime_roots(fake_runtime):
    return [fake_runtime]


# =========================================================================
# detect_runtime_mutation — detection layer
# =========================================================================


class TestDetectRuntimeMutation:
    # --- venv creation ---

    def test_uv_venv_exact_absolute_path_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"uv venv {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None
        assert "venv creation" in reason

    def test_uv_venv_relative_from_parent_flagged(self, runtime_roots):
        parent = os.path.dirname(runtime_roots[0])
        basename = os.path.basename(runtime_roots[0])
        reason = detect_runtime_mutation(
            f"uv venv {basename}",
            cwd=parent,
            runtime_roots=runtime_roots,
        )
        assert reason is not None
        assert "venv creation" in reason

    def test_uv_venv_with_python_flag_still_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"uv venv --python 3.11 {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    def test_uv_venv_unrelated_target_ignored(self, runtime_roots, tmp_path):
        other = tmp_path / "other-project" / ".venv"
        reason = detect_runtime_mutation(
            f"uv venv {other}",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    def test_python_m_venv_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"python -m venv {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    def test_python3_m_venv_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"python3 -m venv {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    def test_virtualenv_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"virtualenv {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    # --- rm -rf ---

    def test_rm_rf_runtime_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"rm -rf {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None
        assert "recursive delete" in reason

    def test_rm_rf_combined_flags_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"rm -fr {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    def test_rm_rf_long_flag_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"rm --recursive --force {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    def test_rm_without_recursive_ignored(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"rm {runtime_roots[0]}/file",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    def test_rm_rf_unrelated_ignored(self, runtime_roots, tmp_path):
        other = tmp_path / "junk"
        other.mkdir()
        reason = detect_runtime_mutation(
            f"rm -rf {other}",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    # --- mv ---

    def test_mv_runtime_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"mv {runtime_roots[0]} /tmp/backup",
            runtime_roots=runtime_roots,
        )
        assert reason is not None
        assert "move" in reason

    def test_mv_into_runtime_as_dest_ignored(self, runtime_roots, tmp_path):
        """Moving something INTO the runtime (as dest) is not destructive."""
        source = tmp_path / "file.txt"
        source.touch()
        reason = detect_runtime_mutation(
            f"mv {source} {runtime_roots[0]}/",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    def test_mv_unrelated_ignored(self, runtime_roots, tmp_path):
        reason = detect_runtime_mutation(
            f"mv {tmp_path}/a {tmp_path}/b",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    # --- uv pip uninstall ---

    def test_uv_pip_uninstall_no_python_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            "uv pip uninstall requests",
            runtime_roots=runtime_roots,
        )
        assert reason is not None
        assert "uv pip uninstall" in reason

    def test_uv_pip_uninstall_with_python_ignored(self, runtime_roots):
        reason = detect_runtime_mutation(
            "uv pip uninstall --python /tmp/other/bin/python3 requests",
            runtime_roots=runtime_roots,
        )
        assert reason is None

    # --- sudo / env wrapper stripping ---

    def test_sudo_rm_rf_runtime_flagged(self, runtime_roots):
        reason = detect_runtime_mutation(
            f"sudo rm -rf {runtime_roots[0]}",
            runtime_roots=runtime_roots,
        )
        assert reason is not None

    # --- edge cases ---

    def test_unparseable_command_returns_none(self, runtime_roots):
        assert (
            detect_runtime_mutation(
                'uv venv "unclosed',
                runtime_roots=runtime_roots,
            )
            is None
        )

    def test_empty_command_returns_none(self, runtime_roots):
        assert detect_runtime_mutation("", runtime_roots=runtime_roots) is None

    def test_no_runtime_roots_returns_none(self):
        assert (
            detect_runtime_mutation(
                "uv venv /anywhere",
                runtime_roots=[],
            )
            is None
        )

    def test_unrelated_command_returns_none(self, runtime_roots):
        assert (
            detect_runtime_mutation(
                "ls -la",
                runtime_roots=runtime_roots,
            )
            is None
        )

    def test_read_only_access_to_runtime_allowed(self, runtime_roots):
        """cat, ls, head etc. against runtime paths must not be blocked."""
        for cmd in ["cat", "ls -la", "head -20", "file"]:
            assert detect_runtime_mutation(
                f"{cmd} {runtime_roots[0]}/bin/python",
                runtime_roots=runtime_roots,
            ) is None


# =========================================================================
# _hard_block_runtime_mutation — enforcement layer
# =========================================================================


class TestHardBlock:
    """Verify _hard_block_runtime_mutation returns proper block dicts."""

    def test_block_returns_hard_blocked_dict(self, monkeypatch, fake_runtime):
        monkeypatch.setattr("tools.approval._runtime_roots", lambda: [fake_runtime])
        result = _hard_block_runtime_mutation(f"rm -rf {fake_runtime}")
        assert result is not None
        assert result["hard_blocked"] is True
        assert result["approved"] is False
        assert "BLOCKED" in result["message"]

    def test_safe_command_returns_none(self, monkeypatch, fake_runtime):
        monkeypatch.setattr("tools.approval._runtime_roots", lambda: [fake_runtime])
        assert _hard_block_runtime_mutation("ls -la") is None


# =========================================================================
# check_all_command_guards integration — non-bypassable
# =========================================================================


class TestCheckAllCommandGuardsIntegration:
    """Verify the hard block fires before yolo/container bypass."""

    def test_hard_block_fires_even_in_yolo_mode(
        self, monkeypatch, fake_runtime
    ):
        monkeypatch.setattr("tools.approval._runtime_roots", lambda: [fake_runtime])
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        result = check_all_command_guards(
            f"rm -rf {fake_runtime}", env_type="local"
        )
        assert result["approved"] is False
        assert result.get("hard_blocked") is True

    def test_hard_block_fires_even_for_containers(
        self, monkeypatch, fake_runtime
    ):
        monkeypatch.setattr("tools.approval._runtime_roots", lambda: [fake_runtime])
        result = check_all_command_guards(
            f"rm -rf {fake_runtime}", env_type="docker"
        )
        assert result["approved"] is False
        assert result.get("hard_blocked") is True

    def test_safe_command_passes_through(self, monkeypatch, fake_runtime):
        monkeypatch.setattr("tools.approval._runtime_roots", lambda: [fake_runtime])
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        result = check_all_command_guards("ls -la", env_type="local")
        assert result["approved"] is True

    def test_rm_rf_against_runtime_still_caught_by_existing_pattern(
        self, fake_runtime
    ):
        # The existing `rm -[^\s]*r` regex pattern already catches any
        # `rm -rf`. Verify we don't break that.
        is_dangerous, _, _ = detect_dangerous_command(f"rm -rf {fake_runtime}")
        assert is_dangerous is True
