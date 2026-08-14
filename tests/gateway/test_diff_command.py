"""End-to-end tests for the gateway ``/diff`` command.

Exercises the real handler against real git repositories (default modes) and
a real checkpoint store (session mode), proving the messaging surface returns
fenced, truncated diff text and degrades to friendly messages when there's
nothing to show.
"""

import shutil
import subprocess

import pytest

import gateway.run as gateway_run
import tools.checkpoint_manager as cpm
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for /diff"
)


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(repo),
                        "PATH": __import__("os").environ["PATH"]})


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    (d / "main.py").write_text("print('hello')\n", encoding="utf-8")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    monkeypatch.setenv("TERMINAL_CWD", str(d))
    return d


def _enable_checkpoints(tmp_path, monkeypatch, enabled=True):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"checkpoints:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", home, raising=False)
    monkeypatch.setattr(cpm, "CHECKPOINT_BASE", tmp_path / "checkpoints")


# ---------------------------------------------------------------------------
# Default (working-tree) mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_long_output_truncated(repo):
    lines = "\n".join(f"line{i} = {i}" for i in range(300)) + "\n"
    (repo / "main.py").write_text(lines, encoding="utf-8")

    result = await _runner()._handle_diff_command(_event("/diff"))

    # Hard-truncated in the handler before the platform senders apply their
    # own message-splitting limits (3-layer tool-progress-style truncation).
    assert "truncated" in result
    assert len(result) < 6000


# ---------------------------------------------------------------------------
# Session mode — checkpoint baseline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diff_session_reports_cumulative_changes(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    # Baseline checkpoint (pre-edit) then an edit, so a diff exists.
    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    assert mgr.ensure_checkpoint(str(project), "baseline") is True
    (project / "main.py").write_text("print('changed')\n", encoding="utf-8")

    result = await _runner()._handle_diff_command(_event("/diff session"))

    assert "-print('hello')" in result
    assert "+print('changed')" in result


@pytest.mark.asyncio
async def test_diff_session_no_changes_message(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    result = await _runner()._handle_diff_command(_event("/diff session"))

    assert "No changes" in result


