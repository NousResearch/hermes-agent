"""Tests for the CLI ``/worktree`` command handler.

``/worktree`` (Copilot CLI-inspired) inspects or creates isolated git
worktrees mid-session. These drive the mixin handler against real git repos:
status/list output, ``new`` creation with cwd + TERMINAL_CWD retargeting,
named-tree collision refusal, and graceful non-repo degradation.
"""

import contextlib
import io
import os
import shutil
import subprocess

import pytest

from agent.i18n import t

import cli as cli_mod
from hermes_cli.cli_commands_mixin import CLICommandsMixin

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required"
)

class _Stub(CLICommandsMixin):
    def __init__(self):
        self.agent = None

def _run(stub, command):
    """Capture handler output.

    Worktree messages route through ``cli._cprint`` (the ANSI-aware
    prompt_toolkit renderer) since the garbled-escape fix; its output binds
    to the prompt_toolkit session output object, not ``sys.stdout``, so
    ``redirect_stdout`` alone cannot capture it. Mirror ``_cprint`` into the
    same buffer to keep asserting the user-visible text.
    """
    import re
    from unittest.mock import patch as _patch

    buf = io.StringIO()
    ansi = re.compile(r"\x1b\[[0-9;]*m")

    def _fake_cprint(text):
        buf.write(ansi.sub("", str(text)) + "\n")

    with contextlib.redirect_stdout(buf), _patch.object(cli_mod, "_cprint", _fake_cprint):
        stub._handle_worktree_command(command)
    return buf.getvalue()

def _git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "HOME": str(repo), "PATH": os.environ["PATH"]},
    )

@pytest.fixture()
def repo(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cli_mod, "_active_worktree", None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    yield repo
    # Leave tmp dir before pytest tears it down (worktree cwd may be inside).
    os.chdir(tmp_path)

def test_status_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_active_worktree", None)
    out = _run(_Stub(), "/worktree")
    assert t("cli.commands.worktree.not_in_repo_note") in out

def test_new_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _run(_Stub(), "/worktree new")
    assert t("cli.commands.worktree.new_requires_repo") in out

@requires_git
def test_list_shows_worktrees(repo):
    out = _run(_Stub(), "/worktree list")
    assert repo.as_posix() in out

@requires_git
def test_new_named_creates_and_retargets(repo):
    out = _run(_Stub(), "/worktree new fix-login")
    wt = repo / ".worktrees" / "fix-login"
    assert t("cli.commands.worktree.ready", path=wt) in out
    assert wt.is_dir()
    assert os.environ["TERMINAL_CWD"] == str(wt)
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(wt))
    assert cli_mod._active_worktree is not None
    assert cli_mod._active_worktree["branch"] == "hermes/fix-login"
    # Same file contents as the base commit.
    assert (wt / "a.txt").read_text() == "hello\n"

@requires_git
def test_new_named_collision_refused(repo):
    _run(_Stub(), "/worktree new dup")
    first = cli_mod._active_worktree
    os.chdir(repo)
    out = _run(_Stub(), "/worktree new dup")
    assert "already exists" in out
    # Active worktree not clobbered by the failed attempt.
    assert cli_mod._active_worktree == first

@requires_git
def test_new_unnamed_uses_random_hermes_prefix(repo):
    out = _run(_Stub(), "/worktree new")
    assert t("cli.commands.worktree.ready", path=cli_mod._active_worktree["path"]) in out
    name = os.path.basename(cli_mod._active_worktree["path"])
    assert name.startswith("hermes-")

@requires_git
def test_new_sanitizes_name(repo):
    _run(_Stub(), "/worktree new " + "weird name!@#")
    name = os.path.basename(cli_mod._active_worktree["path"])
    assert name == "weird-name"


def _gateway_stub(rpc=None):
    """A CLI attached to a (scripted) gateway session: the tools run there, not in this process."""
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tests.hermes_cli.test_gateway_agent import connect_with
    from tests.hermes_cli.test_gateway_session_store import FakeClient
    client = FakeClient(known=("local-s",))
    if rpc is not None:
        client.rpc = rpc.__get__(client)
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    stub = _Stub()
    stub._session_db, stub._gateway_session_open, stub.session_id = GatewaySessionStore(conn), True, "local-s"
    return stub, client, conn


@requires_git
def test_new_moves_the_gateway_sessions_workspace_into_the_tree(repo):
    """CLI-15: the session's tools run in the gateway under its frozen cwd, so /worktree new must move
    that (``session.mutate workspace``, generation-fenced) — a chdir of this process moved nothing."""
    stub, client, conn = _gateway_stub()
    try:
        out = _run(stub, "/worktree new moved")
    finally:
        conn.close()
    wt = repo / ".worktrees" / "moved"
    (params,) = [p for method, p in client.calls if method == "session.mutate"]
    assert (params["operation"], params["payload"], params["session_id"]) == ("workspace", {"cwd": str(wt)}, "local-s")
    assert params["expected_generation"] == 0
    assert t("cli.commands.worktree.tools_in_worktree") in out


@requires_git
def test_new_refused_by_the_gateway_never_claims_the_tools_moved(repo):
    """A refused move (e.g. ``session_busy`` mid-turn) leaves the session where it was: say so, and
    keep this process (and a later /new's cwd) in the original checkout too."""
    from hermes_cli.gateway_client import GatewayRPCError
    from tests.hermes_cli.test_gateway_session_store import snapshot

    async def refuse(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.mutate":
            raise GatewayRPCError("session_busy")
        return snapshot("local-s")
    stub, _client, conn = _gateway_stub(refuse)
    try:
        out = _run(stub, "/worktree new busy")
    finally:
        conn.close()
    assert "session_busy" in out and t("cli.commands.worktree.tools_in_worktree") not in out
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(repo))
    assert "TERMINAL_CWD" not in os.environ and cli_mod._active_worktree is None


@requires_git
def test_a_moved_session_gets_its_clean_tree_back_on_resume(repo):
    """The move freezes the tree as the session's cwd and quit removes a clean tree, so a resume had
    every turn refused ``workspace_unavailable``. The tree is recorded as a ``--tui -w`` one is, so the
    resume recreates it at the same path (N8)."""
    from hermes_cli.tui_worktree_resume import recreate_tui_worktree
    stub, _client, conn = _gateway_stub()
    try:
        _run(stub, "/worktree new kept")
    finally:
        conn.close()
    wt = repo / ".worktrees" / "kept"
    os.chdir(repo)
    cli_mod._cleanup_worktree(cli_mod._active_worktree)  # quit: the tree is clean, so it goes
    assert not wt.exists()
    assert recreate_tui_worktree(str(wt), "local-s") == str(wt) and wt.is_dir()
