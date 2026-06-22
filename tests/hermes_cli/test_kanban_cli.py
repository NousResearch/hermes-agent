"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _resolve_hermes_command() -> list[str]:
    """Locate the command used to drive the real ``hermes kanban create`` CLI.

    Preference order:
      1. ``HERMES_BIN`` env var — full path to a ``hermes`` executable. Lets
         CI/operators pin a specific binary.
      2. ``sys.executable -m hermes_cli.main`` — works in any environment where
         this test's Python interpreter can import ``hermes_cli`` (the typical
         editable-install / PYTHONPATH case). This guarantees the subprocess
         exercises whatever source the test session is bound to, not a sibling
         checkout's venv.

    Always returns a non-empty command list — letting the subprocess raise
    ImportError on a missing ``hermes_cli`` is more diagnostic than silently
    skipping these tests.
    """
    override = os.environ.get("HERMES_BIN")
    if override:
        p = Path(override)
        if p.exists():
            return [str(p)]
    return [sys.executable, "-m", "hermes_cli.main"]


HERMES_CMD = _resolve_hermes_command()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    profiles_dir = home / "profiles"
    profiles_dir.mkdir()
    # The kanban-create lint (card t_ddcf16e1) rejects unknown --assignee
    # values at the CLI entry point with exit 2. Tests in this file that
    # use a placeholder profile name (alice, bob, etc.) need that name to
    # exist as a profile directory under HERMES_HOME; otherwise the lint
    # would correctly reject the create and the test would fail.
    for name in (
        "alice",
        "bob",
        "broken-model",
        "newbie",
        "orig",
    ):
        (profiles_dir / name).mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("scratch",              ("scratch", None)),
        ("worktree",              ("worktree", None)),
        ("worktree:/tmp/wt",       ("worktree", "/tmp/wt")),
        ("dir:/tmp/work",         ("dir", "/tmp/work")),
    ],
)
def test_parse_workspace_flag_valid(value, expected):
    assert kc._parse_workspace_flag(value) == expected


def test_parse_workspace_flag_expands_user():
    kind, path = kc._parse_workspace_flag("dir:~/vault")
    assert kind == "dir"
    assert path.endswith("/vault")
    assert not path.startswith("~")

    kind, path = kc._parse_workspace_flag("worktree:~/trees/t6-wire")
    assert kind == "worktree"
    assert path.endswith("/trees/t6-wire")
    assert not path.startswith("~")

@pytest.mark.parametrize("bad", ["cloud", "dir:", "worktree:", ""])
def test_parse_workspace_flag_rejects(bad):
    if not bad:
        # Empty -> defaults; not an error.
        assert kc._parse_workspace_flag(bad) == ("scratch", None)
        return
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_workspace_flag(bad)


def test_parse_branch_flag_rejects_empty_and_option_like():
    assert kc._parse_branch_flag(None) is None
    assert kc._parse_branch_flag(" wt/t6-wire ") == "wt/t6-wire"
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("   ")
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("-bad")
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("bad branch")


# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------

def test_run_slash_no_args_shows_usage(kanban_home):
    out = kc.run_slash("")
    assert "kanban" in out.lower()
    assert "create" in out.lower() or "subcommand" in out.lower() or "action" in out.lower()


def test_run_slash_create_and_list(kanban_home):
    out = kc.run_slash("create 'ship feature' --assignee alice")
    assert "Created" in out
    out = kc.run_slash("list")
    assert "ship feature" in out
    assert "alice" in out


def test_run_slash_create_worktree_path_and_branch(kanban_home, tmp_path):
    target = tmp_path / ".worktrees" / "t6-wire"
    target_arg = target.as_posix()
    out = kc.run_slash(
        f"create 'ship worktree' --workspace worktree:{target_arg} --branch wt/t6-wire"
    )
    assert "Created" in out

    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    task = tasks[0]
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == target_arg
    assert task.branch_name == "wt/t6-wire"


def test_run_slash_rejects_branch_without_worktree(kanban_home):
    out = kc.run_slash("create 'bad branch' --workspace scratch --branch wt/bad")
    assert "--branch is only valid with --workspace worktree" in out


def test_run_slash_create_with_parent_and_cascade(kanban_home):
    # Parent then child via --parent
    out1 = kc.run_slash("create 'parent' --assignee alice")
    # Extract the "t_xxxx" id from "Created t_xxxx (ready, ...)"
    import re
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    p = m.group(1)
    out2 = kc.run_slash(f"create 'child' --assignee bob --parent {p}")
    assert "todo" in out2  # child starts as todo

    # Complete parent; list should promote child to ready
    kc.run_slash(f"complete {p}")
    # Explicit filter: child should now be ready (was todo before complete).
    ready_list = kc.run_slash("list --status ready")
    assert "child" in ready_list


def test_run_slash_show_includes_comments(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} 'remember to include performance section'")
    show = kc.run_slash(f"show {tid}")
    assert "performance section" in show


def test_run_slash_comment_max_len_trims_long_body(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} '{'x' * 30}' --max-len 20")
    show = kc.run_slash(f"show {tid}")
    assert "trimmed to 20 chars by --max-len" in show
    assert "x" * 30 not in show


def test_run_slash_block_unblock_cycle(kanban_home):
    out = kc.run_slash("create 'x' --assignee alice")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    # Claim first so block() finds it running
    kc.run_slash(f"claim {tid}")
    assert "Blocked" in kc.run_slash(f"block {tid} 'need decision'")
    assert "Unblocked" in kc.run_slash(f"unblock {tid}")


def test_run_slash_json_output(kanban_home):
    out = kc.run_slash("create 'jsontask' --assignee alice --json")
    payload = json.loads(out)
    assert payload["title"] == "jsontask"
    assert payload["assignee"] == "alice"
    assert payload["status"] == "ready"


def test_run_slash_dispatch_dry_run_counts(kanban_home):
    kc.run_slash("create 'a' --assignee alice")
    kc.run_slash("create 'b' --assignee bob")
    out = kc.run_slash("dispatch --dry-run")
    assert "Spawned:" in out


def test_run_slash_context_output_format(kanban_home):
    out = kc.run_slash("create 'tech spec' --assignee alice --body 'write an RFC'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} 'remember to include performance section'")
    ctx = kc.run_slash(f"context {tid}")
    assert "tech spec" in ctx
    assert "write an RFC" in ctx
    assert "performance section" in ctx


def test_run_slash_tenant_filter(kanban_home):
    kc.run_slash("create 'biz-a task' --tenant biz-a --assignee alice")
    kc.run_slash("create 'biz-b task' --tenant biz-b --assignee alice")
    a = kc.run_slash("list --tenant biz-a")
    b = kc.run_slash("list --tenant biz-b")
    assert "biz-a task" in a and "biz-b task" not in a
    assert "biz-b task" in b and "biz-a task" not in b


def test_run_slash_session_filter(kanban_home):
    """`hermes kanban list --session <id>` filters by the originating
    chat session id stamped on tasks created from inside an ACP loop."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="from sess-1 a", assignee="alice", session_id="sess-1"
        )
        kb.create_task(
            conn, title="from sess-1 b", assignee="alice", session_id="sess-1"
        )
        kb.create_task(
            conn, title="from sess-2", assignee="alice", session_id="sess-2"
        )
        kb.create_task(conn, title="cli only", assignee="alice")
    out_1 = kc.run_slash("list --session sess-1")
    out_2 = kc.run_slash("list --session sess-2")
    assert "from sess-1 a" in out_1
    assert "from sess-1 b" in out_1
    assert "from sess-2" not in out_1
    assert "cli only" not in out_1
    assert "from sess-2" in out_2
    assert "from sess-1 a" not in out_2


def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_run_slash_usage_error_returns_message(kanban_home):
    # Missing required argument for create
    out = kc.run_slash("create")
    assert "usage" in out.lower() or "error" in out.lower()


def test_run_slash_assign_reassigns(kanban_home):
    out = kc.run_slash("create 'x' --assignee alice")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    assert "Assigned" in kc.run_slash(f"assign {tid} bob")
    show = kc.run_slash(f"show {tid}")
    assert "bob" in show


def test_run_slash_link_unlink(kanban_home):
    a = kc.run_slash("create 'a'")
    b = kc.run_slash("create 'b'")
    import re
    ta = re.search(r"(t_[a-f0-9]+)", a).group(1)
    tb = re.search(r"(t_[a-f0-9]+)", b).group(1)
    assert "Linked" in kc.run_slash(f"link {ta} {tb}")
    # After link, b is todo
    show = kc.run_slash(f"show {tb}")
    assert "todo" in show
    assert "Unlinked" in kc.run_slash(f"unlink {ta} {tb}")


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kb.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kb.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------

def test_kanban_is_resolvable():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("kanban")
    assert cmd is not None
    assert cmd.name == "kanban"


def test_kanban_bypasses_active_session_guard():
    from hermes_cli.commands import should_bypass_active_session

    assert should_bypass_active_session("kanban")


def test_kanban_in_autocomplete_table():
    from hermes_cli.commands import COMMANDS, SUBCOMMANDS

    assert "/kanban" in COMMANDS
    subs = SUBCOMMANDS.get("/kanban") or []
    assert "create" in subs
    assert "dispatch" in subs


def test_kanban_autocomplete_includes_live_subcommands():
    from prompt_toolkit.document import Document

    from hermes_cli.commands import SlashCommandCompleter

    completer = SlashCommandCompleter()
    doc = Document("/kanban sp", cursor_position=len("/kanban sp"))
    texts = {c.text for c in completer.get_completions(doc, None)}

    assert "specify" in texts

    doc = Document("/kanban re", cursor_position=len("/kanban re"))
    texts = {c.text for c in completer.get_completions(doc, None)}

    assert "reclaim" in texts
    assert "reassign" in texts


def test_kanban_not_gateway_only():
    # kanban is available in BOTH CLI and gateway surfaces.
    from hermes_cli.commands import COMMAND_REGISTRY

    cmd = next(c for c in COMMAND_REGISTRY if c.name == "kanban")
    assert not cmd.cli_only
    assert not cmd.gateway_only


# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()


def test_run_slash_reassign_with_reclaim_flag(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'switch model' --assignee orig")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    tid = m.group(1)

    # Simulate a running claim.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reassign {tid} newbie --reclaim --reason 'switch'")
    assert "Reassigned" in out, out
    out2 = kc.run_slash(f"show {tid}")
    assert "newbie" in out2


# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------

def test_run_slash_specify_end_to_end(kanban_home, monkeypatch):
    """The /kanban specify slash command routes through run_slash, which
    both the interactive CLI and every gateway platform use. This test
    covers both surfaces."""
    from unittest.mock import MagicMock

    # Create a triage task via the same slash surface.
    create_out = kc.run_slash("create 'rough idea' --triage")
    import re
    m = re.search(r"(t_[a-f0-9]+)", create_out)
    assert m, f"no task id in: {create_out!r}"
    tid = m.group(1)

    # Mock the auxiliary client so we don't hit a real provider.
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = (
        '{"title": "Spec: rough idea", "body": "**Goal**\\nShip it."}'
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(return_value=resp)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda *a, **kw: (fake_client, "test-model"),
    )

    # Specify via slash.
    out = kc.run_slash(f"specify {tid}")
    assert "Specified" in out
    assert tid in out

    # Task is promoted and retitled.
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status in {"todo", "ready"}
    assert task.title == "Spec: rough idea"


def test_run_slash_specify_help_is_reachable(kanban_home):
    """`-h`/`--help` on a subcommand returns the actual help text — see
    issue #21794. argparse writes help to stdout and exits 0; run_slash
    must capture both streams and treat exit 0 as success, not error."""
    out = kc.run_slash("specify --help")
    assert "specify" in out.lower()
    # Help dump should NOT come back wrapped as a usage error.
    assert not out.startswith("⚠")


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------

def test_run_slash_bare_returns_curated_help(kanban_home):
    """Bare `/kanban` returns the curated short-help block — not a 5KB
    argparse usage dump."""
    out = kc.run_slash("")
    assert "/kanban" in out
    assert "list" in out
    assert "show" in out
    # Sanity: should be a chat-friendly size, not the raw usage tree.
    assert len(out) < 2000
    # Shouldn't surface argparse's usage-error sentinel.
    assert "usage error" not in out.lower()


@pytest.mark.parametrize("alias", ["help", "--help", "-h", "?"])
def test_run_slash_help_aliases_match_bare(kanban_home, alias):
    """Every documented help alias produces the same curated output."""
    bare = kc.run_slash("")
    out = kc.run_slash(alias)
    assert out == bare


def test_run_slash_subcommand_help_returns_help_text(kanban_home):
    """`/kanban show -h` returns the actual subcommand help, not a
    fake `(usage error: 0)` sentinel."""
    out = kc.run_slash("show -h")
    assert "task_id" in out
    assert "/kanban show" in out
    assert not out.startswith("⚠")


def test_run_slash_unknown_action_friendly_error(kanban_home):
    """Unknown subcommand surfaces a single-line usage error prefixed
    with our marker — no `(usage error: 2)` wrapping, no doubled
    `kanban kanban` prog string."""
    out = kc.run_slash("frobnicate")
    assert "/kanban" in out
    assert "frobnicate" in out
    assert "/kanban-wrap" not in out
    assert "/kanban kanban" not in out
    assert "(usage error: " not in out


def test_run_slash_missing_required_arg_friendly_error(kanban_home):
    """Missing positional argument shows the subcommand-scoped usage
    line, not the top-level kanban tree."""
    out = kc.run_slash("show")
    assert "/kanban show" in out
    assert "task_id" in out


def test_run_slash_board_override_restores_prior_env(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")

    kc.run_slash("--board alpha list")

    assert os.environ.get("HERMES_KANBAN_BOARD") == "beta"


def test_run_slash_board_override_does_not_change_boards_show_current(kanban_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    kb.set_current_board("alpha")

    out = kc.run_slash("--board beta boards show")

    assert "Current board: alpha" in out


# ---------------------------------------------------------------------------
# --assignee lint at the kanban create boundary (card t_ddcf16e1)
# ---------------------------------------------------------------------------
#
# These tests exercise the full subprocess path so we catch the actual exit
# code, stderr marker, and DB state. The lint runs in ``_cmd_create`` BEFORE
# any DB write; a non-resolving --assignee must exit 2 with no card row.
# The CLI subprocess is invoked via ``run_slash`` to match the project's
# "exercise the real entry point both CLI and gateway use" style; the
# underlying ``kanban_command`` returns the int exit code that ``run_slash``
# surfaces through the SystemExit-raising path (we read it back via a small
# wrapper that does NOT swallow SystemExit, so the int is captured exactly).
#
# The tests use the ``kanban_home`` fixture (which seeds ``alice``, ``bob``,
# ``broken-model``, ``newbie``, ``orig``) so a happy-path --assignee resolves
# against a real on-disk profile directory. The reject-path uses a name that
# the fixture deliberately does NOT seed.


class _AssigneeLintSubprocess:
    """Drive the real ``hermes kanban create`` CLI in a temp HERMES_HOME.

    Mirrors the subprocess pattern from ``tests/test_kanban_assignee_lint.py``
    so the lint is exercised through the actual argparse → CLI → DB path,
    not a mocked unit test. Captures (returncode, stdout, stderr) and the
    post-call DB state.
    """

    # Resolved at import time by ``_resolve_hermes_command``: either an
    # explicit ``hermes`` binary from $HERMES_BIN, or the test session's
    # Python interpreter running ``-m hermes_cli.main``. Never a hardcoded
    # absolute path — see card t_1fddf915 for the bug this replaced.
    HERMES_CMD = HERMES_CMD

    def __init__(self, kanban_home_dir: Path):
        # kanban_home is already pointing at our temp HERMES_HOME (with the
        # seeded profile dirs); reuse it directly. We just need to add a
        # dedicated DB path so the subprocess doesn't accidentally share the
        # in-process test DB.
        self.home = kanban_home_dir
        self.db_path = self.home / "kanban.db"

    def run_create(self, *args: str) -> tuple[int, str, str]:
        import subprocess

        env = os.environ.copy()
        # Strip dispatcher-set overrides so the lint is exercised in
        # isolation, not against whatever board the parent test was on.
        for var in (
            "HERMES_KANBAN_DB",
            "HERMES_KANBAN_HOME",
            "HERMES_KANBAN_BOARD",
            "HERMES_KANBAN_TASK",
            "PYTHONPATH",
            "PYTHONHOME",
        ):
            env.pop(var, None)
        env["HERMES_HOME"] = str(self.home)
        env["HERMES_KANBAN_DB"] = str(self.db_path)
        # When invoking ``python -m hermes_cli.main`` we want the subprocess
        # to import the same source the test is bound to. The parent pytest
        # process inherits a PYTHONPATH from its own launch, but the
        # dispatcher / kanban-worker harness may have set it to point at a
        # different checkout. Strip and re-pin to this test's module path so
        # the subprocess always exercises the branch under test.
        import hermes_cli as _hc  # noqa: WPS433 — local import by design

        env["PYTHONPATH"] = (
            str(Path(_hc.__file__).resolve().parent.parent)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        r = subprocess.run(
            [*self.HERMES_CMD, "kanban", "create", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        return r.returncode, r.stdout, r.stderr

    def db_row_count(self) -> int:
        import sqlite3

        if not self.db_path.exists():
            return 0
        conn = sqlite3.connect(str(self.db_path))
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0]
        finally:
            conn.close()

    def db_assignee_for_title(self, title: str) -> object:
        import sqlite3

        if not self.db_path.exists():
            return None
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT assignee FROM tasks WHERE title = ?", (title,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()


@pytest.fixture
def assignee_lint(kanban_home):
    # ``_resolve_hermes_command`` always returns a command (falls back to
    # ``python -m hermes_cli.main``). Only skip when the operator pinned a
    # non-existent $HERMES_BIN — in that case the literal override takes
    # priority and we shouldn't silently fall back to a different binary.
    if (
        os.environ.get("HERMES_BIN")
        and not Path(os.environ["HERMES_BIN"]).exists()
    ):
        pytest.skip(
            f"HERMES_BIN={os.environ['HERMES_BIN']!r} does not exist"
        )
    return _AssigneeLintSubprocess(kanban_home)


def test_create_with_valid_assignee_succeeds(assignee_lint):
    """Happy path: a known profile → rc=0, card persisted with that assignee.

    Acceptance criterion: ``hermes kanban create --assignee <seeded-valid-name>
    \"x\"`` exits 0 and writes a card with the resolved assignee. The card
    shape must be byte-identical to a pre-change valid create (no new columns,
    no schema drift).
    """
    # "alice" is seeded by the kanban_home fixture (see top of this file).
    rc, out, err = assignee_lint.run_create(
        "Test valid", "--assignee", "alice",
        "--body", "smoke", "--created-by", "test",
    )
    assert rc == 0, f"expected rc=0, got {rc}; stdout={out!r}; stderr={err!r}"
    assert "Created" in out, f"missing 'Created' marker in stdout: {out!r}"
    assert assignee_lint.db_row_count() == 1
    assert assignee_lint.db_assignee_for_title("Test valid") == "alice"


def test_create_with_unknown_assignee_exits_2(assignee_lint):
    """Reject path: ghost profile → rc=2, stderr marker, no DB row.

    Acceptance criterion: ``hermes kanban create --assignee ghostname \\"x\\"``
    exits 2 with a stderr marker identifying ``ghostname`` as not a
    registered profile, points at ``hermes profile list``, and writes no
    card.

    Format assertion (unified with the reaper branch's `_validate_assignee`,
    per card t_fcad5872): ``kanban: assignee 'ghostname' is not a
    registered Hermes profile.`` — this is the form produced by
    ``_validate_assignee`` in ``hermes_cli/kanban.py``.
    """
    rc, out, err = assignee_lint.run_create(
        "Test ghost", "--assignee", "ghostname",
        "--body", "smoke", "--created-by", "test",
    )
    assert rc == 2, f"expected rc=2, got {rc}; stdout={out!r}; stderr={err!r}"
    # Unified format: reaper branch's _validate_assignee uses
    # `kanban: assignee '<name>' is not a registered Hermes profile.`
    assert "ghostname" in err, (
        f"expected 'ghostname' in stderr, got: {err!r}"
    )
    assert "is not a registered Hermes profile" in err, (
        f"expected 'is not a registered Hermes profile' in stderr, got: {err!r}"
    )
    # Stderr should also point the operator at the help command advertised
    # by the codebase (see hermes_cli/profiles.py and existing kanban CLI
    # output that references ``hermes profile list``).
    assert "hermes profile list" in err, (
        f"expected stderr to suggest 'hermes profile list', got: {err!r}"
    )
    # No card was written.
    assert assignee_lint.db_row_count() == 0


def test_create_with_assignee_any_bypass_allowed(assignee_lint):
    """Bypass path: ``__any__`` and omitted ``--assignee`` both exit 0.

    Acceptance criterion: ``hermes kanban create --assignee __any__ \"x\"``
    and ``hermes kanban create \"x\"`` both exit 0 and persist the card.

    The bypass whitelist is exactly ``__any__`` and the omitted-flag case.
    ``default`` is NOT a magic value — it's a real profile directory and
    resolves through ``profile_resolver.resolve_assignee`` normally.
    """
    # Pass 1: --assignee __any__ (literal magic value from the bypass whitelist).
    rc1, out1, err1 = assignee_lint.run_create(
        "Test any", "--assignee", "__any__",
        "--body", "smoke", "--created-by", "test",
    )
    assert rc1 == 0, f"expected rc=0, got {rc1}; stdout={out1!r}; stderr={err1!r}"
    assert "Created" in out1
    # Pass 2: --assignee omitted entirely (no flag).
    rc2, out2, err2 = assignee_lint.run_create(
        "Test omitted", "--body", "smoke", "--created-by", "test",
    )
    assert rc2 == 0, f"expected rc=0, got {rc2}; stdout={out2!r}; stderr={err2!r}"
    assert "Created" in out2

    # Both cards present, with the expected assignee values.
    assert assignee_lint.db_row_count() == 2
    assert assignee_lint.db_assignee_for_title("Test any") == "__any__"
    assert assignee_lint.db_assignee_for_title("Test omitted") is None

    assert "Current board: alpha" in out


# ---------------------------------------------------------------------------
# Daemon stranded-config wiring (MEDIUM-2 from the t_f8afafaa WAGS review).
#
# The standalone `hermes kanban daemon --force` path (kept behind a deprecation
# notice for the rare host that can't run the gateway) was the only place
# `kanban.stranded_timeout_seconds` / `kanban.stranded_action` were NOT wired
# through. The fix adds the two kwargs to `kb.run_daemon(...)`, threads them
# into every `dispatch_once(...)` call, and reads them from config in
# `_cmd_daemon` — mirroring the pattern in `_cmd_dispatch`. These tests cover
# the three cases the WAGS review asked for: defaults, custom timeout, and
# custom action.
# ---------------------------------------------------------------------------


def _make_daemon_args(
    *,
    interval: float = 0.0,
    max_spawn: Optional[int] = None,
    failure_limit: Optional[int] = None,
    force: bool = True,
    pidfile: Optional[str] = None,
    verbose: bool = False,
):
    """Build a minimal argparse.Namespace shaped like what
    `hermes kanban daemon --force` parses. The daemon only reads the
    attributes it actually consumes; everything else is None.
    """
    return argparse.Namespace(
        interval=interval,
        max=max_spawn,
        failure_limit=(
            failure_limit if failure_limit is not None
            else kb.DEFAULT_SPAWN_FAILURE_LIMIT
        ),
        force=force,
        pidfile=pidfile,
        verbose=verbose,
    )


class TestDaemonConfig:
    """Verify `_cmd_daemon --force` honours kanban.stranded_* config keys.

    The daemon's main loop (`kb.run_daemon`) is monkey-patched to a stub
    that records the kwargs it received and returns immediately — this
    keeps the test fast and avoids racing the SIGINT/SIGTERM handler the
    real loop installs. The captured kwargs are then asserted against
    the operator's config so we know the wire-through is intact.
    """

    @staticmethod
    def _capture_run_daemon(monkeypatch):
        """Replace `kb.run_daemon` with a recorder. Returns the dict the
        daemon writes its kwargs into. The stub ignores `stop_event`
        because the caller (the daemon command) only sets it for the
        real long-lived loop.
        """
        captured: dict = {}
        calls: list = []

        def _stub(**kwargs):
            captured.update(kwargs)
            calls.append(kwargs)
            # stop_event.wait would block forever; the real loop relies
            # on signal handlers to set the event. The stub returns
            # immediately so the test thread doesn't hang.

        monkeypatch.setattr(kb, "run_daemon", _stub)
        return captured, calls

    def test_defaults_when_no_config(self, kanban_home, monkeypatch, capsys):
        """No config.yaml at all → daemon uses DEFAULT_STRANDED_* values.

        Mirrors the gateway watcher's behaviour when `kanban:` is absent
        from config.yaml (which is the most common deployment).
        """
        # No config file written to kanban_home.
        assert not (kanban_home / "config.yaml").exists()
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert len(calls) == 1
        assert captured["stranded_timeout_seconds"] == (
            kb.DEFAULT_STRANDED_TIMEOUT_SECONDS
        )
        assert captured["stranded_action"] == kb.DEFAULT_STRANDED_ACTION
        # Effective values are surfaced in the startup banner so an
        # operator can sanity-check the daemon saw the right config
        # without having to attach a debugger.
        out = capsys.readouterr().err
        assert f"stranded_timeout_seconds={kb.DEFAULT_STRANDED_TIMEOUT_SECONDS}" in out
        assert f"stranded_action={kb.DEFAULT_STRANDED_ACTION!r}" in out

    def test_defaults_when_kanban_section_empty(
        self, kanban_home, monkeypatch, capsys,
    ):
        """`kanban:` present but missing the stranded keys → still defaults.

        Distinguishes "config file present, keys absent" from "config
        file absent entirely" — both paths must use the defaults so a
        half-written config doesn't accidentally disable the reaper.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  dispatch_in_gateway: true\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_timeout_seconds"] == (
            kb.DEFAULT_STRANDED_TIMEOUT_SECONDS
        )
        assert captured["stranded_action"] == kb.DEFAULT_STRANDED_ACTION

    def test_custom_stranded_timeout_seconds(
        self, kanban_home, monkeypatch, capsys,
    ):
        """`kanban.stranded_timeout_seconds: 5` is honored end-to-end.

        Confirms the operator's tuned value reaches `kb.run_daemon` (and
        therefore every nested `dispatch_once` call) without being
        silently dropped or replaced by the default. Caps the timeout
        at 5s so the test exercises a non-default value distinct from
        the 1800s constant.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  stranded_timeout_seconds: 5\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_timeout_seconds"] == 5
        # Action remains at default; only the timeout was customised.
        assert captured["stranded_action"] == kb.DEFAULT_STRANDED_ACTION
        # Startup banner reflects the tuned value so the operator can
        # confirm the daemon saw the right config without a debugger.
        out = capsys.readouterr().err
        assert "stranded_timeout_seconds=5" in out
        assert f"stranded_action={kb.DEFAULT_STRANDED_ACTION!r}" in out

    def test_custom_stranded_action_archive(
        self, kanban_home, monkeypatch, capsys,
    ):
        """`kanban.stranded_action: archive` is honored end-to-end.

        Mirrors the `auto` (default) → `archive` switch: with this
        setting, a stranded ready task whose original assignee no
        longer maps to an installed profile should be archived, not
        reassigned. The test confirms the value passes through to
        `kb.run_daemon` and is shown in the startup banner.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  stranded_action: archive\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_action"] == "archive"
        # Timeout remains at default; only the action was customised.
        assert captured["stranded_timeout_seconds"] == (
            kb.DEFAULT_STRANDED_TIMEOUT_SECONDS
        )
        out = capsys.readouterr().err
        assert "stranded_action='archive'" in out
        assert (
            f"stranded_timeout_seconds={kb.DEFAULT_STRANDED_TIMEOUT_SECONDS}"
            in out
        )

    def test_custom_stranded_action_reassign(
        self, kanban_home, monkeypatch,
    ):
        """`kanban.stranded_action: reassign` reaches the dispatcher.

        The other valid non-default value. Combined with the
        `archive` test above this proves every entry in
        `kb.VALID_STRANDED_ACTIONS` survives the parse / lowercase /
        validate pipeline.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  stranded_action: reassign\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_action"] == "reassign"

    def test_invalid_stranded_action_falls_back_to_default(
        self, kanban_home, monkeypatch, capsys,
    ):
        """An unrecognised `kanban.stranded_action` falls back to 'auto'.

        A typo in the operator's config (e.g. `archv`) must not crash
        the legacy daemon — the safe default keeps the dispatcher
        moving. The startup banner shows the original (raw) value
        could not be applied; we at least confirm the daemon didn't
        propagate the typo to `kb.run_daemon`.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  stranded_action: archv\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_action"] == kb.DEFAULT_STRANDED_ACTION
        assert captured["stranded_action"] != "archv"

    def test_invalid_stranded_timeout_falls_back_to_default(
        self, kanban_home, monkeypatch,
    ):
        """A non-integer `kanban.stranded_timeout_seconds` falls back.

        e.g. `stranded_timeout_seconds: "5m"` (human string) should not
        crash the daemon — we log the parse failure and use the
        constant. Mirrors the gateway watcher's handling.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  stranded_timeout_seconds: \"5m\"\n", encoding="utf-8",
        )
        captured, calls = self._capture_run_daemon(monkeypatch)
        args = _make_daemon_args()

        rc = kc._cmd_daemon(args)

        assert rc == 0
        assert captured["stranded_timeout_seconds"] == (
            kb.DEFAULT_STRANDED_TIMEOUT_SECONDS
        )

    def test_deprecation_notice_still_prints_without_force(
        self, kanban_home, monkeypatch, capsys,
    ):
        """`hermes kanban daemon` (no --force) still prints the deprecation
        notice and exits 2 — the MEDIUM-2 fix must not regress the
        operator-facing migration path. Even with a fully-tuned config
        the legacy path is gated behind --force.
        """
        (kanban_home / "config.yaml").write_text(
            "kanban:\n"
            "  stranded_timeout_seconds: 5\n"
            "  stranded_action: archive\n",
            encoding="utf-8",
        )
        # No run_daemon stub — the deprecation branch returns before
        # touching it, and any future regression that calls into the
        # loop will be loud (no monkeypatch → real loop tries to start).
        args = _make_daemon_args(force=False)

        rc = kc._cmd_daemon(args)

        assert rc == 2
        out = capsys.readouterr().err
        # Preserved language from the deprecation banner.
        assert "DEPRECATED" in out
        assert "the dispatcher now runs" in out
        assert "--force" in out
