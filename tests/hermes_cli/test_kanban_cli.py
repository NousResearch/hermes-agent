"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
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


@pytest.mark.parametrize("bad", ["cloud", "dir:", "", "worktree:/x"])
def test_parse_workspace_flag_rejects(bad):
    if not bad:
        # Empty -> defaults; not an error.
        assert kc._parse_workspace_flag(bad) == ("scratch", None)
        return
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_workspace_flag(bad)


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
    kc.run_slash(f"comment {tid} 'source is paywalled'")
    show = kc.run_slash(f"show {tid}")
    assert "source is paywalled" in show


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
