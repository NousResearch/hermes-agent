"""A Desktop/backend launched from a kanban worker's shell must not inherit ANY agent-scope env.

Repro: a Desktop reopened with ``open`` from a worker's terminal HERMES_DELEGATED_CHILD_CONTEXT, HERMES_KANBAN_*, HERMES_SESSION_ID/SOURCE,
HERMES_SINGLE_QUERY_SESSION and HERMES_SUPERVISED_CHILD, so every Desktop chat was fenced as a
delegate child and kanban mutations were refused. The scope list lives in one helper in
``agent/delegation_context.py`` (``strip_agent_scope_env``); the Electron main mirrors it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The env the leaked Desktop actually carried (ps eww 59331), plus the rest of the scope family.
POISON = {
    "HERMES_DELEGATED_CHILD_CONTEXT": "/Users/x/.hermes",
    "HERMES_KANBAN_TASK": "t_45d5a361",
    "HERMES_KANBAN_WORKSPACE": "/Users/x/.hermes/kanban/workspaces/t_45d5a361",
    "HERMES_KANBAN_DB": "/Users/x/.hermes/kanban.db",
    "HERMES_KANBAN_BOARD": "default",
    "HERMES_KANBAN_WORKSPACES_ROOT": "/Users/x/.hermes/kanban/workspaces",
    "HERMES_KANBAN_RUN_ID": "2848",
    "HERMES_SESSION_ID": "20261021_185223_3c48bc",
    "HERMES_SESSION_SOURCE": "kanban",
    "HERMES_SESSION_KEY": "k",
    "HERMES_SINGLE_QUERY_SESSION": "1",
    "HERMES_SUPERVISED_CHILD": "1",
    "HERMES_CRON_SESSION": "1",
    "HERMES_PARENT_PID": "84584",
    "HERMES_PARENT_START_MARKER": "x",
    "HERMES_SPAWN": "1",
    "HERMES_RPC_TOKEN": "tok",
    "HERMES_RPC_SOCKET": "/tmp/s",
    "HERMES_KERNEL_DIR": "/tmp/k",
}
# Operator tunables share those prefixes but are configuration, not identity: keep them.
TUNABLES = {
    "HERMES_SESSION_STALL_TIMEOUT": "600",
    "HERMES_CRON_TIMEOUT": "900",
    "HERMES_CRON_MAX_PARALLEL": "2",
    "HERMES_KANBAN_GUARD_HOLD_CAP_SECONDS": "60",
}
SAFE = {"PATH": "/bin", "HOME": "/Users/x", "HERMES_HOME": "/Users/x/.hermes"}


def test_strip_agent_scope_env_removes_whole_family_keeps_tunables():
    from agent.delegation_context import strip_agent_scope_env
    env = {**SAFE, **TUNABLES, **POISON}
    removed = strip_agent_scope_env(env)
    assert set(removed) == set(POISON)
    assert env == {**SAFE, **TUNABLES}


def test_strip_agent_scope_env_is_noop_without_worker_marker():
    """Without a worker/delegate marker, session or board env is a deliberate operator setting."""
    from agent.delegation_context import strip_agent_scope_env
    env = {**SAFE, "HERMES_KANBAN_DB": "/tmp/k.db", "HERMES_SESSION_SOURCE": "desktop",
           "HERMES_SUPERVISED_CHILD": "1"}
    before = dict(env)
    assert strip_agent_scope_env(env) == []
    assert env == before


def test_clean_launch_env_with_poisoned_environ(monkeypatch):
    from hermes_cli.top_level_env import clean_launch_env
    for k, v in {**POISON, **TUNABLES}.items():
        monkeypatch.setenv(k, v)
    env = clean_launch_env()
    assert not (set(POISON) & set(env)), sorted(set(POISON) & set(env))
    assert all(env.get(k) == v for k, v in TUNABLES.items())
    # os.environ itself is untouched: only the copy is cleaned.
    assert os.environ["HERMES_DELEGATED_CHILD_CONTEXT"] == POISON["HERMES_DELEGATED_CHILD_CONTEXT"]


def test_cli_entry_scrubs_before_dispatch_for_top_level_command(monkeypatch):
    """``hermes desktop`` from a worker shell: main() drops the env before the command runs."""
    from hermes_cli import main as hm
    for k, v in POISON.items():
        monkeypatch.setenv(k, v)
    seen = {}

    class _Stop(Exception):
        pass

    def fake_title():
        seen["env"] = dict(os.environ)
        raise _Stop

    monkeypatch.setattr(hm, "_set_process_title", fake_title)
    monkeypatch.setattr(hm.sys, "argv", ["hermes", "desktop"])
    with pytest.raises(_Stop):
        hm.main()
    assert not (set(POISON) & set(seen["env"]))


def test_cli_entry_leaves_worker_env_for_non_top_level_commands(monkeypatch):
    """A worker's own ``hermes kanban ...`` / ``hermes chat`` keep their identity."""
    from hermes_cli.top_level_env import scrub_for_top_level_launch
    for k, v in POISON.items():
        monkeypatch.setenv(k, v)
    for argv in (["kanban", "comment"], ["chat", "-q", "x"], []):
        assert scrub_for_top_level_launch(argv) == []
    assert os.environ["HERMES_KANBAN_TASK"] == POISON["HERMES_KANBAN_TASK"]


@pytest.mark.parametrize("argv", [["serve"], ["desktop"], ["dashboard"], ["gateway", "run"]])
def test_backend_started_with_marker_is_not_a_delegate_child(monkeypatch, argv):
    from agent.delegation_context import is_delegated_child_process_context
    from hermes_cli.top_level_env import scrub_for_top_level_launch
    for k, v in POISON.items():
        monkeypatch.setenv(k, v)
    assert is_delegated_child_process_context()
    scrub_for_top_level_launch(argv)
    assert not is_delegated_child_process_context()
    assert not (set(POISON) & set(os.environ))


def test_chat_kanban_comment_succeeds_after_top_level_scrub(monkeypatch, tmp_path):
    """End to end: a chat in a backend launched with the leaked env can comment on a card."""
    home = tmp_path / ".hermes"
    home.mkdir()
    poison = {**POISON, "HERMES_DELEGATED_CHILD_CONTEXT": str(home),
              "HERMES_KANBAN_DB": str(home / "kanban.db")}
    for k, v in poison.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    def comment(tid: str) -> dict:
        try:
            return json.loads(kt._handle_comment({"task_id": tid, "body": "from chat"}))
        except Exception as exc:  # a _Reject propagates as an exception in some paths
            return {"error": str(exc)}

    from hermes_cli.top_level_env import scrub_for_top_level_launch
    scrub_for_top_level_launch(["serve"])
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="chat target", assignee="someone")
    finally:
        conn.close()
    out = comment(tid)
    assert out.get("ok") is True, out


def _ts_list(name: str, src: str) -> set[str]:
    m = re.search(rf"export const {name}\s*=\s*\[(.*?)\]", src, re.DOTALL)
    assert m, f"{name} missing from leaked-worker-env.ts"
    return set(re.findall(r"'([A-Z0-9_]+)'", m.group(1)))


def test_electron_mirror_matches_python_scope_list():
    from agent import delegation_context as dc
    src = (REPO / "apps/desktop/electron/leaked-worker-env.ts").read_text()
    assert _ts_list("WORKER_MARKER_KEYS", src) == set(dc.AGENT_SCOPE_MARKER_KEYS)
    assert _ts_list("AGENT_SCOPE_EXACT_KEYS", src) == set(dc.AGENT_SCOPE_EXACT_KEYS)
    assert _ts_list("AGENT_SCOPE_PREFIXES", src) == set(dc.AGENT_SCOPE_PREFIXES)
    m = re.search(r"AGENT_SCOPE_TUNABLE_RE\s*=\s*/(.*?)/", src)
    assert m and m.group(1) == dc.AGENT_SCOPE_TUNABLE_RE.pattern
