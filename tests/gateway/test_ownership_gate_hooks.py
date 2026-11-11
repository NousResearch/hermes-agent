"""#47 Phase 2 — C1/C2 Direct-dispatch ownership gate (hook layer).

Unit tests for ``gateway/ownership_gate.py`` + the ``ownership-gate`` plugin
wiring. Every write lands in a private ``tmp_path`` ledger; the production DB is
never touched. The integration suite
(``tests/gateway/test_cross_mode_integration.py``) pins the underlying ledger
and the batch-scan algorithm; this module pins the HOOK contract:

  (a) pre_tool_call claims a Direct dispatch (all scopes) when enforcement is ON
  (b) the batch ``tasks[]`` union is scanned — a nested conflict is NOT bypassed
  (c) a missing / broken ledger FAILS CLOSED (block, never dispatch unowned)
  (d) post_tool_call releases the claim when the call failed (and keeps it on success)
  (e) enforcement defaults OFF — the plugin is inert until the flag is enabled
  (f) the plugin manifest + register() wire both hooks
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cross_session_ownership import (
    STATE_ACTIVE,
    VERDICT_FREE,
    VERDICT_OWNED,
    claim_for_dispatch,
    claim_ownership,
    init_schema,
    inspect_ownership,
    list_ownership,
    release_ownership,
)
from gateway import ownership_gate as gate

REPO_OWNER = "Andrew288369"
REPO_NAME = "ANDREW-GRP"
SCOPE_33 = "gh:andrew288369/andrew-grp#33"
SCOPE_39 = "gh:andrew288369/andrew-grp#39"
SCOPE_47 = "gh:andrew288369/andrew-grp#47"


@pytest.fixture(autouse=True)
def _private_ledger(tmp_path, monkeypatch):
    """Point the ownership ledger at a private tmp_path DB; reset module state."""
    db = tmp_path / "ledger.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", raising=False)
    # The hook verifies the claimed session_id against the AMBIENT session
    # identity; inherit the runner's own HERMES_SESSION_ID and every positive
    # case would fail closed for the wrong reason.
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    gate.reset_state()
    yield db
    gate.reset_state()


def _seed(db_path, scope, session_id):
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    claim_ownership(conn, scope, session_id, ttl_seconds=900)
    conn.close()


def _args(goal, context="", tasks=None):
    batch = {"goal": goal, "context": context}
    if tasks is not None:
        batch["tasks"] = tasks
    return {"batch": batch}


# --------------------------------------------------------------------------
# (e) enforcement defaults OFF — the plugin is inert until enabled
# --------------------------------------------------------------------------
def test_e_enforcement_defaults_off_and_flag_is_readonly_env():
    assert gate.enforcement_enabled() is False
    # OFF: a dispatch-shaped call does not touch the ledger at all.
    res = gate.pre_tool_call(
        tool_name="delegate_task", args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-1",
    )
    assert res is None
    assert gate.claimed_scopes("call-1") == []


# --------------------------------------------------------------------------
# (a) pre_tool_call claims a Direct dispatch
# --------------------------------------------------------------------------
def test_a_pre_tool_call_claims_direct_dispatch(_private_ledger, monkeypatch):
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47 per directive"),
        session_id="session-A", tool_call_id="call-a",
    )
    assert res is None, res
    ins = inspect_ownership(_open(_private_ledger), SCOPE_47)
    assert ins["verdict"] == VERDICT_OWNED and ins["owner"] == "session-A"
    assert gate.claimed_scopes("call-a") == [SCOPE_47]


def test_a2_reexported_claim_for_dispatch_is_the_ledger_primitive():
    assert gate.claim_for_dispatch is claim_for_dispatch


# --------------------------------------------------------------------------
# (b) the batch tasks[] union is scanned — no nested bypass
# --------------------------------------------------------------------------
def test_b_batch_nested_goal_is_scanned_not_bypassed(_private_ledger, monkeypatch):
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    _seed(_private_ledger, SCOPE_33, "session-other")

    # Top-level goal is a FREE issue (#47); the CONFLICT (#33) hides in tasks[].
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args(
            "Work Andrew288369/ANDREW-GRP#47 per directive",
            tasks=[
                {"goal": "Review Andrew288369/ANDREW-GRP#39", "context": ""},
                {"goal": "Implement Andrew288369/ANDREW-GRP#33", "context": ""},
            ],
        ),
        session_id="session-batch", tool_call_id="call-b",
    )
    assert isinstance(res, dict) and res.get("action") == "block", res
    assert "gh:andrew288369/andrew-grp#33" in res["message"], res
    # all-or-nothing: the FREE #47 was NOT claimed by the blocked batch.
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_FREE
    assert gate.claimed_scopes("call-b") == []
    rows = list_ownership(_open(_private_ledger), active_only=False)
    assert len(rows) == 1 and rows[0]["owner_session"] == "session-other"


def test_b2_scan_dispatch_scope_matches_the_tested_union():
    batch = {
        "goal": "Work Andrew288369/ANDREW-GRP#47",
        "context": "",
        "tasks": [
            {"goal": "Review Andrew288369/ANDREW-GRP#39", "context": ""},
            {"goal": "Implement Andrew288369/ANDREW-GRP#33", "context": ""},
        ],
    }
    assert gate.scan_dispatch_scope(batch) == [
        "gh:andrew288369/andrew-grp#33",
        "gh:andrew288369/andrew-grp#39",
        "gh:andrew288369/andrew-grp#47",
    ]
    # include_tasks=False reproduces the v1 defect (F2) the union exists to fix.
    assert gate.scan_dispatch_scope(batch, include_tasks=False) == [
        "gh:andrew288369/andrew-grp#47",
    ]


def test_b3_scan_dispatch_scope_parses_json_string_tasks():
    """P1-B: small models emit ``tasks`` as a JSON-array string. The scan must
    parse it with delegate_task's normalizer; otherwise the string is iterated
    char-by-char, no element is a dict, and every tasks[] scope is skipped."""
    import json

    batch = {
        "goal": "Work Andrew288369/ANDREW-GRP#47",
        "context": "",
        "tasks": json.dumps(
            [
                {"goal": "Review Andrew288369/ANDREW-GRP#39", "context": ""},
                {"goal": "Implement Andrew288369/ANDREW-GRP#33", "context": ""},
            ]
        ),
    }
    assert gate.scan_dispatch_scope(batch) == [
        "gh:andrew288369/andrew-grp#33",
        "gh:andrew288369/andrew-grp#39",
        "gh:andrew288369/andrew-grp#47",
    ]
    # Parsing must not leak tasks[] into the include_tasks=False bypass path.
    assert gate.scan_dispatch_scope(batch, include_tasks=False) == [
        "gh:andrew288369/andrew-grp#47",
    ]


def test_b4_json_string_tasks_conflict_is_not_bypassed(_private_ledger, monkeypatch):
    """P1-B bypass regression: a conflicting issue hidden in a JSON-string tasks[]
    list must still block the batch (not scan as 0 scopes and dispatch unowned)."""
    import json

    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    _seed(_private_ledger, SCOPE_33, "session-other")
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args(
            "Work Andrew288369/ANDREW-GRP#47 per directive",
            tasks=json.dumps(
                [{"goal": "Implement Andrew288369/ANDREW-GRP#33", "context": ""}]
            ),
        ),
        session_id="session-batch", tool_call_id="call-b4",
    )
    assert isinstance(res, dict) and res.get("action") == "block", res
    assert "gh:andrew288369/andrew-grp#33" in res["message"], res
    assert gate.claimed_scopes("call-b4") == []
    # all-or-nothing: the FREE #47 was NOT claimed by the blocked batch.
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_FREE


# --------------------------------------------------------------------------
# (c) a missing / broken ledger FAILS CLOSED
# --------------------------------------------------------------------------
def test_c_missing_ledger_fails_closed(tmp_path, monkeypatch):
    # A path whose parent is a regular file -> sqlite3 cannot open it.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(blocker / "nested" / "ledger.db"))
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")

    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-c",
    )
    assert isinstance(res, dict) and res.get("action") == "block", res
    assert "ledger" in res["message"].lower(), res
    assert gate.claimed_scopes("call-c") == []


def test_c2_broken_ledger_schema_fails_closed(tmp_path, monkeypatch):
    db = tmp_path / "broken.db"
    conn = sqlite3.connect(db, timeout=30, isolation_level=None)
    # Incompatible pre-existing table: init_schema's IF NOT EXISTS is a no-op,
    # the claim then fails. The gate must block, not dispatch unowned.
    conn.execute("CREATE TABLE task_ownership (scope_key TEXT)")
    conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")

    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-c2",
    )
    assert isinstance(res, dict) and res.get("action") == "block", res
    assert gate.claimed_scopes("call-c2") == []


# --------------------------------------------------------------------------
# (d) post_tool_call releases on failure (fail-open), keeps on success
# --------------------------------------------------------------------------
def test_d_post_tool_call_releases_claim_on_error(_private_ledger, monkeypatch):
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    assert gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d",
    ) is None
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_OWNED

    gate.post_tool_call(
        tool_name="delegate_task", args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d",
        status="error", error_type="tool_error", error_message="boom",
    )
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_FREE
    assert gate.claimed_scopes("call-d") == []
    # a foreign session can now take the scope: the release really landed.
    assert claim_ownership(_open(_private_ledger), SCOPE_33, "session-other")["ok"] is True


def test_d2_post_tool_call_success_keeps_the_claim(_private_ledger, monkeypatch):
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    assert gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d2",
    ) is None
    gate.post_tool_call(
        tool_name="delegate_task", args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d2", status="ok",
    )
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_OWNED
    assert gate.claimed_scopes("call-d2") == [SCOPE_47]


def test_d3_post_tool_call_release_is_fail_open(_private_ledger, monkeypatch):
    """A release that cannot reach the ledger must not raise into the tool result."""
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    assert gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d3",
    ) is None
    # Break the ledger path AFTER the claim: the release must be swallowed.
    blocker = _private_ledger.parent / "blocker2"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(blocker / "nested" / "ledger.db"))
    gate.post_tool_call(
        tool_name="delegate_task", args=_args("Work Andrew288369/ANDREW-GRP#47"),
        session_id="session-A", tool_call_id="call-d3", status="error",
        error_type="tool_error",
    )
    # fail-open: no exception propagated (this line is reached), state cleared.
    assert gate.claimed_scopes("call-d3") == []


# --------------------------------------------------------------------------
# (f) the plugin manifest + register() wire both hooks
# --------------------------------------------------------------------------
def test_f_plugin_manifest_and_register_wire_both_hooks():
    import sys

    root = Path(gate.__file__).resolve().parents[1]
    manifest = root / "plugins" / "ownership-gate" / "plugin.yaml"
    text = manifest.read_text(encoding="utf-8")
    assert "kind: backend" in text
    assert "pre_tool_call" in text and "post_tool_call" in text

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib

    mod = importlib.import_module("plugins.ownership-gate")
    registered = {}

    class _Ctx:
        def register_hook(self, name, cb):
            registered[name] = cb

    mod.register(_Ctx())
    assert registered["pre_tool_call"] is gate.pre_tool_call
    assert registered["post_tool_call"] is gate.post_tool_call


def _open(db_path):
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn
