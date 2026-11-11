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
  (g) the Direct-scanner repo derivation: a non-default dispatch-context repo is
      honoured for bare ``GH-n`` / ``#n`` refs, and ``DEFAULT_REPO_*`` applies
      ONLY when the context carries no repo at all
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


# --------------------------------------------------------------------------
# (g) Direct-scanner repo derivation — a non-default dispatch-context repo is
#     honoured; DEFAULT_REPO_* applies ONLY when the context carries no repo.
# --------------------------------------------------------------------------
OTHER_OWNER = "otherowner"
OTHER_REPO = "other-repo"
SCOPE_OTHER_47 = "gh:otherowner/other-repo#47"
SCOPE_OTHER_48 = "gh:otherowner/other-repo#48"


def test_g_scopes_from_text_derives_a_non_default_repo():
    """A bare ``GH-n`` / ``#n`` ref takes its repo from the dispatch context,
    in every accepted shape — not from DEFAULT_REPO_*."""
    assert gate._scopes_from_text("Work GH-47 now", "otherowner/other-repo") == {SCOPE_OTHER_47}
    assert gate._scopes_from_text("Work GH-47 now", ("otherowner", "other-repo")) == {SCOPE_OTHER_47}
    assert gate._scopes_from_text(
        "Work GH-47 now", {"owner": "otherowner", "name": "other-repo"}
    ) == {SCOPE_OTHER_47}
    assert gate._scopes_from_text("Work otherowner/other-repo#47 now", "otherowner/other-repo") == {SCOPE_OTHER_47}
    # Every ref in the text lands on the context repo.
    assert gate._scopes_from_text("GH-47 and GH-48", "otherowner/other-repo") == {
        SCOPE_OTHER_47, SCOPE_OTHER_48,
    }
    # Ref matching is unchanged: the same refs are found, only the repo differs.
    assert gate._scopes_from_text("GH-47 and GH-48") == {
        "gh:andrew288369/andrew-grp#47", "gh:andrew288369/andrew-grp#48",
    }


def test_g2_scopes_from_text_falls_back_to_default_repo_only_without_context():
    """The default repo is a FALLBACK, not an override: it applies only when the
    context carries no repo at all (unset, None, or an unparseable value)."""
    assert (gate.DEFAULT_REPO_OWNER, gate.DEFAULT_REPO_NAME) == (REPO_OWNER, REPO_NAME)
    assert gate._scopes_from_text("Work GH-47 now") == {SCOPE_47}
    assert gate._scopes_from_text("Work GH-47 now", None) == {SCOPE_47}
    # Unparseable context values are treated as "no context".
    assert gate._scopes_from_text("Work GH-47 now", "not-a-repo-slug") == {SCOPE_47}
    assert gate._scopes_from_text("Work GH-47 now", {"owner": "otherowner"}) == {SCOPE_47}
    assert gate._scopes_from_text("Work GH-47 now", ("otherowner",)) == {SCOPE_47}


def test_g3_scan_dispatch_scope_derives_the_repo_per_text():
    """``scan_dispatch_scope`` resolves the repo per text: explicit
    ``default_repo``, else the batch's own repo field, else the individual
    ``tasks[]`` entry's repo field, else ``DEFAULT_REPO_*``."""
    batch = {"goal": "Work GH-47", "context": ""}
    assert gate.scan_dispatch_scope(batch, default_repo="otherowner/other-repo") == [SCOPE_OTHER_47]
    # No context at all -> DEFAULT_REPO_*.
    assert gate.scan_dispatch_scope(batch) == [SCOPE_47]
    # The batch's OWN repo field is honoured even without ``default_repo``.
    batch_repo = {"goal": "Work GH-47", "context": "", "repo": "otherowner/other-repo"}
    assert gate.scan_dispatch_scope(batch_repo) == [SCOPE_OTHER_47]
    # An explicit ``default_repo`` outranks the batch's own repo field.
    batch_both = {"goal": "Work GH-47", "context": "", "repo": "someone/else"}
    assert gate.scan_dispatch_scope(batch_both, default_repo="otherowner/other-repo") == [SCOPE_OTHER_47]
    # A tasks[] entry's own repo applies to its own text; the top-level goal
    # keeps the batch/default repo.
    batch_tasks = {
        "goal": "Work GH-47",
        "context": "",
        "tasks": [{"goal": "Review GH-48", "context": "", "source_repo": "otherowner/other-repo"}],
    }
    assert gate.scan_dispatch_scope(batch_tasks) == [SCOPE_47, SCOPE_OTHER_48]
    # A tasks[] entry with no repo inherits the batch repo.
    batch_tasks_inherit = {
        "goal": "Work GH-47",
        "context": "",
        "tasks": [{"goal": "Review GH-48", "context": ""}],
    }
    assert gate.scan_dispatch_scope(batch_tasks_inherit, default_repo="otherowner/other-repo") == [
        SCOPE_OTHER_47, SCOPE_OTHER_48,
    ]


def test_g4_pre_tool_call_claims_the_context_repo_not_default(_private_ledger, monkeypatch):
    """With enforcement ON, ``pre_tool_call(..., default_repo=...)`` claims the
    bare ref on the CONTEXT repo — the default repo's same-number issue is
    left free."""
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work GH-47 per directive"),
        session_id="session-A", tool_call_id="call-g4",
        default_repo="otherowner/other-repo",
    )
    assert res is None, res
    assert gate.claimed_scopes("call-g4") == [SCOPE_OTHER_47]
    assert inspect_ownership(_open(_private_ledger), SCOPE_OTHER_47)["verdict"] == VERDICT_OWNED
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_FREE


def test_g5_pre_tool_call_falls_back_to_default_repo_without_context(_private_ledger, monkeypatch):
    """The same call with NO context repo claims DEFAULT_REPO_* — the fallback
    the context repo is meant to replace."""
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work GH-47 per directive"),
        session_id="session-A", tool_call_id="call-g5",
    )
    assert res is None, res
    assert gate.claimed_scopes("call-g5") == [SCOPE_47]
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_OWNED
    assert inspect_ownership(_open(_private_ledger), SCOPE_OTHER_47)["verdict"] == VERDICT_FREE


def test_g6_pre_tool_call_context_repo_changes_the_conflict_key(_private_ledger, monkeypatch):
    """The derivation is not cosmetic: a foreign owner on the CONTEXT repo blocks
    the dispatch, while the same-number issue on the default repo stays free
    (and the foreign owner on the other repo must not block a default-repo
    dispatch)."""
    monkeypatch.setenv("HERMES_OWNERSHIP_DIRECT_ENFORCE", "1")
    _seed(_private_ledger, SCOPE_OTHER_47, "session-other")

    # The dispatch is about the OTHER repo -> blocked on the other repo's key.
    res = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work GH-47 per directive"),
        session_id="session-A", tool_call_id="call-g6a",
        default_repo="otherowner/other-repo",
    )
    assert isinstance(res, dict) and res.get("action") == "block", res
    assert SCOPE_OTHER_47 in res["message"], res
    assert gate.claimed_scopes("call-g6a") == []
    # all-or-nothing: neither repo's #47 was claimed.
    assert inspect_ownership(_open(_private_ledger), SCOPE_47)["verdict"] == VERDICT_FREE

    # The same text with NO context repo claims the DEFAULT repo's #47, which is
    # free: the foreign owner on the other repo must not block it.
    res2 = gate.pre_tool_call(
        tool_name="delegate_task",
        args=_args("Work GH-47 per directive"),
        session_id="session-A", tool_call_id="call-g6b",
    )
    assert res2 is None, res2
    assert gate.claimed_scopes("call-g6b") == [SCOPE_47]


def _open(db_path):
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn
