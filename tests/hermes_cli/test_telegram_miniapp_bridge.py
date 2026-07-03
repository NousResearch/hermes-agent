"""Tests for the Telegram Mini App action-gate bridge core.

The bridge is a file-backed, HMAC-authenticated contract between the sidecar
(which posts owner decisions) and the gateway (which resolves them). Everything
here is pure and disabled-by-default; no live gateway is involved.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.telegram_miniapp import bridge as B


BOT_TOKEN = "123456:test-token"


def _pending(session_key="sess-a", command="rm -rf /data", *, requested_at=1_700_000_000, expires_at=1_700_000_300):
    return {
        "session_key": session_key,
        "command": command,
        "description": "Удалить каталог данных",
        "pattern_key": "rm-rf",
        "risk_tier": "critical",
        "requested_at": requested_at,
        "expires_at": expires_at,
    }


def test_bridge_key_depends_on_bot_token():
    assert B.derive_bridge_key(BOT_TOKEN) == B.derive_bridge_key(BOT_TOKEN)
    assert B.derive_bridge_key(BOT_TOKEN) != B.derive_bridge_key("999:other")
    assert isinstance(B.derive_bridge_key(BOT_TOKEN), bytes)


def test_opaque_id_is_stable_and_hides_session_key():
    key = B.derive_bridge_key(BOT_TOKEN)
    a = B.opaque_approval_id(key, session_key="sess-a", command="rm -rf /data", seq=0)
    b = B.opaque_approval_id(key, session_key="sess-a", command="rm -rf /data", seq=0)
    assert a == b
    # Different session/command/seq → different id (timestamp does NOT change it).
    assert a != B.opaque_approval_id(key, session_key="sess-b", command="rm -rf /data", seq=0)
    assert a != B.opaque_approval_id(key, session_key="sess-a", command="ls", seq=0)
    assert a != B.opaque_approval_id(key, session_key="sess-a", command="rm -rf /data", seq=1)
    # The id leaks neither the session key nor the command.
    assert "sess-a" not in a
    assert "rm" not in a and "data" not in a


def test_snapshot_version_stable_across_ticks_for_unchanged_queue():
    # Real gateway approvals have no requested_at; the version must not churn
    # tick-to-tick or in-flight decisions would always go stale.
    key = B.derive_bridge_key(BOT_TOKEN)
    queue = [{"session_key": "sess-a", "command": "rm -rf /data", "risk_tier": "critical"}]
    v1 = B.project_snapshot(key, queue, now=1_700_000_100).public["snapshot_version"]
    v2 = B.project_snapshot(key, queue, now=1_700_000_500).public["snapshot_version"]
    assert v1 == v2


def test_identical_commands_same_second_get_distinct_ids():
    key = B.derive_bridge_key(BOT_TOKEN)
    projected = B.project_snapshot(
        key,
        [
            _pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_000),
            _pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_000),
        ],
        now=1_700_000_100,
    )
    ids = [item["approval_id"] for item in projected.public["items"]]
    assert len(set(ids)) == 2  # per-entry seq disambiguates
    assert len(projected.index) == 2


def test_unknown_risk_tier_floors_to_critical():
    # Fail-closed: a missing / spoofed / unrecognized risk_tier must surface as
    # the MOST dangerous tier, never a safe-looking one — while a legitimate
    # low tier is preserved (not over-escalated).
    key = B.derive_bridge_key(BOT_TOKEN)
    for bad in [None, "", "safe", "read_only_but_lying", "APPROVE", 123]:
        item = B.project_snapshot(
            key, [{"session_key": "s", "command": "c", "risk_tier": bad}], now=1_700_000_100
        ).public["items"][0]
        assert item["risk"] == "critical", bad
        assert item["title"] == "Опасная команда"
    ok = B.project_snapshot(
        key, [{"session_key": "s", "command": "c", "risk_tier": "read_only"}], now=1_700_000_100
    ).public["items"][0]
    assert ok["risk"] == "read_only"  # a valid low tier is NOT escalated


def test_snapshot_projection_redacts_all_sensitive_fields():
    key = B.derive_bridge_key(BOT_TOKEN)
    projected = B.project_snapshot(key, [_pending()], now=1_700_000_100)
    serialized = json.dumps(projected.public)

    for forbidden in ("rm -rf", "/data", "sess-a", "rm-rf", "session_key", "pattern_key", "command", BOT_TOKEN):
        assert forbidden not in serialized, f"snapshot leaked {forbidden!r}"

    assert projected.public["snapshot_version"]
    item = projected.public["items"][0]
    assert set(item) == {
        "approval_id",
        "title",
        "source_label",
        "risk",
        "summary",
        "requested_at",
        "expires_at",
        "allowed_decisions",
    }
    assert item["allowed_decisions"] == ["approve_once", "reject_once"]
    # Title/summary are fixed risk-tier copy, never the raw upstream description.
    assert item["title"] == "Опасная команда"
    assert "Удалить каталог данных" not in serialized
    # Index maps opaque id back to the real session key + FIFO head flag, and
    # stays private.
    ref = projected.index[item["approval_id"]]
    assert ref.session_key == "sess-a" and ref.is_head is True


def test_snapshot_marks_only_oldest_per_session_as_head():
    key = B.derive_bridge_key(BOT_TOKEN)
    projected = B.project_snapshot(
        key,
        [
            _pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_000),
            _pending(session_key="sess-a", command="drop table", requested_at=1_700_000_050),
            _pending(session_key="sess-b", command="ls", requested_at=1_700_000_010),
        ],
        now=1_700_000_100,
    )
    heads = {aid: ref.is_head for aid, ref in projected.index.items()}
    assert list(heads.values()).count(True) == 2  # one head per session


def test_snapshot_version_changes_with_content():
    key = B.derive_bridge_key(BOT_TOKEN)
    v1 = B.project_snapshot(key, [_pending()], now=1_700_000_100).public["snapshot_version"]
    v2 = B.project_snapshot(key, [_pending(), _pending(session_key="sess-b", command="ls")], now=1_700_000_100).public["snapshot_version"]
    assert v1 != v2


def _decision_envelope(key, *, approval_id, decision="approve_once", snapshot_version, client_request_id="uuid-1", issued_at=1_700_000_110):
    payload = {
        "approval_id": approval_id,
        "decision": decision,
        "client_request_id": client_request_id,
        "snapshot_version": snapshot_version,
        "issued_at": issued_at,
    }
    return B.sign_envelope(key, payload)


def test_verify_envelope_rejects_tampering():
    key = B.derive_bridge_key(BOT_TOKEN)
    env = _decision_envelope(key, approval_id="abc", snapshot_version="v1")
    assert B.verify_envelope(key, env) is not None
    # Wrong key.
    assert B.verify_envelope(B.derive_bridge_key("999:other"), env) is None
    # Tampered payload with the old signature.
    tampered = {"payload": {**env["payload"], "decision": "approve_all"}, "sig": env["sig"]}
    assert B.verify_envelope(key, tampered) is None


def test_validate_decision_fail_closed_paths():
    key = B.derive_bridge_key(BOT_TOKEN)
    projected = B.project_snapshot(key, [_pending()], now=1_700_000_100)
    version = projected.public["snapshot_version"]
    approval_id = projected.public["items"][0]["approval_id"]
    applied: set[str] = set()

    def validate(env, *, now=1_700_000_120, cur_version=version, idx=projected.index):
        return B.validate_decision(
            key, env, current_snapshot_version=cur_version, index=idx, now=now, applied=applied
        )

    good = _decision_envelope(key, approval_id=approval_id, snapshot_version=version)
    ok = validate(good)
    assert ok.accepted and ok.session_key == "sess-a" and ok.choice == "once"

    # Bad signature.
    bad_sig = {"payload": good["payload"], "sig": "00" * 32}
    assert not validate(bad_sig).accepted

    # Expired issued_at (TTL 120s).
    stale = _decision_envelope(key, approval_id=approval_id, snapshot_version=version, issued_at=1_700_000_000, client_request_id="uuid-stale")
    assert not validate(stale, now=1_700_000_300).accepted

    # Stale snapshot version.
    old = _decision_envelope(key, approval_id=approval_id, snapshot_version="v-old", client_request_id="uuid-old")
    assert not validate(old).accepted

    # Unknown approval id.
    unknown = _decision_envelope(key, approval_id="deadbeef", snapshot_version=version, client_request_id="uuid-unknown")
    assert not validate(unknown).accepted

    # Disallowed decision.
    allrq = _decision_envelope(key, approval_id=approval_id, decision="approve_all", snapshot_version=version, client_request_id="uuid-all")
    assert not validate(allrq).accepted

    # reject_once maps to deny.
    rej = _decision_envelope(key, approval_id=approval_id, decision="reject_once", snapshot_version=version, client_request_id="uuid-rej")
    r = validate(rej)
    assert r.accepted and r.choice == "deny"


def test_validate_rejects_non_head_approval():
    key = B.derive_bridge_key(BOT_TOKEN)
    projected = B.project_snapshot(
        key,
        [
            _pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_000),
            _pending(session_key="sess-a", command="drop table", requested_at=1_700_000_050),
        ],
        now=1_700_000_100,
    )
    version = projected.public["snapshot_version"]
    # The second item shares the session and is not FIFO-head.
    non_head_id = next(aid for aid, ref in projected.index.items() if not ref.is_head)
    env = _decision_envelope(key, approval_id=non_head_id, snapshot_version=version)
    result = B.validate_decision(
        key, env, current_snapshot_version=version, index=projected.index, now=1_700_000_120, applied=set()
    )
    assert not result.accepted and result.status == "rejected_not_head"


def test_check_target_rejects_stale_or_mismatched_snapshot(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_090)
    snap = br.read_public_snapshot(now=1_700_000_100)
    approval_id = snap["items"][0]["approval_id"]
    version = snap["snapshot_version"]

    assert br.check_target(approval_id, version, now=1_700_000_100) is True
    # Wrong version, unknown id, and an expired snapshot all fail closed.
    assert br.check_target(approval_id, "v-old", now=1_700_000_100) is False
    assert br.check_target("deadbeef", version, now=1_700_000_100) is False
    assert br.check_target(approval_id, version, now=1_700_009_999) is False


def test_process_decisions_resolves_once_and_is_idempotent(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    approval_id = snap["items"][0]["approval_id"]
    version = snap["snapshot_version"]

    env = _decision_envelope(key, approval_id=approval_id, snapshot_version=version)
    br.submit_decision(env)

    resolved: list[tuple[str, str]] = []

    def resolver(session_key, choice, *, expected_approval_id):
        resolved.append((session_key, choice))
        return 1

    receipts = br.process_pending_decisions(resolver, now=1_700_000_120)
    assert resolved == [("sess-a", "once")]
    assert receipts and receipts[0]["status"] == "accepted"

    # Re-submitting the same client_request_id must NOT resolve twice.
    br.submit_decision(env)
    resolved.clear()
    br.process_pending_decisions(resolver, now=1_700_000_121)
    assert resolved == []


def test_process_rejects_when_live_head_changed(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    approval_id = snap["items"][0]["approval_id"]
    env = _decision_envelope(key, approval_id=approval_id, snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)

    resolved: list[tuple[str, str]] = []

    # The live head is now a *different* approval id (queue changed).
    receipts = br.process_pending_decisions(
        lambda s, c, *, expected_approval_id: resolved.append((s, c)) or 1,
        now=1_700_000_120,
        current_head=lambda session_key: "some-other-head-id",
    )
    assert resolved == []
    assert receipts and receipts[0]["status"] == "rejected_not_head"


def test_resolver_receives_expected_id_for_atomic_cas(tmp_path):
    # The resolver must receive the expected approval id so the gateway can
    # compare-and-pop atomically; if its CAS fails (returns 0), the decision is
    # reported rejected_not_pending even though the early head pre-check passed.
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    approval_id = snap["items"][0]["approval_id"]
    env = _decision_envelope(key, approval_id=approval_id, snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)

    seen: list = []

    def cas_resolver(session_key, choice, *, expected_approval_id):
        seen.append(expected_approval_id)
        return 0  # simulate: head changed between check and pop

    receipts = br.process_pending_decisions(cas_resolver, now=1_700_000_120)
    assert seen == [approval_id]  # resolver got the exact target
    assert receipts and receipts[0]["status"] == "rejected_not_pending"


def test_positional_resolver_fails_closed_no_resolve_all(tmp_path):
    # A resolver with the raw resolve_gateway_approval shape (positional third
    # arg = resolve_all) must NOT be usable: expected_approval_id is passed
    # keyword-only, so such a resolver raises TypeError and the decision fails
    # closed instead of resolving the whole session.
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    env = _decision_envelope(key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)

    resolve_all_calls: list = []

    def raw_resolver(session_key, choice, resolve_all=False):
        resolve_all_calls.append(resolve_all)
        return 5  # pretend it unblocked everything

    receipts = br.process_pending_decisions(raw_resolver, now=1_700_000_120)
    assert resolve_all_calls == []  # never invoked as resolve_all
    assert receipts and receipts[0]["status"] == "rejected_resolver_error"


def test_process_reports_no_op_resolution_as_rejected(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    env = _decision_envelope(key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)

    # Resolver returns 0 → nothing was pending (timed out / resolved elsewhere).
    receipts = br.process_pending_decisions(lambda s, c, *, expected_approval_id: 0, now=1_700_000_120)
    assert receipts and receipts[0]["status"] == "rejected_not_pending"


def test_end_to_end_cycle_export_sign_process_resolve(tmp_path):
    """Full loop: gateway exports -> sidecar signs a decision -> gateway
    processes and resolves, with the default (fail-closed) live-head check."""
    key = B.derive_bridge_key(BOT_TOKEN)
    gateway = B.MiniAppBridge(tmp_path, key)
    sidecar = B.MiniAppBridge(tmp_path, key)  # separate process, same files+key

    queue = [_pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_050)]
    gateway.export(queue, now=1_700_000_090)

    # Sidecar reads the signed snapshot and signs a decision for the head.
    snap = sidecar.read_public_snapshot(now=1_700_000_100)
    approval_id = snap["items"][0]["approval_id"]
    env = B.sign_envelope(
        key,
        {
            "approval_id": approval_id,
            "decision": "approve_once",
            "client_request_id": "uuid-e2e",
            "snapshot_version": snap["snapshot_version"],
            "issued_at": 1_700_000_105,
        },
    )
    sidecar.submit_decision(env)

    resolved: list[tuple[str, str]] = []
    # Gateway tick: re-export the (unchanged) queue and process decisions.
    receipts = gateway.run_cycle(queue, lambda s, c, *, expected_approval_id: resolved.append((s, c)) or 1, now=1_700_000_110)
    assert resolved == [("sess-a", "once")]
    assert receipts and receipts[0]["status"] == "accepted"


def test_process_fail_closed_without_explicit_head(tmp_path):
    # Default head check runs even when no callback is passed: a decision for a
    # non-head approval is refused.
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export(
        [
            _pending(session_key="sess-a", command="rm -rf /data", requested_at=1_700_000_000),
            _pending(session_key="sess-a", command="drop table", requested_at=1_700_000_050),
        ],
        now=1_700_000_090,
    )
    snap = br.read_public_snapshot(now=1_700_000_100)
    # Sign for the second (non-head) item.
    non_head = snap["items"][1]["approval_id"]
    env = _decision_envelope(key, approval_id=non_head, snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)

    resolved: list = []
    receipts = br.process_pending_decisions(lambda s, c, *, expected_approval_id: resolved.append((s, c)) or 1, now=1_700_000_110)
    assert resolved == []
    assert receipts and receipts[0]["status"] == "rejected_not_head"


def test_audit_log_is_redacted(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    env = _decision_envelope(key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)
    br.process_pending_decisions(lambda s, c, *, expected_approval_id: 1, now=1_700_000_120)

    audit = (tmp_path / "miniapp" / "audit.jsonl").read_text(encoding="utf-8")
    for forbidden in ("rm -rf", "/data", "sess-a", "rm-rf", BOT_TOKEN):
        assert forbidden not in audit, f"audit leaked {forbidden!r}"


def test_read_snapshot_rejects_tampered_body(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_090)
    snap_path = tmp_path / "miniapp" / "approvals_snapshot.json"

    # Untampered snapshot reads back fine.
    assert br.read_public_snapshot(now=1_700_000_100) is not None

    # A local process without the bot token downgrades the displayed risk while
    # keeping snapshot_version/approval_id and the original signature — must be
    # rejected as unauthenticated.
    envelope = json.loads(snap_path.read_text())
    envelope["payload"]["items"][0]["risk"] = "read_only"
    envelope["payload"]["items"][0]["title"] = "Безобидное чтение"
    snap_path.write_text(json.dumps(envelope))
    assert br.read_public_snapshot(now=1_700_000_100) is None


def test_bridge_directory_permissions(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    mode = (tmp_path / "miniapp").stat().st_mode & 0o777
    assert mode == 0o700
    snap_mode = (tmp_path / "miniapp" / "approvals_snapshot.json").stat().st_mode & 0o777
    assert snap_mode == 0o600


# ── M29: gateway-side seam (CAS adapter, persistent replay, sweeper, e2e) ──

def _clear_gateway_queues():
    from tools import approval as A
    with A._lock:
        A._gateway_queues.clear()


def test_cas_resolver_pops_only_matching_head():
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        entry = A._ApprovalEntry({"command": "rm -rf /data"})
        A._gateway_queues["sess-a"] = [entry]
        head_id = B.opaque_approval_id(key, session_key="sess-a", command="rm -rf /data", seq=0)

        # Wrong expected id → nothing resolved, head untouched.
        assert A.resolve_gateway_approval_cas("sess-a", "once", expected_approval_id="deadbeef", bridge_key=key) == 0
        assert entry.result is None

        # Correct head id → resolves exactly the head.
        assert A.resolve_gateway_approval_cas("sess-a", "once", expected_approval_id=head_id, bridge_key=key) == 1
        assert entry.result == "once"
        assert "sess-a" not in A._gateway_queues  # queue drained
    finally:
        _clear_gateway_queues()


def test_cas_resolver_returns_zero_on_empty_queue():
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    assert A.resolve_gateway_approval_cas("nobody", "once", expected_approval_id="x", bridge_key=key) == 0


def test_end_to_end_seam_resolves_the_selected_head(tmp_path):
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        entry = A._ApprovalEntry({"command": "rm -rf /data"})
        A._gateway_queues["sess-a"] = [entry]
        pendings = [{"session_key": "sess-a", "command": "rm -rf /data", "risk_tier": "critical"}]

        br = B.MiniAppBridge(tmp_path, key)
        br.export(pendings, now=1_700_000_100)
        snap = br.read_public_snapshot(now=1_700_000_110)
        approval_id = snap["items"][0]["approval_id"]
        env = _decision_envelope(key, approval_id=approval_id, snapshot_version=snap["snapshot_version"])
        br.submit_decision(env)

        def cas(session_key, choice, *, expected_approval_id):
            return A.resolve_gateway_approval_cas(session_key, choice, expected_approval_id=expected_approval_id, bridge_key=key)

        receipts = br.run_cycle(pendings, cas, now=1_700_000_120)
        assert receipts and receipts[0]["status"] == "accepted"
        assert entry.result == "once"  # the real gateway entry was unblocked
        assert "sess-a" not in A._gateway_queues
    finally:
        _clear_gateway_queues()


def test_persistent_replay_guard_survives_restart(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    env = _decision_envelope(key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)
    br.process_pending_decisions(lambda s, c, *, expected_approval_id: 1, now=1_700_000_120)

    # Simulate a restart: fresh bridge (empty in-memory _applied) sees the
    # receipt on disk and must NOT re-resolve a re-submitted decision.
    br2 = B.MiniAppBridge(tmp_path, key)
    br2.export([_pending()], now=1_700_000_130)
    br2.submit_decision(env)
    called: list = []
    receipts = br2.process_pending_decisions(
        lambda s, c, *, expected_approval_id: called.append(1) or 1, now=1_700_000_140
    )
    assert called == []
    assert receipts and receipts[0]["status"] == "already_resolved"


def test_sweeper_removes_expired_decision_files(tmp_path):
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_100)
    env = _decision_envelope(
        key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"], issued_at=1_700_000_000
    )
    br.submit_decision(env)
    assert list((tmp_path / "miniapp" / "decisions").glob("*.json"))

    # Far past the decision TTL → swept.
    swept = br.sweep_stale_decisions(now=1_700_009_999)
    assert swept == 1
    assert not list((tmp_path / "miniapp" / "decisions").glob("*.json"))


def test_sweeper_removes_future_dated_decision(tmp_path):
    # A future-dated decision (issued_at ahead of the gateway clock) is rejected
    # by validate_decision forever (two-sided abs TTL); the sweeper must use the
    # SAME two-sided window so it does not linger as a permanent orphan.
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_100)
    env = _decision_envelope(
        key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"], issued_at=1_700_099_999
    )
    br.submit_decision(env)
    assert list((tmp_path / "miniapp" / "decisions").glob("*.json"))

    swept = br.sweep_stale_decisions(now=1_700_000_100)
    assert swept == 1
    assert not list((tmp_path / "miniapp" / "decisions").glob("*.json"))


def test_authoritative_receipt_not_overwritten_after_snapshot_change(tmp_path):
    # A decision resolved in a prior run leaves an `accepted` receipt. If it is
    # resubmitted after a restart AND the queue/snapshot has changed (so live
    # validation would say stale), the guard must still report already_resolved
    # and MUST NOT overwrite the authoritative `accepted` receipt.
    key = B.derive_bridge_key(BOT_TOKEN)
    br = B.MiniAppBridge(tmp_path, key)
    br.export([_pending()], now=1_700_000_100)
    snap = br.read_public_snapshot(now=1_700_000_110)
    env = _decision_envelope(key, approval_id=snap["items"][0]["approval_id"], snapshot_version=snap["snapshot_version"])
    br.submit_decision(env)
    br.process_pending_decisions(lambda s, c, *, expected_approval_id: 1, now=1_700_000_120)
    receipt_path = next((tmp_path / "miniapp" / "receipts").glob("*.json"))
    original = receipt_path.read_text(encoding="utf-8")
    assert '"accepted"' in original

    # Restart with a CHANGED snapshot, then resubmit the same decision.
    br2 = B.MiniAppBridge(tmp_path, key)
    br2.export([_pending(command="something else entirely")], now=1_700_000_130)
    br2.submit_decision(env)
    called: list = []
    receipts = br2.process_pending_decisions(
        lambda s, c, *, expected_approval_id: called.append(1) or 1, now=1_700_000_140
    )
    assert called == []  # never re-resolved
    assert receipts and receipts[0]["status"] == "already_resolved"
    assert receipt_path.read_text(encoding="utf-8") == original  # accepted receipt preserved


def test_decision_id_is_128_bit():
    key = B.derive_bridge_key(BOT_TOKEN)
    env = _decision_envelope(key, approval_id="abc", snapshot_version="v1")
    assert len(B._decision_id(env["payload"])) == 32


# ── M29-fix: opaque-id seq must be per-session/per-instance, never global ──

def test_projection_prefers_instance_nonce_over_positional_fallback():
    # Guards the export-side half of the fix: project_snapshot MUST fold in the
    # entry's per-instance nonce, NOT the positional fallback. A regression back
    # to positional seq would make `exported` equal the seq-0 id and flip both
    # assertions — which no other test would catch.
    key = B.derive_bridge_key(BOT_TOKEN)
    proj = B.project_snapshot(
        key, [{"session_key": "s", "command": "deploy", "risk_tier": "critical", "_nonce": "abc123"}], now=1_700_000_100
    )
    exported = proj.public["items"][0]["approval_id"]
    assert exported == B.opaque_approval_id(key, session_key="s", command="deploy", nonce="abc123")
    assert exported != B.opaque_approval_id(key, session_key="s", command="deploy", seq=0)


def test_stale_decision_rejected_after_restart_and_reissue(tmp_path):
    # End-to-end for the cross-restart replay: owner signs a decision for one
    # approval instance; the gateway "restarts" (fresh bridge, empty in-memory
    # queue) and an identical command is re-requested as a NEW instance (new
    # nonce). The old signed decision must be rejected, not silently applied to
    # the fresh instance.
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        # Pre-crash: instance with nonce n1, owner signs a decision for it.
        pre = [{"session_key": "sess-a", "command": "deploy", "risk_tier": "critical", "_nonce": "n1"}]
        br1 = B.MiniAppBridge(tmp_path, key)
        br1.export(pre, now=1_700_000_100)
        snap1 = br1.read_public_snapshot(now=1_700_000_100)
        env = _decision_envelope(
            key, approval_id=snap1["items"][0]["approval_id"], snapshot_version=snap1["snapshot_version"]
        )

        # Restart: fresh bridge, empty queue; an identical command reissues with a
        # NEW nonce. The gateway resolver only sees the reissued instance.
        reissued = A._ApprovalEntry({"command": "deploy"}, nonce="n2")
        A._gateway_queues["sess-a"] = [reissued]
        post = [{"session_key": "sess-a", "command": "deploy", "risk_tier": "critical", "_nonce": "n2"}]
        br2 = B.MiniAppBridge(tmp_path, key)
        br2.submit_decision(env)

        def cas(session_key, choice, *, expected_approval_id):
            return A.resolve_gateway_approval_cas(session_key, choice, expected_approval_id=expected_approval_id, bridge_key=key)

        receipts = br2.run_cycle(post, cas, now=1_700_000_120)
        assert receipts and receipts[0]["status"] != "accepted"  # stale decision refused
        assert reissued.result is None  # fresh instance untouched
    finally:
        _clear_gateway_queues()


def test_conflicting_decisions_reject_wins_fail_closed(tmp_path):
    # If the owner submits BOTH approve_once and reject_once for the same
    # approval before a gateway tick (racing taps), the fail-closed DENY must
    # win — never the approve, regardless of decision-id hash ordering.
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        entry = A._ApprovalEntry({"command": "rm -rf /data"}, nonce="n1")
        A._gateway_queues["sess-a"] = [entry]
        pendings = [{"session_key": "sess-a", "command": "rm -rf /data", "risk_tier": "critical", "_nonce": "n1"}]
        br = B.MiniAppBridge(tmp_path, key)
        br.export(pendings, now=1_700_000_100)
        snap = br.read_public_snapshot(now=1_700_000_100)
        aid, ver = snap["items"][0]["approval_id"], snap["snapshot_version"]

        approve = _decision_envelope(key, approval_id=aid, decision="approve_once", snapshot_version=ver, client_request_id="uuid-approve")
        reject = _decision_envelope(key, approval_id=aid, decision="reject_once", snapshot_version=ver, client_request_id="uuid-reject")
        br.submit_decision(approve)
        br.submit_decision(reject)

        def cas(session_key, choice, *, expected_approval_id):
            return A.resolve_gateway_approval_cas(session_key, choice, expected_approval_id=expected_approval_id, bridge_key=key)

        br.run_cycle(pendings, cas, now=1_700_000_120)
        assert entry.result == "deny"  # reject won; the command was NOT approved
    finally:
        _clear_gateway_queues()


def test_snapshot_seq_is_per_session_not_global():
    # Regression for the global-enumerate bug: with two sessions pending, EACH
    # session head must be exported with per-session seq 0 so the gateway CAS
    # resolver (which recomputes the head from its own seq) can match it. A
    # global index would give the second session's head seq 1 and make it
    # permanently unresolvable.
    key = B.derive_bridge_key(BOT_TOKEN)
    proj = B.project_snapshot(
        key,
        [
            {"session_key": "sess-a", "command": "cmd-a", "risk_tier": "critical"},
            {"session_key": "sess-b", "command": "cmd-b", "risk_tier": "critical"},
        ],
        now=1_700_000_100,
    )
    a_id = proj.public["items"][0]["approval_id"]
    b_id = proj.public["items"][1]["approval_id"]
    assert a_id == B.opaque_approval_id(key, session_key="sess-a", command="cmd-a", seq=0)
    assert b_id == B.opaque_approval_id(key, session_key="sess-b", command="cmd-b", seq=0)


def test_cas_resolves_non_index_zero_session_head():
    # The exact multi-session repro the reviewers flagged: the SECOND session's
    # head (not at global export index 0) must still resolve — and must not
    # disturb the first session. Each entry carries its own per-instance nonce,
    # exactly as the real enqueue path assigns it.
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        a = A._ApprovalEntry({"command": "cmd-a"}, nonce="nonce-a")
        b = A._ApprovalEntry({"command": "cmd-b"}, nonce="nonce-b")
        A._gateway_queues["sess-a"] = [a]
        A._gateway_queues["sess-b"] = [b]
        pendings = [
            {"session_key": "sess-a", "command": "cmd-a", "risk_tier": "critical", "_nonce": "nonce-a"},
            {"session_key": "sess-b", "command": "cmd-b", "risk_tier": "critical", "_nonce": "nonce-b"},
        ]
        expected_b = B.project_snapshot(key, pendings, now=1_700_000_100).public["items"][1]["approval_id"]

        assert A.resolve_gateway_approval_cas("sess-b", "once", expected_approval_id=expected_b, bridge_key=key) == 1
        assert b.result == "once"
        assert a.result is None  # untouched
        assert "sess-a" in A._gateway_queues
    finally:
        _clear_gateway_queues()


def test_cas_rejects_stale_decision_for_reissued_command():
    # Instance-binding: a decision signed for a timed-out approval (old nonce)
    # must NOT resolve a freshly re-requested identical command (new nonce) —
    # this holds across turns AND restarts because the nonce is random per
    # instance, never a counter that resets. The current instance's id works.
    from tools import approval as A

    _clear_gateway_queues()
    key = B.derive_bridge_key(BOT_TOKEN)
    try:
        stale_id = B.opaque_approval_id(key, session_key="sess-a", command="deploy", nonce="nonce-old")
        reissued = A._ApprovalEntry({"command": "deploy"}, nonce="nonce-new")
        A._gateway_queues["sess-a"] = [reissued]

        assert A.resolve_gateway_approval_cas("sess-a", "once", expected_approval_id=stale_id, bridge_key=key) == 0
        assert reissued.result is None

        fresh_id = B.opaque_approval_id(key, session_key="sess-a", command="deploy", nonce="nonce-new")
        assert A.resolve_gateway_approval_cas("sess-a", "once", expected_approval_id=fresh_id, bridge_key=key) == 1
        assert reissued.result == "once"
    finally:
        _clear_gateway_queues()


def test_reissued_command_gets_distinct_id_across_snapshots():
    # The stale-decision defense at the projection layer: two DIFFERENT instances
    # of an identical (session, command) — e.g. one that timed out and one
    # re-requested a turn or a restart later — must project to DIFFERENT approval
    # ids, so a decision signed against the first snapshot cannot validate against
    # the second (unknown approval id / changed snapshot_version).
    key = B.derive_bridge_key(BOT_TOKEN)
    first = B.project_snapshot(
        key, [{"session_key": "s", "command": "deploy", "risk_tier": "critical", "_nonce": "n1"}], now=1_700_000_100
    )
    second = B.project_snapshot(
        key, [{"session_key": "s", "command": "deploy", "risk_tier": "critical", "_nonce": "n2"}], now=1_700_000_200
    )
    assert first.public["items"][0]["approval_id"] != second.public["items"][0]["approval_id"]
    assert first.public["snapshot_version"] != second.public["snapshot_version"]


def test_real_enqueue_stamps_unique_nonce_and_keeps_it_off_notify_payload():
    # Drive the REAL gateway enqueue (`_await_gateway_decision`) — the notify
    # callback resolves the just-enqueued head so the wait returns immediately —
    # and confirm: (1) each instance gets a distinct, unguessable 128-bit nonce on
    # the ENTRY even for an identical (session, command); (2) the nonce is NEVER in
    # the `approval_data` handed to the notify callback (which TUI/API/SSE emit
    # verbatim), so it stays an internal bridge field, never client-facing.
    from tools import approval as A

    _clear_gateway_queues()
    try:
        nonces: list[str] = []

        def make_notify():
            def notify_cb(data):
                # The notify payload MUST NOT carry the internal CAS nonce.
                assert "_nonce" not in data
                with A._lock:
                    entry = A._gateway_queues["sess-a"][-1]
                nonces.append(str(entry.nonce))  # nonce lives on the entry, not data
                entry.result = "once"
                entry.event.set()
            return notify_cb

        for _ in range(2):
            A._await_gateway_decision("sess-a", make_notify(), {"command": "deploy", "description": "d", "pattern_key": "p"})

        assert len(nonces) == 2
        assert nonces[0] != nonces[1]
        assert all(len(n) == 32 for n in nonces)  # 16 random bytes, hex
    finally:
        _clear_gateway_queues()
