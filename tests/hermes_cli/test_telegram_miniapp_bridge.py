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
