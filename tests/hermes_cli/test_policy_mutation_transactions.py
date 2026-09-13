"""Policy broker regressions with real SQLite, files, and crash boundaries."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from hermes_cli import policy_mutation as policy


def _mint(broker, key="approvals.mode", session="s1", operation="set"):
    request = broker.request(session, key, operation)
    confirmation = f"confirm-{request.request_id}"
    broker.record_settlement(request.request_id, confirmation)
    with policy._operator_settlement_scope(request.request_id, confirmation):
        return broker.operator_confirm(request.request_id)


@pytest.fixture
def broker(tmp_path):
    (tmp_path / "config.yaml").write_bytes(b"approvals:\n  mode: manual\n")
    return policy.PolicyMutationBroker(db_path=tmp_path / "state.db")


def _consumed(broker, proof):
    with sqlite3.connect(broker.db_path) as db:
        return db.execute(
            "SELECT consumed_at FROM policy_mutation_proofs WHERE nonce=?", (proof.nonce,)
        ).fetchone()[0]


@pytest.mark.parametrize("key", [
    "approvals", "approvals.mode", "security", "security.tirith_enabled",
    "command_allowlist", "command_allowlist.commands", "yolo", "yolo.enabled",
    "persistent", "persistent.yolo", "persistent.yolo.enabled",
])
def test_policy_ancestors_and_descendants_are_guarded(key):
    assert policy.is_policy_config_key(key)
    with pytest.raises(policy.PolicyMutationDenied, match="proof required"):
        policy.require_policy_proof(key, "unset")


@pytest.mark.parametrize("key", [
    "display.skin", "agent.max_turns", "approvals_extra", "security_notes",
    "persistent.theme", "display.approvals", r"approvals\.mode", "yolov8",
])
def test_unrelated_keys_remain_unguarded(key):
    assert not policy.is_policy_config_key(key)
    policy.require_policy_proof(key, "set")


@pytest.mark.parametrize(("before", "after", "expected"), [
    ({}, {"security": {}}, ["security"]),
    ({"security": {}}, {}, ["security"]),
    ({"approvals": {"mode": False}}, {"approvals": {"mode": 0}}, ["approvals.mode"]),
    ({"approvals": {"rule.name": "same"}}, {"approvals": {"rule.name": "same"}}, []),
    ({"approvals": {"rule.name": "old"}}, {"approvals": {"rule.name": "new"}},
     [r"approvals.rule\.name"]),
    ({"approvals": {"rule\\name": True}}, {"approvals": {"rule\\name": False}},
     [r"approvals.rule\\name"]),
    ({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "manual"}}, []),
    ({"persistent": {"theme": "old"}}, {"persistent": {"theme": "new"}}, []),
    ({"security": {"enabled": True}}, {"security": None}, ["security", "security.enabled"]),
])
def test_payload_diff_preserves_structure_and_scalar_types(before, after, expected):
    assert policy.policy_changed_keys(before, after) == expected


def test_unreadable_policy_is_not_an_empty_policy(tmp_path):
    path = tmp_path / "config.yaml"
    path.mkdir()
    with pytest.raises(policy.PolicyMutationDenied, match="policy.*unavailable"):
        policy.policy_config_digest(path)


def test_missing_policy_file_remains_a_valid_initial_state(tmp_path):
    import hashlib
    assert policy.policy_config_digest(tmp_path / "absent.yaml") == hashlib.sha256(b"").hexdigest()


def test_policy_read_failure_refuses_before_effect(broker, tmp_path):
    proof = _mint(broker)
    (tmp_path / "config.yaml").unlink()
    (tmp_path / "config.yaml").mkdir()
    effects = []
    with pytest.raises(policy.PolicyMutationDenied, match="policy.*unavailable"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", lambda: effects.append(True))
    assert effects == []
    assert _consumed(broker, proof) is None


def test_ledger_failure_after_initialization_is_a_refusal(broker):
    proof = _mint(broker)
    broker.db_path.write_bytes(b"not a database")
    effects = []
    with pytest.raises(policy.PolicyMutationDenied, match="ledger unavailable"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", lambda: effects.append(True))
    assert effects == []


def test_stale_policy_refuses_before_effect_and_preserves_newer_bytes(broker, tmp_path):
    proof = _mint(broker)
    path = tmp_path / "config.yaml"
    newer = b"approvals:\n  mode: smart\n"
    path.write_bytes(newer)
    with pytest.raises(policy.PolicyMutationDenied, match="digest mismatch"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", lambda: path.write_bytes(b"wrong"))
    assert path.read_bytes() == newer
    assert _consumed(broker, proof) is None


@pytest.mark.parametrize("field", ["target_key", "operation", "session_id"])
def test_wrong_binding_does_not_spend_proof(broker, field):
    proof = _mint(broker)
    bad = replace(proof, **{field: "wrong"})
    with pytest.raises(policy.PolicyMutationDenied, match="binding mismatch"):
        broker.consume(bad, "approvals.mode", "set", "s1")
    assert _consumed(broker, proof) is None
    assert broker.consume(proof, "approvals.mode", "set", "s1")


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), "not a time", None])
def test_malformed_proof_expiry_cannot_reach_writer(broker, expiry):
    proof = _mint(broker)
    effects = []
    with pytest.raises(policy.PolicyMutationDenied):
        broker.consume_for_write(replace(proof, expires_at=expiry), "approvals.mode", "set", "s1",
                                 lambda: effects.append(True))
    assert effects == []
    assert _consumed(broker, proof) is None


def test_expiry_is_bound_to_ledger_not_just_in_the_future(broker):
    proof = _mint(broker)
    with pytest.raises(policy.PolicyMutationDenied, match="binding mismatch"):
        broker.consume(replace(proof, expires_at=proof.expires_at + 1), "approvals.mode", "set", "s1")
    assert _consumed(broker, proof) is None


def test_unicode_session_binding_keeps_its_identity(broker):
    proof = _mint(broker, session="session-\u03c0")
    assert broker.consume(proof, "approvals.mode", "set", "session-\u03c0")


def test_settlement_must_match_the_recorded_request_and_confirmation(broker):
    request = broker.request("s1", "approvals.mode", "set")
    with policy._operator_settlement_scope(request.request_id, "unrecorded"):
        with pytest.raises(policy.PolicyMutationDenied, match="receipt"):
            broker.operator_confirm(request.request_id)
    broker.record_settlement(request.request_id, "correct")
    with policy._operator_settlement_scope(request.request_id, "wrong"):
        with pytest.raises(policy.PolicyMutationDenied, match="receipt"):
            broker.operator_confirm(request.request_id)
    with policy._operator_settlement_scope(request.request_id, "correct"):
        proof = broker.operator_confirm(request.request_id)
    assert broker.consume(proof, "approvals.mode", "set", "s1")


def test_context_restored_after_settlement(broker):
    request = broker.request("s1", "approvals.mode", "set")
    broker.record_settlement(request.request_id, "correct")
    with policy._operator_settlement_scope(request.request_id, "correct"):
        pass
    with pytest.raises(policy.PolicyMutationDenied, match="receipt"):
        broker.operator_confirm(request.request_id)


def test_expired_proof_has_no_effect(broker, monkeypatch):
    proof = _mint(broker)
    monkeypatch.setattr(policy.time, "time", lambda: proof.expires_at + 1)
    effects = []
    with pytest.raises(policy.PolicyMutationDenied, match="expired"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", lambda: effects.append(True))
    assert effects == []
    assert _consumed(broker, proof) is None


def test_independent_brokers_reserve_before_either_callback_can_duplicate(broker, tmp_path):
    proof = _mint(broker)
    other = policy.PolicyMutationBroker(db_path=broker.db_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()

    def write_first():
        first_entered.set()
        if not release_first.wait(5):
            raise TimeoutError("test did not release first writer")
        (tmp_path / "effect-first").write_bytes(b"one")

    def write_second():
        second_entered.set()
        (tmp_path / "effect-second").write_bytes(b"two")

    def run(owner, write, started=None):
        if started is not None:
            started.set()
        try:
            return owner.consume_for_write(proof, "approvals.mode", "set", "s1", write)
        except policy.PolicyMutationDenied:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, broker, write_first)
        try:
            assert first_entered.wait(5)
            second = pool.submit(run, other, write_second, second_started)
            assert second_started.wait(5)
            second_entered.wait(0.5)
        finally:
            release_first.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert results.count(True) == 1
    assert len(list(tmp_path.glob("effect-*"))) == 1
    assert not second_entered.is_set()


def test_callback_exception_after_effect_never_reopens_proof(broker, tmp_path):
    proof = _mint(broker)
    effect = tmp_path / "effect"

    def write_then_fail():
        effect.write_bytes(b"already happened")
        raise OSError("writer failed after an effect")

    with pytest.raises(RuntimeError, match="indeterminate"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", write_then_fail)
    assert effect.read_bytes() == b"already happened"
    assert _consumed(broker, proof) is not None
    with pytest.raises(policy.PolicyMutationDenied, match="replay"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", lambda: effect.write_bytes(b"replayed"))
    assert effect.read_bytes() == b"already happened"


def test_process_death_after_effect_never_reopens_proof(broker, tmp_path):
    proof = _mint(broker)
    effect = tmp_path / "crash-effect"
    code = """
import json, os, sys
from pathlib import Path
from hermes_cli.policy_mutation import PolicyMutationBroker, PolicyMutationProof
broker = PolicyMutationBroker(db_path=Path(sys.argv[1]))
proof = PolicyMutationProof(**json.load(sys.stdin))
def write():
    Path(sys.argv[2]).write_bytes(b'already happened')
    os._exit(23)
broker.consume_for_write(proof, 'approvals.mode', 'set', 's1', write)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(broker.db_path), str(effect)],
        input=json.dumps(asdict(proof)), capture_output=True, text=True, timeout=15,
        env=dict(os.environ), check=False,
    )
    assert result.returncode == 23, result.stderr
    assert effect.read_bytes() == b"already happened"
    restarted = policy.PolicyMutationBroker(db_path=broker.db_path)
    with pytest.raises(policy.PolicyMutationDenied, match="replay"):
        restarted.consume_for_write(proof, "approvals.mode", "set", "s1",
                                    lambda: effect.write_bytes(b"replayed"))
    assert effect.read_bytes() == b"already happened"


def test_consumption_is_durable_before_callback(broker):
    proof = _mint(broker)
    observations = []
    assert broker.consume_for_write(proof, "approvals.mode", "set", "s1",
                                    lambda: observations.append(_consumed(broker, proof)))
    assert len(observations) == 1
    assert observations[0] is not None


def test_async_writer_is_not_reported_as_a_successful_write(broker):
    proof = _mint(broker)

    async def write():
        raise AssertionError("async callback must not run")

    with pytest.raises(TypeError, match="synchronous"):
        broker.consume_for_write(proof, "approvals.mode", "set", "s1", write)
    assert _consumed(broker, proof) is None


def test_symlinked_ledger_does_not_retarget_the_policy_file(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "config.yaml").write_bytes(b"approvals:\n  mode: off\n")
    target = policy.PolicyMutationBroker(db_path=storage / "state.db")
    home = tmp_path / "home"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_bytes(b"approvals:\n  mode: manual\n")
    try:
        (home / "state.db").symlink_to(target.db_path)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not permit test symlinks")
    owner = policy.PolicyMutationBroker(db_path=home / "state.db")
    assert owner.policy_digest == policy.policy_config_digest(config_path)
    assert owner.policy_digest != target.policy_digest


def test_schema_upgrade_preserves_old_proofs_and_does_not_invent_success(tmp_path):
    import hashlib
    import time

    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(b"approvals:\n  mode: manual\n")
    db_path = tmp_path / "state.db"
    token = "historical-test-token"
    now = time.time()
    proof = policy.PolicyMutationProof(
        token=token, policy_digest=policy.policy_config_digest(config_path),
        target_key="approvals.mode", operation="set", session_id="s1",
        nonce="old-ready", expires_at=now + 30,
    )
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE policy_mutation_proofs (
            nonce TEXT PRIMARY KEY, request_id TEXT NOT NULL,
            session_id TEXT NOT NULL, target_key TEXT NOT NULL,
            operation TEXT NOT NULL, policy_digest TEXT NOT NULL,
            token_hash TEXT, expires_at REAL NOT NULL DEFAULT 0,
            consumed_at REAL, settled INTEGER NOT NULL DEFAULT 0,
            confirmation_id TEXT
        )""")
        for nonce, consumed_at in ((proof.nonce, None), ("old-spent", now)):
            db.execute(
                "INSERT INTO policy_mutation_proofs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (nonce, f"request-{nonce}", proof.session_id, proof.target_key,
                 proof.operation, proof.policy_digest, hashlib.sha256(token.encode()).hexdigest(),
                 proof.expires_at, consumed_at, 1, f"confirm-{nonce}"),
            )
    owner = policy.PolicyMutationBroker(db_path=db_path)
    effect = tmp_path / "effect"
    assert owner.consume_for_write(proof, "approvals.mode", "set", "s1",
                                   lambda: effect.write_bytes(b"applied"))
    with sqlite3.connect(db_path) as db:
        records = dict(db.execute("SELECT nonce, completed_at FROM policy_mutation_proofs"))
        historical_consumed = db.execute(
            "SELECT consumed_at FROM policy_mutation_proofs WHERE nonce='old-spent'"
        ).fetchone()[0]
    assert effect.read_bytes() == b"applied"
    assert records[proof.nonce] is not None
    assert records["old-spent"] is None
    assert historical_consumed == now
