from __future__ import annotations

import os
import threading
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_author_and_apply as author
from scripts.canary import owner_gate_author_journal as journal_module
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust as trust
from tests.scripts.canary import test_owner_gate_pre_foundation as helpers


RELEASE = "a" * 40
PLAN = "b" * 64


class SimulatedCrash(BaseException):
    pass


class _Capabilities:
    def __init__(self, root: Path) -> None:
        self.release_key = Ed25519PrivateKey.generate()
        self.network_key = Ed25519PrivateKey.generate()
        self.journal = journal_module.OwnerGateAuthorJournal(
            _root=root,
            _owner_uid=os.geteuid(),
            _owner_gid=os.getegid(),
        )
        self.clock = 2_000_000_000
        self.apply_calls = 0
        self.apply_results: list[object] = []
        self.reconcile_calls = 0
        self.reconcile_results: list[Mapping[str, Any]] = []

    def now_unix(self) -> int:
        self.clock += 1
        return self.clock

    def manifest(self) -> Mapping[str, Any]:
        raise AssertionError("_author is replaced in orchestration tests")

    def release_private_key(self) -> Ed25519PrivateKey:
        return self.release_key

    def network_public_key(self):
        return self.network_key.public_key()

    def owner_reauthentication(self, private_key):
        raise AssertionError

    def interpreter_evidence(self, collected_at_unix):
        raise AssertionError

    def network_evidence(self, collected_at_unix):
        raise AssertionError

    def ancestry_evidence(self, collected_at_unix, receipt, release_public_key):
        raise AssertionError

    def apply(self, **_kwargs: Any) -> Mapping[str, Any]:
        self.apply_calls += 1
        if self.apply_results:
            result = self.apply_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            assert isinstance(result, Mapping)
            return result
        return {
            "completed_at_unix": self.clock,
            "foundation_apply_receipt_sha256": "c" * 64,
        }

    def reconcile_foundation_terminal(self, **_kwargs: Any) -> Mapping[str, Any]:
        self.reconcile_calls += 1
        if self.reconcile_results:
            return self.reconcile_results.pop(0)
        return {"state": "absent", "transaction_id": "e" * 64}


def _root(tmp_path: Path) -> Path:
    parent = tmp_path / ".hermes"
    parent.mkdir(mode=0o700)
    os.chown(parent, os.geteuid(), os.getegid(), follow_symlinks=False)
    parent.chmod(0o700)
    return parent / "authoring"


def _authored(ordinal: int = 1) -> author._AuthoredArtifacts:
    return author._AuthoredArtifacts(
        owner_reauth={"kind": "owner-reauth", "ordinal": ordinal},
        interpreter={"kind": "interpreter", "ordinal": ordinal},
        network={"kind": "network", "ordinal": ordinal},
        ancestry={"kind": "ancestry", "ordinal": ordinal},
        authority={
            "kind": "authority",
            "ordinal": ordinal,
            "issued_at_unix": 2_000_000_000,
        },
        plan_sha256=PLAN,
    )


def _patch_author(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: _Capabilities,
) -> list[int]:
    calls: list[int] = []

    def fake(_release: str, _capabilities: object):
        ordinal = len(calls) + 1
        calls.append(ordinal)
        return (
            _authored(ordinal),
            capabilities.release_key.public_key(),
            capabilities.network_key.public_key(),
        )

    monkeypatch.setattr(author, "_author", fake)
    monkeypatch.setattr(
        foundation_apply,
        "decode_validated_foundation_a_chain",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        foundation_apply,
        "_decode_validated_foundation_apply_chain",
        lambda **kwargs: SimpleNamespace(
            apply_receipt=json.loads(kwargs["apply_receipt_raw"])
        ),
    )
    monkeypatch.setattr(
        foundation_apply,
        "validate_failure_receipt",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        author,
        "_validate_failure_receipt_for_artifacts",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        author,
        "_validate_success_receipt_for_artifacts",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        owner_reauth,
        "validate_owner_reauth_receipt",
        lambda *_args, **_kwargs: {"expires_at_unix": 2_000_001_000},
    )
    return calls


def test_production_capabilities_translate_owner_reauthentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = object.__new__(author._ProductionCapabilities)
    capabilities.release_revision = RELEASE
    capabilities.executable = object()
    capabilities.configuration = object()

    def fail(**_kwargs: Any) -> Mapping[str, Any]:
        raise owner_reauth.OwnerGateOwnerReauthError(
            "owner_gate_owner_reauth_runtime_invalid"
        )

    monkeypatch.setattr(owner_reauth, "produce_owner_reauth_receipt", fail)

    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_owner_reauthentication_failed",
    ):
        capabilities.owner_reauthentication(Ed25519PrivateKey.generate())


def test_success_is_append_only_and_replays_without_second_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    author_calls = _patch_author(monkeypatch, capabilities)

    first = author._author_and_apply_with_capabilities(RELEASE, capabilities)
    second = author._author_and_apply_with_capabilities(RELEASE, capabilities)

    assert second == first
    assert capabilities.apply_calls == 1
    assert author_calls == [1]
    with capabilities.journal.release_lease(RELEASE):
        transactions = capabilities.journal.list_transactions(RELEASE)
    assert len(transactions) == 1
    artifacts = next(iter(transactions.values()))
    assert artifacts["terminal"]["state"] == "succeeded"
    assert artifacts["terminal"]["terminal_sha256"] == author._sha256(
        author._canonical({
            key: value
            for key, value in artifacts["terminal"].items()
            if key != "terminal_sha256"
        })
    )


def test_crash_after_inner_success_recovers_without_redispatch_after_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    _patch_author(monkeypatch, capabilities)
    durable_receipt = {
        "completed_at_unix": capabilities.clock,
        "foundation_apply_receipt_sha256": "c" * 64,
    }
    capabilities.apply_results = [durable_receipt]
    capabilities.reconcile_results = [{
        "state": "succeeded",
        "transaction_id": "e" * 64,
        "receipt": durable_receipt,
    }]
    original = capabilities.journal.publish
    crashed = False

    def publish(*args: Any, **kwargs: Any):
        nonlocal crashed
        name = args[2]
        if name == "apply-receipt" and not crashed:
            crashed = True
            raise SimulatedCrash()
        return original(*args, **kwargs)

    monkeypatch.setattr(capabilities.journal, "publish", publish)
    with pytest.raises(SimulatedCrash):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)

    monkeypatch.setattr(
        owner_reauth,
        "validate_owner_reauth_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            owner_reauth.OwnerGateOwnerReauthError(
                "owner_gate_owner_reauth_receipt_expired"
            )
        ),
    )
    receipt = author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert receipt["foundation_apply_receipt_sha256"] == "c" * 64
    assert capabilities.apply_calls == 1
    assert capabilities.reconcile_calls == 1


def test_partial_artifact_publication_is_terminalized_before_fresh_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    calls = _patch_author(monkeypatch, capabilities)
    original = capabilities.journal.publish
    crashed = False

    def publish(*args: Any, **kwargs: Any):
        nonlocal crashed
        if args[2] == "network-evidence" and not crashed:
            crashed = True
            raise SimulatedCrash()
        return original(*args, **kwargs)

    monkeypatch.setattr(capabilities.journal, "publish", publish)
    with pytest.raises(SimulatedCrash):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)

    author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert calls == [1, 2]
    with capabilities.journal.release_lease(RELEASE):
        states = [
            artifacts["terminal"]["state"]
            for artifacts in capabilities.journal.list_transactions(RELEASE).values()
        ]
    assert sorted(states) == ["failed", "succeeded"]


def test_failure_receipt_survives_crash_before_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    calls = _patch_author(monkeypatch, capabilities)
    failure = {
        "foundation_apply_failure_receipt_sha256": "d" * 64,
        "terminal_state": "rolled_back_clean",
    }
    capabilities.apply_results = [foundation_apply.FoundationApplyFailed(failure)]
    capabilities.reconcile_results = [{
        "state": "failed",
        "transaction_id": "e" * 64,
        "receipt": failure,
    }]
    original = capabilities.journal.publish
    crashed = False

    def crash_terminal(*args: Any, **kwargs: Any):
        nonlocal crashed
        if args[2] == "terminal" and not crashed:
            crashed = True
            raise SimulatedCrash()
        return original(*args, **kwargs)

    monkeypatch.setattr(capabilities.journal, "publish", crash_terminal)
    with pytest.raises(SimulatedCrash):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)

    author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert calls == [1, 2]
    with capabilities.journal.release_lease(RELEASE):
        transactions = capabilities.journal.list_transactions(RELEASE)
    failed = next(
        item for item in transactions.values() if item["terminal"]["state"] == "failed"
    )
    assert failed["apply-failure-receipt"] == failure
    assert failed["terminal"]["terminal_receipt_kind"] == (
        "foundation_apply_failure"
    )


def test_tampered_canonical_artifact_breaks_terminal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    _patch_author(monkeypatch, capabilities)
    author._author_and_apply_with_capabilities(RELEASE, capabilities)
    transaction = next(iter((capabilities.journal.root / RELEASE).iterdir()))
    authority_path = transaction / "authority.json"
    authority_path.write_bytes(author._canonical({"tampered": True}))
    authority_path.chmod(0o600)

    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_intent_artifact_mismatch",
    ):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)


def test_no_private_seed_or_bearer_material_enters_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    _patch_author(monkeypatch, capabilities)
    author._author_and_apply_with_capabilities(RELEASE, capabilities)
    payload = b"".join(
        path.read_bytes() for path in capabilities.journal.root.rglob("*.json")
    )
    assert b"private_seed" not in payload
    assert b"access_token" not in payload
    assert capabilities.release_key.private_bytes_raw() not in payload


def test_keyboard_interrupt_leaves_nonterminal_then_partial_inner_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    calls = _patch_author(monkeypatch, capabilities)
    capabilities.apply_results = [KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)
    with capabilities.journal.release_lease(RELEASE):
        transactions = capabilities.journal.list_transactions(RELEASE)
    assert len(transactions) == 1
    assert "terminal" not in next(iter(transactions.values()))

    capabilities.reconcile_results = [{
        "state": "in_progress",
        "transaction_id": "e" * 64,
        "failure_intent_present": False,
    }]
    monkeypatch.setattr(
        owner_reauth,
        "validate_owner_reauth_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("freshness_must_not_precede_inner_reconciliation")
        ),
    )
    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_manual_reconciliation_required",
    ):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert calls == [1]
    with capabilities.journal.release_lease(RELEASE):
        terminal = next(
            iter(capabilities.journal.list_transactions(RELEASE).values())
        )["terminal"]
    assert terminal["state"] == "manual_reconciliation_required"

    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_manual_reconciliation_required",
    ):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert capabilities.reconcile_calls == 1
    assert calls == [1]


def test_generic_apply_error_stays_nonterminal_until_absence_is_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _Capabilities(_root(tmp_path))
    calls = _patch_author(monkeypatch, capabilities)
    capabilities.apply_results = [RuntimeError("provider detail must not leak")]

    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_apply_outcome_unknown",
    ):
        author._author_and_apply_with_capabilities(RELEASE, capabilities)
    with capabilities.journal.release_lease(RELEASE):
        first = next(
            iter(capabilities.journal.list_transactions(RELEASE).values())
        )
    assert "terminal" not in first

    author._author_and_apply_with_capabilities(RELEASE, capabilities)
    assert calls == [1, 2]
    assert capabilities.apply_calls == 2
    with capabilities.journal.release_lease(RELEASE):
        states = sorted(
            item["terminal"]["state"]
            for item in capabilities.journal.list_transactions(RELEASE).values()
        )
    assert states == ["failed", "succeeded"]


def test_valid_signed_failure_from_another_transaction_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        helpers.RELEASE_KEY_ID,
    )
    authority_value, _plan, _network = helpers._authority()
    owner_value = helpers._owner_reauth_receipt()
    network_value = helpers._signed_network_evidence()
    ancestry_value = json.loads(helpers._signed_ancestry_raw())
    artifacts = {
        "authority": authority_value,
        "owner-reauth": owner_value,
        "network-evidence": network_value,
        "ancestry-evidence": ancestry_value,
    }
    chain = foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=author._canonical(authority_value),
        owner_reauthentication_receipt_raw=author._canonical(owner_value),
        network_evidence_raw=author._canonical(network_value),
        project_ancestry_evidence_raw=author._canonical(ancestry_value),
        release_public_key=helpers.RELEASE_KEY.public_key(),
        network_collector_public_key=helpers.NETWORK_KEY.public_key(),
        project_ancestry_collector_public_key=(
            helpers.NETWORK_KEY.public_key()
        ),
        now_unix=helpers.NOW + 1,
    )
    body = {
        "schema": foundation_apply.FAILURE_RECEIPT_SCHEMA,
        "purpose": foundation_apply.FAILURE_RECEIPT_PURPOSE,
        "transaction_id": foundation_apply._transaction_id(chain),
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "inert_plan_sha256": pre_foundation.inert_plan_sha256(chain.plan),
        "foundation_source_revision": chain.foundation_source_revision,
        "foundation_source_tree_oid": chain.foundation_source_tree_oid,
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "ancestry_chain_sha256": chain.ancestry_evidence.value[
            "stable_chain_sha256"
        ],
        "started_at_unix": helpers.NOW + 1,
        "failed_at_unix": helpers.NOW + 2,
        "failed_step_name": chain.plan.foundation_steps[0].name,
        "failure_code": "owner_gate_foundation_test_failure",
        "completed_step_receipts": [],
        "rollback_step_receipts": [],
        "terminal_state": "rolled_back_clean",
        "partial_unknown_state": False,
        "mutation_iam_binding_created": False,
        "package_deployed": False,
        "service_started": False,
        "signer_key_id": helpers.RELEASE_KEY_ID,
    }
    receipt = foundation_apply._sign_failure_receipt(
        body,
        private_key=helpers.RELEASE_KEY,
    )
    assert author._validate_failure_receipt_for_artifacts(
        receipt,
        artifacts=artifacts,
        release_public=helpers.RELEASE_KEY.public_key(),
        network_public=helpers.NETWORK_KEY.public_key(),
    ) == receipt

    substituted = foundation_apply._sign_failure_receipt(
        {**body, "transaction_id": "f" * 64},
        private_key=helpers.RELEASE_KEY,
    )
    with pytest.raises(
        author.OwnerGateAuthorAndApplyError,
        match="owner_gate_author_apply_failure_receipt_mismatch",
    ):
        author._validate_failure_receipt_for_artifacts(
            substituted,
            artifacts=artifacts,
            release_public=helpers.RELEASE_KEY.public_key(),
            network_public=helpers.NETWORK_KEY.public_key(),
        )


def test_partial_pending_write_is_safely_replaced(tmp_path: Path) -> None:
    store = journal_module.OwnerGateAuthorJournal(
        _root=_root(tmp_path),
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )
    transaction_id = "f" * 64
    with store.release_lease(RELEASE):
        store.publish(RELEASE, transaction_id, "intent", {"value": 1})
        transaction = store.root / RELEASE / transaction_id
        pending = transaction / ".authority.pending"
        pending.write_bytes(b'{"partial"')
        pending.chmod(0o600)
        assert store.publish(
            RELEASE,
            transaction_id,
            "authority",
            {"value": 2},
        ) == {"value": 2}
        assert not pending.exists()


def test_invalid_pending_with_final_fails_closed(tmp_path: Path) -> None:
    store = journal_module.OwnerGateAuthorJournal(
        _root=_root(tmp_path),
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )
    transaction_id = "f" * 64
    with store.release_lease(RELEASE):
        store.publish(RELEASE, transaction_id, "authority", {"value": 1})
        transaction = store.root / RELEASE / transaction_id
        pending = transaction / ".authority.pending"
        pending.write_bytes(b'{"partial"')
        pending.chmod(0o600)
        with pytest.raises(
            journal_module.OwnerGateAuthorJournalError,
            match="owner_gate_author_journal_json_invalid",
        ):
            store.list_artifacts(RELEASE, transaction_id)


def test_release_lease_serializes_two_independent_journal_instances(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    first = journal_module.OwnerGateAuthorJournal(
        _root=root,
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )
    second = journal_module.OwnerGateAuthorJournal(
        _root=root,
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def holder() -> None:
        with first.release_lease(RELEASE):
            order.append("first")
            entered.set()
            assert release.wait(2)

    def waiter() -> None:
        assert entered.wait(2)
        with second.release_lease(RELEASE):
            order.append("second")

    one = threading.Thread(target=holder)
    two = threading.Thread(target=waiter)
    one.start()
    two.start()
    assert entered.wait(2)
    assert order == ["first"]
    release.set()
    one.join(2)
    two.join(2)
    assert order == ["first", "second"]
