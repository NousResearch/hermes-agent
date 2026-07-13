"""Focused fail-closed contracts for the honest full-canary live driver."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import runpy
import sqlite3
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.canonical_full_canary_live_driver as live
import gateway.canonical_full_canary_runtime as runtime
from gateway.canonical_full_canary_runtime import (
    ExactArtifact,
    FullCanaryIdentities,
    FullCanaryOwnerApproval,
    FullCanaryPlan,
)


REVISION = "a" * 40
RELEASE_SHA256 = "b" * 64
FIXTURE_SHA256 = "c" * 64
PLAN_SHA256 = "d" * 64
SESSION_KEY = "session-key-that-never-enters-json"
NOW_MS = 2_000_000_000_000


def _identities() -> FullCanaryIdentities:
    return FullCanaryIdentities(
        writer_user="writer",
        writer_group="writer",
        writer_uid=2101,
        writer_gid=2201,
        gateway_user="gateway",
        gateway_group="gateway",
        gateway_uid=2102,
        gateway_gid=2202,
        socket_client_group="clients",
        socket_client_gid=2203,
        edge_user="muncho-discord-egress",
        edge_group="muncho-discord-egress",
        edge_uid=2103,
        edge_gid=2204,
    )


def _plan(tmp_path: Path, *, writer_digest: str = "e" * 64) -> FullCanaryPlan:
    identities = _identities()
    return FullCanaryPlan(
        revision=REVISION,
        release={"artifact_sha256": RELEASE_SHA256},
        identities=identities,
        writer_activation_plan={},
        writer_activation_receipt={},
        writer_activation_receipt_file_sha256="f" * 64,
        artifacts={
            "writer_config": ExactArtifact(
                source_path=tmp_path / "writer.json",
                target_path=Path("/etc/muncho/full-canary/writer.json"),
                sha256=writer_digest,
                mode=0o440,
                uid=0,
                gid=identities.writer_gid,
            ),
            "e2e_fixture": ExactArtifact(
                source_path=tmp_path / "fixture.json",
                target_path=Path("/etc/muncho/full-canary/fixture.json"),
                sha256=FIXTURE_SHA256,
                mode=0o440,
                uid=0,
                gid=identities.gateway_gid,
            ),
        },
        allowed_previous_sha256={},
        unit_bundle=None,  # type: ignore[arg-type]
        unit_paths={},
        e2e_verifier_module="gateway.canonical_full_canary_e2e",
        sha256=PLAN_SHA256,
    )


def _approval(plan: FullCanaryPlan) -> FullCanaryOwnerApproval:
    now = int(time.time())
    return FullCanaryOwnerApproval.from_mapping({
        "schema": "muncho-full-canary-owner-approval.v1",
        "scope": "full_canary_runtime_start",
        "plan_sha256": plan.sha256,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": "1" * 64,
        "approval_source_sha256": "2" * 64,
        "nonce_sha256": "3" * 64,
        "approved_at_unix": now - 1,
        "expires_at_unix": now + 300,
    })


def _fixture() -> dict[str, Any]:
    return {
        "release_sha": REVISION,
        "release_artifact_sha256": RELEASE_SHA256,
        "canary_run_id": "11111111-1111-4111-8111-111111111111",
        "case_id": "case:full-canary:1",
        "valid_from_unix_ms": NOW_MS - 10_000,
        "valid_until_unix_ms": NOW_MS + 60_000,
        "task_policy": {"prompt": "do the exact task", "prompt_sha256": "4" * 64},
        "public_routeback": {"canonical_idempotency_key": "routeback:1"},
    }


def test_prepare_stages_session_digest_before_plan_and_approval(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    captured: dict[str, Any] = {}

    def writer(path: Path, payload: bytes, **metadata: Any) -> None:
        order.append("writer")
        captured.update(path=path, payload=payload, metadata=metadata)

    def build() -> FullCanaryPlan:
        order.append("plan")
        assert "payload" in captured
        digest = hashlib.sha256(captured["payload"]).hexdigest()
        return _plan(tmp_path, writer_digest=digest)

    def approve(plan: FullCanaryPlan) -> FullCanaryOwnerApproval:
        order.append("approval")
        return _approval(plan)

    prepared = live.prepare_session_bound_plan(
        writer_config={
            "database": {
                "host": "10.0.0.8",
                "tls_server_name": "db.internal",
                "port": 5432,
                "database": "muncho_canary_brain",
            },
            "canary_scope_preapproval": {
                "grant_id": "grant:live-driver",
                "case_id": _fixture()["case_id"],
                "release_sha256": RELEASE_SHA256,
                "fixture_sha256": FIXTURE_SHA256,
                "run_id": _fixture()["canary_run_id"],
                "session_key_sha256": "0" * 64,
                "expires_at": "2026-01-01T00:00:00+00:00",
                "approved_by": "1279454038731264061",
                "approval_source_sha256": "2" * 64,
                "provisioning_receipt_sha256": "3" * 64,
                "bootstrap_database_user": ("canonical_brain_canary_bootstrap_login"),
            },
        },
        fixture=_fixture(),
        writer_gid=_identities().writer_gid,
        bootstrap_sql_sha256="9" * 64,
        bootstrap_retire_sql_sha256="8" * 64,
        staged_writer_config=tmp_path / "writer.json",
        plan_builder=build,
        approval_provider=approve,
        session_key_factory=lambda: SESSION_KEY,
        writer=writer,
        process_guard=lambda: order.append("harden"),
    )

    assert order == ["harden", "writer", "plan", "approval"]
    assert prepared.session_key == SESSION_KEY
    assert SESSION_KEY not in repr(prepared)
    assert SESSION_KEY.encode() not in captured["payload"]
    staged = json.loads(captured["payload"])
    assert staged["canary_scope_preapproval"]["session_key_sha256"] == (
        hashlib.sha256(SESSION_KEY.encode()).hexdigest()
    )
    assert staged["canary_scope_preapproval"][
        "provisioning_receipt_sha256"
    ] == live.canonical_canary_bootstrap_authorization_sha256(
        staged,
        bootstrap_sql_sha256="9" * 64,
        bootstrap_retire_sql_sha256="8" * 64,
    )
    assert captured["metadata"] == {
        "mode": 0o440,
        "uid": 0,
        "gid": _identities().writer_gid,
        "expected_existing_sha256": None,
    }


def _old_staged_config() -> dict[str, Any]:
    return {
        "service": {"writer_uid": 2101, "writer_gid": 2201},
        "database": {
            "host": "10.0.0.8",
            "tls_server_name": "db.internal",
            "port": 5432,
            "database": "muncho_canary_brain",
            "user": "canonical_brain_writer_login",
        },
        "canary_scope_preapproval": {
            "grant_id": "grant:old",
            "case_id": "case:old",
            "release_sha256": "1" * 64,
            "fixture_sha256": "2" * 64,
            "run_id": "run:old",
            "session_key_sha256": "3" * 64,
            "expires_at": "2026-07-13T12:00:00+00:00",
            "approved_by": "1279454038731264061",
            "approval_source_sha256": "4" * 64,
            "provisioning_receipt_sha256": "5" * 64,
        },
    }


def _fake_stage_stat() -> SimpleNamespace:
    return SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat.S_IFREG | 0o440,
        st_nlink=1,
        st_uid=0,
        st_gid=2201,
        st_size=123,
        st_mtime_ns=4,
        st_ctime_ns=5,
    )


def test_different_staged_plan_is_preserved_without_fresh_db_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_raw = live._canonical_bytes(_old_staged_config())
    monkeypatch.setattr(live.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        live,
        "_read_staged_writer_config",
        lambda *_a, **_k: (old_raw, _fake_stage_stat()),
    )

    with pytest.raises(
        live.LiveCanaryError,
        match="staged_writer_config_unreconciled",
    ):
        live._reconcile_existing_staged_writer_config(
            Path("/etc/muncho/full-canary/staged/writer.json"),
            b'{"different":true}',
            mode=0o440,
            uid=0,
            gid=2201,
            reconciler=None,
        )


def test_different_staged_plan_requires_fresh_exact_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _old_staged_config()
    old_raw = live._canonical_bytes(old)
    order: list[str] = []
    validated: dict[str, Any] = {}
    monkeypatch.setattr(live.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        live,
        "_read_staged_writer_config",
        lambda *_a, **_k: (old_raw, _fake_stage_stat()),
    )
    monkeypatch.setattr(
        live,
        "observe_canary_preclaim_reconciliation_generation",
        lambda _path: (9,),
    )

    def validate(**kwargs: Any) -> dict[str, Any]:
        order.append("validate")
        validated.update(kwargs)
        return {"result": {"outcome": "retired", "authority_active": False}}

    monkeypatch.setattr(
        live,
        "validate_canary_preclaim_reconciliation_receipt",
        validate,
    )
    digest = live._reconcile_existing_staged_writer_config(
        Path("/etc/muncho/full-canary/staged/writer.json"),
        b'{"different":true}',
        mode=0o440,
        uid=0,
        gid=2201,
        reconciler=lambda: order.append("reconcile"),
    )

    assert digest == hashlib.sha256(old_raw).hexdigest()
    assert order == ["reconcile", "validate"]
    assert validated["source_config_raw"] == old_raw
    assert validated["writer_config"] == old
    assert validated["allowed_outcomes"] == frozenset({"retired", "claimed"})
    assert validated["prior_generation"] == (9,)
    assert validated["require_fresh_generation"] is True


def test_different_staged_plan_accepts_prior_durable_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _old_staged_config()
    old_raw = live._canonical_bytes(old)
    durable = {"result": {"outcome": "retired"}}
    events: list[str] = []
    monkeypatch.setattr(live.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        live,
        "_read_staged_writer_config",
        lambda *_a, **_k: (old_raw, _fake_stage_stat()),
    )
    monkeypatch.setattr(
        live,
        "observe_canary_preclaim_reconciliation_generation",
        lambda _path: events.append("observe_prior") or (1,),
    )

    def validate(value, **kwargs: Any) -> dict[str, Any]:
        assert value is durable
        assert kwargs["source_config_raw"] == old_raw
        assert kwargs["writer_config"] == old
        assert kwargs["allowed_outcomes"] == frozenset({"retired", "claimed"})
        events.append("validate_durable")
        return durable

    monkeypatch.setattr(
        live,
        "_validate_canary_preclaim_reconciliation_value",
        validate,
    )

    digest = live._reconcile_existing_staged_writer_config(
        Path("/etc/muncho/full-canary/staged/writer.json"),
        b'{"different":true}',
        mode=0o440,
        uid=0,
        gid=2201,
        reconciler=lambda: events.append("reuse") or durable,
    )

    assert digest == hashlib.sha256(old_raw).hexdigest()
    assert events == ["observe_prior", "reuse", "validate_durable"]


def test_fresh_approval_callback_waits_for_a_new_exact_plan_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    path = tmp_path / "owner-approval.json"
    monkeypatch.setattr(live, "DEFAULT_APPROVAL_PATH", path)
    stale = SimpleNamespace(
        st_dev=1,
        st_ino=1,
        st_mode=stat.S_IFREG | 0o400,
        st_nlink=1,
        st_uid=0,
        st_gid=0,
        st_size=100,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )
    fresh = SimpleNamespace(**{**stale.__dict__, "st_ino": 2, "st_mtime_ns": 2})
    identities = iter((stale, fresh, fresh))
    monkeypatch.setattr(Path, "lstat", lambda _self: next(identities))
    approval = _approval(plan)

    observed = live.wait_for_fresh_owner_approval(
        plan,
        path=path,
        timeout_seconds=1,
        loader=lambda observed_path: (
            approval if observed_path == path else pytest.fail("wrong path")
        ),
        monotonic=lambda: 1.0,
        now=lambda: float(approval.value["approved_at_unix"]),
        sleeper=lambda _seconds: pytest.fail("fresh approval should not sleep"),
        process_guard=lambda: None,
    )
    assert observed is approval


def test_root_guard_hardens_process_before_secret_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(live.sys, "platform", "linux")
    monkeypatch.setattr(live.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        live,
        "harden_current_process_against_dumping",
        lambda: calls.append("harden"),
    )
    live._require_root_linux()
    assert calls == ["harden"]


def test_posix_identity_helpers_fail_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(live.os, "geteuid")
    monkeypatch.delattr(live.os, "getegid")
    with pytest.raises(PermissionError, match="posix_uid"):
        live._effective_uid()
    with pytest.raises(PermissionError, match="posix_gid"):
        live._effective_gid()


def test_atomic_root_writer_rejects_zero_progress_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    real_lstat = Path.lstat

    def root_parent(path: Path):
        item = real_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=item.st_mode, st_uid=0)
        return item

    monkeypatch.setattr(Path, "lstat", root_parent)
    monkeypatch.setattr(live.os, "fchown", lambda _fd, _uid, _gid: None)
    monkeypatch.setattr(live.os, "write", lambda _fd, _payload: 0)
    with pytest.raises(live.LiveCanaryError, match="root_artifact_write_stalled"):
        live._atomic_write_root(target, b"payload", mode=0o400)
    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_atomic_stage_replacement_rejects_zero_progress_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "writer.json"
    target.write_bytes(b"old")
    item = target.lstat()
    real_lstat = Path.lstat

    def root_parent(path: Path):
        observed = real_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=observed.st_mode, st_uid=0)
        return observed

    monkeypatch.setattr(Path, "lstat", root_parent)
    monkeypatch.setattr(
        live,
        "_read_staged_writer_config",
        lambda *_args, **_kwargs: (b"old", item),
    )
    monkeypatch.setattr(live.os, "fchown", lambda _fd, _uid, _gid: None)
    monkeypatch.setattr(live.os, "write", lambda _fd, _payload: 0)
    with pytest.raises(
        live.LiveCanaryError,
        match="staged_writer_config_write_stalled",
    ):
        live._atomic_stage_writer_config(
            target,
            b"new",
            mode=0o440,
            uid=0,
            gid=2201,
            expected_existing_sha256=hashlib.sha256(b"old").hexdigest(),
        )
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.stage.*"))


def test_loopback_client_idempotently_clears_both_raw_keys() -> None:
    client = live.LoopbackCanaryClient(
        control_key="control-key",
        session_key=SESSION_KEY,
    )
    assert client.secrets_cleared is False
    client.clear_secrets()
    client.clear_secrets()
    assert client.secrets_cleared is True
    with pytest.raises(live.LiveCanaryError, match="api_client_secrets_consumed"):
        client._headers(request_id="request:1")


class _FakeCollector:
    def __init__(self, order: list[str], *, plugin_error: bool = False) -> None:
        self.order = order
        self.plugin_error = plugin_error
        self.frames: tuple[Any, ...] = ()
        self.chain_head_sha256 = "5" * 64
        snapshot = live.JournalSnapshot(1, "6" * 64)
        self.private_snapshots = (snapshot, snapshot)

    def start(self) -> None:
        self.order.append("collector.start")

    def wait_plugin_ready(self) -> None:
        self.order.append("collector.plugin_ready")
        if self.plugin_error:
            raise live.LiveCanaryError("plugin_readiness_failed")

    def wait_session_end(self) -> None:
        self.order.append("collector.session_end")

    def close(self) -> None:
        self.order.append("collector.close")


class _FakeLifecycle:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def start(self, _approval: FullCanaryOwnerApproval) -> dict[str, str]:
        self.order.append("lifecycle.start")
        return {"receipt_path": "/tmp/start.json"}

    def stop(self, *, reason: str) -> dict[str, Any]:
        self.order.append(f"lifecycle.stop:{reason}")
        return {}

    def verify_and_stop(self, **_kwargs: Any) -> dict[str, Any]:
        self.order.append("lifecycle.verify_and_stop")
        return {"verified": True}


def _prepared(
    tmp_path: Path,
    *,
    session_key_sha256: str | None = None,
) -> live.SessionBoundPlan:
    plan = _plan(tmp_path)
    return live.SessionBoundPlan(
        session_key=SESSION_KEY,
        session_key_sha256=(
            hashlib.sha256(SESSION_KEY.encode()).hexdigest()
            if session_key_sha256 is None
            else session_key_sha256
        ),
        writer_config_sha256=plan.artifacts["writer_config"].sha256,
        plan=plan,
        approval=_approval(plan),
    )


class _BootstrapProvisioner:
    def __init__(self) -> None:
        self.abort_count = 0
        self.active = True

    def provision(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("provision must not run in this test")

    def reconcile(
        self,
        _request: Any,
        _provisioning_receipt: Any,
    ) -> dict[str, Any]:
        raise AssertionError("reconcile must not run in this test")

    def abort(self) -> None:
        self.abort_count += 1
        self.active = False


def test_driver_default_lifecycle_retains_blocked_admin_boundary(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    driver = live.HonestFullCanaryDriver(prepared, root_guard=lambda: None)

    lifecycle = driver._lifecycle(prepared.plan)

    with pytest.raises(
        runtime.FullCanaryBootstrapAdminUnavailable,
        match="ephemeral bootstrap admin connection is unavailable",
    ):
        lifecycle.bootstrap_provisioner.provision(None)
    prepared.discard_session_key()


def test_driver_rejects_factory_and_preopened_provisioner_together(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    provisioner = _BootstrapProvisioner()

    with pytest.raises(TypeError, match="mutually exclusive"):
        live.HonestFullCanaryDriver(
            prepared,
            lifecycle_factory=lambda plan: runtime.FullCanaryLifecycle(plan),
            bootstrap_provisioner=provisioner,
        )

    prepared.discard_session_key()


def test_driver_aborts_and_releases_preopened_admin_on_earliest_failure(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    provisioner = _BootstrapProvisioner()
    driver = live.HonestFullCanaryDriver(
        prepared,
        bootstrap_provisioner=provisioner,
        root_guard=lambda: (_ for _ in ()).throw(
            PermissionError("process hardening failed")
        ),
    )

    with pytest.raises(PermissionError, match="process hardening failed"):
        driver.run()

    assert provisioner.abort_count == 1
    assert provisioner.active is False
    assert driver._bootstrap_provisioner is None
    assert prepared.session_key is None


def test_driver_retries_transient_preopened_admin_close_failure(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)

    class CloseFailsOnceProvisioner(_BootstrapProvisioner):
        def abort(self) -> None:
            self.abort_count += 1
            if self.abort_count == 1:
                raise RuntimeError("transient close failure")
            self.active = False

    provisioner = CloseFailsOnceProvisioner()
    driver = live.HonestFullCanaryDriver(
        prepared,
        bootstrap_provisioner=provisioner,
        root_guard=lambda: (_ for _ in ()).throw(
            PermissionError("process hardening failed")
        ),
    )

    with pytest.raises(PermissionError, match="process hardening failed"):
        driver.run()

    assert provisioner.abort_count == 2
    assert provisioner.active is False
    assert driver._bootstrap_provisioner is None


def test_driver_injects_only_the_explicit_preopened_provisioner(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    provisioner = _BootstrapProvisioner()
    driver = live.HonestFullCanaryDriver(
        prepared,
        bootstrap_provisioner=provisioner,
        root_guard=lambda: None,
    )

    lifecycle = driver._lifecycle(prepared.plan)

    assert lifecycle.bootstrap_provisioner is provisioner
    prepared.discard_session_key()
    provisioner.abort()


def _patch_driver_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
    order: list[str],
) -> None:
    monkeypatch.setattr(live, "_validated_e2e_fixture", lambda _plan: fixture)
    monkeypatch.setattr(
        live,
        "load_start_receipt",
        lambda _path, plan: SimpleNamespace(
            value={"sealed": True}, file_sha256="7" * 64
        ),
    )
    monkeypatch.setattr(
        live,
        "assemble_live_evidence",
        lambda **_kwargs: {"schema": "test-evidence"},
    )
    monkeypatch.setattr(
        live,
        "verify_evidence",
        lambda *_args, **_kwargs: {"ok": True},
    )


def test_consumed_key_digest_is_checked_before_any_live_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    monkeypatch.setattr(live, "_validated_e2e_fixture", lambda _plan: fixture)
    prepared = _prepared(tmp_path, session_key_sha256="0" * 64)
    driver = live.HonestFullCanaryDriver(
        prepared,
        lifecycle_factory=lambda _plan: pytest.fail("lifecycle must not start"),
        collector_factory=lambda _plan: pytest.fail("collector must not start"),
        client_factory=lambda *_args: pytest.fail("client must not receive key"),
        root_guard=lambda: None,
    )
    with pytest.raises(live.LiveCanaryError, match="prepared_plan_binding_invalid"):
        driver.run()
    assert prepared.session_key is None
    assert prepared.consumed is True


def test_root_guard_failure_discards_key_without_reaching_live_dependencies(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    driver = live.HonestFullCanaryDriver(
        prepared,
        lifecycle_factory=lambda _plan: pytest.fail("lifecycle must not start"),
        collector_factory=lambda _plan: pytest.fail("collector must not start"),
        client_factory=lambda *_args: pytest.fail("client must not receive key"),
        root_guard=lambda: (_ for _ in ()).throw(
            PermissionError("process hardening failed")
        ),
    )

    with pytest.raises(PermissionError, match="process hardening failed"):
        driver.run()

    assert prepared.session_key is None
    assert prepared.consumed is True
    with pytest.raises(
        live.LiveCanaryError,
        match="session_bound_plan_already_consumed",
    ):
        prepared.consume_session_key()


def test_driver_success_rechecks_listener_after_run_and_stops_via_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    order: list[str] = []
    _patch_driver_dependencies(monkeypatch, fixture, order)
    monkeypatch.setattr(
        live,
        "_verify_listener_binding",
        lambda _receipt: order.append("listener.verify"),
    )
    lifecycle = _FakeLifecycle(order)
    collector = _FakeCollector(order)
    routeback_call: dict[str, Any] = {}
    client_ref: dict[str, Any] = {}

    class Client:
        def __init__(self, control_key: str, session_key: str) -> None:
            self.control_key = control_key
            self.session_key = session_key

        def run(self, *, fixture: dict[str, Any]) -> object:
            order.append("client.run")
            return object()

        def clear_secrets(self) -> None:
            order.append("client.clear")
            self.control_key = None
            self.session_key = None

    def client_factory(control_key: str, session_key: str) -> Client:
        client = Client(control_key, session_key)
        client_ref["value"] = client
        return client

    def routeback(path: Path, **kwargs: Any) -> dict[str, Any]:
        order.append("routeback.read")
        routeback_call.update(path=path, **kwargs)
        return {"signed": True}

    prepared = _prepared(tmp_path)
    driver = live.HonestFullCanaryDriver(
        prepared,
        lifecycle_factory=lambda _plan: lifecycle,
        collector_factory=lambda _plan: collector,
        client_factory=client_factory,
        control_key_reader=lambda: ("control-key", "8" * 64),
        projection_exporter=lambda _plan: [],
        edge_routeback_reader=routeback,
        evidence_writer=lambda _plan, _evidence: (
            Path("/tmp/evidence.json"),
            "9" * 64,
        ),
        root_guard=lambda: order.append("root.guard"),
    )
    result = driver.run()

    assert result["ok"] is True
    assert prepared.session_key is None
    assert prepared.consumed is True
    assert client_ref["value"].control_key is None
    assert client_ref["value"].session_key is None
    assert order.count("listener.verify") == 2
    assert order.index("client.run") < order.index("listener.verify", 4)
    assert order[-1] == "collector.close"
    assert "lifecycle.verify_and_stop" in order
    assert routeback_call["expected_uid"] == _identities().edge_uid
    assert routeback_call["expected_gid"] == _identities().edge_gid
    with pytest.raises(
        live.LiveCanaryError,
        match="session_bound_plan_already_consumed",
    ):
        driver.run()


def test_post_run_listener_drift_fails_closed_before_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    order: list[str] = []
    _patch_driver_dependencies(monkeypatch, fixture, order)
    checks = iter((None, live.LiveCanaryError("api_listener_not_owned_by_gateway")))

    def listener(_receipt: Any) -> None:
        outcome = next(checks)
        order.append("listener.verify")
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(live, "_verify_listener_binding", listener)
    lifecycle = _FakeLifecycle(order)
    collector = _FakeCollector(order)

    class Client:
        def run(self, *, fixture: dict[str, Any]) -> object:
            order.append("client.run")
            return object()

        def clear_secrets(self) -> None:
            order.append("client.clear")

    prepared = _prepared(tmp_path)
    driver = live.HonestFullCanaryDriver(
        prepared,
        lifecycle_factory=lambda _plan: lifecycle,
        collector_factory=lambda _plan: collector,
        client_factory=lambda *_args: Client(),
        control_key_reader=lambda: ("control", "1" * 64),
        projection_exporter=lambda _plan: order.append("projection") or [],
        root_guard=lambda: None,
    )
    with pytest.raises(RuntimeError, match="failed closed"):
        driver.run()
    assert prepared.session_key is None
    assert prepared.consumed is True
    assert "projection" not in order
    assert "lifecycle.stop:verification_failed" in order
    assert order[-1] == "collector.close"


@pytest.mark.parametrize("barrier", ["plugin", "client"])
def test_driver_failure_barriers_always_reverse_stop_and_close(
    barrier: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    order: list[str] = []
    _patch_driver_dependencies(monkeypatch, fixture, order)
    monkeypatch.setattr(live, "_verify_listener_binding", lambda _value: None)
    lifecycle = _FakeLifecycle(order)
    collector = _FakeCollector(order, plugin_error=barrier == "plugin")
    client_ref: dict[str, Any] = {}

    class Client:
        def __init__(self, control_key: str, session_key: str) -> None:
            self.control_key = control_key
            self.session_key = session_key

        def run(self, *, fixture: dict[str, Any]) -> object:
            order.append("client.run")
            raise live.LiveCanaryError("api_sse_terminal_invalid")

        def clear_secrets(self) -> None:
            order.append("client.clear")
            self.control_key = None
            self.session_key = None

    def client_factory(control_key: str, session_key: str) -> Client:
        client = Client(control_key, session_key)
        client_ref["value"] = client
        return client

    prepared = _prepared(tmp_path)
    driver = live.HonestFullCanaryDriver(
        prepared,
        lifecycle_factory=lambda _plan: lifecycle,
        collector_factory=lambda _plan: collector,
        client_factory=client_factory,
        control_key_reader=lambda: ("control", "1" * 64),
        root_guard=lambda: None,
    )
    with pytest.raises(RuntimeError, match="failed closed"):
        driver.run()
    assert prepared.session_key is None
    assert prepared.consumed is True
    if barrier == "client":
        assert client_ref["value"].control_key is None
        assert client_ref["value"].session_key is None
    assert "lifecycle.stop:verification_failed" in order
    assert order[-1] == "collector.close"


def _readback_event(event_id: str, event_type: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "case_id": "case:full-canary:1",
        "payload": {},
    }


def _complete_readback(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": "77777777-7777-4777-8777-777777777777",
        "status": "ok",
        "events": [copy.deepcopy(event)],
        "support_events": [],
        "view": "resume_bundle",
        "bounded": True,
        "event_count": 1,
        "truncated": False,
        "candidate_cases_truncated": False,
        "support_incomplete_reasons": [],
        "missing_verification_event_ids": [],
    }


def test_live_readback_is_derived_and_cross_checked_with_post_revoke_projection() -> (
    None
):
    event = _readback_event("event:before", "task.plan.updated")
    revoke = _readback_event("event:revoke", "canary.scope.revoked")
    readback = _complete_readback(event)
    incomplete, missing = live._validate_live_readback(
        readback,
        payload={"query_view": "resume_bundle", "query_limit": 200},
        projection_events=[event, revoke],
    )
    assert incomplete is False
    assert missing == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("support_incomplete_reasons", ["support_byte_budget_exceeded"]),
        ("missing_verification_event_ids", ["missing-event"]),
        ("truncated", True),
    ],
)
def test_incomplete_live_readback_cannot_be_relabelled_complete(
    field: str,
    value: Any,
) -> None:
    event = _readback_event("event:before", "task.plan.updated")
    readback = _complete_readback(event)
    readback[field] = value
    with pytest.raises(live.LiveCanaryError):
        live._validate_live_readback(
            readback,
            payload={"query_view": "resume_bundle", "query_limit": 200},
            projection_events=[
                event,
                _readback_event("event:revoke", "canary.scope.revoked"),
            ],
        )


def _create_edge_journal(path: Path) -> str:
    key = "canonical-routeback:" + "a" * 64
    request = json.dumps({"request": "signed"}, sort_keys=True, separators=(",", ":"))
    receipt = json.dumps({"receipt": "signed"}, sort_keys=True, separators=(",", ":"))
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE discord_edge_journal_meta_v1 (
            singleton INTEGER, marker_id TEXT, schema_version INTEGER
        );
        CREATE TABLE discord_edge_idempotency_v1 (
            idempotency_key TEXT, request_envelope_sha256 TEXT,
            request_envelope_json TEXT, request_id TEXT, capability_id TEXT,
            request_sha256 TEXT, content_sha256 TEXT, state TEXT,
            receipt_json TEXT, blocker_code TEXT,
            created_at_unix_ms INTEGER, updated_at_unix_ms INTEGER
        );
        CREATE TABLE discord_edge_receipt_history_v1 (
            idempotency_key TEXT, sequence INTEGER, receipt_json TEXT,
            recorded_at_unix_ms INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO discord_edge_journal_meta_v1 VALUES (1, 'marker', 1)"
    )
    connection.execute(
        "INSERT INTO discord_edge_idempotency_v1 VALUES "
        "(?, '', ?, '', '', '', '', 'verified', ?, NULL, 1, 2)",
        (key, request, receipt),
    )
    connection.execute(
        "INSERT INTO discord_edge_receipt_history_v1 VALUES (?, 1, ?, 2)",
        (key, receipt),
    )
    connection.commit()
    connection.close()
    path.chmod(0o600)
    return key


def test_edge_journal_reads_bind_exact_owner_mode_and_signed_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal.sqlite3"
    key = _create_edge_journal(journal)
    monkeypatch.setattr(live, "DEFAULT_EDGE_JOURNAL", journal)
    journal_stat = journal.stat()
    uid, gid = journal_stat.st_uid, journal_stat.st_gid
    snapshot = live._logical_journal_snapshot(
        journal, expected_uid=uid, expected_gid=gid
    )
    assert snapshot.record_count == 1
    routeback = live._read_edge_routeback(
        journal,
        idempotency_key=key,
        expected_uid=uid,
        expected_gid=gid,
    )
    assert routeback["discord_edge_receipt"] == {"receipt": "signed"}
    with pytest.raises(live.LiveCanaryError, match="edge_journal_identity_invalid"):
        live._logical_journal_snapshot(journal, expected_uid=uid + 1, expected_gid=gid)


def test_edge_journal_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.sqlite3"
    _create_edge_journal(target)
    link = tmp_path / "journal-link.sqlite3"
    link.symlink_to(target)
    monkeypatch.setattr(live, "DEFAULT_EDGE_JOURNAL", link)
    with pytest.raises(live.LiveCanaryError, match="edge_journal_identity_invalid"):
        live._logical_journal_snapshot(
            link, expected_uid=os.getuid(), expected_gid=os.getgid()
        )


@pytest.mark.parametrize("reader", ["snapshot", "routeback"])
def test_edge_journal_replacement_during_read_is_rejected(
    reader: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal.sqlite3"
    key = _create_edge_journal(journal)
    monkeypatch.setattr(live, "DEFAULT_EDGE_JOURNAL", journal)
    journal_stat = journal.stat()
    uid, gid = journal_stat.st_uid, journal_stat.st_gid
    real = live._edge_journal_identity(journal, expected_uid=uid, expected_gid=gid)
    identities = iter((real, replace(real, inode=real.inode + 1)))
    monkeypatch.setattr(
        live, "_edge_journal_identity", lambda *_a, **_k: next(identities)
    )
    with pytest.raises(live.LiveCanaryError, match="edge_journal_replaced"):
        if reader == "snapshot":
            live._logical_journal_snapshot(journal, expected_uid=uid, expected_gid=gid)
        else:
            live._read_edge_routeback(
                journal,
                idempotency_key=key,
                expected_uid=uid,
                expected_gid=gid,
            )


def test_collector_rejects_frame_not_bound_to_readiness_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    monkeypatch.setattr(live, "_validated_e2e_fixture", lambda _plan: fixture)
    collector = live.RootEvidenceCollector(_plan(tmp_path))
    collector._collector_readiness = {
        "service_identity_sha256": "1" * 64,
        "edge_service_identity_sha256": "2" * 64,
    }
    peer = live.PeerIdentity(
        pid=4242,
        uid=_identities().gateway_uid,
        gid=_identities().gateway_gid,
        start_time_ticks=31337,
    )
    frame = {
        "schema": live.PLUGIN_FRAME_SCHEMA,
        "sequence": 1,
        "event": "plugin_ready",
        "release_sha": REVISION,
        "release_sha256": RELEASE_SHA256,
        "canary_run_id": fixture["canary_run_id"],
        "case_id": fixture["case_id"],
        "fixture_sha256": FIXTURE_SHA256,
        "collector_service_identity_sha256": "1" * 64,
        "discord_edge_service_identity_sha256": "2" * 64,
        "session_id": None,
        "turn_id": None,
        "observed_at_unix_ms": NOW_MS,
        "payload": {"gateway_pid": peer.pid},
    }
    collector._validate_common_frame(frame, peer)
    bad = {**frame, "collector_service_identity_sha256": "3" * 64}
    collector._gateway_peer = None
    with pytest.raises(live.LiveCanaryError, match="collector_frame_binding_invalid"):
        collector._validate_common_frame(bad, peer)


def test_collector_cleanup_never_unlinks_a_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / name for name in ("plugin", "collector", "socket")]
    for path in paths:
        path.write_text("ours", encoding="utf-8")
    monkeypatch.setattr(live, "DEFAULT_PLUGIN_READINESS_PATH", paths[0])
    monkeypatch.setattr(live, "DEFAULT_COLLECTOR_READINESS_PATH", paths[1])
    monkeypatch.setattr(live, "DEFAULT_COLLECTOR_SOCKET", paths[2])
    collector = object.__new__(live.RootEvidenceCollector)
    collector._closed = threading.Event()
    collector._listener = None
    collector._thread = None
    collector._created_runtime_paths = {
        path: collector._runtime_path_identity(path) for path in paths
    }
    replacement = tmp_path / "replacement"
    replacement.write_text("not ours", encoding="utf-8")
    os.replace(replacement, paths[0])
    with pytest.raises(
        live.LiveCanaryError, match="collector_runtime_artifact_replaced"
    ):
        collector.close()
    assert paths[0].read_text(encoding="utf-8") == "not ours"


def test_assembler_recomputes_frame_digest_before_accepting_evidence() -> None:
    frame = live.CollectedFrame(
        value={"sequence": 1},
        sha256="0" * 64,
        chain_head_sha256="0" * 64,
        peer=live.PeerIdentity(1, 1, 1, 1),
    )
    with pytest.raises(live.LiveCanaryError, match="collector_frame_digest_invalid"):
        live.assemble_live_evidence(
            plan=object(),  # type: ignore[arg-type]
            fixture={},
            fixture_sha256="0" * 64,
            start_receipt={},
            start_receipt_file_sha256="0" * 64,
            frames=[frame],
            conversation=object(),  # type: ignore[arg-type]
            credential_provenance_sha256="0" * 64,
            private_before=live.JournalSnapshot(0, "0" * 64),
            private_after=live.JournalSnapshot(0, "0" * 64),
            projection_events=[],
            public_routeback={},
            collector_chain_head_sha256="0" * 64,
        )


def test_live_assembler_output_passes_the_packaged_offline_verifier(
    tmp_path: Path,
) -> None:
    bundle = runpy.run_path(
        str(Path(__file__).with_name("test_canonical_full_canary_e2e.py"))
    )["_bundle"]
    fixture, template = bundle()
    truth = template["canonical_truth"]
    case_id = fixture["case_id"]

    def iso(value: int) -> str:
        return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).isoformat(
            timespec="milliseconds"
        )

    projection: list[dict[str, Any]] = []
    scope_keys = {
        "canary.scope.preapproved": "canary_scope_preapproval",
        "canary.scope.claimed": "canary_scope_claim",
        "canary.scope.revoked": "canary_scope_revocation",
    }
    for event in truth["scope_events"]:
        scope = copy.deepcopy(event["scope"])
        scope.pop("session_tombstone_recorded", None)
        if "expires_at_unix_ms" in scope:
            scope["expires_at"] = iso(scope.pop("expires_at_unix_ms"))
        projection.append({
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "case_id": case_id,
            "occurred_at": iso(event["occurred_at_unix_ms"]),
            "payload": {scope_keys[event["event_type"]]: scope},
            "safety": (
                {"session_tombstone_recorded": True}
                if event["event_type"] == "canary.scope.revoked"
                else {}
            ),
        })
    for event in truth["plan_events"]:
        plan_value = copy.deepcopy(event["plan"])
        criterion_ids = plan_value.pop("criterion_ids")
        plan_value["objective"] = "model-authored objective"
        plan_value["success_criteria"] = [
            {"id": criterion_id, "content": "model-authored criterion"}
            for criterion_id in criterion_ids
        ]
        projection.append({
            "event_id": event["event_id"],
            "event_type": "task.plan.updated",
            "case_id": case_id,
            "occurred_at": iso(
                fixture["valid_from_unix_ms"] + 100 + event["plan"]["revision"]
            ),
            "payload": {"plan": plan_value},
            "safety": {},
        })
    for event in truth["verification_events"]:
        verification = copy.deepcopy(event["verification"])
        verification["receipt"] = {
            "kind": verification["receipt"]["kind"],
            "live": True,
        }
        projection.append({
            "event_id": event["event_id"],
            "event_type": "task.verification.recorded",
            "case_id": case_id,
            "occurred_at": iso(fixture["valid_from_unix_ms"] + 200),
            "payload": {"verification": verification},
            "safety": {},
        })
    routeback = truth["routeback_event"]
    projection.append({
        "event_id": routeback["event_id"],
        "event_type": "route_back.sent",
        "case_id": case_id,
        "occurred_at": iso(fixture["valid_from_unix_ms"] + 300),
        "payload": {
            "authorization_id": routeback["authorization_id"],
            "route_back": {"target_ref": routeback["target_ref"]},
            "receipt": routeback["receipt"],
        },
        "safety": {},
    })

    readback_events = [
        copy.deepcopy(event)
        for event in projection
        if event["event_type"] != "canary.scope.revoked"
    ]
    readback = {
        "request_id": truth["writer_query_request_id"],
        "status": "ok",
        "events": readback_events,
        "support_events": [],
        "view": "resume_bundle",
        "bounded": True,
        "event_count": len(readback_events),
        "truncated": False,
        "candidate_cases_truncated": False,
        "support_incomplete_reasons": [],
        "missing_verification_event_ids": [],
    }
    session_id = template["source_receipt"]["session_id"]
    turn_id = template["source_receipt"]["turn_id"]
    peer = live.PeerIdentity(4242, 2102, 2202, 31337)
    collector_identity = "4" * 64
    edge_identity = template["runtime_provenance"][
        "discord_edge_service_identity_sha256"
    ]
    raw_frames: list[dict[str, Any]] = []

    def add_frame(
        event: str,
        payload: dict[str, Any],
        *,
        session: str | None = session_id,
        turn: str | None = turn_id,
        observed: int | None = None,
    ) -> None:
        raw_frames.append({
            "schema": live.PLUGIN_FRAME_SCHEMA,
            "sequence": len(raw_frames) + 1,
            "event": event,
            "release_sha": fixture["release_sha"],
            "release_sha256": fixture["release_artifact_sha256"],
            "canary_run_id": fixture["canary_run_id"],
            "case_id": case_id,
            "fixture_sha256": template["fixture_sha256"],
            "collector_service_identity_sha256": collector_identity,
            "discord_edge_service_identity_sha256": edge_identity,
            "session_id": session,
            "turn_id": turn,
            "observed_at_unix_ms": (
                template["collected_at_unix_ms"] if observed is None else observed
            ),
            "payload": payload,
        })

    add_frame("plugin_ready", {"gateway_pid": peer.pid}, session=None, turn=None)
    claim = copy.deepcopy(truth["scope_events"][1]["scope"])
    claim.update(success=True, claimed_at="writer-authored")
    add_frame("canonical_scope_claim", claim, session=session_id, turn=None)
    add_frame("private_target_probe_ready", {}, session=session_id, turn=None)
    private_result = copy.deepcopy(template["private_denial"])
    for name in (
        "schema",
        "provenance",
        "release_sha",
        "canary_run_id",
        "session_id",
        "turn_id",
        "journal_snapshot_before",
        "journal_snapshot_after",
    ):
        private_result.pop(name)
    add_frame(
        "private_target_probe_result",
        private_result,
        session=session_id,
        turn=None,
        observed=private_result["observed_at_unix_ms"],
    )
    for model_call in template["model_calls"]:
        api_request_id = f"api:{model_call['request_ordinal']}"
        add_frame(
            "pre_api_request",
            {
                "api_request_id": api_request_id,
                "request_ordinal": model_call["request_ordinal"],
                "provider": model_call["provider"],
                "api_mode": model_call["api_mode"],
                "base_url": model_call["base_url"],
                "model": model_call["model"],
                "reasoning_effort": model_call["reasoning_effort"],
                "api_request_sha256": model_call["api_request_sha256"],
            },
        )
        add_frame(
            "post_api_request",
            {
                "api_request_id": api_request_id,
                "request_ordinal": model_call["request_ordinal"],
                "response_payload_sha256": model_call["response_payload_sha256"],
                "response_model": model_call["response_model"],
                "response_observed_at_unix_ms": model_call[
                    "response_observed_at_unix_ms"
                ],
                "assistant_tool_call_ids": model_call["assistant_tool_call_ids"],
            },
        )
        for call_id in model_call["assistant_tool_call_ids"]:
            tool_payload = {
                "api_request_id": api_request_id,
                "produced_by_model_call_ordinal": model_call["request_ordinal"],
                "tool_call_id": call_id,
                "tool_name": (
                    "todo" if call_id == "call:reasoning" else "canonical_brain_record"
                ),
                "result_sha256": "5" * 64,
            }
            if call_id == "call:reasoning":
                tool_payload.update(
                    reasoning_directive={
                        "effort": "xhigh",
                        "reason_code": "model-authored",
                    },
                    reasoning_control=template["reasoning_directive"][
                        "reasoning_control"
                    ],
                )
            add_frame("post_tool_call", tool_payload)
    add_frame(
        "canonical_case_readback",
        {
            "writer_request_id": truth["writer_query_request_id"],
            "query_view": "resume_bundle",
            "query_limit": 200,
            "readback_sha256": live._sha256_json(readback),
            "readback": readback,
        },
    )
    add_frame("session_end", {"completed": True, "interrupted": False})

    frames: list[live.CollectedFrame] = []
    chain_head = live.COLLECTOR_ZERO_CHAIN_SHA256
    for frame in raw_frames:
        frame_sha256 = live._sha256_json(frame)
        chain_head = live._sha256_json({
            "schema": live.COLLECTOR_CHAIN_SCHEMA,
            "previous_sha256": chain_head,
            "sequence": frame["sequence"],
            "frame_sha256": frame_sha256,
            "peer_pid": peer.pid,
            "peer_start_time_ticks": peer.start_time_ticks,
        })
        frames.append(live.CollectedFrame(frame, frame_sha256, chain_head, peer))

    plan = replace(
        _plan(tmp_path),
        revision=fixture["release_sha"],
        release={"artifact_sha256": fixture["release_artifact_sha256"]},
        artifacts={
            "e2e_fixture": ExactArtifact(
                source_path=tmp_path / "fixture.json",
                target_path=tmp_path / "fixture.json",
                sha256=template["fixture_sha256"],
                mode=0o440,
                uid=0,
                gid=_identities().gateway_gid,
            )
        },
    )
    start_receipt = {
        "service_identity_receipts": {
            "gateway": {
                "receipt": template["writer_readiness"],
                "sha256": template["runtime_provenance"][
                    "gateway_service_identity_sha256"
                ],
            },
            "writer": {
                "receipt": {},
                "sha256": template["runtime_provenance"][
                    "canonical_writer_service_identity_sha256"
                ],
            },
            "edge": {"receipt": {}, "sha256": edge_identity},
        },
        "collector_readiness_receipt": {
            "receipt_sha256": template["runtime_provenance"]["collector_receipt_sha256"]
        },
    }
    outcome = template["task_outcome"]
    conversation = live.SSEConversation(
        session_id=session_id,
        session_create_request_id=template["source_receipt"][
            "session_create_request_id"
        ],
        chat_stream_request_id=template["source_receipt"]["chat_stream_request_id"],
        api_run_id=template["source_receipt"]["api_run_id"],
        api_message_id=template["source_receipt"]["api_message_id"],
        events=(),
        assistant_completed={"content": "done", "status": "completed"},
        run_completed=copy.deepcopy(outcome),
        observed_at_unix_ms=template["source_receipt"]["observed_at_unix_ms"],
        completed_at_unix_ms=outcome["completed_at_unix_ms"],
    )
    private_before = live.JournalSnapshot(
        **template["private_denial"]["journal_snapshot_before"]
    )
    evidence = live.assemble_live_evidence(
        plan=plan,
        fixture=fixture,
        fixture_sha256=template["fixture_sha256"],
        start_receipt=start_receipt,
        start_receipt_file_sha256=template["runtime_provenance"][
            "full_canary_start_receipt_sha256"
        ],
        frames=frames,
        conversation=conversation,
        credential_provenance_sha256=template["source_receipt"][
            "credential_provenance_receipt_sha256"
        ],
        private_before=private_before,
        private_after=private_before,
        projection_events=projection,
        public_routeback=template["public_routeback"],
        collector_chain_head_sha256=chain_head,
        collected_at_unix_ms=template["collected_at_unix_ms"],
    )
    evidence_sha256 = live._sha256_json(evidence)
    verified = live.verify_evidence(
        fixture,
        evidence,
        start_receipt_sha256=template["runtime_provenance"][
            "full_canary_start_receipt_sha256"
        ],
        fixture_sha256=template["fixture_sha256"],
        evidence_sha256=evidence_sha256,
    )
    assert verified["ok"] is True
    assert evidence["canonical_truth"]["support_incomplete"] is False
    assert evidence["task_outcome"]["partial"] is False
