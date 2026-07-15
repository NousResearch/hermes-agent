from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import operational_edge_readiness as readiness
from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
    InMemoryCanonicalWriterBackend,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_edge_writer_authority import CanonicalWriterDiscordAuthority
from gateway.operational_edge_assets import (
    PACKAGED_ASSET_VERIFICATION_SCHEMA,
    OperationalEdgeAssetError,
    package_operational_assets,
    verify_packaged_operational_assets,
)
from gateway.operational_edge_bootstrap import (
    OperationalEdgeBootstrapError,
    build_operational_edge_foundation,
    stage_operational_edge_key_foundation,
    stage_operational_edge_key_foundation_from_staged_writer_private,
)
from gateway.operational_edge_catalog import (
    CREDENTIALS_BY_DOMAIN,
    OPERATION_PURPOSES,
    WEBSITE_RELEASE_CONTRACT_BLOCKER,
    asset_catalog,
    build_operation_argv,
    catalog_public_contract,
    operation_catalog,
    required_cron_operations,
)
from gateway.operational_edge_client import (
    OperationalEdgeClient,
    OperationalEdgeClientError,
)
from gateway.operational_edge_protocol import (
    OperationalAccess,
    OperationalCapability,
    OperationalIntent,
    OperationalProtocolError,
    OperationalRequest,
    operational_command_sha256,
    sha256_json,
    sign_envelope,
    verify_envelope,
    verify_mutation_capability,
)
from gateway.operational_edge_readiness import (
    OPERATIONAL_EDGE_READINESS_SCHEMA,
    OperationalEdgeReadinessError,
    _collect_operational_edge_readiness_as_service_peer,
    _require_current_collector_identity,
    _validate_process_status,
    build_operational_edge_readiness,
    collect_operational_edge_readiness,
    validate_operational_edge_readiness,
)
from gateway.operational_edge_service import (
    OperationalEdgePeer,
    OperationalEdgeService,
    OperationalEdgeServiceConfig,
    OperationalEdgeServiceError,
    load_config,
)
from gateway.operational_edge_units import (
    OperationalEdgeUnitError,
    render_operational_edge_units,
    service_identity_name,
    socket_group_name,
)
from ops.muncho.runtime import github_refs_collector, skyvision_email_utf8_search


REVISION = "a" * 40
BOOT_SHA = "b" * 64
NOW = 2_000_000_000


def _edge_identities() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    services: dict[str, dict[str, object]] = {}
    sockets: dict[str, dict[str, object]] = {}
    for index, domain in enumerate(sorted(CREDENTIALS_BY_DOMAIN)):
        services[domain] = {
            "user": service_identity_name(domain),
            "group": service_identity_name(domain),
            "uid": 1100 + index,
            "gid": 1200 + index,
        }
        sockets[domain] = {
            "group": socket_group_name(domain),
            "gid": 1300 + index,
        }
    return services, sockets


def _readiness_jobs() -> list[dict[str, object]]:
    catalog = operation_catalog()
    rows: list[dict[str, object]] = []
    services, sockets = _edge_identities()
    for job_id, operation_id in required_cron_operations().items():
        operation = catalog[operation_id]
        rows.append(
            {
                "source_job_id": job_id,
                "operation_id": operation_id,
                "domain": operation.domain,
                "service_unit": f"muncho-operational-edge-{operation.domain}.service",
                "service_uid": services[operation.domain]["uid"],
                "service_gid": services[operation.domain]["gid"],
                "socket_path": f"/run/muncho-operational-edge/{operation.domain}/edge.sock",
                "socket_uid": services[operation.domain]["uid"],
                "socket_gid": sockets[operation.domain]["gid"],
                "socket_mode": "0660",
                "main_pid": 1234,
                "peer_round_trip": True,
                "probe_operation_id": operation.probe_operation_id
                or operation.operation_id,
                "probe_return_code": 0,
                "probe_packet_schema": "muncho-operational-edge-probe-packet.v1",
                "probe_packet_sha256": "c" * 64,
                "meaningful_packet": True,
                "error_only_packet": False,
            }
        )
    return rows


def test_catalog_is_exactly_implemented_and_credential_scoped() -> None:
    catalog = operation_catalog()
    assets = asset_catalog()
    required = required_cron_operations()

    assert len(catalog) == 62
    assert len(required) == 14
    assert set(CREDENTIALS_BY_DOMAIN) == {item.domain for item in catalog.values()}
    assert all(item.asset_id in assets for item in catalog.values())
    assert all(required[job] == operation.operation_id for operation in catalog.values() if (job := operation.cron_source_job_id))
    assert CREDENTIALS_BY_DOMAIN["canonical"] == ()
    assert all(
        CREDENTIALS_BY_DOMAIN[domain]
        for domain in set(CREDENTIALS_BY_DOMAIN) - {"canonical"}
    )

    mutations = [item for item in catalog.values() if item.access is OperationalAccess.MUTATION]
    assert len(mutations) == 8
    assert all(item.probe_operation_id for item in mutations)
    assert all(
        catalog[item.probe_operation_id].access is OperationalAccess.READ
        for item in mutations
    )
    assert catalog["cron.canonical.heartbeat"].access is OperationalAccess.MECHANICAL
    assert catalog["cron.canonical.projections"].access is OperationalAccess.MECHANICAL
    assert catalog["infra.contabo.observe"].argv_prefix == ("instances",)
    assert catalog["infra.alwyzon.observe"].argv_prefix == ("status",)


def test_every_operation_has_static_model_readable_purpose_metadata() -> None:
    catalog = operation_catalog()
    rows = {
        row["operation_id"]: row
        for row in catalog_public_contract()["operations"]
    }

    assert set(OPERATION_PURPOSES) == set(catalog) == set(rows)
    assert all(
        operation.purpose == OPERATION_PURPOSES[operation_id]
        == rows[operation_id]["purpose"]
        for operation_id, operation in catalog.items()
    )
    assert all(operation.purpose.strip() == operation.purpose for operation in catalog.values())
    assert all(operation.purpose.endswith(".") for operation in catalog.values())


def test_invoice_lookup_requires_invoice_or_order_id_and_exposes_constraint() -> None:
    operation = operation_catalog()["skyvision.panel.invoice_lookup"]
    common = {"case_id": "case:invoice-1", "requester": "Emo"}

    with pytest.raises(
        ValueError,
        match="operation requires at least one of: invoice_id, order_id",
    ):
        build_operation_argv(operation, common)
    with pytest.raises(
        ValueError,
        match="operation requires at least one of: invoice_id, order_id",
    ):
        build_operation_argv(
            operation,
            {**common, "invoice_id": None, "order_id": None},
        )

    assert build_operation_argv(operation, {**common, "invoice_id": "inv-1"})[-2:] == (
        "--requester",
        "Emo",
    )
    assert "--invoice-id" in build_operation_argv(
        operation, {**common, "invoice_id": "inv-1"}
    )
    assert "--order-id" in build_operation_argv(
        operation, {**common, "order_id": "order-1"}
    )
    row = next(
        item
        for item in catalog_public_contract()["operations"]
        if item["operation_id"] == operation.operation_id
    )
    assert row["requires_any_of"] == [["invoice_id", "order_id"]]


def test_skyvision_deploy_gate_is_model_visible_and_fails_before_transport() -> None:
    catalog = operation_catalog()
    preflight = catalog["skyvision.deploy.preflight"]
    blocked = (
        catalog["skyvision.deploy.request_approval"],
        catalog["skyvision.deploy.execute"],
    )

    assert preflight.available is True
    assert preflight.access is OperationalAccess.READ
    assert preflight.blocker_code == ""
    assert all(operation.available is False for operation in blocked)
    assert all(
        operation.blocker_code == WEBSITE_RELEASE_CONTRACT_BLOCKER
        for operation in blocked
    )
    assert all(
        all(term in operation.availability_requirement for term in ("Node", "npm", "PM2", "canary", "soak", "rollback"))
        for operation in blocked
    )

    client = object.__new__(OperationalEdgeClient)
    client.config = SimpleNamespace(domain="skyvision_gitlab")
    with pytest.raises(
        OperationalEdgeClientError,
        match=WEBSITE_RELEASE_CONTRACT_BLOCKER,
    ) as client_error:
        client.invoke(
            "skyvision.deploy.execute",
            {},
            idempotency_key="deploy:block:1",
        )
    assert client_error.value.dispatch_uncertain is False

    intent = OperationalIntent(
        operation_id="skyvision.deploy.request_approval",
        arguments={},
        arguments_sha256=sha256_json({}),
        idempotency_key="deploy:block:2",
    )
    request = OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=0,
        deadline_unix_ms=2_000,
        intent=intent,
        capability=None,
    )
    service = object.__new__(OperationalEdgeService)
    service.operations = catalog
    with pytest.raises(
        OperationalEdgeServiceError,
        match=WEBSITE_RELEASE_CONTRACT_BLOCKER,
    ):
        service.dispatch(request, OperationalEdgePeer(pid=1, uid=1, gid=1))


def test_typed_argv_rejects_unknown_or_invalid_arguments() -> None:
    operation = operation_catalog()["skyvision.gitlab.branches"]
    assert build_operation_argv(operation, {"project": "skyvision/site", "limit": 10}) == (
        "branches",
        "--project",
        "skyvision/site",
        "--limit",
        "10",
    )
    with pytest.raises(ValueError):
        build_operation_argv(operation, {"project": "skyvision/site", "shell": "id"})
    with pytest.raises(ValueError):
        build_operation_argv(operation, {"project": "skyvision/site", "limit": 101})


def test_mutation_capability_is_exactly_bound_to_writer_plan_intent() -> None:
    private_key = Ed25519PrivateKey.generate()
    intent = OperationalIntent(
        operation_id="bitrix.crm.lead_add",
        arguments={"title": "Exact model-authored title"},
        arguments_sha256=sha256_json({"title": "Exact model-authored title"}),
        idempotency_key="case-1:step-1",
    )
    capability = OperationalCapability(
        authority_kind="canonical_owner_plan",
        authority_ref="plan:case-1:lease-7",
        operation_id=intent.operation_id,
        arguments_sha256=intent.arguments_sha256,
        idempotency_key=intent.idempotency_key,
        issued_at_unix_ms=1_000,
        expires_at_unix_ms=2_000,
    )
    envelope = sign_envelope(
        capability.to_mapping(), key_id="canonical-writer-v1", private_key=private_key
    )
    request = OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=0,
        deadline_unix_ms=2_000,
        intent=intent,
        capability=envelope,
    )
    verified = verify_mutation_capability(
        request,
        key_id="canonical-writer-v1",
        public_key=private_key.public_key(),
        now_unix_ms=1_500,
    )
    assert verified.authority_ref == "plan:case-1:lease-7"

    changed = OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=1,
        deadline_unix_ms=2_000,
        intent=OperationalIntent(
            operation_id=intent.operation_id,
            arguments=intent.arguments,
            arguments_sha256=intent.arguments_sha256,
            idempotency_key="case-1:step-2",
        ),
        capability=envelope,
    )
    with pytest.raises(OperationalProtocolError, match="capability_intent_mismatch"):
        verify_mutation_capability(
            changed,
            key_id="canonical-writer-v1",
            public_key=private_key.public_key(),
            now_unix_ms=1_500,
        )


def test_existing_writer_consume_issues_edge_capability_from_same_plan_lease() -> None:
    now = dt.datetime(2026, 7, 15, 8, 0, tzinfo=dt.timezone.utc)
    writer_key = Ed25519PrivateKey.generate()
    backend = InMemoryCanonicalWriterBackend(clock=lambda: now)
    authority = CanonicalWriterDiscordAuthority(
        capability_private_key=writer_key,
        edge_receipt_public_key=Ed25519PrivateKey.generate().public_key(),
        clock_unix_ms=lambda: int(now.timestamp() * 1000),
    )
    handlers = CanonicalWriterHandlers(
        backend,
        discord_edge_authority=authority,
    )
    runtime = RuntimeContext(
        request_id="request-1",
        platform="discord",
        session_key_sha256="1" * 64,
        capability_epoch_sha256="2" * 64,
        user_id="owner-1",
        chat_id="public-channel-1",
        thread_id="public-thread-1",
        message_id="message-1",
        owner_authenticated=True,
    )
    plan = {
        "plan_id": "plan:edge-1",
        "revision": 1,
        "objective": "Execute one exact approved operational mutation",
        "state": "active",
        "success_criteria": [{"id": "receipt", "content": "Receipt verified"}],
        "steps": [
            {
                "id": "execute",
                "content": "Execute exact bounded mutation",
                "status": "in_progress",
                "depends_on": [],
            }
        ],
        "current_step_id": "execute",
        "resume_cursor": {
            "next_step_id": "execute",
            "summary": "Execute through operational edge",
        },
    }
    transition = handlers.dispatch(
        CanonicalWriterOperation.PLAN_TRANSITION.value,
        {
            "case_id": "case:edge-1",
            "summary": "Activate owner-reviewed edge plan",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {"plan": plan},
            "idempotency_key": "edge-plan:1",
        },
        runtime=runtime,
    )
    assert transition["ok"] is True

    arguments = {
        "title": "Owner-approved lead",
        "requester": "Emo",
        "reason": "Approved plan step",
        "execute": True,
    }
    intent = OperationalIntent.from_mapping(
        {
            "operation_id": "bitrix.crm.lead_add",
            "arguments": arguments,
            "arguments_sha256": sha256_json(arguments),
            "idempotency_key": "edge-mutation:1",
        }
    )
    command_sha256 = operational_command_sha256(intent)
    granted = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        {
            "approval_id": "approval:edge-1",
            "case_id": "case:edge-1",
            "plan_id": "plan:edge-1",
            "plan_revision": 1,
            "approval_source_sha256": "3" * 64,
            "command_hashes": [command_sha256],
            "expires_at": (now + dt.timedelta(minutes=5)).isoformat(),
            "max_uses": 1,
        },
        runtime=runtime,
    )
    assert granted["ok"] is True

    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        {
            "command_sha256": command_sha256,
            "idempotency_key": "consume-edge-mutation:1",
            "operational_edge_intent": intent.to_mapping(),
        },
        runtime=runtime,
    )
    assert consumed["ok"] is True
    assert consumed["result"]["authorized"] is True
    assert consumed["result"]["remaining_uses"] == 0
    assert consumed["result"]["operational_edge_intent_sha256"] == command_sha256
    key_id = ed25519_public_key_id(writer_key.public_key())
    payload = verify_envelope(
        consumed["result"]["operational_edge_capability"],
        key_id=key_id,
        public_key=writer_key.public_key(),
        code="invalid_test_capability",
    )
    capability = OperationalCapability.from_mapping(payload)
    capability.require(intent, now_unix_ms=int(now.timestamp() * 1000))
    assert capability.authority_kind == "canonical_owner_plan"
    assert capability.authority_ref.startswith("canonical-owner-plan:")


def test_readiness_v2_is_fresh_boot_bound_and_portably_validatable() -> None:
    value = build_operational_edge_readiness(
        revision=REVISION,
        required_jobs=required_cron_operations(),
        jobs=_readiness_jobs(),
        boot_id_sha256=BOOT_SHA,
        observed_at_unix=NOW,
        collector_nonce="f1438b18-df67-46ea-ae46-5e4f3f863f09",
    )
    assert value["schema"] == OPERATIONAL_EDGE_READINESS_SCHEMA
    assert validate_operational_edge_readiness(
        value,
        revision=REVISION,
        required_jobs=required_cron_operations(),
        expected_boot_id_sha256=BOOT_SHA,
        now_unix=NOW + 120,
    ) == value
    with pytest.raises(OperationalEdgeReadinessError):
        validate_operational_edge_readiness(
            value,
            revision=REVISION,
            required_jobs=required_cron_operations(),
            expected_boot_id_sha256="d" * 64,
            now_unix=NOW,
        )
    with pytest.raises(OperationalEdgeReadinessError):
        validate_operational_edge_readiness(
            value,
            revision=REVISION,
            required_jobs=required_cron_operations(),
            expected_boot_id_sha256=BOOT_SHA,
            now_unix=NOW + 121,
        )


def test_collector_nonce_prevents_probe_receipt_replay() -> None:
    catalog = operation_catalog()
    keys: list[str] = []

    def observe(domain: str, _unit: str) -> dict[str, object]:
        return {
            "service_uid": 1001,
            "service_gid": 1002,
            "socket_path": f"/run/muncho-operational-edge/{domain}/edge.sock",
            "socket_uid": 1001,
            "socket_gid": 1003,
            "socket_mode": "0660",
            "main_pid": 1234,
            "peer_round_trip": False,
        }

    def probe(operation_id: str, key: str) -> dict[str, object]:
        keys.append(key)
        return {
            "operation_id": operation_id,
            "domain": catalog[operation_id].domain,
            "outcome": "succeeded",
            "return_code": 0,
            "readback_verified": True,
            "service_pid": 1234,
            "secret_material_recorded": False,
            "stdout_b64": base64.b64encode(b"{}\n").decode("ascii"),
        }

    first = collect_operational_edge_readiness(
        revision=REVISION,
        required_jobs=required_cron_operations(),
        service_observer=observe,
        probe_runner=probe,
        boot_id_sha256=BOOT_SHA,
        observed_at_unix=NOW,
    )
    second = collect_operational_edge_readiness(
        revision=REVISION,
        required_jobs=required_cron_operations(),
        service_observer=observe,
        probe_runner=probe,
        boot_id_sha256=BOOT_SHA,
        observed_at_unix=NOW,
    )
    first_keys, second_keys = keys[:14], keys[14:]
    assert first["collector_nonce"] != second["collector_nonce"]
    assert set(first_keys).isdisjoint(second_keys)
    assert all(f":{first['collector_nonce']}:" in key for key in first_keys)
    assert all(f":{second['collector_nonce']}:" in key for key in second_keys)


def test_process_identity_requires_all_real_effective_saved_fs_uids_and_gids() -> None:
    _validate_process_status(
        "Name:\tedge\nUid:\t1001\t1001\t1001\t1001\nGid:\t1002\t1002\t1002\t1002\n",
        expected_uid=1001,
        expected_gid=1002,
    )
    with pytest.raises(OperationalEdgeReadinessError):
        _validate_process_status(
            "Uid:\t1001\t1001\t1001\t1001\nGid:\t1002\t1002\t0\t1002\n",
            expected_uid=1001,
            expected_gid=1002,
        )


def test_root_is_publisher_only_and_never_a_live_socket_probe_peer() -> None:
    services, sockets = _edge_identities()
    supplementary = tuple(
        sorted(int(item["gid"]) for item in sockets.values())
    )
    configs = {
        domain: SimpleNamespace(
            service_uid=services[domain]["uid"],
            service_gid=services[domain]["gid"],
            socket_gid=sockets[domain]["gid"],
            probe_uid=1001,
            probe_gid=1002,
            probe_supplementary_gids=supplementary,
        )
        for domain in CREDENTIALS_BY_DOMAIN
    }
    assert _require_current_collector_identity(
        configs,
        effective_uid=1001,
        effective_gid=1002,
        effective_supplementary_gids=supplementary,
    ) == (1001, 1002, supplementary)
    with pytest.raises(
        OperationalEdgeReadinessError,
        match="operational_edge_collector_peer_unauthorized",
    ):
        _require_current_collector_identity(
            configs,
            effective_uid=0,
            effective_gid=0,
            effective_supplementary_gids=(),
        )


def test_release_pinned_root_publish_cli_never_uses_the_root_socket_peer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    receipt = {"schema": OPERATIONAL_EDGE_READINESS_SCHEMA, "ok": True}

    def publish(*, revision: str, required_jobs: dict[str, str]):
        calls.append((revision, dict(required_jobs)))
        return receipt

    monkeypatch.setattr(readiness.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        readiness,
        "collect_and_publish_operational_edge_readiness",
        publish,
    )
    monkeypatch.setattr(
        readiness,
        "collect_operational_edge_readiness_live",
        lambda **_kwargs: pytest.fail("root must not collect from sockets"),
    )
    assert readiness.main(["--publish", "--revision", REVISION]) == 0
    assert calls == [(REVISION, dict(required_cron_operations()))]
    assert json.loads(capsys.readouterr().out) == receipt


def test_packaged_cloud_collector_drops_to_exact_gateway_probe_identity() -> None:
    services, sockets = _edge_identities()
    supplementary = tuple(
        sorted(int(item["gid"]) for item in sockets.values())
    )
    configs = {
        domain: SimpleNamespace(
            service_uid=services[domain]["uid"],
            service_gid=services[domain]["gid"],
            socket_gid=sockets[domain]["gid"],
            probe_uid=1001,
            probe_gid=1002,
            probe_supplementary_gids=supplementary,
        )
        for domain in CREDENTIALS_BY_DOMAIN
    }
    receipt = build_operational_edge_readiness(
        revision=REVISION,
        required_jobs=required_cron_operations(),
        jobs=_readiness_jobs(),
        boot_id_sha256=BOOT_SHA,
        observed_at_unix=NOW,
        collector_nonce="37b73ca9-9c06-4e5e-a429-a4065e540a44",
    )
    raw = json.dumps(
        receipt,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, raw, b"")

    release = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{REVISION[:12]}"
    )
    interpreter = release / ".venv/bin/python"
    assert _collect_operational_edge_readiness_as_service_peer(
        revision=REVISION,
        required_jobs=required_cron_operations(),
        configs=configs,
        interpreter=interpreter,
        runner=runner,
        expected_boot_id_sha256=BOOT_SHA,
        now_unix=NOW,
    ) == receipt
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        str(interpreter),
        "-I",
        "-B",
        "-m",
        "gateway.operational_edge_readiness",
        "--collect-child",
        "--revision",
        REVISION,
    ]
    assert kwargs["cwd"] == str(release)
    assert kwargs["user"] == 1001
    assert kwargs["group"] == 1002
    assert kwargs["extra_groups"] == supplementary
    assert kwargs["umask"] == 0o077
    assert kwargs["start_new_session"] is True


def test_create_only_key_stager_and_pure_foundation_bind_real_key_ids(
    tmp_path: Path,
) -> None:
    key_root = tmp_path / "etc/muncho/keys"
    trust_root = tmp_path / "etc/muncho/operational-edge/trust"
    key_root.mkdir(parents=True, mode=0o755)
    os.chown(key_root, os.geteuid(), os.getegid())
    os.chown(key_root, os.geteuid(), os.getegid())
    os.chmod(key_root, 0o755)
    writer = Ed25519PrivateKey.generate()
    writer_public = writer.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    writer_path = key_root / "writer-capability-public.pem"
    writer_path.write_bytes(writer_public)
    os.chmod(writer_path, 0o440)
    writer_id = ed25519_public_key_id(writer.public_key())
    staged_root = tmp_path / "staged/keys"
    staged_root.mkdir(parents=True, mode=0o700)
    os.chown(staged_root, os.geteuid(), os.getegid())
    os.chmod(staged_root, 0o700)
    staged_writer_private = staged_root / "writer-capability-private.pem"
    staged_writer_private.write_bytes(
        writer.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(staged_writer_private, 0o400)
    pre_owner = (
        stage_operational_edge_key_foundation_from_staged_writer_private(
            expected_writer_public_key_id=writer_id,
            staged_writer_private_path=staged_writer_private,
            staging_root=staged_root,
            require_root=False,
        )
    )
    assert pre_owner["writer_public_key_id"] == writer_id
    assert pre_owner["private_content_or_digest_recorded"] is False
    assert all(
        Path(row["private_path"]).parent == staged_root
        for row in pre_owner["keys"]
    )
    with pytest.raises(OperationalEdgeBootstrapError):
        stage_operational_edge_key_foundation_from_staged_writer_private(
            expected_writer_public_key_id="0" * 64,
            staged_writer_private_path=staged_writer_private,
            staging_root=staged_root,
            require_root=False,
        )
    first = stage_operational_edge_key_foundation(
        expected_writer_public_key_id=writer_id,
        writer_public_key_gid=os.getegid(),
        writer_public_key_path=writer_path,
        key_root=key_root,
        trust_root=trust_root,
        require_root=False,
    )
    assert first["key_count"] == len(CREDENTIALS_BY_DOMAIN)
    assert all(row["created"] is True for row in first["keys"])
    assert len({row["public_key_id"] for row in first["keys"]}) == len(
        CREDENTIALS_BY_DOMAIN
    )
    assert writer_id not in {row["public_key_id"] for row in first["keys"]}
    assert first["private_content_or_digest_recorded"] is False
    second = stage_operational_edge_key_foundation(
        expected_writer_public_key_id=writer_id,
        writer_public_key_gid=os.getegid(),
        writer_public_key_path=writer_path,
        key_root=key_root,
        trust_root=trust_root,
        require_root=False,
    )
    assert all(row["created"] is False for row in second["keys"])
    assert [row["public_key_id"] for row in second["keys"]] == [
        row["public_key_id"] for row in first["keys"]
    ]

    asset_files = [
        {
            "asset_id": asset_id,
            "path": str(
                Path("/opt/adventico-ai-platform/hermes-agent-releases")
                / f"hermes-agent-{REVISION[:12]}"
                / asset.packaged_relative
            ),
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "mode": "0555",
            "size": 1,
            "sha256": hashlib.sha256(asset_id.encode("ascii")).hexdigest(),
        }
        for asset_id, asset in sorted(asset_catalog().items())
    ]
    asset_unsigned = {
        "schema": PACKAGED_ASSET_VERIFICATION_SCHEMA,
        "release_revision": REVISION,
        "manifest_sha256": "9" * 64,
        "expected_uid": os.geteuid(),
        "expected_gid": os.getegid(),
        "files": asset_files,
        "file_count": len(asset_files),
        "all_payloads_verified": True,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    asset_verification = {
        **asset_unsigned,
        "verification_sha256": hashlib.sha256(
            json.dumps(
                asset_unsigned,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    services, sockets = _edge_identities()
    foundation = build_operational_edge_foundation(
        revision=REVISION,
        service_identities=services,
        socket_groups=sockets,
        read_peer_uids=(1004,),
        gateway_uid=1004,
        gateway_gid=1005,
        release_owner_uid=os.geteuid(),
        release_owner_gid=os.getegid(),
        writer_public_key_id=writer_id,
        key_foundation=pre_owner,
        asset_verification=asset_verification,
        key_root=staged_root,
        trust_root=staged_root,
        key_uid=os.geteuid(),
        key_gid=os.getegid(),
    )
    assert foundation.manifest["key_foundation_sha256"] == pre_owner["receipt_sha256"]
    assert foundation.manifest["asset_manifest_sha256"] == "9" * 64
    assert foundation.manifest["receipt_public_key_ids"] == {
        row["domain"]: row["public_key_id"] for row in pre_owner["keys"]
    }
    assert foundation.manifest["artifact_count"] == 19
    config_artifacts = [
        item
        for item in foundation.artifacts
        if item.path.parent == Path("/etc/muncho/operational-edge")
        and item.path.suffix == ".json"
    ]
    assert len(config_artifacts) == len(CREDENTIALS_BY_DOMAIN)
    assert all(item.mode == 0o400 for item in config_artifacts)


def test_asset_packager_seals_all_62_operation_dependencies(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes-home"
    canonical = tmp_path / "canonical-brain"
    release = tmp_path / "release"
    release.mkdir()
    os.chown(release, os.geteuid(), os.getegid())
    (release / ".codex-source-commit").write_text(REVISION + "\n", encoding="ascii")
    for asset in asset_catalog().values():
        root = {"hermes": hermes_home, "canonical": canonical, "release": release}[
            asset.source_root
        ]
        source = root / asset.source_relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((f"asset:{asset.asset_id}\n").encode("ascii"))
        os.chmod(source, 0o755)
    manifest = package_operational_assets(
        release_root=release,
        revision=REVISION,
        hermes_home=hermes_home,
        canonical_brain=canonical,
    )
    assert manifest["asset_count"] == len(asset_catalog())
    assert manifest["operation_count"] == 62
    assert manifest["all_operations_implemented"] is True
    assert manifest["credential_material_packaged"] is False
    assert all(
        (release / row["packaged_relative"]).stat().st_mode & 0o777 == 0o555
        for row in manifest["assets"]
    )
    verification = verify_packaged_operational_assets(
        release_root=release,
        revision=REVISION,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    assert verification["file_count"] == len(asset_catalog())
    assert verification["all_payloads_verified"] is True
    reported_release = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{REVISION[:12]}"
    )
    reported = verify_packaged_operational_assets(
        release_root=release,
        revision=REVISION,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        reported_release_root=reported_release,
    )
    assert all(
        Path(row["path"]).is_relative_to(reported_release)
        for row in reported["files"]
    )
    tampered = release / manifest["assets"][0]["packaged_relative"]
    os.chmod(tampered, 0o755)
    with pytest.raises(OperationalEdgeAssetError):
        verify_packaged_operational_assets(
            release_root=release,
            revision=REVISION,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )


def test_units_load_only_domain_credentials_and_attest_service_gid() -> None:
    receipt_key_ids = {
        domain: f"{index:064x}"
        for index, domain in enumerate(sorted(CREDENTIALS_BY_DOMAIN), start=1)
    }
    services, sockets = _edge_identities()
    bundle = render_operational_edge_units(
        revision=REVISION,
        service_identities=services,
        socket_groups=sockets,
        release_owner_uid=1006,
        release_owner_gid=1007,
        read_peer_uids=(1004,),
        mutation_peer_uid=1004,
        mutation_peer_gid=1005,
        receipt_public_key_ids=receipt_key_ids,
        writer_key_id="f" * 64,
    )
    assert bundle.manifest["operation_count"] == 62
    assert bundle.manifest["release_owner_uid"] == 1006
    assert bundle.manifest["release_owner_gid"] == 1007
    identity = bundle.manifest["identity_contract"]
    assert identity["services"] == services
    assert identity["probe_runner"]["uid"] == 1004
    assert identity["probe_runner"]["gid"] == 1005
    assert identity["probe_runner"]["supplementary_gids"] == sorted(
        item["gid"] for item in sockets.values()
    )
    assert identity["mutation_peer"] == {
        "user": "ai-platform-brain",
        "uid": 1004,
        "gid": 1005,
        "distinct_from_services": True,
    }
    assert identity["sockets"] == {
        domain: {**row, "mode": "0660"}
        for domain, row in sockets.items()
    }
    assert identity["cross_domain_service_access"] is False
    assert identity["allowed_read_peer_uids_by_domain"] == {
        domain: [1004] for domain in CREDENTIALS_BY_DOMAIN
    }
    assert identity["root_socket_peer_allowed"] is False
    assert bundle.manifest["receipt_public_key_ids"] == receipt_key_ids
    client = json.loads(bundle.client_config)
    assert client["schema"] == "muncho-operational-edge-client-config.v3"
    assert len({row["service_uid"] for row in client["domains"].values()}) == 9
    assert len({row["service_gid"] for row in client["domains"].values()}) == 9
    assert len({row["socket_gid"] for row in client["domains"].values()}) == 9
    for raw in bundle.configs.values():
        config = json.loads(raw)
        domain = config["domain"]
        sibling_uids = {
            row["uid"] for sibling, row in services.items() if sibling != domain
        }
        assert config["allowed_read_peer_uids"] == [1004]
        assert sibling_uids.isdisjoint(config["allowed_read_peer_uids"])
        assert 0 not in config["allowed_read_peer_uids"]
        assert config["mutation_peer_uid"] == 1004
        assert config["release_owner_uid"] == 1006
        assert config["release_owner_gid"] == 1007
        assert config["receipt_key_id"] == receipt_key_ids[config["domain"]]
    for domain, credentials in CREDENTIALS_BY_DOMAIN.items():
        unit = bundle.units[f"muncho-operational-edge-{domain}.service"].decode()
        assert "EnvironmentFile=" not in unit
        assert "PassEnvironment=" not in unit
        assert f"User={service_identity_name(domain)}\n" in unit
        assert f"Group={service_identity_name(domain)}\n" in unit
        assert f"SupplementaryGroups={socket_group_name(domain)}\n" in unit
        for sibling in set(CREDENTIALS_BY_DOMAIN) - {domain}:
            assert f"User={service_identity_name(sibling)}\n" not in unit
            assert f"SupplementaryGroups={socket_group_name(sibling)}\n" not in unit
            assert (
                f"ReadWritePaths=/var/lib/muncho-operational-edge/{sibling}\n"
                not in unit
            )
            assert (
                f"/run/credentials/muncho-operational-edge-{sibling}.service/"
                not in unit
            )
        assert (
            f"ReadWritePaths=/var/lib/muncho-operational-edge/{domain}\n"
            in unit
        )
        assert "# ReleaseOwnerUID=1006\n" in unit
        assert "# ReleaseOwnerGID=1007\n" in unit
        assert (
            f"LoadCredential=service-config:/etc/muncho/operational-edge/"
            f"{domain}.json\n"
        ) in unit
        assert (
            f"--config /run/credentials/muncho-operational-edge-{domain}.service/"
            "service-config\n"
        ) in unit
        assert (
            "LoadCredential=writer-public-key:"
            "/etc/muncho/keys/writer-capability-public.pem\n"
        ) in unit
        assert "RuntimeDirectoryMode=0711\n" in unit
        if domain == "canonical":
            assert "ReadOnlyPaths=/opt/adventico-ai-platform/canonical-brain\n" in unit
            assert "ReadWritePaths=/opt/adventico-ai-platform/canonical-brain/state\n" in unit
        else:
            assert "InaccessiblePaths=-/opt/adventico-ai-platform/canonical-brain\n" in unit
            assert "ReadOnlyPaths=/opt/adventico-ai-platform/canonical-brain\n" not in unit
        for credential in credentials:
            assert f"LoadCredential={credential.name}:{credential.source_path}\n" in unit
        foreign = {
            credential.name
            for other, values in CREDENTIALS_BY_DOMAIN.items()
            if other != domain
            for credential in values
        } - {credential.name for credential in credentials}
        assert all(f"LoadCredential={name}:" not in unit for name in foreign)


def test_every_rendered_service_config_loads_with_gateway_only_peer(
    tmp_path: Path,
) -> None:
    services, sockets = _edge_identities()
    receipt_key_ids = {
        domain: f"{index:064x}"
        for index, domain in enumerate(
            sorted(CREDENTIALS_BY_DOMAIN), start=1
        )
    }
    bundle = render_operational_edge_units(
        revision=REVISION,
        service_identities=services,
        socket_groups=sockets,
        release_owner_uid=1006,
        release_owner_gid=1007,
        read_peer_uids=(1004,),
        mutation_peer_uid=1004,
        mutation_peer_gid=1005,
        receipt_public_key_ids=receipt_key_ids,
        writer_key_id="f" * 64,
    )
    for domain in sorted(CREDENTIALS_BY_DOMAIN):
        path = tmp_path / f"{domain}.json"
        path.write_bytes(
            bundle.configs[
                f"/etc/muncho/operational-edge/{domain}.json"
            ]
        )
        path.chmod(0o400)
        parsed = load_config(
            path,
            expected_owner_uid=os.geteuid(),
            require_service_credential_path=False,
        )
        assert parsed.domain == domain
        assert parsed.service_uid == services[domain]["uid"]
        assert parsed.service_gid == services[domain]["gid"]
        assert parsed.socket_gid == sockets[domain]["gid"]
        assert parsed.allowed_read_peer_uids == frozenset({1004})
        assert parsed.mutation_peer_uid == 1004


def test_service_config_accepts_only_exact_production_or_canary_release_root(
    tmp_path: Path,
) -> None:
    services, sockets = _edge_identities()
    receipt_key_ids = {
        domain: f"{index:064x}"
        for index, domain in enumerate(
            sorted(CREDENTIALS_BY_DOMAIN), start=1
        )
    }
    bundle = render_operational_edge_units(
        revision=REVISION,
        service_identities=services,
        socket_groups=sockets,
        release_owner_uid=1006,
        release_owner_gid=1007,
        read_peer_uids=(1003, 1004),
        mutation_peer_uid=1004,
        mutation_peer_gid=1005,
        receipt_public_key_ids=receipt_key_ids,
        writer_key_id="f" * 64,
    )
    raw = next(iter(bundle.configs.values()))
    production = json.loads(raw)

    def load(value: dict[str, object], name: str = "config.json"):
        path = tmp_path / name
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        path.chmod(0o400)
        return load_config(
            path,
            expected_owner_uid=os.geteuid(),
            require_service_credential_path=False,
        )

    assert load(dict(production)).release_root == Path(
        "/opt/adventico-ai-platform/hermes-agent-releases"
    ) / f"hermes-agent-{REVISION[:12]}"
    canary = {
        **production,
        "release_root": f"/opt/muncho-canary-releases/{REVISION}",
        "release_owner_uid": 0,
        "release_owner_gid": 0,
    }
    parsed = load(canary, "canary.json")
    assert parsed.release_root == Path("/opt/muncho-canary-releases") / REVISION
    assert parsed.allowed_read_peer_uids == frozenset({1003, 1004})
    assert parsed.mutation_peer_uid == 1004

    with pytest.raises(OperationalEdgeServiceError):
        load(
            {
                **canary,
                "release_owner_uid": 1006,
                "release_owner_gid": 1007,
            },
            "canary-non-root-owner.json",
        )
    with pytest.raises(OperationalEdgeServiceError):
        load(
            {
                **production,
                "release_owner_uid": 0,
                "release_owner_gid": 0,
            },
            "production-root-owner.json",
        )

    rejected = (
        f"/opt/muncho-canary-releases/{REVISION[:12]}",
        f"/opt/muncho-canary-releases/{'b' * 40}",
        f"/opt/muncho-canary-releases/{REVISION}-sibling",
        f"/opt/muncho-canary-releases/../muncho-canary-releases/{REVISION}",
    )
    for index, release_root in enumerate(rejected):
        with pytest.raises(OperationalEdgeServiceError):
            load(
                {**production, "release_root": release_root},
                f"rejected-{index}.json",
            )

    target = tmp_path / "target.json"
    target.write_text(
        json.dumps(canary, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    target.chmod(0o400)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(OperationalEdgeServiceError):
        load_config(
            symlink,
            expected_owner_uid=os.geteuid(),
            require_service_credential_path=False,
        )


def test_unit_renderer_rejects_any_cross_domain_identity_or_socket_alias() -> None:
    services, sockets = _edge_identities()
    domains = sorted(CREDENTIALS_BY_DOMAIN)
    receipt_key_ids = {
        domain: f"{index:064x}"
        for index, domain in enumerate(domains, start=1)
    }
    aliased_services = {
        domain: dict(row) for domain, row in services.items()
    }
    aliased_services[domains[1]]["uid"] = aliased_services[domains[0]]["uid"]
    with pytest.raises(OperationalEdgeUnitError):
        render_operational_edge_units(
            revision=REVISION,
            service_identities=aliased_services,
            socket_groups=sockets,
            release_owner_uid=1006,
            release_owner_gid=1007,
            read_peer_uids=(1004,),
            mutation_peer_uid=1004,
            mutation_peer_gid=1005,
            receipt_public_key_ids=receipt_key_ids,
            writer_key_id="f" * 64,
        )
    aliased_sockets = {
        domain: dict(row) for domain, row in sockets.items()
    }
    aliased_sockets[domains[1]]["gid"] = aliased_sockets[domains[0]]["gid"]
    with pytest.raises(OperationalEdgeUnitError):
        render_operational_edge_units(
            revision=REVISION,
            service_identities=services,
            socket_groups=aliased_sockets,
            release_owner_uid=1006,
            release_owner_gid=1007,
            read_peer_uids=(1004,),
            mutation_peer_uid=1004,
            mutation_peer_gid=1005,
            receipt_public_key_ids=receipt_key_ids,
            writer_key_id="f" * 64,
        )


@pytest.mark.parametrize(
    "read_peer_uids",
    (
        (),
        (1004, 1004),
        (1004, 1003),
        (1003,),
        (1004, 1100),
        tuple(range(1000, 1017)),
    ),
)
def test_unit_renderer_rejects_malformed_or_unauthorized_reader_sets(
    read_peer_uids: tuple[int, ...],
) -> None:
    services, sockets = _edge_identities()
    receipt_key_ids = {
        domain: f"{index:064x}"
        for index, domain in enumerate(
            sorted(CREDENTIALS_BY_DOMAIN), start=1
        )
    }
    with pytest.raises(OperationalEdgeUnitError):
        render_operational_edge_units(
            revision=REVISION,
            service_identities=services,
            socket_groups=sockets,
            release_owner_uid=1006,
            release_owner_gid=1007,
            read_peer_uids=read_peer_uids,
            mutation_peer_uid=1004,
            mutation_peer_gid=1005,
            receipt_public_key_ids=receipt_key_ids,
            writer_key_id="f" * 64,
        )


def test_sibling_service_uid_is_rejected_before_cross_domain_dispatch() -> None:
    services, sockets = _edge_identities()
    target_domain, sibling_domain = sorted(CREDENTIALS_BY_DOMAIN)[:2]
    operation_id = next(
        operation_id
        for operation_id, operation in operation_catalog().items()
        if operation.domain == target_domain
    )
    target = services[target_domain]
    config = OperationalEdgeServiceConfig(
        domain=target_domain,
        release_revision=REVISION,
        release_root=Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{REVISION[:12]}",
        release_owner_uid=1006,
        release_owner_gid=1007,
        socket_path=Path("/run/muncho-operational-edge")
        / target_domain
        / "edge.sock",
        socket_gid=int(sockets[target_domain]["gid"]),
        service_uid=int(target["uid"]),
        service_gid=int(target["gid"]),
        allowed_read_peer_uids=frozenset({1004}),
        mutation_peer_uid=1004,
        journal_path=Path("/var/lib/muncho-operational-edge")
        / target_domain
        / "journal.sqlite3",
        subprocess_home=Path("/opt/adventico-ai-platform"),
        receipt_private_key_file=Path("/run/credentials/receipt-private"),
        receipt_key_id="e" * 64,
        writer_public_key_file=Path("/run/credentials/writer-public"),
        writer_key_id="f" * 64,
        maximum_output_bytes=1024,
        maximum_connections=1,
    )
    service = object.__new__(OperationalEdgeService)
    service.config = config
    service.operations = {
        operation_id: operation_catalog()[operation_id]
    }
    intent = OperationalIntent(
        operation_id=operation_id,
        arguments={},
        arguments_sha256=sha256_json({}),
        idempotency_key="cross-domain-negative-test",
    )
    request = OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=0,
        deadline_unix_ms=int(dt.datetime.now().timestamp() * 1000) + 10_000,
        intent=intent,
        capability=None,
    )
    with pytest.raises(OperationalEdgeServiceError, match="peer_unauthorized"):
        service.dispatch(
            request,
            OperationalEdgePeer(
                pid=4321,
                uid=int(services[sibling_domain]["uid"]),
                gid=int(services[sibling_domain]["gid"]),
            ),
        )


def test_cyrillic_email_search_uses_bounded_pop3_local_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    helper = tmp_path / "skyvision_email_ops.py"
    helper.write_text(
        """
MAX_LIMIT = 25
def account_email(value): return value + '@example.test'
def cmd_search(args): raise AssertionError('IMAP must not receive Unicode criteria')
def pop3_search(address, args, limit, reason):
    return {'status':'PASS','transport':'pop3_fallback','address':address,
            'text':args.text,'limit':limit,'reason':reason}
""".lstrip(),
        encoding="utf-8",
    )
    code = skyvision_email_utf8_search.main(
        [
            "--asset",
            str(helper),
            "--account",
            "office",
            "--text",
            "резервация Иван",
            "--limit",
            "500",
        ]
    )
    emitted = json.loads(capsys.readouterr().out)
    assert code == 0
    assert emitted["transport"] == "pop3_fallback"
    assert emitted["text"] == "резервация Иван"
    assert emitted["limit"] == 25
    assert emitted["reason"] == "imap_ascii_criteria_incompatible"


def test_github_refs_collector_calls_exact_fixed_endpoint_table(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b'{"object":{"sha":"abc"}}', b"")

    monkeypatch.setattr(github_refs_collector.subprocess, "run", run)
    value, code = github_refs_collector.collect(Path("/sealed/gh-hermes"))
    assert code == 0
    assert value["status"] == "PASS"
    assert calls == [
        ["/sealed/gh-hermes", "api", endpoint]
        for endpoint in github_refs_collector.ENDPOINTS.values()
    ]
