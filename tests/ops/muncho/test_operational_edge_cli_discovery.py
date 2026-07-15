from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.operational_edge_catalog import (
    CREDENTIALS_BY_DOMAIN,
    WEBSITE_RELEASE_CONTRACT_BLOCKER,
    operation_catalog,
)
from gateway.operational_edge_units import (
    render_operational_edge_units,
    service_identity_name,
    socket_group_name,
)
from gateway.operational_edge_protocol import (
    OperationalIntent,
    operational_command_sha256,
)
from ops.muncho.runtime import operational_edge_cli as cli
from tools import skills_sync


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40


def _rendered_client_config() -> dict:
    domains = sorted(CREDENTIALS_BY_DOMAIN)
    services = {
        domain: {
            "user": service_identity_name(domain),
            "group": service_identity_name(domain),
            "uid": 2100 + index,
            "gid": 2200 + index,
        }
        for index, domain in enumerate(domains)
    }
    sockets = {
        domain: {
            "group": socket_group_name(domain),
            "gid": 2300 + index,
        }
        for index, domain in enumerate(domains)
    }
    bundle = render_operational_edge_units(
        revision=REVISION,
        service_identities=services,
        socket_groups=sockets,
        release_owner_uid=1000,
        release_owner_gid=1001,
        read_peer_uids=(1000,),
        mutation_peer_uid=1000,
        mutation_peer_gid=1001,
        receipt_public_key_ids={
            domain: f"{index:064x}"
            for index, domain in enumerate(domains, start=1)
        },
        writer_key_id="f" * 64,
    )
    return json.loads(bundle.client_config)


def test_bundled_skill_sync_exposes_the_exact_cli_to_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "bundled"
    source = ROOT / "skills/muncho-operational-edge"
    shutil.copytree(source, bundled / source.name)
    home = tmp_path / "hermes-home"
    installed = home / "skills"
    monkeypatch.setattr(skills_sync, "HERMES_HOME", home)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", installed)
    monkeypatch.setattr(
        skills_sync,
        "MANIFEST_FILE",
        installed / ".bundled_manifest",
    )
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(
        skills_sync,
        "_build_external_skill_index",
        lambda: set(),
    )

    result = skills_sync.sync_skills(quiet=True)
    assert "muncho-operational-edge" in result["copied"]
    skill = (installed / "muncho-operational-edge/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "GPT chooses the operation and arguments semantically" in skill
    assert "/opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops" in skill
    assert "muncho-ops catalog" in skill
    assert "muncho-ops schema --operation" in skill
    assert "muncho-ops authorization-hash" in skill
    assert "Discord DMs remain forbidden" in skill


def test_packaged_console_entry_and_catalog_are_exact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["muncho-ops"] == (
        "ops.muncho.runtime.operational_edge_cli:main"
    )
    assert cli.main(["catalog"]) == 0
    contract = json.loads(capsys.readouterr().out)
    assert contract["semantic_routing"] is False
    assert contract["unknown_operation_fails_closed"] is True
    assert len(contract["operations"]) == len(operation_catalog()) == 62
    assert all(row["purpose"] for row in contract["operations"])

    assert cli.main(
        ["schema", "--operation", "skyvision.db.query"]
    ) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["operation_id"] == "skyvision.db.query"
    assert schema["access"] == "read"
    assert schema["purpose"]
    assert {item["name"] for item in schema["arguments"]} >= {
        "db",
        "query",
        "case_id",
        "requester",
        "purpose",
    }

    assert cli.main(
        ["schema", "--operation", "skyvision.panel.invoice_lookup"]
    ) == 0
    invoice_schema = json.loads(capsys.readouterr().out)
    assert invoice_schema["requires_any_of"] == [["invoice_id", "order_id"]]


@pytest.mark.parametrize(
    "operation_id",
    (
        "skyvision.deploy.request_approval",
        "skyvision.deploy.execute",
    ),
)
def test_site_deploy_mutations_refuse_before_authority_or_transport(
    operation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "_consume_approved_capability",
        lambda _intent: pytest.fail("blocked deploy must not consume authority"),
    )
    monkeypatch.setattr(
        cli,
        "_config",
        lambda *_args: pytest.fail("blocked deploy must not load transport config"),
    )
    monkeypatch.setattr(
        cli,
        "OperationalEdgeClient",
        lambda *_args, **_kwargs: pytest.fail("blocked deploy must not create a client"),
    )

    with pytest.raises(SystemExit) as blocked:
        cli.main(
            [
                "invoke",
                "--operation",
                operation_id,
                "--arguments-json",
                "{}",
                "--idempotency-key",
                "site-deploy:blocked:1",
            ]
        )
    assert blocked.value.code == WEBSITE_RELEASE_CONTRACT_BLOCKER


def test_site_deploy_preflight_remains_available_in_model_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(
        ["schema", "--operation", "skyvision.deploy.preflight"]
    ) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["available"] is True
    assert schema["blocker_code"] is None

    for operation_id in (
        "skyvision.deploy.request_approval",
        "skyvision.deploy.execute",
    ):
        assert cli.main(["schema", "--operation", operation_id]) == 0
        blocked_schema = json.loads(capsys.readouterr().out)
        assert blocked_schema["available"] is False
        assert blocked_schema["blocker_code"] == WEBSITE_RELEASE_CONTRACT_BLOCKER
        requirement = blocked_schema["availability_requirement"]
        assert all(
            term in requirement
            for term in ("Node", "npm", "PM2", "canary", "soak", "rollback")
        )


def test_rendered_v3_client_config_reaches_cli_invoke_without_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rendered = _rendered_client_config()
    monkeypatch.setattr(cli, "_stable_json", lambda *_args, **_kwargs: rendered)
    monkeypatch.setattr(
        cli,
        "AttestedMainPidFileProvider",
        lambda *_args, **_kwargs: "mainpid-provider",
    )
    observed: dict[str, object] = {}

    class Client:
        def __init__(self, config, *, main_pid_provider):
            observed["config"] = config
            assert main_pid_provider == "mainpid-provider"

        def invoke(self, operation, arguments, *, idempotency_key, capability):
            observed["invoke"] = (
                operation,
                arguments,
                idempotency_key,
                capability,
            )
            return {"outcome": "succeeded", "receipt_sha256": "a" * 64}

    monkeypatch.setattr(cli, "OperationalEdgeClient", Client)
    monkeypatch.setattr(
        cli,
        "_consume_approved_capability",
        lambda _intent: pytest.fail("read must not consume owner authority"),
    )
    arguments = {
        "db": "skyvisio_fp",
        "query": "SELECT 1",
        "case_id": "case:v3-client-config",
        "requester": "Emo",
        "purpose": "Exercise the rendered v3 CLI transport boundary",
    }
    assert cli.main(
        [
            "invoke",
            "--operation",
            "skyvision.db.query",
            "--arguments-json",
            json.dumps(arguments),
            "--idempotency-key",
            "read:v3-client-config",
        ]
    ) == 0
    config = observed["config"]
    assert config.domain == "skyvision_db"
    assert config.probe_uid == 1000
    assert config.probe_gid == 1001
    assert config.probe_supplementary_gids == tuple(
        sorted(row["socket_gid"] for row in rendered["domains"].values())
    )
    assert observed["invoke"] == (
        "skyvision.db.query",
        arguments,
        "read:v3-client-config",
        None,
    )
    assert json.loads(capsys.readouterr().out)["outcome"] == "succeeded"


def test_cli_rejects_old_v2_rows_before_socket_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = json.loads(json.dumps(_rendered_client_config()))
    for row in stale["domains"].values():
        row.pop("probe_uid")
        row.pop("probe_gid")
        row.pop("probe_supplementary_gids")
    stale["schema"] = "muncho-operational-edge-client-config.v2"
    monkeypatch.setattr(cli, "_stable_json", lambda *_args, **_kwargs: stale)
    with pytest.raises(
        ValueError,
        match="operational_edge_client_config_invalid",
    ):
        cli._config(cli.DEFAULT_CLIENT_CONFIG, "skyvision_db")


def test_read_invoke_never_contacts_canonical_writer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, dict, str, object]] = []

    class Client:
        def __init__(self, config, *, main_pid_provider):
            assert config == "skyvision_db-config"
            assert main_pid_provider == "mainpid-provider"

        def invoke(self, operation, arguments, *, idempotency_key, capability):
            calls.append((operation, arguments, idempotency_key, capability))
            return {"outcome": "succeeded", "receipt_sha256": "a" * 64}

    monkeypatch.setattr(cli, "_config", lambda _path, domain: f"{domain}-config")
    monkeypatch.setattr(
        cli,
        "AttestedMainPidFileProvider",
        lambda *_args, **_kwargs: "mainpid-provider",
    )
    monkeypatch.setattr(cli, "OperationalEdgeClient", Client)
    monkeypatch.setattr(
        cli,
        "_consume_approved_capability",
        lambda _intent: pytest.fail("read must not consume owner authority"),
    )
    arguments = {
        "db": "skyvisio_fp",
        "query": "SELECT 1",
        "case_id": "case:read-1",
        "requester": "Emo",
        "purpose": "Verify one exact read path",
    }
    assert cli.main(
        [
            "invoke",
            "--operation",
            "skyvision.db.query",
            "--arguments-json",
            json.dumps(arguments),
            "--idempotency-key",
            "read-1",
        ]
    ) == 0
    assert calls == [("skyvision.db.query", arguments, "read-1", None)]
    assert json.loads(capsys.readouterr().out)["outcome"] == "succeeded"


def test_mutation_invoke_consumes_only_the_exact_approved_writer_binding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    envelope = {"payload": "signed-writer-capability"}

    def consume(intent: OperationalIntent):
        observed["intent"] = intent
        return envelope

    class Client:
        def __init__(self, _config, *, main_pid_provider):
            assert main_pid_provider == "mainpid-provider"

        def invoke(self, operation, arguments, *, idempotency_key, capability):
            observed["invoke"] = (
                operation,
                arguments,
                idempotency_key,
                capability,
            )
            return {"outcome": "succeeded", "receipt_sha256": "b" * 64}

    monkeypatch.setattr(cli, "_consume_approved_capability", consume)
    monkeypatch.setattr(cli, "_config", lambda *_args: "bitrix-config")
    monkeypatch.setattr(
        cli,
        "AttestedMainPidFileProvider",
        lambda *_args, **_kwargs: "mainpid-provider",
    )
    monkeypatch.setattr(cli, "OperationalEdgeClient", Client)
    arguments = {
        "title": "Approved lead",
        "requester": "Emo",
        "reason": "Execute approved plan step",
        "execute": True,
    }
    assert cli.main(
        [
            "invoke",
            "--operation",
            "bitrix.crm.lead_add",
            "--arguments-json",
            json.dumps(arguments),
            "--idempotency-key",
            "lead-add:approved:1",
        ]
    ) == 0
    intent = observed["intent"]
    assert isinstance(intent, OperationalIntent)
    assert intent.operation_id == "bitrix.crm.lead_add"
    assert intent.arguments == arguments
    assert intent.idempotency_key == "lead-add:approved:1"
    assert observed["invoke"] == (
        "bitrix.crm.lead_add",
        arguments,
        "lead-add:approved:1",
        envelope,
    )
    assert json.loads(capsys.readouterr().out)["outcome"] == "succeeded"


def test_writer_consume_payload_is_the_authorization_hash_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict, str]] = []
    arguments = {
        "entity_type": "deal",
        "entity_id": "42",
        "comment": "Approved note",
        "requester": "Emo",
        "reason": "Approved plan",
        "execute": True,
    }
    intent = OperationalIntent(
        operation_id="bitrix.crm.timeline_add",
        arguments=arguments,
        arguments_sha256=cli.sha256_json(arguments),
        idempotency_key="timeline:42:approved:1",
    )
    expected_hash = operational_command_sha256(intent)

    def writer_call(operation, payload, *, idempotency_key):
        calls.append((operation, dict(payload), idempotency_key))
        return {
            "authorized": True,
            "operational_edge_capability": {"signed": True},
        }

    monkeypatch.setattr(
        "gateway.canonical_writer_boundary.canonical_writer_call",
        writer_call,
    )
    assert cli._consume_approved_capability(intent) == {"signed": True}
    assert calls == [
        (
            "capability.consume",
            {
                "command_sha256": expected_hash,
                "idempotency_key": "operational-edge-consume:" + expected_hash,
                "operational_edge_intent": intent.to_mapping(),
            },
            "operational-edge-consume:" + expected_hash,
        )
    ]
