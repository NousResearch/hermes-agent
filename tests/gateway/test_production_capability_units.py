from __future__ import annotations

import hashlib
import json

import pytest

from gateway import production_capability_units as units
from gateway.browser_controller import (
    BrowserControllerConfig,
    CONFIG_SCHEMA as BROWSER_CONTROLLER_CONFIG_SCHEMA,
)
from gateway.mac_ops_edge_service import DEFAULT_PROJECT_ID
from gateway.production_capability_prerequisites import (
    BROWSER_UNIT,
    MAC_OPS_UNIT,
    PHASE_B_UNIT,
    ROUTEBACK_EDGE_UNIT,
)


REVISION = "a" * 40
NODE_SHA256 = "b" * 64
WRAPPER_SHA256 = "c" * 64
NATIVE_SHA256 = "d" * 64
CHROME_SHA256 = "e" * 64
AGENT_BROWSER_CONFIG_SHA256 = "f" * 64
RELEASE = "/opt/adventico-ai-platform/hermes-agent-releases/hermes-agent-aaaaaaaaaaaa"


def _browser_artifact_arguments() -> dict[str, str]:
    return {
        "browser_node_path": (
            f"{RELEASE}/ops/muncho/runtime/dependencies/node-linux-x64/bin/node"
        ),
        "browser_node_sha256": NODE_SHA256,
        "browser_wrapper_path": (
            f"{RELEASE}/node_modules/agent-browser/bin/agent-browser.js"
        ),
        "browser_wrapper_sha256": WRAPPER_SHA256,
        "browser_native_path": (
            f"{RELEASE}/node_modules/agent-browser/bin/agent-browser-linux-x64"
        ),
        "browser_native_sha256": NATIVE_SHA256,
        "browser_chrome_path": (
            f"{RELEASE}/ops/muncho/runtime/dependencies/chrome-linux64/chrome"
        ),
        "browser_chrome_sha256": CHROME_SHA256,
        "agent_browser_config_path": (
            f"{RELEASE}/ops/muncho/runtime/dependencies/agent-browser.json"
        ),
        "agent_browser_config_sha256": AGENT_BROWSER_CONFIG_SHA256,
    }


def _bundle() -> units.ProductionCapabilityUnitBundle:
    return units.render_production_capability_units(
        revision=REVISION,
        database_ip="10.37.4.9",
        gateway_user="hermes-cloud-gateway",
        gateway_group="hermes-cloud-gateway",
        gateway_uid=1001,
        gateway_gid=1001,
        routeback_user="muncho-discord-egress",
        routeback_group="muncho-discord-egress",
        routeback_uid=1002,
        routeback_gid=1002,
        mac_ops_user="muncho-mac-ops-edge",
        mac_ops_group="muncho-mac-ops-edge",
        mac_ops_uid=1003,
        mac_ops_gid=1003,
        browser_user="muncho-capability-browser",
        browser_group="muncho-capability-browser",
        browser_uid=1004,
        browser_gid=1004,
        socket_client_group="muncho-mac-ops-edge",
        **_browser_artifact_arguments(),
    )


def test_bundle_is_exact_release_addressed_and_self_hashed() -> None:
    bundle = _bundle()
    assert str(bundle.release_root) == RELEASE
    assert str(bundle.interpreter) == f"{RELEASE}/.venv/bin/python"
    assert set(bundle.units()) == {
        PHASE_B_UNIT,
        ROUTEBACK_EDGE_UNIT,
        MAC_OPS_UNIT,
        BROWSER_UNIT,
    }
    for name, payload in bundle.units().items():
        assert payload.endswith(b"\n")
        assert bundle.unit_sha256()[name] == hashlib.sha256(payload).hexdigest()
        assert RELEASE.encode() in payload
    assert bundle.configs() == {
        "/etc/muncho/browser-controller.json": bundle.browser_config
    }
    assert (
        bundle.browser_config_sha256
        == hashlib.sha256(bundle.browser_config).hexdigest()
    )
    manifest = bundle.manifest()
    assert manifest["units"] == bundle.unit_sha256()
    assert manifest["secret_material_recorded"] is False
    assert manifest["secret_digest_recorded"] is False
    assert manifest["configs"] == {
        "/etc/muncho/browser-controller.json": {
            "gid": 1004,
            "mode": "0440",
            "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
            "sha256": bundle.browser_config_sha256,
            "uid": 0,
        }
    }
    assert manifest["browser_controller"] == {
        "allowed_client_uid": 1001,
        "service_gid": 1004,
        "service_uid": 1004,
        "socket_path": "/run/muncho-browser-controller/controller.sock",
        "state_path": "/var/lib/muncho-browser-controller",
    }
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    assert (
        bundle.bundle_sha256
        == hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    )


def test_phase_b_is_root_read_only_preflight_with_exact_network_allow() -> None:
    text = _bundle().phase_b_unit.decode()
    assert "User=root\nGroup=root\n" in text
    assert "gateway.canonical_writer_production_cutover phase-b-preflight" in text
    assert "IPAddressDeny=any\nIPAddressAllow=10.37.4.9/32\n" in text
    assert "ProtectSystem=strict" in text
    assert (
        "LoadCredential=postgresql-pgpass:/etc/muncho-production-cutover/pgpass"
    ) in text
    assert (
        "BindReadOnlyPaths=/run/credentials/"
        f"{PHASE_B_UNIT}/postgresql-pgpass:"
        "/etc/muncho-production-cutover/pgpass"
    ) in text
    assert "ReadWritePaths=/var/lib/muncho-production-legacy-cutover/plans" in text
    assert "ReadWritePaths=/var/lib/muncho/canonical-writer-phase-b" in text
    assert "database_apply" not in text


def test_persistent_units_are_durable_and_not_reused_from_bounded_runtime() -> None:
    bundle = _bundle()
    for payload in (
        bundle.routeback_unit,
        bundle.mac_ops_unit,
        bundle.browser_unit,
    ):
        text = payload.decode()
        assert text.count("Restart=on-failure") == 1
        assert text.count("RestartSec=5s") == 1
        assert text.count("StartLimitIntervalSec=300s") == 1
        assert text.count("StartLimitBurst=5") == 1
        assert "Restart=no" not in text
        assert "RuntimeMaxSec=" not in text
        assert "capability canary" not in text.casefold()
        assert "/opt/muncho-canary" not in text


def test_routeback_and_mac_credentials_are_loadcredential_only() -> None:
    bundle = _bundle()
    routeback = bundle.routeback_unit.decode()
    mac = bundle.mac_ops_unit.decode()
    assert (
        "LoadCredential=discord-bot-token:"
        "/etc/muncho/discord-edge-credentials/bot-token"
    ) in routeback
    assert (
        "LoadCredential=discord-edge-receipt-private-key:"
        "/etc/muncho/keys/discord-edge-receipt-private.pem"
    ) in routeback
    assert "gateway.production_discord_edge_bootstrap" in routeback
    assert "gateway.full_canary_discord_edge_bootstrap" not in routeback
    assert (
        "InaccessiblePaths=/etc/muncho/discord-edge-credentials/bot-token"
    ) in routeback
    assert (
        "LoadCredential=mac-ops-gitlab-env:"
        "/etc/muncho/mac-ops-edge-credentials/gitlab.env"
    ) in mac
    assert ("InaccessiblePaths=/etc/muncho/mac-ops-edge-credentials/gitlab.env") in mac
    for text in (routeback, mac):
        assert "EnvironmentFile=" not in text
        assert "PassEnvironment=" not in text
        assert "Environment=DISCORD_BOT_TOKEN=" not in text
        assert "Environment=GITLAB_TOKEN=" not in text


def test_browser_config_is_exact_controller_schema_and_root_owned() -> None:
    bundle = _bundle()
    value = json.loads(bundle.browser_config)
    assert bundle.browser_config == json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert set(value) == {
        "schema",
        "socket_path",
        "socket_runtime_root",
        "socket_gid",
        "allowed_client_uid",
        "session_root",
        "release_root",
        "node_path",
        "node_sha256",
        "wrapper_path",
        "wrapper_sha256",
        "native_path",
        "native_sha256",
        "chrome_path",
        "chrome_sha256",
        "agent_browser_config_path",
        "agent_browser_config_sha256",
        "command_timeout_seconds",
        "idle_timeout_seconds",
        "max_connections",
        "max_sessions",
        "session_quota_bytes",
        "session_quota_entries",
    }
    assert value == {
        "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
        "socket_path": "/run/muncho-browser-controller/controller.sock",
        "socket_runtime_root": "/run/muncho-browser-controller",
        "socket_gid": 1004,
        "allowed_client_uid": 1001,
        "session_root": "/var/lib/muncho-browser-controller",
        "release_root": RELEASE,
        "node_path": _browser_artifact_arguments()["browser_node_path"],
        "node_sha256": NODE_SHA256,
        "wrapper_path": _browser_artifact_arguments()["browser_wrapper_path"],
        "wrapper_sha256": WRAPPER_SHA256,
        "native_path": _browser_artifact_arguments()["browser_native_path"],
        "native_sha256": NATIVE_SHA256,
        "chrome_path": _browser_artifact_arguments()["browser_chrome_path"],
        "chrome_sha256": CHROME_SHA256,
        "agent_browser_config_path": _browser_artifact_arguments()[
            "agent_browser_config_path"
        ],
        "agent_browser_config_sha256": AGENT_BROWSER_CONFIG_SHA256,
        "command_timeout_seconds": 120,
        "idle_timeout_seconds": 900,
        "max_connections": 8,
        "max_sessions": 4,
        "session_quota_bytes": 256 * 1024 * 1024,
        "session_quota_entries": 4096,
    }
    parsed = BrowserControllerConfig.from_mapping(value)
    assert parsed.allowed_client_uid == 1001
    assert parsed.socket_gid == 1004
    assert bundle.browser_config_path == units.BROWSER_CONFIG_PATH
    assert bundle.browser_config_uid == 0
    assert bundle.browser_config_gid == 1004
    assert bundle.browser_config_mode == 0o440
    assert bundle.browser_service_uid == 1004
    assert bundle.browser_service_gid == 1004
    assert bundle.browser_allowed_client_uid == 1001


def test_browser_unit_runs_only_dedicated_controller_and_keeps_chrome_sandbox() -> None:
    bundle = _bundle()
    text = bundle.browser_unit.decode()
    assert f"# ControllerConfigSHA256={bundle.browser_config_sha256}" in text
    assert (
        f"ExecStart={RELEASE}/.venv/bin/python -B -P -s -m "
        "gateway.browser_controller --config "
        "/etc/muncho/browser-controller.json"
    ) in text
    assert "remote-debugging" not in text
    assert "9222" not in text
    assert "SocketBindAllow=" not in text
    assert "--no-sandbox" not in text
    assert "RestrictNamespaces=yes" not in text
    assert "User=muncho-capability-browser" in text
    assert "User=hermes-cloud-gateway" not in text
    assert "Type=notify" in text
    assert "NotifyAccess=main" in text
    assert "Type=simple" not in text
    assert "RuntimeDirectory=muncho-browser-controller" in text
    assert "RuntimeDirectoryMode=0750" in text
    assert "StateDirectory=muncho-browser-controller" in text
    assert "StateDirectoryMode=0700" in text
    assert "KillMode=control-group" in text
    assert "KillMode=mixed" not in text
    assert "MemoryMax=2G" in text
    assert "MemorySwapMax=512M" in text
    assert f"ReadOnlyPaths={RELEASE}" in text
    assert "ReadOnlyPaths=/etc/muncho/browser-controller.json" in text
    assert "ReadWritePaths=/run/muncho-browser-controller" in text
    assert "ReadWritePaths=/var/lib/muncho-browser-controller" in text
    assert "ProtectHome=yes" in text


def test_browser_unit_uses_stub_dns_and_kernel_public_only_egress() -> None:
    text = _bundle().browser_unit.decode()
    allow = "IPAddressAllow=127.0.0.53/32"
    assert allow in text
    assert (
        "BindReadOnlyPaths=/run/systemd/resolve/stub-resolv.conf:/etc/resolv.conf"
    ) in text
    assert "AssertPathExists=/run/systemd/resolve/stub-resolv.conf" in text
    assert "IPAddressDeny=any" not in text
    assert text.index(allow) < text.index("IPAddressDeny=127.0.0.0/8")
    for network in units.BROWSER_NETWORK_DENY_RANGES:
        assert text.count(f"IPAddressDeny={network}\n") == 1
    for required in (
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    ):
        assert f"IPAddressDeny={required}" in text


def test_browser_unit_has_no_secret_or_inherited_proxy_surface() -> None:
    text = _bundle().browser_unit.decode()
    assert "LoadCredential=" not in text
    assert "EnvironmentFile=" not in text
    assert "PassEnvironment=" not in text
    assert "UnsetEnvironment=ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY" in text
    for path in (
        "/opt/adventico-ai-platform/hermes-home",
        "/opt/adventico-ai-platform/hermes-home/auth.json",
        "/run/credentials",
        "/etc/muncho/discord-connector-credentials",
        "/etc/muncho/discord-edge-credentials",
        "/etc/muncho/mac-ops-edge-credentials",
        "/etc/muncho/keys",
        "/etc/muncho-production-cutover",
    ):
        assert f"InaccessiblePaths={path}\n" in text
    assert "InaccessiblePaths=/etc/muncho\n" not in text
    assert "ReadOnlyPaths=/etc/muncho/browser-controller.json" in text


def test_mac_runtime_socket_uses_dedicated_client_group() -> None:
    text = _bundle().mac_ops_unit.decode()
    assert "User=muncho-mac-ops-edge" in text
    assert "Group=muncho-mac-ops-edge" in text
    assert "SupplementaryGroups=" not in text
    assert "RuntimeDirectoryMode=0750" in text


def test_configs_are_canonical_and_reference_only_runtime_credentials() -> None:
    bundle = _bundle()
    routeback = units.render_production_routeback_config(
        gateway_uid=1001,
        routeback_uid=1002,
        routeback_gid=1002,
        writer_capability_public_key_id="c" * 64,
        edge_receipt_public_key_id="d" * 64,
        connection_timeout_seconds=10,
        max_connections=4,
        api_timeout_seconds=5,
        journal_busy_timeout_ms=5_000,
        max_proof_age_ms=10_000,
    )
    routeback_value = json.loads(routeback)
    assert routeback == json.dumps(
        routeback_value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    credential_root = f"/run/credentials/{ROUTEBACK_EDGE_UNIT}"
    assert routeback_value["discord"] == {
        "api_timeout_seconds": 5.0,
        "credentials_directory": credential_root,
        "target_policy": "guild_acl",
        "token_file": f"{credential_root}/discord-bot-token",
    }
    assert routeback_value["keys"]["edge_receipt_private_key_file"] == (
        f"{credential_root}/discord-edge-receipt-private-key"
    )
    assert b"/etc/muncho/discord-edge-credentials/bot-token" not in routeback

    mac = units.render_production_mac_ops_config(
        gateway_uid=1001,
        socket_gid=1003,
        service_identity_sha256=bundle.mac_ops_sha256,
        max_connections=4,
        project_id=DEFAULT_PROJECT_ID,
        timeout_seconds=20,
        journal_busy_timeout_ms=5_000,
    )
    mac_value = json.loads(mac)
    assert mac == json.dumps(
        mac_value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert mac_value["gitlab"]["env_file"] == (
        f"/run/credentials/{MAC_OPS_UNIT}/mac-ops-gitlab-env"
    )
    assert mac_value["service"]["service_identity_sha256"] == (bundle.mac_ops_sha256)
    assert b"/etc/muncho/mac-ops-edge-credentials/gitlab.env" not in mac


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"revision": "a" * 39}, "revision"),
        ({"database_ip": "cloudsql.internal"}, "IPv4"),
        ({"database_ip": "127.0.0.1"}, "IPv4"),
        ({"gateway_user": "bad user"}, "gateway user"),
        ({"browser_node_sha256": "B" * 64}, "SHA-256"),
        (
            {
                "browser_chrome_path": (
                    "/opt/adventico-ai-platform/hermes-agent-releases/"
                    "hermes-agent-bbbbbbbbbbbb/ops/muncho/runtime/dependencies/"
                    "chrome-linux64/chrome"
                )
            },
            "release-local",
        ),
        (
            {"agent_browser_config_path": f"{RELEASE}/config.json"},
            "release-local",
        ),
    ],
)
def test_bundle_rejects_unbound_release_network_identity_or_digest(
    override: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "revision": REVISION,
        "database_ip": "10.37.4.9",
        "gateway_user": "hermes-cloud-gateway",
        "gateway_group": "hermes-cloud-gateway",
        "gateway_uid": 1001,
        "gateway_gid": 1001,
        "routeback_user": "muncho-discord-egress",
        "routeback_group": "muncho-discord-egress",
        "routeback_uid": 1002,
        "routeback_gid": 1002,
        "mac_ops_user": "muncho-mac-ops-edge",
        "mac_ops_group": "muncho-mac-ops-edge",
        "mac_ops_uid": 1003,
        "mac_ops_gid": 1003,
        "browser_user": "muncho-capability-browser",
        "browser_group": "muncho-capability-browser",
        "browser_uid": 1004,
        "browser_gid": 1004,
        "socket_client_group": "muncho-mac-ops-edge",
        **_browser_artifact_arguments(),
    }
    values.update(override)
    with pytest.raises(units.ProductionCapabilityUnitError, match=match):
        units.render_production_capability_units(**values)  # type: ignore[arg-type]


def test_config_renderers_do_not_accept_guessed_or_unpinned_identities() -> None:
    routeback_args = {
        "gateway_uid": 1001,
        "routeback_uid": 1001,
        "routeback_gid": 1002,
        "writer_capability_public_key_id": "c" * 64,
        "edge_receipt_public_key_id": "d" * 64,
        "connection_timeout_seconds": 10,
        "max_connections": 4,
        "api_timeout_seconds": 5,
        "journal_busy_timeout_ms": 5_000,
        "max_proof_age_ms": 10_000,
    }
    with pytest.raises(units.ProductionCapabilityUnitError, match="distinct"):
        units.render_production_routeback_config(**routeback_args)

    with pytest.raises(units.ProductionCapabilityUnitError, match="project"):
        units.render_production_mac_ops_config(
            gateway_uid=1001,
            socket_gid=1003,
            service_identity_sha256="e" * 64,
            max_connections=4,
            project_id="guessed-project",
            timeout_seconds=20,
            journal_busy_timeout_ms=5_000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("browser_user", "hermes-cloud-gateway"),
        ("browser_group", "muncho-discord-egress"),
        ("browser_uid", 1002),
        ("browser_gid", 1003),
    ],
)
def test_bundle_rejects_aliased_service_principals(field: str, value: object) -> None:
    arguments = {
        "revision": REVISION,
        "database_ip": "10.37.4.9",
        "gateway_user": "hermes-cloud-gateway",
        "gateway_group": "hermes-cloud-gateway",
        "gateway_uid": 1001,
        "gateway_gid": 1001,
        "routeback_user": "muncho-discord-egress",
        "routeback_group": "muncho-discord-egress",
        "routeback_uid": 1002,
        "routeback_gid": 1002,
        "mac_ops_user": "muncho-mac-ops-edge",
        "mac_ops_group": "muncho-mac-ops-edge",
        "mac_ops_uid": 1003,
        "mac_ops_gid": 1003,
        "browser_user": "muncho-capability-browser",
        "browser_group": "muncho-capability-browser",
        "browser_uid": 1004,
        "browser_gid": 1004,
        "socket_client_group": "muncho-mac-ops-edge",
        **_browser_artifact_arguments(),
    }
    arguments[field] = value
    with pytest.raises(units.ProductionCapabilityUnitError, match="pairwise distinct"):
        units.render_production_capability_units(**arguments)
