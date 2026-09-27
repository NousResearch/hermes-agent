"""Security boundary tests for dashboard MCP catalog credential writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import hermes_yaml as yaml
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app


HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


@pytest.fixture
def catalog_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolate_hermes_home):
    """Install one synthetic API-key catalog entry in the isolated test home."""
    from hermes_constants import get_hermes_home
    from hermes_cli.config import invalidate_env_cache

    catalog = tmp_path / "optional-mcps"
    entry_dir = catalog / "demo"
    entry_dir.mkdir(parents=True)
    (entry_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "name": "demo",
                "description": "Synthetic dashboard boundary fixture",
                "source": "https://example.test/demo",
                "transport": {
                    "type": "stdio",
                    "command": "demo-mcp",
                },
                "auth": {
                    "type": "api_key",
                    "env": [
                        {
                            "name": "DEMO_API_KEY",
                            "prompt": "Demo API key",
                            "secret": True,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(catalog))
    invalidate_env_cache()
    return get_hermes_home()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_catalog_rejects_undeclared_key_before_any_write_or_install(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hermes_cli.mcp_catalog as mcp_catalog

    installs: list[str] = []
    monkeypatch.setattr(
        mcp_catalog,
        "install_entry",
        lambda entry, enable=True, preloaded_env=None: installs.append(entry.name),
    )

    response = client.post(
        "/api/mcp/catalog/install",
        headers=HEADERS,
        json={
            "name": "demo",
            "env": {
                "DEMO_API_KEY": "valid-demo-value",
                "UNRELATED_SETTING": "must-not-land",
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "UNRELATED_SETTING" in detail
    assert "valid-demo-value" not in detail
    assert "must-not-land" not in detail
    assert installs == []
    env_path = catalog_env / ".env"
    assert not env_path.exists() or env_path.read_text(encoding="utf-8") == ""


def test_catalog_cannot_declare_reserved_control_key(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hermes_cli.mcp_catalog as mcp_catalog

    catalog_root = Path(os.environ["HERMES_OPTIONAL_MCPS"])
    manifest_path = catalog_root / "demo" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["auth"]["env"].append(
        {
            "name": "HERMES_YOLO_MODE",
            "prompt": "Unsafe control",
            "secret": False,
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    installs: list[str] = []
    monkeypatch.setattr(
        mcp_catalog,
        "install_entry",
        lambda entry, enable=True, preloaded_env=None: installs.append(entry.name),
    )

    response = client.post(
        "/api/mcp/catalog/install",
        headers=HEADERS,
        json={"name": "demo", "env": {"HERMES_YOLO_MODE": "1"}},
    )

    assert response.status_code == 400
    assert "denylist" in response.json()["detail"]
    assert installs == []
    env_path = catalog_env / ".env"
    assert not env_path.exists() or "HERMES_YOLO_MODE" not in env_path.read_text(
        encoding="utf-8"
    )


def test_catalog_accepts_declared_credential(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hermes_cli.mcp_config as mcp_config
    from agent.secret_scope import get_secret
    from tools.connectors.mcp import _CatalogBackend

    probes: list[str] = []

    def probe(name, cfg, **_kwargs):
        # The credential is in scope for the probe, and nothing is saved before it answers.
        assert get_secret("DEMO_API_KEY") == "valid-demo-value"
        assert not (catalog_env / ".env").exists()
        assert "demo" not in mcp_config._get_mcp_servers()
        probes.append(name)
        return [("demo_tool", "")]

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)

    assert _CatalogBackend().install(
        "demo", {"DEMO_API_KEY": "valid-demo-value"}
    ) == ["demo_tool"]
    assert probes == ["demo"]
    assert "demo" in mcp_config._get_mcp_servers()
    assert "DEMO_API_KEY=valid-demo-value" in (
        catalog_env / ".env"
    ).read_text(encoding="utf-8")


def test_catalog_non_secret_env_never_lands_in_env_file(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-secret declared env vars (e.g. a base URL) are not written to .env:
    install_entry inlines them into the server config instead."""
    import hermes_cli.mcp_catalog as mcp_catalog

    catalog_root = Path(os.environ["HERMES_OPTIONAL_MCPS"])
    manifest_path = catalog_root / "demo" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["auth"]["env"].append(
        {
            "name": "DEMO_BASE_URL",
            "prompt": "Demo base URL",
            "secret": False,
        }
    )
    # The transport references the non-secret var; install_entry inlines it.
    # (HTTP transport so the var lands in the server url.)
    manifest["transport"] = {"type": "http", "url": "${DEMO_BASE_URL}"}
    manifest["auth"]["type"] = "api_key"
    manifest["auth"]["env"] = [
        {"name": "MCP_DEMO_API_KEY", "prompt": "Demo API key", "secret": True},
        {"name": "DEMO_BASE_URL", "prompt": "Demo base URL", "secret": False},
    ]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    # The real install_entry probes the server after writing config; avoid
    # launching a nonexistent binary in tests.
    monkeypatch.setattr(
        mcp_catalog,
        "_probe_tools",
        lambda name: None,
    )

    response = client.post(
        "/api/mcp/catalog/install",
        headers=HEADERS,
        json={
            "name": "demo",
            "env": {
                "MCP_DEMO_API_KEY": "valid-demo-value",
                "DEMO_BASE_URL": "https://demo.example.test",
            },
        },
    )

    assert response.status_code == 200
    env_text = (catalog_env / ".env").read_text(encoding="utf-8")
    assert "MCP_DEMO_API_KEY=valid-demo-value" in env_text
    assert "DEMO_BASE_URL" not in env_text
    assert "https://demo.example.test" not in env_text
    # The non-secret is inlined into config.yaml (server config carries the
    # literal; the raw file never stores it and never keeps a ${VAR} ref).
    from hermes_cli.config import load_config

    server = load_config()["mcp_servers"]["demo"]
    assert server["url"] == "https://demo.example.test"
    assert "${DEMO_BASE_URL}" not in (
        catalog_env / "config.yaml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "protected_key",
    [
        "HERMES_YOLO_MODE",
        "HERMES_OPTIONAL_MCPS",
        "HERMES_COPILOT_ACP_COMMAND",
        "HERMES_COPILOT_ACP_ARGS",
    ],
)
def test_generic_env_endpoint_rejects_protected_key(
    client: TestClient,
    catalog_env: Path,
    protected_key: str,
):
    response = client.put(
        "/api/env",
        headers=HEADERS,
        json={"key": protected_key, "value": "must-not-land"},
    )

    assert response.status_code == 400
    env_path = catalog_env / ".env"
    assert not env_path.exists() or protected_key not in env_path.read_text(
        encoding="utf-8"
    )


def test_process_supplied_catalog_root_remains_supported(catalog_env: Path):
    from hermes_cli.mcp_catalog import get_entry

    assert get_entry("demo") is not None


def test_rejected_copilot_controls_do_not_change_live_resolvers(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from agent.copilot_acp_client import _resolve_args, _resolve_command

    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/trusted/copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")
    expected_command = _resolve_command()
    expected_args = _resolve_args()

    attempts = {
        "HERMES_COPILOT_ACP_COMMAND": "/tmp/attacker-command",
        "HERMES_COPILOT_ACP_ARGS": "--malicious-transport",
    }
    for key, value in attempts.items():
        response = client.put(
            "/api/env",
            headers=HEADERS,
            json={"key": key, "value": value},
        )
        assert response.status_code == 400

    assert _resolve_command() == expected_command
    assert _resolve_args() == expected_args
    env_path = catalog_env / ".env"
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        assert "/tmp/attacker-command" not in env_text
        assert "--malicious-transport" not in env_text


def test_preexisting_copilot_controls_remain_usable(
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from agent.copilot_acp_client import _resolve_args, _resolve_command
    from hermes_cli.env_loader import load_hermes_dotenv

    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "parent-placeholder")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--parent-placeholder")
    (catalog_env / ".env").write_text(
        "HERMES_COPILOT_ACP_COMMAND=/opt/operator/copilot\n"
        "HERMES_COPILOT_ACP_ARGS=--acp --stdio --operator-mode\n",
        encoding="utf-8",
    )

    load_hermes_dotenv(
        hermes_home=catalog_env,
        load_external_secrets=False,
    )

    assert _resolve_command() == "/opt/operator/copilot"
    assert _resolve_args() == ["--acp", "--stdio", "--operator-mode"]


def test_connection_card_install_keeps_env_file_secrets_only(
    client: TestClient,
    catalog_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The connector-card backend (Desktop/TUI/CLI setup card) makes the same secrets-only split
    as the terminal install: a declared non-secret lands in the server block, never in .env."""
    import hermes_cli.mcp_config as mcp_config
    from tools.connectors.mcp import _CatalogBackend

    catalog_root = Path(os.environ["HERMES_OPTIONAL_MCPS"])
    manifest_path = catalog_root / "demo" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["transport"]["env"] = {"DEMO_BASE_URL": "${DEMO_BASE_URL}"}
    manifest["auth"]["env"] = [
        {"name": "DEMO_API_KEY", "prompt": "Demo API key", "secret": True},
        {"name": "DEMO_BASE_URL", "prompt": "Demo base URL", "secret": False},
    ]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(mcp_config, "_probe_single_server", lambda name, cfg, **_k: [("demo_tool", "")])

    _CatalogBackend().install(
        "demo", {"DEMO_API_KEY": "valid-demo-value", "DEMO_BASE_URL": "https://demo.example.test"}
    )

    env_text = (catalog_env / ".env").read_text(encoding="utf-8")
    assert "DEMO_API_KEY=valid-demo-value" in env_text
    assert "DEMO_BASE_URL" not in env_text and "https://demo.example.test" not in env_text
    assert mcp_config._get_mcp_servers()["demo"]["env"]["DEMO_BASE_URL"] == "https://demo.example.test"
