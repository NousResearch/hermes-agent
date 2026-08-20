"""Security boundary tests for dashboard MCP catalog credential writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
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
        lambda entry, enable=True: installs.append(entry.name),
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
        lambda entry, enable=True: installs.append(entry.name),
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
    import hermes_cli.mcp_catalog as mcp_catalog

    installs: list[str] = []
    monkeypatch.setattr(
        mcp_catalog,
        "install_entry",
        lambda entry, enable=True: installs.append(entry.name),
    )

    response = client.post(
        "/api/mcp/catalog/install",
        headers=HEADERS,
        json={"name": "demo", "env": {"DEMO_API_KEY": "valid-demo-value"}},
    )

    assert response.status_code == 200
    assert installs == ["demo"]
    assert "DEMO_API_KEY=valid-demo-value" in (
        catalog_env / ".env"
    ).read_text(encoding="utf-8")


def test_generic_env_endpoint_rejects_yolo_control_key(
    client: TestClient,
    catalog_env: Path,
):
    response = client.put(
        "/api/env",
        headers=HEADERS,
        json={"key": "HERMES_YOLO_MODE", "value": "1"},
    )

    assert response.status_code == 400
    env_path = catalog_env / ".env"
    assert not env_path.exists() or "HERMES_YOLO_MODE" not in env_path.read_text(
        encoding="utf-8"
    )