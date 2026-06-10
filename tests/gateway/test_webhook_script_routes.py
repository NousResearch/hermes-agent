"""Tests for deterministic webhook script routes.

Script routes let signed webhooks trigger one preconfigured local script without
spawning an agent or granting the webhook platform terminal tools.
"""

import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _install_script(tmp_path: Path, name: str, body: str) -> Path:
    scripts_dir = tmp_path / ".hermes" / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / name
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "admin"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_script_route_executes_without_agent(fake_home, tmp_path):
    marker = tmp_path / "marker.json"
    script = _install_script(
        fake_home,
        "webhook_script_ok.py",
        """
import json, os, pathlib, sys
body = sys.stdin.read()
pathlib.Path(r'__MARKER__').write_text(json.dumps({
    'route': os.environ.get('HERMES_WEBHOOK_ROUTE'),
    'event': os.environ.get('HERMES_WEBHOOK_EVENT_TYPE'),
    'delivery': os.environ.get('HERMES_WEBHOOK_DELIVERY_ID'),
    'payload': json.loads(body),
}), encoding='utf-8')
print('import ok')
""".replace("__MARKER__", str(marker)),
    )
    routes = {
        "import": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["export.completed"],
            "script": str(script),
            "script_timeout_seconds": 10,
        }
    }
    adapter = _make_adapter(routes)
    handle_message_calls = []

    async def _capture(event):
        handle_message_calls.append(event)

    adapter.handle_message = _capture
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/import",
            json={"event_type": "export.completed", "note": {"title": "A"}},
            headers={"X-GitHub-Delivery": "delivery-script-1"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["status"] == "script_executed"
    assert data["route"] == "import"
    assert data["event"] == "export.completed"
    assert data["stdout_bytes"] > 0
    assert handle_message_calls == []
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_data["route"] == "import"
    assert marker_data["event"] == "export.completed"
    assert marker_data["delivery"] == "delivery-script-1"
    assert marker_data["payload"]["note"]["title"] == "A"


@pytest.mark.asyncio
async def test_script_route_failure_returns_502(fake_home):
    script = _install_script(fake_home, "webhook_script_fail.py", "import sys; print('bad'); sys.exit(7)\n")
    routes = {
        "import": {
            "secret": _INSECURE_NO_AUTH,
            "script": str(script),
        }
    }
    adapter = _make_adapter(routes)
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/import",
            json={"event_type": "export.completed"},
            headers={"X-GitHub-Delivery": "delivery-script-fail-1"},
        )
        data = await resp.json()

    assert resp.status == 502
    assert data["status"] == "error"
    assert data["error"] == "Script failed"
    assert data["returncode"] == 7


def test_script_route_rejects_paths_outside_hermes_scripts(fake_home):
    outside = fake_home / "outside.py"
    outside.write_text("print('no')\n", encoding="utf-8")
    adapter = _make_adapter({})

    assert (
        adapter._validate_script_route({"script": str(outside)})
        == "script path is outside allowed Hermes scripts directories"
    )


def test_script_route_rejects_deliver_only_combo(fake_home):
    script = _install_script(fake_home, "webhook_script_ok.py", "print('ok')\n")
    adapter = _make_adapter({})

    assert adapter._validate_script_route({"script": str(script), "deliver_only": True}) == (
        "cannot combine 'script' with deliver_only=true"
    )
