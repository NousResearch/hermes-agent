

# ---------------------------------------------------------------------------
# Memory provider Desktop config endpoints
# ---------------------------------------------------------------------------

import json

import pytest


class TestMemoryProviderConfigEndpoints:
    """GET/PUT /api/memory/providers/{name}/config — generic, ABC-delegating.

    Uses a fake provider injected via plugins.memory.load_memory_provider so the
    tests don't depend on any specific bundled provider's schema.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self._monkeypatch = monkeypatch

    def _install_provider(self, provider):
        import plugins.memory as pm

        self._monkeypatch.setattr(pm, "load_memory_provider", lambda name: provider)

    @staticmethod
    def _fake_provider():
        from typing import Any, Dict, List
        from agent.memory_provider import MemoryProvider

        class _Fake(MemoryProvider):
            display_label = "Demo"

            def __init__(self):
                self.saved = None

            @property
            def name(self) -> str:
                return "demo"

            def is_available(self) -> bool:
                return True

            def initialize(self, session_id: str, **kwargs) -> None:
                pass

            def get_tool_schemas(self) -> List[Dict[str, Any]]:
                return []

            def get_config_schema(self) -> List[Dict[str, Any]]:
                return [
                    {"key": "api_key", "secret": True, "env_var": "DEMO_KEY", "description": "key"},
                    {"key": "api_url", "default": "https://default"},
                    {"key": "mode", "choices": ["cloud", "local"]},
                ]

            def save_config(self, values, hermes_home):
                self.saved = dict(values)

        return _Fake()

    def _field_map(self, payload):
        return {f["key"]: f for f in payload["fields"]}

    def test_get_returns_enriched_surface(self):
        self._install_provider(self._fake_provider())
        resp = self.client.get("/api/memory/providers/demo/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "demo"
        assert data["label"] == "Demo"
        fields = self._field_map(data)
        assert fields["api_key"]["kind"] == "secret"
        assert fields["api_url"]["value"] == "https://default"
        assert fields["mode"]["kind"] == "select"
        assert {o["value"] for o in fields["mode"]["options"]} == {"cloud", "local"}

    def test_get_never_returns_secret_value(self, monkeypatch):
        monkeypatch.setenv("DEMO_KEY", "super-secret")
        self._install_provider(self._fake_provider())
        resp = self.client.get("/api/memory/providers/demo/config")
        data = resp.json()
        fields = self._field_map(data)
        assert fields["api_key"]["value"] == ""
        assert fields["api_key"]["is_set"] is True
        assert "super-secret" not in json.dumps(data)

    def test_get_unknown_provider_returns_empty_schema(self):
        self._install_provider(None)
        resp = self.client.get("/api/memory/providers/nope/config")
        assert resp.status_code == 200
        assert resp.json()["fields"] == []

    def test_put_persists_config_and_secret(self):
        from hermes_cli.config import load_env

        provider = self._fake_provider()
        self._install_provider(provider)
        resp = self.client.put(
            "/api/memory/providers/demo/config",
            json={"values": {"api_url": "https://custom", "mode": "local", "api_key": "k-123"}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert provider.saved == {"api_url": "https://custom", "mode": "local"}
        assert load_env()["DEMO_KEY"] == "k-123"

    def test_put_does_not_change_active_provider(self):
        from hermes_cli.config import load_config

        before = load_config().get("memory", {}).get("provider")
        self._install_provider(self._fake_provider())
        self.client.put(
            "/api/memory/providers/demo/config",
            json={"values": {"api_url": "https://x"}},
        )
        after = load_config().get("memory", {}).get("provider")
        assert after == before  # saving settings != activating

    def test_put_rejects_invalid_select_value(self):
        self._install_provider(self._fake_provider())
        resp = self.client.put(
            "/api/memory/providers/demo/config",
            json={"values": {"mode": "bogus"}},
        )
        assert resp.status_code == 400
        assert "mode" in json.dumps(resp.json())

    def test_put_unknown_provider_404(self):
        self._install_provider(None)
        resp = self.client.put("/api/memory/providers/nope/config", json={"values": {}})
        assert resp.status_code == 404

    def test_put_blank_secret_does_not_overwrite(self):
        from hermes_cli.config import save_env_value, load_env

        save_env_value("DEMO_KEY", "existing")
        provider = self._fake_provider()
        self._install_provider(provider)
        self.client.put(
            "/api/memory/providers/demo/config",
            json={"values": {"api_url": "https://y", "api_key": ""}},
        )
        # blank secret left the existing env value untouched
        assert load_env()["DEMO_KEY"] == "existing"

    @staticmethod
    def _mode_gated_provider():
        """Provider with the same when-gated shape as Hindsight (api_url 2x)."""
        from typing import Any, Dict, List
        from agent.memory_provider import MemoryProvider

        class _Gated(MemoryProvider):
            def __init__(self):
                self.saved = None

            @property
            def name(self) -> str:
                return "gated"

            def is_available(self) -> bool:
                return True

            def initialize(self, session_id: str, **kwargs) -> None:
                pass

            def get_tool_schemas(self) -> List[Dict[str, Any]]:
                return []

            def get_config_schema(self) -> List[Dict[str, Any]]:
                return [
                    {"key": "mode", "choices": ["cloud", "local_external"], "default": "cloud"},
                    {"key": "api_url", "default": "https://cloud", "when": {"mode": "cloud"}},
                    {"key": "api_url", "default": "http://local", "when": {"mode": "local_external"}},
                ]

            def save_config(self, values, hermes_home):
                self.saved = dict(values)

        return _Gated()

    def test_put_when_gating_persists_only_matching_field(self):
        provider = self._mode_gated_provider()
        self._install_provider(provider)
        # mode=cloud -> only the cloud api_url applies; local row is skipped
        resp = self.client.put(
            "/api/memory/providers/gated/config",
            json={"values": {"mode": "cloud", "api_url": "https://my-cloud"}},
        )
        assert resp.status_code == 200
        assert provider.saved == {"mode": "cloud", "api_url": "https://my-cloud"}
