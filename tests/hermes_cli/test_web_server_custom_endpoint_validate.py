"""Custom-endpoint validate-route key resolution tests (issue #108785).

Extracted from test_web_server.py when that file passed its FILE_LINES ratchet
cap: the validate route's per-profile saved-key fallback, profile scoping,
destination binding, and the masked key previews live here.
"""

import pytest

import hermes_yaml as yaml

__all__ = ["TestCustomEndpointValidate"]


class TestCustomEndpointValidate:
    """The /validate route's key semantics: saved-key fallback, profile scoping, destination binding."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        """Same TestClient harness as TestWebServerEndpoints, without inheriting its ~180 tests."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    @staticmethod
    def _probe_stub(monkeypatch, status_code=200, models=("m",)):
        """Stub ``httpx.AsyncClient`` for the /validate probe; returns the captured request."""
        captured = {}

        class _Resp:
            def __init__(self):
                self.status_code = status_code
                self.is_success = 200 <= status_code < 300

            def json(self):
                return {"data": [{"id": m} for m in models]}

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, *args, headers=None, **kwargs):
                captured["url"] = url
                captured["headers"] = headers
                return _Resp()

        monkeypatch.setattr("httpx.AsyncClient", _Client)
        return captured

    def _save_proxy(self, api_key="sk-saved-key-0123456789"):
        self.client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "proxy",
                "name": "Proxy",
                "base_url": "https://llm.example.com/v1",
                "model": "m",
                "api_key": api_key,
            },
        )

    def test_validate_uses_the_saved_key_when_the_field_is_blank(self, monkeypatch):
        """Save moves the key to .env and clears the form field ("Leave blank to
        keep current key"); Test on that saved endpoint must send the saved key,
        not go out unauthenticated and report the key as rejected."""
        self._save_proxy()
        captured = self._probe_stub(monkeypatch)

        response = self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"id": "proxy", "name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m"},
        )

        assert response.json()["ok"] is True
        assert captured["headers"]["Authorization"] == "Bearer sk-saved-key-0123456789"

    def test_validate_prefers_a_submitted_key_over_the_saved_one(self, monkeypatch):
        self._save_proxy(api_key="sk-old-key-0123456789")
        captured = self._probe_stub(monkeypatch)

        self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={
                "id": "proxy",
                "name": "Proxy",
                "base_url": "https://llm.example.com/v1",
                "model": "m",
                "api_key": "sk-new-key-0123456789",
            },
        )

        assert captured["headers"]["Authorization"] == "Bearer sk-new-key-0123456789"

    def test_validate_does_not_borrow_a_key_for_an_unsaved_endpoint(self, monkeypatch):
        """No ``id`` means a new endpoint: never attach some other entry's key."""
        self._save_proxy()
        captured = self._probe_stub(monkeypatch)

        self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m"},
        )

        assert "Authorization" not in captured["headers"]

    def test_validate_401_names_whether_a_key_was_sent(self, monkeypatch):
        """Three different situations used to collapse into "rejected the API key"."""
        self._probe_stub(monkeypatch, status_code=401)
        unsaved = self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"name": "New", "base_url": "https://llm.example.com/v1", "model": "m"},
        ).json()
        assert unsaved["ok"] is False and unsaved["reachable"] is True
        assert "none is saved" in unsaved["message"]

        self._save_proxy()
        self._probe_stub(monkeypatch, status_code=401)
        saved = self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"id": "proxy", "name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m"},
        ).json()
        assert "rejected the saved API key" in saved["message"]

        submitted = self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"id": "proxy", "name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m", "api_key": "sk-typed"},
        ).json()
        assert submitted["message"] == "The endpoint rejected the API key."

    def test_validate_falls_back_to_the_direct_config_model_block(self, monkeypatch):
        """The synthesized ``custom`` row (provider: custom on ``model``) has no
        providers entry; its key lives on the model block."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["model"] = {
            "provider": "custom",
            "default": "m",
            "base_url": "https://llm.example.com/v1",
            "api_key": "sk-direct-key-0123456789",
        }
        cfg.pop("providers", None)
        save_config(cfg)
        captured = self._probe_stub(monkeypatch)

        self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"id": "custom", "name": "Custom", "base_url": "https://llm.example.com/v1", "model": "m"},
        )

        assert captured["headers"]["Authorization"] == "Bearer sk-direct-key-0123456789"

    def test_validate_uses_the_key_of_a_legacy_custom_providers_row(self, monkeypatch):
        """A legacy ``custom_providers:`` list entry gets a row (its id is the slugged
        name), so Test on that row must find the key in the list entry that produced
        it — not go out unauthenticated (#108785 review)."""
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.pop("providers", None)
        cfg["custom_providers"] = [
            {"name": "Old Box", "base_url": "https://llm.example.com/v1", "model": "qwen", "api_key": "sk-leg...6789"},
        ]
        save_config(cfg)
        captured = self._probe_stub(monkeypatch)

        response = self.client.post(
            "/api/providers/custom-endpoints/validate",
            json={"id": "old-box", "name": "Old Box", "base_url": "https://llm.example.com/v1", "model": "qwen"},
        )

        assert response.json()["ok"] is True
        assert captured["headers"]["Authorization"] == "Bearer sk-leg...6789"

    def test_custom_endpoint_response_masks_the_value_behind_key_env(self):
        """The row shows what the key_env resolves to (masked), not the bare
        ``${VAR}`` template, and says so when the var is missing."""
        from hermes_cli.config import custom_endpoint_key_env, remove_env_value

        self._save_proxy(api_key="sk-live-key-0123456789abcdef")
        listed = self.client.get("/api/providers/custom-endpoints").json()
        endpoint = next(e for e in listed["endpoints"] if e["id"] == "proxy")
        assert endpoint["has_api_key"] is True
        assert endpoint["api_key_preview"] == "sk-l...cdef"
        assert "sk-live-key-0123456789abcdef" not in endpoint["api_key_preview"]

        remove_env_value(custom_endpoint_key_env("proxy"))
        listed = self.client.get("/api/providers/custom-endpoints").json()
        endpoint = next(e for e in listed["endpoints"] if e["id"] == "proxy")
        assert endpoint["has_api_key"] is False
        assert endpoint["api_key_preview"] == "${HERMES_CUSTOM_PROXY_API_KEY} (not set)"

    @staticmethod
    def _write_profile(name, config, env):
        """Create ``profiles/<name>`` with a config.yaml and .env written straight to disk (a POST
        would also mirror the key into os.environ, which is exactly what these tests must not rely on)."""
        from hermes_cli import profiles as profiles_mod

        home = profiles_mod.get_profile_dir(name)
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
        return home

    _PROXY_CFG = {
        "providers": {"proxy": {"name": "Proxy", "base_url": "https://llm.worker.example/v1", "model": "m", "key_env": "HERMES_CUSTOM_PROXY_API_KEY"}},
    }

    def _validate(self, profile=None, **overrides):
        body = {"id": "proxy", "name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m"}
        body.update(overrides)
        query = f"?profile={profile}" if profile else ""
        return self.client.post(f"/api/providers/custom-endpoints/validate{query}", json=body).json()

    def test_validate_uses_the_requested_profiles_key_not_the_process_profiles(self, monkeypatch):
        """``?profile=worker`` must probe with the WORKER's saved key. The process profile has the
        same env var (in its .env and in os.environ, as save_env_value publishes it); resolving
        through the process environment sent the parent's key to the worker's endpoint."""
        self._save_proxy(api_key="sk-parent-key-0123456789")
        monkeypatch.setenv("HERMES_CUSTOM_PROXY_API_KEY", "sk-parent-key-0123456789")
        self._write_profile("worker", self._PROXY_CFG, {"HERMES_CUSTOM_PROXY_API_KEY": "sk-worker-key-0123456789"})
        captured = self._probe_stub(monkeypatch)

        self._validate(profile="worker", base_url="https://llm.worker.example/v1")

        assert captured["headers"]["Authorization"] == "Bearer sk-worker-key-0123456789"
        listed = self.client.get("/api/providers/custom-endpoints?profile=worker").json()
        row = next(e for e in listed["endpoints"] if e["id"] == "proxy")
        assert row["api_key_preview"] == "sk-w...6789"

    def test_validate_never_borrows_the_process_key_when_the_profile_has_none(self, monkeypatch):
        """A worker whose .env lacks the var gets NO key — not the parent's os.environ value —
        and the row says the var is unset. Same for a hand-written ``api_key: ${VAR}`` ref."""
        monkeypatch.setenv("HERMES_CUSTOM_PROXY_API_KEY", "sk-parent-key-0123456789")
        monkeypatch.setenv("WORKER_REF_KEY", "sk-parent-ref-0123456789")
        self._write_profile("worker", self._PROXY_CFG, {})
        captured = self._probe_stub(monkeypatch, status_code=401)

        result = self._validate(profile="worker", base_url="https://llm.worker.example/v1")

        assert "Authorization" not in captured["headers"]
        assert "none is saved" in result["message"]
        listed = self.client.get("/api/providers/custom-endpoints?profile=worker").json()
        row = next(e for e in listed["endpoints"] if e["id"] == "proxy")
        assert row["has_api_key"] is False
        assert row["api_key_preview"] == "${HERMES_CUSTOM_PROXY_API_KEY} (not set)"

        ref_cfg = {"providers": {"proxy": {"name": "Proxy", "base_url": "https://llm.worker.example/v1", "model": "m", "api_key": "${WORKER_REF_KEY}"}}}
        self._write_profile("worker", ref_cfg, {})
        captured = self._probe_stub(monkeypatch)
        self._validate(profile="worker", base_url="https://llm.worker.example/v1")
        assert "Authorization" not in captured["headers"]

        self._write_profile("worker", ref_cfg, {"WORKER_REF_KEY": "sk-worker-ref-0123456789"})
        captured = self._probe_stub(monkeypatch)
        self._validate(profile="worker", base_url="https://llm.worker.example/v1")
        assert captured["headers"]["Authorization"] == "Bearer sk-worker-ref-0123456789"

    def test_validate_only_sends_the_saved_key_to_the_saved_destination(self, monkeypatch):
        """Editing the URL keeps the form's ``id``; Test-before-Save must not forward the old
        host's secret to the new host. Trailing slash / host case are not a different destination."""
        self._save_proxy()
        captured = self._probe_stub(monkeypatch, status_code=401)

        result = self._validate(base_url="https://attacker.example.com/v1")

        assert "Authorization" not in captured["headers"]
        assert "URL differs from the saved endpoint" in result["message"]

        for same in ("https://llm.example.com/v1/", "https://LLM.Example.com/v1"):
            captured = self._probe_stub(monkeypatch)
            self._validate(base_url=same)
            assert captured["headers"]["Authorization"] == "Bearer sk-saved-key-0123456789", same

        for other in ("http://llm.example.com/v1", "https://llm.example.com:8443/v1", "https://llm.example.com/v2"):
            captured = self._probe_stub(monkeypatch)
            self._validate(base_url=other)
            assert "Authorization" not in captured["headers"], other

    def test_validate_prefers_key_env_over_a_stale_inline_key_like_the_runtime(self, monkeypatch):
        """``runtime_provider_custom._match_new_style_provider`` resolves key_env first and uses the
        inline api_key only as a fallback; Test and the row preview must agree with it."""
        from hermes_cli.config import get_config_path, get_env_path

        get_config_path().write_text(yaml.safe_dump({
            "providers": {"proxy": {
                "name": "Proxy", "base_url": "https://llm.example.com/v1", "model": "m",
                "api_key": "sk-inline-stale-0123456789", "key_env": "PROXY_LIVE_KEY",
            }},
        }), encoding="utf-8")
        get_env_path().write_text("PROXY_LIVE_KEY=sk-env-live-0123456789\n", encoding="utf-8")
        captured = self._probe_stub(monkeypatch)

        self._validate()

        assert captured["headers"]["Authorization"] == "Bearer sk-env-live-0123456789"
        row = next(e for e in self.client.get("/api/providers/custom-endpoints").json()["endpoints"] if e["id"] == "proxy")
        assert row["api_key_preview"] == "sk-e...6789"

        # Env var gone: the inline key is the runtime's fallback, so it is Test's too.
        get_env_path().write_text("", encoding="utf-8")
        captured = self._probe_stub(monkeypatch)
        self._validate()
        assert captured["headers"]["Authorization"] == "Bearer sk-inline-stale-0123456789"
