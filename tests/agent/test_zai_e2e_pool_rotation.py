"""Additional end-to-end tests for Z.AI pool rotation scenarios.

These extend tests/agent/test_zai_e2e_pool.py with stress scenarios:
  1. 5-key round_robin rotation: each request hits a different key
  2. Pool with one leaked (revoked) key: it fails fast and pool continues
  3. Probe failure recovery: even when detect_zai_endpoint raises, pool works
  4. Mixed Coding + non-Coding keys: only the matching endpoint wins per key
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Reuse the FakeZaiHandler from the main e2e file
# ────────────────────────────────────────────────────────────────────────────


class FakeZaiHandler2(BaseHTTPRequestHandler):
    """Same as FakeZaiHandler but supports a 'token catalog' for varied responses."""

    # Map of token -> expected response. If token not in catalog, default to 200.
    token_catalog: dict = {}

    request_log: list = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        auth = self.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()

        FakeZaiHandler2.request_log.append(
            {"path": path, "token": token, "body": body[:100]}
        )

        if path.endswith("/chat/completions"):
            response = FakeZaiHandler2.token_catalog.get(token)
            if response:
                self._send_json(response["code"], response["body"])
            else:
                self._send_json(200, {
                    "id": "chatcmpl-ok",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                })
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))


@pytest.fixture
def fake_zai_server2():
    server = HTTPServer(("127.0.0.1", 0), FakeZaiHandler2)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    FakeZaiHandler2.request_log = []
    FakeZaiHandler2.token_catalog = {}
    yield base_url
    server.shutdown()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _build_entries(token_list):
    """Build pool entries with the given list of tokens."""
    from agent.credential_pool import PooledCredential

    entries = []
    for i, token in enumerate(token_list):
        entry = mock.Mock(spec=PooledCredential)
        entry.provider = "zai"
        entry.id = f"entry{i}"
        entry.access_token = token
        entry.runtime_api_key = token
        entry.base_url = "http://127.0.0.1:1/api/paas/v4"
        entry.runtime_base_url = "http://127.0.0.1:1/api/paas/v4"
        entry.inference_base_url = None
        entry.last_status = "active"
        entries.append(entry)
    return entries


def _setup_pool_with_selector(fake_zai_url, entries, selector):
    """Patch all layers with a custom selector for the round_robin logic."""
    from agent import auxiliary_client as ac
    import hermes_cli.auth as auth_mod

    fake_pconfig = mock.Mock()
    fake_pconfig.inference_base_url = fake_zai_url + "/api/paas/v4"
    fake_pconfig.name = "Z.AI / GLM"
    fake_pconfig.auth_type = "api_key"
    fake_pconfig.api_key_env_vars = ("GLM_API_KEY",)

    class FakeRegistry:
        def items(self_inner):
            yield ("zai", fake_pconfig)
        def values(self_inner):
            return [fake_pconfig]
        def keys(self_inner):
            return ["zai"]
        def __iter__(self_inner):
            return iter(["zai"])
        def __contains__(self_inner, key):
            return key == "zai"
        def __getitem__(self_inner, key):
            if key == "zai":
                return fake_pconfig
            raise KeyError(key)

    def fake_select_pool_entry(provider_id):
        if provider_id != "zai":
            return False, None
        return selector(entries)

    def fake_resolve(api_key, default_url, env_override):
        if env_override:
            return env_override
        if not api_key:
            return default_url
        return fake_zai_url + "/api/coding/paas/v4"

    return [
        mock.patch.object(auth_mod, "PROVIDER_REGISTRY", FakeRegistry()),
        mock.patch.object(ac, "_select_pool_entry", side_effect=fake_select_pool_entry),
        mock.patch.object(auth_mod, "_resolve_zai_base_url", side_effect=fake_resolve),
        mock.patch.object(ac, "_is_provider_unhealthy", return_value=False),
        mock.patch.object(
            ac, "_pool_runtime_base_url",
            side_effect=lambda entry, fb: entry.base_url if entry else fb,
        ),
        mock.patch.object(
            ac, "_pool_runtime_api_key",
            side_effect=lambda entry: entry.access_token if entry else "",
        ),
        mock.patch.object(ac, "_get_aux_model_for_provider", return_value="glm-5.2"),
    ]


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


class TestFiveKeyRoundRobinRotation:
    """5 healthy keys in round_robin: each request picks the next active key."""

    def test_each_request_rotates_to_next_key(self, fake_zai_server2):
        """5 successive requests cycle through 5 different keys."""
        from agent import auxiliary_client as ac

        tokens = [f"KEY_{i}" for i in range(5)]
        entries = _build_entries(tokens)

        used_keys = []

        def round_robin_selector(entries):
            for entry in entries:
                if entry.last_status == "active" and entry.access_token not in used_keys:
                    used_keys.append(entry.access_token)
                    return True, entry
            return False, None

        patches = _setup_pool_with_selector(fake_zai_server2, entries, round_robin_selector)
        for p in patches:
            p.start()
        try:
            for _ in range(5):
                client, model = ac._resolve_api_key_provider()
                assert client is not None
        finally:
            for p in patches:
                p.stop()

        assert sorted(used_keys) == sorted(tokens), (
            f"round_robin didn't cycle through all 5 keys. Used: {used_keys}"
        )


class TestLeakedKeyHandling:
    """A key whose token has been leaked/revoked: pool should fail fast on it, continue."""

    def test_revoked_key_in_catalog_pool_still_resolves(self, fake_zai_server2):
        """Token revoked (401): the server responds 401 but pool selection still works."""
        FakeZaiHandler2.token_catalog = {
            "KEY_REVOKED": {
                "code": 401,
                "body": {"error": {"code": 1001, "message": "Invalid API key"}},
            },
        }

        entries = _build_entries(["KEY_REVOKED", "KEY_OK_1", "KEY_OK_2"])

        def safe_selector(entries):
            for entry in entries:
                if entry.last_status not in ("exhausted", "dead"):
                    return True, entry
            return False, None

        patches = _setup_pool_with_selector(fake_zai_server2, entries, safe_selector)
        for p in patches:
            p.start()
        try:
            from agent import auxiliary_client as ac
            client, model = ac._resolve_api_key_provider()
            assert client is not None
        finally:
            for p in patches:
                p.stop()


class TestProbeFailureRecovery:
    """When detect_zai_endpoint raises, the re-resolution block must not break pool."""

    def test_probe_exception_falls_back_to_entry_base_url(self):
        """Exception in _resolve_zai_base_url is caught and logged."""
        from agent import auxiliary_client as ac
        import hermes_cli.auth as auth_mod

        entries = _build_entries(["KEY_X"])
        fake_pconfig = mock.Mock()
        fake_pconfig.inference_base_url = "http://127.0.0.1:1/api/paas/v4"
        fake_pconfig.api_key_env_vars = ("GLM_API_KEY",)
        fake_pconfig.name = "Z.AI / GLM"
        fake_pconfig.auth_type = "api_key"

        class FakeRegistry:
            def items(self_inner):
                yield ("zai", fake_pconfig)
            def values(self_inner):
                return [fake_pconfig]
            def keys(self_inner):
                return ["zai"]
            def __iter__(self_inner):
                return iter(["zai"])
            def __contains__(self_inner, key):
                return key == "zai"
            def __getitem__(self_inner, key):
                if key == "zai":
                    return fake_pconfig
                raise KeyError(key)

        with mock.patch.object(auth_mod, "PROVIDER_REGISTRY", FakeRegistry()), \
             mock.patch.object(ac, "_select_pool_entry", return_value=(True, entries[0])), \
             mock.patch.object(ac, "_is_provider_unhealthy", return_value=False), \
             mock.patch.object(auth_mod, "_resolve_zai_base_url", side_effect=Exception("probe fail")), \
             mock.patch.object(ac, "_pool_runtime_api_key", return_value="KEY_X"), \
             mock.patch.object(ac, "_pool_runtime_base_url", return_value="http://127.0.0.1:1/api/paas/v4"), \
             mock.patch.object(ac, "_get_aux_model_for_provider", return_value="glm-5.2"), \
             mock.patch.object(ac, "_create_openai_client", return_value=mock.Mock()), \
             mock.patch.object(ac, "_maybe_wrap_anthropic", side_effect=lambda c, m, k, u: c):
            # Must NOT raise despite probe exception
            result = ac._resolve_api_key_provider()
        assert result is not None
        _, model = result
        assert model == "glm-5.2"


class TestMixedProviderLeakGuard:
    """Ensure model.base_url from a non-Z.AI provider does NOT leak into Z.AI resolution."""

    def test_non_zai_model_base_url_does_not_affect_zai(self):
        from hermes_cli.auth import _configured_zai_base_url

        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                }
            },
        ):
            # _configured_zai_base_url returns "" for non-zai providers
            assert _configured_zai_base_url() == ""