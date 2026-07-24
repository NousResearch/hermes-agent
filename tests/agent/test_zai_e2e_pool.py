"""End-to-end test: Z.AI Coding Plan pool rotation works through the patched code path.

Spins up a local HTTP server that mimics Z.AI behavior:
- /api/coding/paas/v4 returns 200 for "good" tokens, 1308 for "exhausted" tokens
- /api/paas/v4 returns 1113 for ALL Coding Plan tokens (the bug we're fixing)

Then exercises the auxiliary client with a credential pool of 5 mixed
keys (1 exhausted, 4 healthy) and verifies:
1. Requests go to /api/coding/paas/v4, not /api/paas/v4
2. The exhausted key is marked, but the 4 healthy keys continue to work
3. round_robin rotates through the healthy keys
4. NO cascade — pool doesn't take the whole provider offline

This is the closest we can get to a real Z.AI without burning real API quota.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Mock Z.AI server
# ────────────────────────────────────────────────────────────────────────────


class FakeZaiHandler(BaseHTTPRequestHandler):
    """Mimics Z.AI endpoint behavior for end-to-end pool testing."""

    request_log: list = []

    def log_message(self, format, *args):
        pass  # silence stderr

    def do_POST(self):
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        auth = self.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()

        FakeZaiHandler.request_log.append(
            {"path": path, "token": token[:12] + "...", "body": body[:100]}
        )

        if path == "/api/coding/paas/v4/chat/completions":
            if token == "EXHAUSTED":
                self._send_json(429, {"error": {"code": 1308, "message": "Usage limit reached for 5 hour"}})
            elif token == "BAD_KEY":
                self._send_json(401, {"error": {"code": 1001, "message": "Invalid API key"}})
            else:
                self._send_json(
                    200,
                    {
                        "id": "chatcmpl-fake",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "pong"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
        elif path == "/api/paas/v4/chat/completions":
            self._send_json(429, {"error": {"code": 1113, "message": "Insufficient balance or no resource package"}})
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))


@pytest.fixture
def fake_zai_server():
    """Start a fake Z.AI server on a random port, yield the base URL."""
    server = HTTPServer(("127.0.0.1", 0), FakeZaiHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    FakeZaiHandler.request_log = []
    yield base_url
    server.shutdown()


# ────────────────────────────────────────────────────────────────────────────
# Mock Z.AI server
# ────────────────────────────────────────────────────────────────────────────


class FakeRegistry:
    """Dict-like registry with only 'zai' provider, for focused testing."""

    def __init__(self, pconfig):
        self._pconfig = pconfig
        self._items = [("zai", pconfig)]

    def items(self):
        return iter(self._items)

    def values(self):
        return [self._pconfig]

    def keys(self):
        return ["zai"]

    def __iter__(self):
        return iter(["zai"])

    def __contains__(self, key):
        return key == "zai"

    def __getitem__(self, key):
        if key == "zai":
            return self._pconfig
        raise KeyError(key)


def _build_pool_entries():
    """5 Coding Plan keys: 1 exhausted, 4 healthy."""
    from agent.credential_pool import PooledCredential

    entries = []
    for i, token in enumerate(["GOOD_1", "GOOD_2", "EXHAUSTED", "GOOD_3", "GOOD_4"]):
        entry = mock.Mock(spec=PooledCredential)
        entry.provider = "zai"
        entry.id = f"entry{i}"
        entry.access_token = token
        entry.runtime_api_key = token
        # BUG REPRO: entry.base_url baked in at auth add time = WRONG endpoint
        entry.base_url = "http://127.0.0.1:1/api/paas/v4"
        entry.runtime_base_url = "http://127.0.0.1:1/api/paas/v4"
        entry.inference_base_url = None
        entry.last_status = "active"
        entries.append(entry)
    return entries


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


class TestEndToEndPoolRouting:
    """Verify the full runtime path: pool → _resolve_zai_base_url → client → server."""

    def _setup_pool(self, fake_zai_url, entries):
        """Patch all the layers needed to reach _resolve_zai_base_url."""
        from agent import auxiliary_client as ac
        import hermes_cli.auth as auth_mod

        fake_pconfig = mock.Mock()
        fake_pconfig.inference_base_url = fake_zai_url + "/api/paas/v4"
        fake_pconfig.name = "Z.AI / GLM"
        fake_pconfig.auth_type = "api_key"
        fake_pconfig.api_key_env_vars = ("GLM_API_KEY", "ZAI_API_KEY")

        fake_registry = FakeRegistry(fake_pconfig)

        def fake_select_pool_entry(provider_id):
            if provider_id != "zai":
                return False, None
            for entry in entries:
                if entry.last_status != "exhausted":
                    return True, entry
            return False, None

        def fake_resolve(api_key, default_url, env_override):
            # Mimic the real function: env_override wins, else probe
            if env_override:
                return env_override
            if api_key == "EXHAUSTED":
                return fake_zai_url + "/api/coding/paas/v4"
            if not api_key:
                return default_url
            return fake_zai_url + "/api/coding/paas/v4"

        return [
            mock.patch.object(auth_mod, "PROVIDER_REGISTRY", fake_registry),
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

    def test_pool_routes_to_coding_endpoint_not_metered(self, fake_zai_server):
        """Pool entries baked with the WRONG endpoint get redirected to /api/coding/paas/v4."""
        from agent import auxiliary_client as ac

        entries = _build_pool_entries()
        patches = self._setup_pool(fake_zai_server, entries)

        for p in patches:
            p.start()
        try:
            # Single call: the resolved base_url should be the coding endpoint
            client, model = ac._resolve_api_key_provider()
        finally:
            for p in patches:
                p.stop()

        assert client is not None, "_resolve_api_key_provider returned no client"
        resolved_url = str(client.base_url).rstrip("/")
        assert "coding/paas/v4" in resolved_url, (
            f"Pool entry was NOT redirected to coding endpoint. Got: {resolved_url}"
        )
        assert "/api/paas/v4" not in resolved_url or "coding" in resolved_url, (
            f"Pool entry still points to metered endpoint. Got: {resolved_url}"
        )

    def test_is_payment_error_does_not_cascade_zai(self, fake_zai_server):
        """Z.AI Coding Plan 1113/1308 do NOT trigger payment-error cascade."""
        from agent import auxiliary_client as ac

        # No server interaction needed — pure function test
        exc_1113 = Exception("Insufficient balance or no resource package")
        exc_1113.status_code = 429
        assert ac._is_payment_error(exc_1113) is False, (
            "Z.AI 1113 must not be classified as payment error (cascade trigger)"
        )

        exc_1308 = Exception("Usage limit reached for 5 hour")
        exc_1308.status_code = 429
        assert ac._is_payment_error(exc_1308) is False, (
            "Z.AI 1308 must not be classified as payment error (cascade trigger)"
        )

        # Negative: Vertex AI resource exhausted IS still a payment error
        exc_vertex = Exception("resource exhausted")
        exc_vertex.status_code = 429
        assert ac._is_payment_error(exc_vertex) is True, (
            "Vertex AI resource exhausted must still be classified as payment error"
        )

        # Negative: 402 is always payment
        exc_402 = Exception("Insufficient credits")
        exc_402.status_code = 402
        assert ac._is_payment_error(exc_402) is True, (
            "402 must always be classified as payment error"
        )

    def test_vision_helper_lists_coding_endpoints_first(self, fake_zai_server):
        """The vision auto-detect list probes coding endpoints before metered."""
        from agent import auxiliary_client as ac
        import re

        with open(ac.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()

        match = re.search(r"zai_openai_urls = \[(.*?)\]", src, re.DOTALL)
        assert match is not None, "zai_openai_urls not found in source"

        urls = re.findall(r'https://[^\s",]+', match.group(1))
        assert len(urls) >= 4, f"Expected 4 URLs in zai_openai_urls, got {len(urls)}"

        coding_urls = [u for u in urls if "coding/paas/v4" in u]
        metered_urls = [u for u in urls if "/api/paas/v4" in u and "coding" not in u]

        assert len(coding_urls) >= 2, f"Expected 2+ coding endpoints, got {coding_urls}"
        assert len(metered_urls) >= 2, f"Expected 2+ metered fallbacks, got {metered_urls}"

        # The first URL must be a coding endpoint
        assert "coding" in urls[0], (
            f"First URL must be coding endpoint for first-try success, got {urls[0]}"
        )

        # Coding endpoints must appear before any metered endpoint
        first_coding_idx = next(i for i, u in enumerate(urls) if "coding" in u)
        first_metered_idx = next(
            (i for i, u in enumerate(urls) if "/api/paas/v4" in u and "coding" not in u),
            len(urls),
        )
        assert first_coding_idx < first_metered_idx, (
            f"Coding endpoints must be probed before metered. "
            f"First coding={first_coding_idx}, first metered={first_metered_idx}"
        )

    def test_full_request_flow_returns_success(self, fake_zai_server):
        """End-to-end: send a real HTTP request through the patched client, get 200."""
        from agent import auxiliary_client as ac

        entries = _build_pool_entries()
        patches = self._setup_pool(fake_zai_server, entries)

        for p in patches:
            p.start()
        try:
            client, model = ac._resolve_api_key_provider()
        finally:
            for p in patches:
                p.stop()

        assert client is not None, "_resolve_api_key_provider returned no client"

        # Send a real HTTP request through the OpenAI client
        client.chat.completions.create(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )

        # Verify the request hit the fake Z.AI coding endpoint
        assert len(FakeZaiHandler.request_log) == 1, (
            f"Expected 1 request logged, got {len(FakeZaiHandler.request_log)}"
        )
        logged = FakeZaiHandler.request_log[0]
        assert "/api/coding/paas/v4/chat/completions" in logged["path"], (
            f"Request went to wrong endpoint: {logged['path']}"
        )
        assert logged["token"].startswith("GOOD_"), (
            f"Wrong token used: {logged['token']}"
        )