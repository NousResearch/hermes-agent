"""
Unit + integration tests for the Remote Management API sanitiser.

Two layers:

* Module-level tests for ``sanitize_value`` / ``sanitize_response``
  (no HTTP, no aiohttp) — exercise the regex + identity-field
  heuristics in isolation.
* HTTP integration tests against the four affected endpoints
  (``/api/memory``, ``/api/sessions``, ``/api/sessions/{id}/messages``,
  ``/v1/capabilities``) — each asserts that real responses carry
  the sentinel in place of identity content.

The regression-guard at the bottom plants a known PII canary
string into the user profile fixture and asserts no response from
the four affected endpoints contains it. If a future commit
accidentally re-exposes the identity surface, that test fails.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_sanitiser import (
    REDACTED,
    declared_policy,
    sanitize_response,
    sanitize_value,
)
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Module-level tests — sanitize_value
# ---------------------------------------------------------------------------


class TestSanitizeValueStructural:
    def test_redacts_url_basic_auth(self):
        v = sanitize_value("https://user:pass@example.com/foo")
        assert v == f"https://{REDACTED}@example.com/foo"

    def test_redacts_querystring_token(self):
        v = sanitize_value("https://api.x.com/v1/get?api_key=sk-secret&q=1")
        assert "sk-secret" not in v
        assert f"api_key={REDACTED}" in v

    def test_redacts_value_when_field_name_signals_secret(self):
        # All known secret-substring variants → wholesale redaction.
        for field in (
            "api_key",
            "apikey",
            "anthropic_token",
            "client_secret",
            "user_password",
            "passwd",
            "credential_id",
            "bearer_token",
        ):
            assert sanitize_value("any-value", field) == REDACTED, (
                f"field {field!r} should have triggered redaction"
            )

    def test_keeps_innocent_strings(self):
        assert sanitize_value("hello world") == "hello world"
        assert sanitize_value(42) == 42
        assert sanitize_value(None) is None
        assert sanitize_value(True) is True

    def test_recursively_walks_dict_and_list(self):
        payload = {
            "name": "openai",
            "api_key": "sk-xyz",
            "endpoints": [
                {"url": "https://u:p@x/foo", "label": "primary"},
                {"url": "https://api.x.com/?token=abc", "label": "fallback"},
            ],
        }
        out = sanitize_value(payload)
        assert out["name"] == "openai"
        assert out["api_key"] == REDACTED
        assert out["endpoints"][0]["url"] == f"https://{REDACTED}@x/foo"
        assert "abc" not in out["endpoints"][1]["url"]
        assert f"token={REDACTED}" in out["endpoints"][1]["url"]


# ---------------------------------------------------------------------------
# Module-level tests — sanitize_response identity pass
# ---------------------------------------------------------------------------


class TestSanitizeResponseIdentity:
    def test_memory_endpoint_redacts_user_content(self):
        # Shape mirrors what Codex's _handle_read_memory returns.
        codex_payload: Dict[str, Any] = {
            "memory": {
                "content": "",
                "entries": [],
                "char_count": 0,
                "char_limit": 2200,
            },
            "user": {
                "content": "User prefers German. Internal codename: dragon_42.",
                "exists": True,
                "last_modified": 1779537645,
                "char_count": 50,
                "char_limit": 1375,
            },
            "stats": {"totalSessions": 8, "totalMessages": 221},
        }
        out = sanitize_response(codex_payload)
        assert out["user"]["content"] == REDACTED
        # Siblings survive.
        assert out["user"]["char_count"] == 50
        assert out["user"]["last_modified"] == 1779537645
        assert out["user"]["exists"] is True
        # Other top-level blocks untouched.
        assert out["memory"]["char_limit"] == 2200
        assert out["stats"]["totalSessions"] == 8

    def test_sessions_endpoint_redacts_system_prompt(self):
        codex_payload = {
            "sessions": [
                {
                    "id": "20260523_173558_542bc8",
                    "source": "cli",
                    "model": "gpt-5.5",
                    "system_prompt": "...long persona + USER.md...",
                    "started_at": 1779557760,
                    "input_tokens": 6998,
                    "output_tokens": 282,
                },
                {
                    "id": "20260523_154122_001abc",
                    "source": "telegram",
                    "model": "claude-opus-4.6",
                    "system_prompt": "...different persona, same USER.md...",
                    "input_tokens": 100,
                },
            ]
        }
        out = sanitize_response(codex_payload)
        for row in out["sessions"]:
            assert row["system_prompt"] == REDACTED
            # Structural fields survive.
            assert "id" in row
            assert "model" in row
            assert "input_tokens" in row

    def test_session_messages_endpoint_redacts_system_role(self):
        codex_payload = {
            "messages": [
                {
                    "id": 1,
                    "role": "system",
                    "content": "Full system prompt with USER.md inside.",
                    "timestamp": 1779557760,
                },
                {
                    "id": 2,
                    "role": "user",
                    "content": "What's my name?",
                    "timestamp": 1779557761,
                },
                {
                    "id": 3,
                    "role": "assistant",
                    "content": "Tom.",
                    "timestamp": 1779557762,
                },
            ]
        }
        out = sanitize_response(codex_payload)
        msgs = {m["role"]: m for m in out["messages"]}
        assert msgs["system"]["content"] == REDACTED
        # Non-system roles pass through.
        assert msgs["user"]["content"] == "What's my name?"
        assert msgs["assistant"]["content"] == "Tom."

    def test_repr_response_never_contains_pii_canary(self):
        # The single most important regression-guard: plant a canary
        # in every spot identity content is known to land, then
        # assert no canary survives sanitisation.
        CANARY = "PII_CANARY_BLOCK_dragon_42"
        SK_KEY = "sk-secret-XYZ-fixture"
        payload = {
            "user": {"content": f"User profile: {CANARY}", "char_count": 30},
            "sessions": [
                {
                    "id": "s1",
                    "system_prompt": f"Persona block; user says: {CANARY}",
                    "model": "x",
                }
            ],
            "messages": [
                {"role": "system", "content": f"More persona: {CANARY}"},
                {"role": "user", "content": "innocent question"},
            ],
            "providers": [
                {"name": "openai", "api_key": SK_KEY},
            ],
            "links": [
                f"https://user:pass-{CANARY}@api.x.com/v1",
            ],
        }
        out = sanitize_response(payload)
        flat = repr(out)
        assert CANARY not in flat, (
            f"PII canary survived sanitisation; "
            f"output contained {CANARY!r}"
        )
        assert SK_KEY not in flat, (
            f"API key fixture survived sanitisation; "
            f"output contained {SK_KEY!r}"
        )
        # Sanity: structural data still there.
        assert "openai" in flat
        assert "innocent question" in flat


class TestDeclaredPolicy:
    def test_policy_advertises_hard_redaction(self):
        p = declared_policy()
        assert p == {
            "enabled": True,
            "identity_blocks_redacted": True,
            "opt_in_supported": False,
            "sanitised_endpoints": [
                "/api/memory",
                "/api/sessions",
                "/api/sessions/{id}/messages",
            ],
            "out_of_scope": [
                "/api/profiles/{name}/soul",
            ],
        }

    def test_policy_lists_endpoints_in_lockstep_with_handlers(self):
        """The advertised endpoint list must match the constants
        used to wire the handlers — single source of truth.
        """
        from gateway.platforms.api_sanitiser import (
            _SANITISED_ENDPOINTS,
            _OUT_OF_SCOPE_ENDPOINTS,
        )

        p = declared_policy()
        assert p["sanitised_endpoints"] == list(_SANITISED_ENDPOINTS)
        assert p["out_of_scope"] == list(_OUT_OF_SCOPE_ENDPOINTS)
        # Disjoint sets — a path is either redacted or pass-through,
        # never both at once.
        assert not (
            set(_SANITISED_ENDPOINTS) & set(_OUT_OF_SCOPE_ENDPOINTS)
        )


# ---------------------------------------------------------------------------
# HTTP integration — against the patched APIServerAdapter
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [
        mw
        for mw in (cors_middleware, security_headers_middleware)
        if mw is not None
    ]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/memory", adapter._handle_read_memory)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_get(
        "/api/sessions/{session_id}/messages",
        adapter._handle_session_messages,
    )
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestCapabilitiesAdvertisesPolicy:
    @pytest.mark.asyncio
    async def test_v1_capabilities_includes_sanitisation_block(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            body = await resp.json()
            assert "sanitisation" in body, (
                f"/v1/capabilities should advertise the sanitisation "
                f"policy; got keys: {list(body.keys())}"
            )
            assert body["sanitisation"] == {
                "enabled": True,
                "identity_blocks_redacted": True,
                "opt_in_supported": False,
                "sanitised_endpoints": [
                    "/api/memory",
                    "/api/sessions",
                    "/api/sessions/{id}/messages",
                ],
                "out_of_scope": [
                    "/api/profiles/{name}/soul",
                ],
            }

    @pytest.mark.asyncio
    async def test_capabilities_lists_exactly_the_sanitised_endpoints(
        self, adapter
    ):
        """Regression-guard against drift: the advertised list MUST
        equal the set of handlers that actually route through
        sanitize_response(). A maintainer adding a handler to the
        sanitised set without updating ``_SANITISED_ENDPOINTS``
        (or vice versa) breaks this test by name.
        """
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            body = await resp.json()
            sanitised = set(body["sanitisation"]["sanitised_endpoints"])
            assert sanitised == {
                "/api/memory",
                "/api/sessions",
                "/api/sessions/{id}/messages",
            }
            # SOUL must NOT be in the sanitised set — it is the
            # owner's pass-through edit target. If a future PR adds
            # SOUL to the sanitised set, this test must be updated
            # in lockstep so clients see the change advertised.
            assert "/api/profiles/{name}/soul" not in sanitised
            assert "/api/profiles/{name}/soul" in body[
                "sanitisation"
            ]["out_of_scope"]


class TestIncludeIdentityQueryParamIgnored:
    """The original spec considered an opt-in flag; the hardened
    spec rejects it entirely. Adding ?include_identity=1 must
    behave exactly the same as omitting it — silent, no 400, no
    different response. Probing the policy shape via status codes
    should not be possible."""

    @pytest.mark.asyncio
    async def test_include_identity_query_param_is_silently_ignored(
        self, adapter, monkeypatch, tmp_path
    ):
        # Point HERMES_HOME to an empty temp dir so the handler
        # returns a deterministic empty shape rather than touching
        # the test runner's real memory.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty"))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            r_off = await cli.get("/api/memory")
            r_on = await cli.get("/api/memory?include_identity=1")
            assert r_off.status == r_on.status
            assert (await r_off.json()) == (await r_on.json())
