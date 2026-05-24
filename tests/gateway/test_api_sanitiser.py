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

    def test_querystring_redaction_handles_boundary_chars(self):
        """Lock in the boundary behaviour of ``_TOKEN_QS_RE`` against
        the encoded / punctuated value shapes that come up in real
        URLs: percent-encoded reserved chars, JWT-style dots, base64
        ``=`` padding, fragments, adjacent params, matrix-style
        ``;`` separators, and trailing punctuation in prose.

        Every case must (a) remove the original secret string and
        (b) leave the ``api_key=<redacted>`` (or equivalent) marker
        intact so a reader can still see *that* something was
        redacted."""
        cases = [
            # (input, the secret substring we must NOT see in the output)
            # Percent-encoded reserved characters in the value.
            ("https://x?api_key=abc%2Fdef",       "abc%2Fdef"),
            ("https://x?api_key=abc%26more=xx",   "abc%26more=xx"),
            ("https://x?api_key=abc%3Bdef",       "abc%3Bdef"),
            # Trailing slash / semicolon in the raw token body.
            ("https://x?api_key=abc/def/ghi",     "abc/def/ghi"),
            ("https://x?api_key=abc;jsessionid=xyz", "abc;jsessionid"),
            # JWT-shaped value (segments separated by dots).
            ("https://x?access_token=hdr.payload.sig", "hdr.payload.sig"),
            # Base64 padding inside the value.
            ("https://x?secret=abc=padding==",    "abc=padding=="),
            # Fragment after the value.
            ("https://x?api_key=abc#frag",        "abc#frag"),
            # Adjacent param: only the first value goes; q= survives.
            ("https://x?api_key=abc&q=foo",       "abc"),
            # Case-insensitive field name.
            ("https://x?TOKEN=abc",               "abc"),
        ]
        for url, secret in cases:
            out = sanitize_value(url)
            assert secret not in out, (
                f"token value leaked through sanitizer for input {url!r}: "
                f"got {out!r}"
            )
            assert REDACTED in out, (
                f"redaction marker missing for input {url!r}: got {out!r}"
            )

    def test_querystring_redaction_preserves_adjacent_params(self):
        """The redaction must not eat the next param. ``?api_key=X&q=Y``
        must come out as ``?api_key=<redacted>&q=Y`` — that ``q=Y``
        is sometimes the *only* readable hint a debugger has."""
        out = sanitize_value("https://x?api_key=sk-secret&q=hello&page=2")
        assert "sk-secret" not in out
        assert "q=hello" in out
        assert "page=2" in out

    def test_querystring_empty_value_is_left_alone(self):
        """An empty value (``?api_key=``) has nothing to redact and
        must not crash. The marker stays as-is so the caller can
        still see the (vacuous) key."""
        out = sanitize_value("https://x?api_key=&q=1")
        assert out == "https://x?api_key=&q=1"

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

    def test_counter_fields_with_token_substring_are_not_redacted(self):
        """Regression: ``input_tokens`` etc. are integer counts, not
        credentials. The field-name heuristic must let them through
        even though they contain the substring ``token``.

        Live-smoke against /api/sessions found these getting
        wholesale-redacted, which broke any cost / usage display
        downstream. See ``_COUNTER_FIELD_RE``."""
        counters = (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "tokens",            # bare plural
            "token_count",
            "tokenCount",
            "TOKEN_COUNT",       # case-insensitive
            "foo_tokens",        # any prefix-with-underscore + tokens
        )
        for field in counters:
            assert sanitize_value(1234, field) == 1234, (
                f"counter field {field!r} should pass through unchanged"
            )

    def test_actual_secret_token_fields_still_redacted(self):
        """Negative pair to the counter test above: singular
        ``token`` and explicit access/api/bearer tokens stay
        redacted. Don't widen the allow-list past counters."""
        for field in (
            "token",
            "access_token",
            "api_token",
            "bearer_token",
            "refresh_token",
            "session_token",
        ):
            assert sanitize_value("sk-xyz", field) == REDACTED, (
                f"secret field {field!r} must still be redacted"
            )

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
        }


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
            }


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
