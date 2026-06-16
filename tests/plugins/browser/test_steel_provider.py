"""Tests for the Steel cloud browser provider plugin.

Focuses on the Steel-specific behaviour that the shared
``test_browser_provider_plugins.py`` suite does not cover:

- ``_normalize_cdp_url`` — the slash-before-query fix that makes Steel's
  ``websocketUrl`` connectable by CDP clients (the linchpin of the plugin).
- ``create_session`` request shaping (proxy / captcha / timeout knobs) and the
  returned metadata contract.
- ``close_session`` hitting the ``/release`` endpoint.
- Explicit-only registry selection (Steel is never auto-detected).

Network calls are faked at the ``requests.post`` boundary, mirroring
``tests/tools/test_managed_browserbase_and_modal.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.browser.steel import provider as steel_module
from plugins.browser.steel.provider import SteelBrowserProvider, _normalize_cdp_url


class _Response:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload or {}
        self.text = "" if payload else "error body"

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "STEEL_API_KEY",
        "STEEL_BASE_URL",
        "STEEL_USE_PROXY",
        "STEEL_SOLVE_CAPTCHA",
        "STEEL_SESSION_TIMEOUT",
        # Cleared so the explicit-only resolution tests can't be perturbed by
        # another provider's credentials in the ambient environment.
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_GATEWAY_URL",
        "FIRECRAWL_API_KEY",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# _normalize_cdp_url — the slash fix
# ---------------------------------------------------------------------------


class TestNormalizeCdpUrl:
    def test_inserts_slash_before_query_and_appends_api_key(self) -> None:
        url = _normalize_cdp_url(
            "wss://connect.steel.dev?sessionId=abc", "secret"
        )
        assert url == "wss://connect.steel.dev/?sessionId=abc&apiKey=secret"

    def test_preserves_existing_path(self) -> None:
        url = _normalize_cdp_url(
            "wss://connect.steel.dev/cdp?sessionId=abc", "secret"
        )
        assert url == "wss://connect.steel.dev/cdp?sessionId=abc&apiKey=secret"

    def test_handles_self_hosted_host(self) -> None:
        url = _normalize_cdp_url(
            "wss://steel.internal.example?sessionId=xyz", "k"
        )
        assert url == "wss://steel.internal.example/?sessionId=xyz&apiKey=k"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_without_api_key(self) -> None:
        assert SteelBrowserProvider().is_available() is False

    def test_true_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "k")
        assert SteelBrowserProvider().is_available() is True


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


def _ok_session_payload() -> dict:
    return {
        "id": "sess-123",
        "websocketUrl": "wss://connect.steel.dev?sessionId=sess-123",
        "sessionViewerUrl": "https://app.steel.dev/sessions/sess-123",
    }


class TestCreateSession:
    def test_returns_normalized_cdp_url_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(200, _ok_session_payload())
        ) as post:
            session = SteelBrowserProvider().create_session("task-1")

        assert session["bb_session_id"] == "sess-123"
        assert session["cdp_url"] == (
            "wss://connect.steel.dev/?sessionId=sess-123&apiKey=secret"
        )
        assert session["session_name"].startswith("hermes_task-1_")
        assert session["session_viewer_url"].endswith("sess-123")
        # No optional knobs set → all features off.
        assert session["features"] == {
            "proxy": False,
            "captcha_solving": False,
            "custom_timeout": False,
        }
        # API key travels in the Steel-Api-Key header.
        assert post.call_args.kwargs["headers"]["Steel-Api-Key"] == "secret"

    def test_request_body_reflects_env_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        monkeypatch.setenv("STEEL_USE_PROXY", "true")
        monkeypatch.setenv("STEEL_SOLVE_CAPTCHA", "1")
        monkeypatch.setenv("STEEL_SESSION_TIMEOUT", "600000")

        with patch.object(
            steel_module.requests, "post", return_value=_Response(200, _ok_session_payload())
        ) as post:
            session = SteelBrowserProvider().create_session("task-2")

        body = post.call_args.kwargs["json"]
        assert body["useProxy"] is True
        assert body["solveCaptcha"] is True
        assert body["timeout"] == 600000
        assert session["features"] == {
            "proxy": True,
            "captcha_solving": True,
            "custom_timeout": True,
        }

    def test_invalid_timeout_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        monkeypatch.setenv("STEEL_SESSION_TIMEOUT", "not-a-number")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(200, _ok_session_payload())
        ) as post:
            SteelBrowserProvider().create_session("task-3")
        assert "timeout" not in post.call_args.kwargs["json"]

    def test_missing_api_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="STEEL_API_KEY"):
            SteelBrowserProvider().create_session("task-4")

    def test_non_ok_response_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(402, None)
        ):
            with pytest.raises(RuntimeError, match="Failed to create Steel session"):
                SteelBrowserProvider().create_session("task-5")

    def test_response_missing_fields_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(200, {"id": "x"})
        ):
            with pytest.raises(RuntimeError, match="missing id/websocketUrl"):
                SteelBrowserProvider().create_session("task-6")


# ---------------------------------------------------------------------------
# close_session / emergency_cleanup
# ---------------------------------------------------------------------------


class TestCloseSession:
    def test_release_endpoint_called_and_true_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(204, {})
        ) as post:
            ok = SteelBrowserProvider().close_session("sess-123")
        assert ok is True
        assert post.call_args.args[0].endswith("/v1/sessions/sess-123/release")

    def test_false_when_api_key_missing(self) -> None:
        assert SteelBrowserProvider().close_session("sess-123") is False

    def test_false_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STEEL_API_KEY", "secret")
        with patch.object(
            steel_module.requests, "post", return_value=_Response(500, None)
        ):
            assert SteelBrowserProvider().close_session("sess-123") is False


# ---------------------------------------------------------------------------
# Picker schema
# ---------------------------------------------------------------------------


class TestSetupSchema:
    def test_schema_shape(self) -> None:
        schema = SteelBrowserProvider().get_setup_schema()
        assert schema["name"] == "Steel"
        assert schema["post_setup"] == "agent_browser"
        keys = [e["key"] for e in schema["env_vars"]]
        assert keys == ["STEEL_API_KEY"]


# ---------------------------------------------------------------------------
# Registry: explicit-only selection
# ---------------------------------------------------------------------------


class TestRegistrySelection:
    def test_explicit_steel_resolves_even_when_unavailable(self) -> None:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        from agent.browser_registry import _resolve

        provider = _resolve("steel")
        assert provider is not None
        assert provider.name == "steel"
        assert provider.is_available() is False

    def test_steel_not_auto_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Steel is explicit-only — having STEEL_API_KEY set must not
        auto-route a user to the paid cloud browser."""
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        from agent.browser_registry import _resolve

        monkeypatch.setenv("STEEL_API_KEY", "k")
        assert _resolve(None) is None
