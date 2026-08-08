"""Tests for ACP ``authenticate`` → provider ``api_key`` forwarding.

Wire shape: ``params._meta.oauth_token`` (not a top-level ``params.token`` —
``AuthenticateRequest`` declares only ``methodId`` + ``_meta``). The SDK
router spreads ``_meta`` into the handler's kwargs before dispatch, so the
handler sees ``kwargs["oauth_token"]``.

Covers:
  * ``SessionManager.set_auth_token`` / ``get_auth_token`` semantics
    (normalization, overwrite, None-safety).
  * ``_make_agent`` installing a refreshing closure on ``runtime["api_key"]``
    when a token has been deposited, falling back to the resolver's static
    string otherwise.
  * Rotation: a second ``set_auth_token`` is observed by the previously
    installed closure on the next call, *without* a rebuild.
  * The native-Gemini path keeps the static string (``GeminiNativeClient``
    cannot consume a callable api_key) and emits a warning.
  * The ``authenticate`` handler deposits tokens correctly, overwrites on
    repeat, and never logs the token literal.

These pin the contract described in the plan at
``C:\\Users\\lixin4\\.claude\\plans\\melodic-swimming-reddy.md`` §2–§4.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict
from unittest.mock import patch

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def _make_runtime(
    *,
    provider: str = "openrouter",
    api_key: str = "resolved-static-key",
    api_mode: str = "chat_completions",
    base_url: str = "https://openrouter.ai/api/v1",
) -> Dict[str, Any]:
    """Build a runtime dict in the shape ``resolve_runtime_provider`` returns."""
    return {
        "provider": provider,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": api_key,
        "command": None,
        "args": [],
    }


class _RecordingAgent:
    """Captures ``api_key`` so tests can introspect what was handed to AIAgent."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model", "fake")
        self._print_fn = print


def _patch_aiagent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap ``run_agent.AIAgent`` so ``_make_agent`` never builds the real thing."""
    import run_agent  # noqa: F401 — ensure the module is importable before patch
    monkeypatch.setattr("run_agent.AIAgent", _RecordingAgent)


def _patch_resolver(
    monkeypatch: pytest.MonkeyPatch, runtime: Dict[str, Any]
) -> Dict[str, Any]:
    """Stub ``resolve_runtime_provider`` and return the call-log dict."""
    seen: Dict[str, Any] = {}

    def _fake(**kwargs: Any) -> Dict[str, Any]:
        seen.update(kwargs)
        return dict(runtime)  # shallow copy so test mutations don't bleed

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _fake
    )
    return seen


def _patch_config(monkeypatch: pytest.MonkeyPatch, provider: str = "openrouter") -> None:
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "any-model", "provider": provider}, "mcp_servers": {}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# SessionManager.set_auth_token / get_auth_token
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionManagerAuthTokens:
    def test_set_and_get_roundtrip(self):
        mgr = SessionManager()
        mgr.set_auth_token("openai", "sk-abc")
        assert mgr.get_auth_token("openai") == "sk-abc"

    def test_method_id_is_normalized(self):
        mgr = SessionManager()
        mgr.set_auth_token("  OpenAI  ", "sk-abc")
        assert mgr.get_auth_token("openai") == "sk-abc"
        assert mgr.get_auth_token("OPENAI") == "sk-abc"

    def test_overwrite_replaces_prior(self):
        mgr = SessionManager()
        mgr.set_auth_token("openai", "t1")
        mgr.set_auth_token("openai", "t2")
        assert mgr.get_auth_token("openai") == "t2"

    def test_empty_token_is_rejected(self):
        mgr = SessionManager()
        mgr.set_auth_token("openai", "")
        mgr.set_auth_token("openai", "   ")
        assert mgr.get_auth_token("openai") is None

    def test_none_method_id_returns_none(self):
        mgr = SessionManager()
        assert mgr.get_auth_token(None) is None

    def test_non_string_method_id_is_ignored(self):
        mgr = SessionManager()
        mgr.set_auth_token(123, "sk-abc")  # type: ignore[arg-type]
        assert mgr.get_auth_token("123") is None


# ─────────────────────────────────────────────────────────────────────────────
# _make_agent: explicit_api_key forwarding + refreshing closure
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeAgentForwardsTokens:
    def test_no_token_passes_none_explicit_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Back-compat: when no token has been deposited, the resolver is
        called with ``explicit_api_key=None`` (current behavior preserved)."""
        _patch_config(monkeypatch)
        seen = _patch_resolver(monkeypatch, _make_runtime())
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        agent = mgr._make_agent(session_id="s1", cwd=".")

        assert seen.get("explicit_api_key") is None
        assert agent.api_key == "resolved-static-key"

    def test_token_forwarded_to_resolver_as_string(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The resolver must see a *string* (it does ``(x or '').strip()``)."""
        _patch_config(monkeypatch)
        seen = _patch_resolver(monkeypatch, _make_runtime(api_key="sk-deposited"))
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("openrouter", "sk-deposited")
        mgr._make_agent(session_id="s1", cwd=".")

        assert seen.get("explicit_api_key") == "sk-deposited"

    def test_api_key_becomes_callable_after_deposit(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``_make_agent`` swaps the resolved string for a closure so refreshes
        propagate to the AIAgent without rebuild."""
        _patch_config(monkeypatch)
        _patch_resolver(monkeypatch, _make_runtime(api_key="sk-t1"))
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("openrouter", "sk-t1")
        agent = mgr._make_agent(session_id="s1", cwd=".")

        assert callable(agent.api_key)
        assert agent.api_key() == "sk-t1"

    def test_rotation_is_picked_up_without_rebuild(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The closure must observe a *later* ``set_auth_token`` call."""
        _patch_config(monkeypatch)
        _patch_resolver(monkeypatch, _make_runtime(api_key="sk-t1"))
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("openrouter", "sk-t1")
        agent = mgr._make_agent(session_id="s1", cwd=".")
        assert agent.api_key() == "sk-t1"

        # Simulate the client re-issuing `authenticate` with a fresh bearer
        # mid-session. The existing closure on the previously-built agent
        # must observe the new value — no new _make_agent call.
        mgr.set_auth_token("openrouter", "sk-t2")
        assert agent.api_key() == "sk-t2"

    def test_closure_falls_back_to_resolved_string_if_token_evaporates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """If the dict entry disappears (defensive), the closure returns the
        original resolved key rather than an empty string. This guards
        against ``Authorization: Bearer `` (empty) escaping to the wire."""
        _patch_config(monkeypatch)
        _patch_resolver(monkeypatch, _make_runtime(api_key="sk-fallback"))
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("openrouter", "sk-fallback")
        agent = mgr._make_agent(session_id="s1", cwd=".")

        # Simulate evaporation (no public delete API — reach into the dict).
        with mgr._lock:
            mgr._auth_tokens.clear()

        assert agent.api_key() == "sk-fallback"

    def test_native_gemini_keeps_static_string_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """``GeminiNativeClient.__init__`` rejects callables; we must not
        install the refreshing closure on that path."""
        _patch_config(monkeypatch, provider="google")
        _patch_resolver(
            monkeypatch,
            _make_runtime(
                provider="google",
                api_key="sk-gem",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
        )
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("google", "sk-gem")
        with caplog.at_level(logging.WARNING, logger="acp_adapter.session"):
            agent = mgr._make_agent(session_id="s1", cwd=".")

        assert isinstance(agent.api_key, str)
        assert agent.api_key == "sk-gem"
        assert any(
            "will not auto-refresh" in record.message for record in caplog.records
        )

    def test_resolver_callable_api_key_is_left_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Azure Entra and similar paths already return a callable from the
        resolver. We must not wrap a callable in another callable."""
        _patch_config(monkeypatch, provider="azure-foundry")
        entra_callable = lambda: "entra-jwt"  # noqa: E731
        _patch_resolver(
            monkeypatch,
            _make_runtime(
                provider="azure-foundry",
                api_key=entra_callable,  # type: ignore[arg-type]
                base_url="https://r.openai.azure.com/openai/v1",
            ),
        )
        _patch_aiagent(monkeypatch)

        mgr = SessionManager()
        mgr.set_auth_token("azure-foundry", "deposited-but-irrelevant")
        agent = mgr._make_agent(session_id="s1", cwd=".")

        assert agent.api_key is entra_callable


# ─────────────────────────────────────────────────────────────────────────────
# HermesACPAgent.authenticate: token extraction + deposit
# ─────────────────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestAuthenticateDepositsToken:
    def test_authenticate_with_token_deposits_into_session_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        resp = _run(agent.authenticate("openrouter", oauth_token="sk-abc"))
        assert resp is not None
        assert mgr.get_auth_token("openrouter") == "sk-abc"

    def test_authenticate_without_token_is_noop_on_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """No ``oauth_token`` kwarg → auth still succeeds, no deposit.

        This also pins the legacy-shape behavior: a client sending
        ``{"methodId": "openrouter", "token": "sk-x"}`` (top-level ``token``,
        no ``_meta``) reaches us via the same kwargs shape — pydantic
        validation against ``AuthenticateRequest`` drops the unknown
        top-level field before the router calls us. So the misconfigured
        client gets ``AuthenticateResponse()`` back ("auth succeeded"),
        but no token is deposited; the bug surfaces on the next outbound
        provider request as an upstream-auth error.

        Guards against a future change that adds a top-level ``token``
        field to ``AuthenticateRequest`` and silently revives ambiguity
        between the legacy and ``_meta.oauth_token`` channels.
        """
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        resp = _run(agent.authenticate("openrouter"))
        assert resp is not None
        assert mgr.get_auth_token("openrouter") is None

    def test_authenticate_repeated_call_overwrites_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        _run(agent.authenticate("openrouter", oauth_token="t1"))
        _run(agent.authenticate("openrouter", oauth_token="t2"))
        assert mgr.get_auth_token("openrouter") == "t2"

    def test_authenticate_does_not_deposit_when_provider_mismatches(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Method-id gate runs first; mismatched method_id must not deposit."""
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        resp = _run(agent.authenticate("anthropic", oauth_token="sk-abc"))
        assert resp is None
        assert mgr.get_auth_token("anthropic") is None
        assert mgr.get_auth_token("openrouter") is None

    def test_authenticate_does_not_log_token_literal(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        secret = "sk-very-secret-token-zzz"
        with caplog.at_level(logging.DEBUG):
            _run(agent.authenticate("openrouter", oauth_token=secret))

        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_authenticate_accepts_oauth_token_kwarg_from_meta_spread(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Mirrors what ``acp/router.py`` does after validating an
        ``AuthenticateRequest`` whose JSON payload is::

            {"methodId": "openrouter", "_meta": {"oauth_token": "sk-meta"}}

        The router calls ``params = {k: getattr(model, k) for k in fields
        if k != "field_meta"}`` then ``params.update(field_meta)`` and
        finally ``await handler(**params)``. So our handler observes
        ``method_id="openrouter", oauth_token="sk-meta"`` as kwargs — this
        test exercises exactly that calling shape and asserts the deposit
        happens by the ``_meta`` channel rather than any top-level field.
        """
        from acp_adapter import server as _server

        monkeypatch.setattr(_server, "detect_provider", lambda: "openrouter")
        mgr = SessionManager()
        agent = HermesACPAgent(session_manager=mgr)

        # Simulate router spread: method_id positionally, _meta entries as kwargs.
        router_kwargs = {"oauth_token": "sk-meta"}
        resp = _run(agent.authenticate("openrouter", **router_kwargs))

        assert resp is not None
        assert mgr.get_auth_token("openrouter") == "sk-meta"
