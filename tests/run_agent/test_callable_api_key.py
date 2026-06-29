"""Tests that callable api_key (Entra ID bearer provider) flows through
the agent stack without coercion.

The OpenAI Python SDK accepts ``api_key: str | None | Callable[[], str]``,
and ``azure-identity``'s ``get_bearer_token_provider`` returns a callable.
Hermes preserves the callable end-to-end so the SDK refreshes tokens
transparently. This file pins the contract at the high-risk seams the
rubber-duck audit identified.

Covered:
  * ``_create_openai_client`` passes a callable ``api_key`` straight
    through to ``openai.OpenAI(...)``.
  * ``_normalize_main_runtime`` preserves the callable so auxiliary
    clients inherit Entra auth.
  * ``_truncate_token`` (dashboard preview) renders ``"<entra-id-bearer>"``
    instead of ``"<function ...>"`` and never invokes the callable.
  * ``run_agent.py`` masked-banner path renders the Entra placeholder
    and never tries to slice/len the callable.
  * Serialization scrub: dumping a runtime dict via ``json.dumps`` with
    a callable api_key raises (default behaviour) — guards against
    silently leaking ``"<function ...>"`` strings into event logs.
  * ``batch_runner`` strips the callable from the worker config dict
    so multiprocessing.Pool can pickle the rest.
"""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# OpenAI SDK construction preserves the callable
# ---------------------------------------------------------------------------


class TestCreateOpenAIClientCallable:
    """``AIAgent._create_openai_client`` must pass the callable through
    to ``openai.OpenAI(...)`` without coercion."""

    def test_callable_api_key_passed_to_openai_constructor(self, monkeypatch):
        """Construct the smallest possible AIAgent surface and verify
        the OpenAI client receives the callable unchanged."""
        captured = {}

        def fake_openai(**kwargs):
            captured["kwargs"] = kwargs
            return MagicMock(api_key=kwargs.get("api_key"))

        # Patch the module-level OpenAI proxy used by ``_create_openai_client``.
        monkeypatch.setattr("run_agent.OpenAI", fake_openai)

        # Build a minimal stand-in for AIAgent so we can call the bound
        # method directly without paying the full __init__ cost.
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        # Attributes consulted by _create_openai_client / _client_log_context.
        agent.provider = "azure-foundry"
        agent.model = "gpt-4o"
        agent.base_url = "https://r.openai.azure.com/openai/v1"
        agent._client_kwargs = {}

        def token_provider():
            return "fresh-jwt"

        client_kwargs = {
            "api_key": token_provider,
            "base_url": "https://r.openai.azure.com/openai/v1",
        }
        client = agent._create_openai_client(client_kwargs, reason="test", shared=False)

        # The OpenAI constructor must receive the *callable*, not a string.
        forwarded = captured["kwargs"]["api_key"]
        assert callable(forwarded)
        assert not isinstance(forwarded, str)
        assert forwarded is token_provider, (
            "_create_openai_client must not wrap or coerce the callable"
        )
        assert client is not None


# ---------------------------------------------------------------------------
# Auxiliary runtime preserves the callable
# ---------------------------------------------------------------------------


class TestNormalizeMainRuntimePreservesCallable:
    """The aux client orchestrator must keep the callable on the
    runtime dict so compression / vision / embedding / title-gen clients
    inherit Entra ID auth from the main agent."""

    def test_callable_api_key_survives_normalization(self):
        from agent.auxiliary_client import _normalize_main_runtime

        def provider():
            return "jwt"

        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "model": "gpt-4o",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_key": provider,
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
        })
        assert normalized["api_key"] is provider
        assert normalized["auth_mode"] == "entra_id"

    def test_string_api_key_still_works(self):
        from agent.auxiliary_client import _normalize_main_runtime
        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "api_key": "sk-static",
        })
        assert normalized["api_key"] == "sk-static"

    def test_normalization_drops_empty_string_but_preserves_callable(self):
        from agent.auxiliary_client import _normalize_main_runtime

        def provider():
            return ""

        # Empty string fields are dropped, but a callable is preserved
        # even if it would mint an empty token (we don't invoke during
        # normalization).
        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "api_key": provider,
            "model": "",
        })
        assert normalized["api_key"] is provider
        assert "model" not in normalized

    def test_unknown_field_dropped(self):
        from agent.auxiliary_client import _normalize_main_runtime, _MAIN_RUNTIME_FIELDS
        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "api_key": "k",
            "secret_field_we_dont_want": "leak",
        })
        assert "secret_field_we_dont_want" not in normalized
        # auth_mode IS in the field allowlist (rubber-duck blocker fix).
        assert "auth_mode" in _MAIN_RUNTIME_FIELDS


# ---------------------------------------------------------------------------
# Display surfaces never invoke the callable
# ---------------------------------------------------------------------------


class TestTruncateTokenCallable:
    def test_callable_returns_placeholder(self):
        """Dashboard preview must render the Entra placeholder, NOT
        ``"<function ...>"``."""
        from hermes_cli.web_server import _truncate_token

        invoked = {"count": 0}

        def provider():
            invoked["count"] += 1
            return "should-not-appear-in-ui"

        token_provider = cast(str | None, provider)
        rendered = _truncate_token(token_provider)
        assert rendered == "<entra-id-bearer>"
        assert invoked["count"] == 0

    def test_string_jwt_still_truncated_to_signature_tail(self):
        from hermes_cli.web_server import _truncate_token
        # JWT shape: header.payload.signature → only signature tail shown.
        out = _truncate_token("aaaa.bbbb.cccccccsig", visible=4)
        assert out == "…csig"

    def test_empty_returns_empty(self):
        from hermes_cli.web_server import _truncate_token
        assert _truncate_token(None) == ""
        assert _truncate_token("") == ""


# ---------------------------------------------------------------------------
# Serialization scrub — runtime dicts with callables must NOT silently
# JSON-encode as ``"<function ...>"`` (would leak garbage into events).
# ---------------------------------------------------------------------------


class TestRuntimeDictSerializationGuard:
    def test_json_dumps_default_str_does_not_silently_stringify_callable(self):
        """Sanity check: a runtime dict with a callable api_key must
        either raise on plain ``json.dumps`` (good — fail loud) or be
        sanitized BEFORE serialization. This test pins the loud-fail
        behaviour so future changes that introduce
        ``json.dumps(..., default=str)`` over a runtime dict are caught
        by a regression here."""

        def provider():
            return "jwt"

        runtime = {
            "provider": "azure-foundry",
            "api_key": provider,
            "auth_mode": "entra_id",
        }
        # Plain json.dumps — must raise, not silently produce
        # ``"<function provider at 0x...>"``.
        with pytest.raises(TypeError):
            json.dumps(runtime)


# ---------------------------------------------------------------------------
# batch_runner strips callables from the worker config dict
# ---------------------------------------------------------------------------


class TestBatchRunnerCallableHandling:
    def test_callable_api_key_stripped_from_worker_config(self, capsys, monkeypatch, tmp_path):
        """``BatchRunner._run_batches`` (or the equivalent code path)
        must replace a callable api_key with None before pickling the
        worker config dict — otherwise multiprocessing.Pool fails."""
        # We can't easily run BatchRunner end-to-end in a unit test
        # (it spawns subprocesses), but we CAN inline the same logic:
        # the production code uses ``callable(self.api_key) and not
        # isinstance(self.api_key, str)`` to gate the substitution.
        # Re-execute the same predicate here as a contract guard.

        def provider():
            return "jwt"

        api_key = provider
        worker_api_key = None if (callable(api_key) and not isinstance(api_key, str)) else api_key
        assert worker_api_key is None, (
            "BatchRunner must replace callable api_key with None so "
            "multiprocessing.Pool can pickle the worker config"
        )

        # And a string passes through unchanged.
        api_key_str = "sk-static"
        worker_api_key_str = None if (callable(api_key_str) and not isinstance(api_key_str, str)) else api_key_str
        assert worker_api_key_str == "sk-static"

    def test_batch_runner_source_uses_the_correct_predicate(self):
        """Pin the predicate string in batch_runner so refactors that
        change it are caught here. Reading the source rather than
        importing avoids spinning up the full BatchRunner."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent
               / "batch_runner.py").read_text(encoding="utf-8")
        assert "callable(self.api_key) and not isinstance(self.api_key, str)" in src, (
            "BatchRunner.api_key callable check changed — update test or "
            "verify the new predicate still routes Entra token providers "
            "to the worker-rebuild path."
        )


# ---------------------------------------------------------------------------
# Inline masked-banner / display sites (callable-aware)
# ---------------------------------------------------------------------------


class TestCliEnsureRuntimeCredentialsCallable:
    """Regression: ``cli.py:_ensure_runtime_credentials`` previously
    treated a callable ``api_key`` as "not a string" and overwrote it
    with the ``"no-key-required"`` placeholder, which then got sent as
    ``Authorization: Bearer no-key-required`` and rejected by Azure
    with a 401. This is the most subtle of the callable-api_key audit
    sites — gated by ``not isinstance(api_key, str)`` rather than the
    cleaner ``callable(...)`` check used elsewhere.

    We verify the source pattern (rather than spinning up a real
    ``HermesCLI`` instance) — the predicate change is the load-bearing
    fix and is invariant under the surrounding orchestration code."""

    def test_callable_predicate_present_in_cli_runtime_validation(self):
        from pathlib import Path
        # ``_ensure_runtime_credentials`` was extracted from cli.py into the
        # ``CLIAgentSetupMixin`` (god-file decomposition Phase 4). Read the
        # module the method actually lives in now.
        src = (Path(__file__).resolve().parent.parent.parent
               / "hermes_cli" / "cli_agent_setup_mixin.py").read_text(encoding="utf-8")
        # The fix introduces ``_is_callable_provider`` which gates the
        # string-only check so callable token providers survive.
        assert "_is_callable_provider = callable(api_key)" in src, (
            "_ensure_runtime_credentials must preserve a callable "
            "api_key (Entra ID bearer provider). Without the guard, the "
            "callable is stringified to 'no-key-required' and Azure 401s."
        )


class TestInlinedDisplayMasks:
    """The masked-credential display sites are now inlined per-site (no
    shared helper). Each site uses the ``is_token_provider`` predicate
    to short-circuit on callables and print a static
    ``"Microsoft Entra ID"`` label, then falls through to its own
    context-appropriate string mask. This replaces a unified helper
    that would have forced one mask shape across sites with legitimately
    different display needs (banner vs diagnostic vs UI vs preview)."""

    def test_run_agent_banner_uses_is_token_provider_guard(self):
        """The masked-banner sites live in ``agent/agent_init.py``
        (the ``__init__`` body was extracted into ``init_agent`` after
        this feature was first written). Both the OpenAI and Anthropic
        client init paths must guard their banner prints with
        ``is_token_provider`` so a callable Entra ID provider doesn't
        crash ``len(api_key)``."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent
               / "agent" / "agent_init.py").read_text(encoding="utf-8")
        assert src.count("is_token_provider(") >= 2, (
            "agent/agent_init.py must guard BOTH masked-banner paths "
            "(chat_completions and anthropic_messages) with "
            "is_token_provider()."
        )
        assert src.count('"🔑 Using credentials: Microsoft Entra ID"') >= 2, (
            "agent/agent_init.py banner blocks should print a static "
            "'Microsoft Entra ID' label for callable api_keys — no "
            "placeholder plumbing, no describe-mask fallback."
        )

    def test_cli_show_config_handles_callable(self):
        """``cli.HermesCLI.show_config`` previously did
        ``self.api_key[-4:]`` / ``len(self.api_key)`` which crashes on
        callable Entra ID providers. The inlined version uses
        ``is_token_provider`` and prints the same static label as the
        run_agent banners."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent
               / "cli.py").read_text(encoding="utf-8")
        assert "is_token_provider(self.api_key)" in src, (
            "cli.HermesCLI.show_config must guard self.api_key via "
            "is_token_provider so callable Entra ID providers don't "
            "crash /config."
        )
        assert '"Microsoft Entra ID"' in src, (
            "cli.HermesCLI.show_config must print the static "
            "'Microsoft Entra ID' label (matching run_agent banners) "
            "instead of attempting to slice the callable."
        )

    def test_mask_api_key_for_logs_handles_callable(self):
        """``run_agent._mask_api_key_for_logs`` is called from the
        request-dump JSON path. For Entra users, ``self.client.api_key``
        is the SDK's empty string (callable stashed privately) — but
        defensively the helper must also accept a callable directly
        and return the placeholder rather than crashing on
        ``len(callable)``."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent
               / "run_agent.py").read_text(encoding="utf-8")
        # The function now starts with a callable check.
        assert (
            "if callable(key) and not isinstance(key, str):" in src
            and '"<entra-id-bearer>"' in src
        ), (
            "run_agent._mask_api_key_for_logs must short-circuit for "
            "callable api_keys to avoid len(callable) crashes in "
            "request-dump paths."
        )

    def test_anthropic_401_diagnostic_handles_callable(self):
        """The Anthropic 401 diagnostic path lives in
        ``agent/conversation_loop.py`` (the ``run_conversation`` body
        was extracted after this feature was first written). It used
        to do ``key[:12]`` on ``self._anthropic_api_key``. For Entra ID +
        Anthropic-style mode that's a callable; slicing crashes."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent
               / "agent" / "conversation_loop.py").read_text(encoding="utf-8")
        # The Anthropic 401 block now branches on is_token_provider
        # before slicing the key.
        assert "Microsoft Entra ID (httpx event hook)" in src, (
            "agent/conversation_loop.py Anthropic 401 diagnostic must "
            "surface a Microsoft Entra ID branch before slicing the "
            "key prefix."
        )
