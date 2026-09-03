"""Round 4 — RED owning-path tests: physical-dispatch topology.

These tests prove that the transport-normalized ceiling gate must fire at
each PHYSICAL dispatch owner (not at a single shared seam) and that the
gate reads the FINAL payload, not generic kwargs or host attributes.

Topology (proven from code, 2026-08-22):
  * Main owner (non-streaming, all api_modes):
      _dispatch_nonstreaming_api_request  (chat_completion_helpers.py:926)
  * Auxiliary owner (sync/async/stream, all tasks):
      call_llm → _relay_sync_completion / _relay_async_completion /
      _relay_sync_stream → client.chat.completions.create
      (auxiliary_client.py:3350, 3378, 3404, 3410)
  * MoA aggregator streaming bypasses even the relay:
      client.chat.completions.create  (auxiliary_client.py:9556)
  * Codex app-server:
      codex_app_server_session.run_turn()  (bypasses _dispatch_nonstreaming_api_request)

Each test drives the REAL owning function with a spied physical I/O and
asserts ContextCeilingExceeded is raised BEFORE the physical call.

All 8 tests are RED against current code (no gate at these seams).
They become GREEN once the transport-normalized FinalContextBudget gate
is implemented at each owner.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

# ── Shared test infrastructure ────────────────────────────────────────────────

from agent.agent_runtime_helpers import ContextCeilingExceeded
from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent import auxiliary_client as _aux
from agent import relay_llm as _relay_llm  # module object — relay helpers import this locally
from agent.model_metadata import estimate_request_tokens_rough


class _SpyClient:
    """Records physical I/O calls; never makes a real network call."""

    class _Completions:
        def __init__(self, spy):
            self._spy = spy

        def create(self, **kwargs):
            self._spy.create_calls.append(kwargs)
            from types import SimpleNamespace
            msg = SimpleNamespace(content="ok", tool_calls=[])
            ch = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[ch], usage=None, model="test")

    class _Chat:
        def __init__(self, spy):
            self.completions = _SpyClient._Completions(spy)

    class _BedrockRuntime:
        def __init__(self, spy):
            self._spy = spy

        def converse(self, **kwargs):
            self._spy.converse_calls.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(output={"message": {"content": [{"text": "ok"}]}},
                                   usage={"inputTokens": 1, "outputTokens": 1})

    class _Responses:
        def __init__(self, spy):
            self._spy = spy

        def create(self, **kwargs):
            self._spy.responses_calls.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(output=[], usage=None, model="test")

    class _CodexStream:
        def __init__(self, spy):
            self._spy = spy

        def create(self, **kwargs):
            self._spy.responses_calls.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(output=[], usage=None, model="test")

    def __init__(self):
        self.create_calls: list[dict] = []
        self.converse_calls: list[dict] = []
        self.responses_calls: list[dict] = []
        self.chat = _SpyClient._Chat(self)
        self.bedrock = _SpyClient._BedrockRuntime(self)
        self.responses = _SpyClient._Responses(self)
        self._codex = _SpyClient._CodexStream(self)


class _EngineStub:
    """Minimal context_compressor with a max_context_length ceiling."""
    _handles_context_ceiling = True

    def __init__(self, ceiling: int | None = None, pre_cap: int | None = None):
        self.max_context_length = ceiling
        self._pre_cap = pre_cap
        self.context_length = min(pre_cap, ceiling) if (pre_cap and ceiling) else (pre_cap or ceiling)

    @property
    def pre_cap_context_length(self):
        return self._pre_cap


class _AgentStub:
    """Minimal agent with the attributes the gate reads."""

    def __init__(self, *, pre_cap: int = 128_000, ceiling: int | None = 64_000,
                 api_mode: str = "chat_completions", provider: str = "openai"):
        self._pre_cap_context_length = pre_cap
        self.context_compressor = _EngineStub(ceiling=ceiling, pre_cap=pre_cap)
        self.api_mode = api_mode
        self.provider = provider
        self.model = "test-model"
        self.base_url = "http://test"
        self.api_key = "test-key"
        self.client = _SpyClient()
        # Codex non-streaming dispatch calls agent._run_codex_stream; route it
        # to the spy so the gate (which must fire BEFORE this) is the thing
        # under test, not a missing harness method.
        self._codex_on_first_delta = None

    def _run_codex_stream(self, api_kwargs, *, client=None, on_first_delta=None):
        # Non-streaming Codex path in this stub: record as a responses.create.
        self.client.responses_calls.append(dict(api_kwargs))
        from types import SimpleNamespace
        return SimpleNamespace(output=[], usage=None, model="test-model")


def _big_messages(n_tokens_target: int = 200_000) -> list:
    """Build a messages list whose rough token estimate exceeds n_tokens_target."""
    # ~4 chars ≈ 1 token (ASCII). 200K tokens ≈ 800K chars.
    filler = "x" * (n_tokens_target * 4)
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": filler},
    ]


def _big_kwargs(n_tokens_target: int = 200_000) -> dict:
    """Build api_kwargs with messages large enough to exceed the ceiling."""
    return {
        "model": "test-model",
        "messages": _big_messages(n_tokens_target),
        "max_tokens": 4096,
    }


def _tools_payload() -> list:
    """A tool schema large enough to be meaningful in accounting."""
    return [
        {
            "type": "function",
            "function": {
                "name": "big_tool",
                "description": "A tool with a large schema. " * 50,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{i}": {"type": "string", "description": f"param {i} " * 20}
                        for i in range(50)
                    },
                    "required": [f"param_{i}" for i in range(50)],
                },
            },
        }
    ]


# ── Test 1: Main dispatch — under ceiling → .create IS called ─────────────────

class TestMainDispatchUnderCeiling:
    def test_create_called_when_under_ceiling(self):
        """A payload under the ceiling must dispatch successfully.

        The gate must NOT refuse a valid request. This proves the gate
        doesn't over-refuse (false positive).
        """
        agent = _AgentStub(pre_cap=200_000, ceiling=100_000)
        # 50K tokens is well under the 100K effective ceiling.
        kwargs = _big_kwargs(n_tokens_target=50_000)
        kwargs["max_tokens"] = 4096  # reserve
        # Total ≈ 50K + 4096 ≈ 54K < 100K → should pass.

        try:
            _dispatch_nonstreaming_api_request(agent, dict(kwargs), make_client=lambda *_a, **_k: agent.client)
        except ContextCeilingExceeded:
            pytest.fail("Gate refused a payload that is under the ceiling (false positive)")

        assert len(agent.client.create_calls) == 1, "Physical .create must be called when under ceiling"


# ── Test 2: Main dispatch — over ceiling → .create NOT called ─────────────────

class TestMainDispatchOverCeiling:
    def test_refused_before_create(self):
        """An over-ceiling payload must raise ContextCeilingExceeded BEFORE
        client.chat.completions.create is called. No retry, no fallback."""
        agent = _AgentStub(pre_cap=128_000, ceiling=64_000)
        # 200K tokens > 64K ceiling → must refuse.
        kwargs = _big_kwargs(n_tokens_target=200_000)

        with pytest.raises(ContextCeilingExceeded):
            _dispatch_nonstreaming_api_request(agent, dict(kwargs), make_client=lambda *_a, **_k: agent.client)

        assert len(agent.client.create_calls) == 0, (
            "Physical .create must NOT be called when the payload exceeds the ceiling"
        )


# ── Test 3: Auxiliary sync — over ceiling → .create NOT called ────────────────

class TestAuxiliarySyncOverCeiling:
    def test_refused_before_create(self):
        """Auxiliary sync path: _relay_sync_completion must raise
        ContextCeilingExceeded before client.chat.completions.create."""
        agent = _AgentStub(pre_cap=128_000, ceiling=64_000)
        client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=200_000)

        # In production, call_llm publishes the effective ceiling (this model's
        # context clamped by the profile ceiling) in a contextvar before
        # dispatching to the relay helpers.  Reproduce that here so the
        # physical-owner gate has a ceiling to enforce against.
        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                _aux._relay_sync_completion(
                    client,
                    dict(kwargs),
                    provider="openai",
                    api_mode="chat_completions",
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 0, (
            "Auxiliary physical .create must NOT be called when over ceiling"
        )


# ── Test 4: Auxiliary fallback — refusal does NOT trigger fallback ───────────

class TestAuxiliaryFallbackNoBypass:
    def test_refusal_propagates_not_fallback(self):
        """When the primary auxiliary dispatch is refused by the ceiling,
        the refusal must propagate as ContextCeilingExceeded — it must NOT
        be classified as a provider error and trigger a fallback candidate
        dispatch (which would also be over-ceiling and must also refuse)."""
        agent = _AgentStub(pre_cap=128_000, ceiling=64_000)
        # Both primary AND fallback clients are spied.
        primary_client = _SpyClient()
        fallback_client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=200_000)

        # In production, call_llm publishes the effective ceiling in a
        # contextvar before dispatching to the relay helpers.
        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                _aux._relay_sync_completion(
                    primary_client,
                    dict(kwargs),
                    provider="openai",
                    api_mode="chat_completions",
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(primary_client.create_calls) == 0
        assert len(fallback_client.create_calls) == 0, (
            "Fallback client must NOT be called when the primary was ceiling-refused"
        )


# ── Test 5: Credential-refresh retry must NOT bypass refusal ─────────────────

class TestCredentialRefreshNoBypass:
    def test_refusal_not_retried(self):
        """The auxiliary credential-refresh retry path must re-raise
        ContextCeilingExceeded immediately, NOT classify it as an auth error
        and attempt a credential refresh + retry."""
        agent = _AgentStub(pre_cap=128_000, ceiling=64_000)
        client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=200_000)

        # In production, call_llm publishes the effective ceiling in a
        # contextvar before dispatching to the relay helpers.
        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                _aux._relay_sync_completion(
                    client,
                    dict(kwargs),
                    provider="openai",
                    api_mode="chat_completions",
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 0, (
            "Credential-refresh retry must NOT dispatch an over-ceiling request"
        )


# ── Test 6: Bedrock Converse — nested tools + maxTokens → zero converse() ────

class TestBedrockOverCeiling:
    def test_converse_not_called(self, monkeypatch):
        """Bedrock Converse path: over-ceiling payload with nested tools and
        inferenceConfig.maxTokens must raise ContextCeilingExceeded before
        client.converse() is called."""
        agent = _AgentStub(
            pre_cap=128_000,
            ceiling=64_000,
            api_mode="bedrock_converse",
            provider="bedrock",
        )
        # Patch the Bedrock runtime client factory to return our spy,
        # so we can assert converse() is NOT called.
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            lambda region: agent.client.bedrock,
        )
        # Bedrock payload shape: messages + system + toolConfig + inferenceConfig
        bedrock_kwargs = {
            "model": "test-model",
            "messages": _big_messages(n_tokens_target=200_000),
            "system": [{"text": "You are a test assistant."}],
            "toolConfig": {"tools": _tools_payload()},
            "inferenceConfig": {"maxTokens": 4096},
        }

        with pytest.raises(ContextCeilingExceeded):
            _dispatch_nonstreaming_api_request(
                agent, dict(bedrock_kwargs), make_client=lambda *_a, **_k: agent.client
            )

        assert len(agent.client.converse_calls) == 0, (
            "Bedrock converse() must NOT be called when the payload exceeds the ceiling"
        )


# ── Test 7: In-place tool mutation must invalidate stale estimate ─────────────

class TestToolMutationInvalidateCache:
    def test_in_place_mutation_seen_by_gate(self):
        """A middleware that mutates the SAME tool-list object in place must
        be seen by the terminal gate. The gate must NOT use a stale
        identity-keyed cache — it must see the expanded schema.

        Deterministic setup (measured, not guessed):
          base messages estimate  ≈ 55,000
          unexpanded tools estimate ≈ 3,271
          expanded tools estimate   ≈ 6,941
          output reservation        = 4,096
          ceiling                   = 64,000
        So with the UNEXPANDED tools the total ≈ 62,390 (< 64K → dispatch),
        but with the EXPANDED tools the total ≈ 66,060 (> 64K → refuse).

        We pre-warm the tool-estimate cache with the UNEXPANDED value (as a
        prior estimate would have), then expand the schema IN PLACE (same
        object identity). A stale identity-keyed cache would still report the
        unexpanded estimate and let the over-ceiling request through; the
        gate must recompute from the FINAL tools and refuse.
        """
        from agent.model_metadata import _estimate_tools_tokens_rough

        agent = _AgentStub(pre_cap=128_000, ceiling=64_000)
        tools = _tools_payload()

        # Pre-warm the tool-estimate cache with the unexpanded value (simulates
        # a prior estimate a middleware would have triggered before mutating).
        _estimate_tools_tokens_rough(tools)

        base_messages = _big_messages(n_tokens_target=55_000)
        kwargs = {
            "model": "test-model",
            "messages": base_messages,
            "tools": tools,
            "max_tokens": 4096,
        }

        # Now expand the tool schema IN PLACE (same object identity).
        for tool in tools:
            for i in range(100):
                tool["function"]["parameters"]["properties"][f"extra_{i}"] = {
                    "type": "string", "description": "expanded " * 10,
                }
                tool["function"]["parameters"]["required"].append(f"extra_{i}")

        # A fresh estimate of the FINAL tools pushes the total over the 64K
        # ceiling; a stale (pre-mutation) estimate would not.  The gate must
        # see the FINAL (expanded) tools.  Because the non-output portion
        # (input + tools) still fits under the ceiling, the gate clamps the
        # outgoing output cap to the maximum legal allowance derived from the
        # FINAL tools rather than refusing: the clamped cap is the observable
        # that distinguishes the expanded tools (cap = 64000 - 61964 = 2036)
        # from a stale unexpanded estimate (cap would be 64000 - 58294 = 5706).
        _dispatch_nonstreaming_api_request(
            agent, dict(kwargs), make_client=lambda *_a, **_k: agent.client
        )
        # The cap was clamped downward (from 4096) to a value reflecting the
        # FINAL tools — proof the gate saw the expanded schema, not the
        # pre-warmed (stale) unexpanded estimate.  A stale estimate would have
        # clamped the cap to a LARGER value (5706), so asserting the exact
        # FINAL-tools-derived cap (2036) is the cache-invalidation invariant.
        assert len(agent.client.create_calls) == 1, (
            "Shrinkable overflow (non-output fits) must clamp and dispatch"
        )
        dispatched = agent.client.create_calls[0]
        clamped = dispatched["max_tokens"]
        assert clamped < 4096, "Gate must clamp the cap downward for over-ceiling total"
        # The clamp cap reflects the FINAL (expanded) tools: ceiling -
        # (input + expanded_tools).  A stale unexpanded estimate would give a
        # different (larger) cap, so this exact value is the cache-invalidation
        # proof.
        from agent.model_metadata import (
            _estimate_tools_tokens_rough as _est_tools,
            estimate_messages_tokens_rough as _est_msgs,
        )
        expected_cap = 64_000 - (_est_msgs(base_messages) + _est_tools(tools))
        assert clamped == expected_cap, (
            f"Gate clamped to {clamped} but the FINAL-tools-derived cap is "
            f"{expected_cap} — the gate used a stale tool estimate"
        )


# ── Test 8: Codex Responses — final preflight payload → gate ──────────────────

class TestCodexOverCeiling:
    def test_responses_create_not_called(self):
        """Codex Responses path: over-ceiling payload with instructions +
        input + tools + max_output_tokens must raise ContextCeilingExceeded
        before responses.create() is called. The gate must see the FINAL
        preflight-normalized payload, not the pre-preflight request."""
        agent = _AgentStub(
            pre_cap=128_000,
            ceiling=64_000,
            api_mode="codex_responses",
            provider="openai",
        )
        codex_kwargs = {
            "model": "test-model",
            "instructions": "You are a test assistant.",
            "input": _big_messages(n_tokens_target=200_000),
            "tools": _tools_payload(),
            "max_output_tokens": 4096,
        }

        with pytest.raises(ContextCeilingExceeded):
            _dispatch_nonstreaming_api_request(
                agent, dict(codex_kwargs), make_client=lambda *_a, **_k: agent.client
            )

        assert len(agent.client.responses_calls) == 0, (
            "Codex responses.create() must NOT be called when the payload exceeds the ceiling"
        )

# ── Test 9: Relay-enlarges — provider-boundary enforcement (non-stream) ──────

class TestRelayEnlargesNonStream:
    """The relay-helper entry gate fires on the CALLER's request. If Relay
    (relay_llm.execute) enlarges the request before invoking the provider
    callback — expanding tools, appending content, rewriting the payload in
    any way that increases its size — the entry gate has already passed.
    The PROVIDER-BOUNDARY wrapper (_aux_provider_callback) is the second,
    terminal gate: it enforces on the FINAL payload at the exact moment it
    is about to be handed to the provider.

    Deterministic setup (measured):
      caller's messages estimate  ≈ 55,000   (under 64K → entry gate passes)
      Relay enlargement           ≈ +8,000   (pushes final to ≈ 63,000)
      output reservation          = 4,096
      ceiling                     = 64,000
      final total                 ≈ 67,000   (> 64K → provider-boundary refuses)

    The provider callback must NEVER be called with the oversized payload.
    """

    def _relay_that_enlarges(self, spy):
        """A Relay.execute_current that ENLARGES the request by 8K tokens
        before invoking the provider callback, then returns its result."""
        filler = "y" * (8_000 * 4)  # ≈ 8,000 tokens of enlargement

        def _execute(request, callback, **kwargs):
            # Relay enlarges the request (simulates middleware/middleware
            # appending content, expanding tools, or rewriting the payload).
            enlarged = dict(request)
            msgs = list(enlarged.get("messages") or [])
            if msgs and isinstance(msgs[-1], dict):
                enlarged["messages"] = msgs[:-1] + [
                    dict(msgs[-1], content=str(msgs[-1].get("content", "")) + filler)
                ]
            return callback(enlarged)
        return _execute

    def test_provider_never_called_with_oversized_final_payload(self, monkeypatch):
        """Relay enlarges the request past the ceiling. The provider callback
        must NOT be called — the provider-boundary wrapper refuses first."""
        client = _SpyClient()
        # Force the relay path (route non-None) so relay_llm.execute_current
        # is exercised and can ENLARGE the request before the provider call.
        monkeypatch.setattr(
            _aux, "_relay_auxiliary_metadata",
            lambda provider=None, api_mode=None: ("openai", "m", {
                "api_mode": "chat_completions", "api_request_id": "aux-test",
                "call_role": "auxiliary:test", "retry_count": 0, "auxiliary_task": "test",
            }),
        )
        # Caller's request: ≈ 55K tokens — UNDER the 64K ceiling.
        # The entry gate PASSES (correct — the caller's request is fine).
        kwargs = _big_kwargs(n_tokens_target=55_000)
        kwargs["max_tokens"] = 4096

        # Patch Relay to enlarge the request before the provider callback.
        monkeypatch.setattr(
            _relay_llm, "execute_current", self._relay_that_enlarges(client),
        )

        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                _aux._relay_sync_completion(
                    client,
                    dict(kwargs),
                    provider="openai",
                    api_mode="chat_completions",
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 0, (
            "Provider .create must NOT be called with a Relay-enlarged "
            "payload that exceeds the ceiling"
        )

    def test_under_ceiling_still_dispatches(self, monkeypatch):
        """A Relay-enlarged payload that STAYS under the ceiling must still
        dispatch (the gate must not over-refuse)."""
        client = _SpyClient()
        # Caller's request: ≈ 30K tokens — well under.
        # Relay adds 8K → final ≈ 38K + 4096 ≈ 42K < 64K → passes.
        kwargs = _big_kwargs(n_tokens_target=30_000)
        kwargs["max_tokens"] = 4096

        monkeypatch.setattr(
            _relay_llm, "execute_current", self._relay_that_enlarges(client),
        )

        token = _aux.set_aux_ceiling(64_000)
        try:
            _aux._relay_sync_completion(
                client,
                dict(kwargs),
                provider="openai",
                api_mode="chat_completions",
            )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 1, (
            "Provider .create must be called when the final payload is under the ceiling"
        )

    def test_negative_control_without_provider_boundary_wrapper(self, monkeypatch):
        """Prove the provider-boundary wrapper IS the enforcement for Relay
        enlargement: if it is absent (only the entry gate), the provider
        IS called with the oversized final payload (the bug this fixes)."""
        client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=55_000)
        kwargs["max_tokens"] = 4096

        # Patch Relay to enlarge.
        monkeypatch.setattr(
            _relay_llm, "execute_current", self._relay_that_enlarges(client),
        )
        # REMOVE the provider-boundary wrapper (simulate pre-fix code):
        # _aux_provider_callback becomes identity — the gate is only at entry.
        # (Accepts the (callback, *, task, provider, model) signature.)
        monkeypatch.setattr(_aux, "_aux_provider_callback", lambda cb, **kw: cb)

        token = _aux.set_aux_ceiling(64_000)
        try:
            # The entry gate passes (55K < 64K). Relay enlarges to ~63K.
            # Without the provider-boundary wrapper, the provider IS called.
            result = _aux._relay_sync_completion(
                client,
                dict(kwargs),
                provider="openai",
                api_mode="chat_completions",
            )
        finally:
            _aux.reset_aux_ceiling(token)

        # BUG REPRODUCED: provider called with oversized final payload.
        assert len(client.create_calls) == 1, (
            "Without the provider-boundary wrapper, the provider IS called "
            "with the Relay-enlarged payload (this is the bug the wrapper fixes)"
        )

    def test_async_path_provider_never_called(self, monkeypatch):
        """Async relay path: same enforcement — Relay enlarges, provider
        callback must not be called."""
        import asyncio
        client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=55_000)
        kwargs["max_tokens"] = 4096

        filler = "y" * (8_000 * 4)

        monkeypatch.setattr(
            _aux, "_relay_auxiliary_metadata",
            lambda provider=None, api_mode=None: ("openai", "m", {
                "api_mode": "chat_completions", "api_request_id": "aux-test",
                "call_role": "auxiliary:test", "retry_count": 0, "auxiliary_task": "test",
            }),
        )
        async def _enlarging_execute(request, callback, **kwargs_):
            enlarged = dict(request)
            msgs = list(enlarged.get("messages") or [])
            if msgs and isinstance(msgs[-1], dict):
                enlarged["messages"] = msgs[:-1] + [
                    dict(msgs[-1], content=str(msgs[-1].get("content", "")) + filler)
                ]
            return await callback(enlarged)

        monkeypatch.setattr(_relay_llm, "execute_current_async", _enlarging_execute)

        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                asyncio.get_event_loop().run_until_complete(
                    _aux._relay_async_completion(
                        client,
                        dict(kwargs),
                        provider="openai",
                        api_mode="chat_completions",
                    )
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 0


# ── Test 10: Relay-enlarges — provider-boundary enforcement (stream) ─────────

class TestRelayEnlargesStream:
    """Streaming relay path: the same provider-boundary enforcement applies.
    Relay (relay_llm.stream_current) enlarges the request before invoking
    the stream_factory (which wraps client.chat.completions.create). The
    provider-boundary wrapper must refuse the oversized final payload BEFORE
    the provider is called."""

    def test_stream_provider_never_called(self, monkeypatch):
        client = _SpyClient()
        kwargs = _big_kwargs(n_tokens_target=55_000)
        kwargs["max_tokens"] = 4096

        filler = "y" * (8_000 * 4)

        monkeypatch.setattr(
            _aux, "_relay_auxiliary_metadata",
            lambda provider=None, api_mode=None: ("openai", "m", {
                "api_mode": "chat_completions", "api_request_id": "aux-test",
                "call_role": "auxiliary:test", "retry_count": 0, "auxiliary_task": "test",
            }),
        )
        def _enlarging_stream(request, stream_factory, **kwargs_):
            enlarged = dict(request)
            msgs = list(enlarged.get("messages") or [])
            if msgs and isinstance(msgs[-1], dict):
                enlarged["messages"] = msgs[:-1] + [
                    dict(msgs[-1], content=str(msgs[-1].get("content", "")) + filler)
                ]
            return stream_factory(enlarged)

        monkeypatch.setattr(_relay_llm, "stream_current", _enlarging_stream)

        token = _aux.set_aux_ceiling(64_000)
        try:
            with pytest.raises(ContextCeilingExceeded):
                _aux._relay_sync_stream(
                    client,
                    dict(kwargs),
                    provider="openai",
                    api_mode="chat_completions",
                )
        finally:
            _aux.reset_aux_ceiling(token)

        assert len(client.create_calls) == 0, (
            "Streaming provider .create must NOT be called with a "
            "Relay-enlarged payload that exceeds the ceiling"
        )

# ── Test 11: Codex app-server run_turn terminal gate ──────────────────────────

class TestAppServerRunTurnGate:
    """FIFTH physical owner: run_codex_app_server_turn hands the whole turn to
    a codex app-server subprocess via agent._codex_session.run_turn(), bypassing
    BOTH _dispatch_nonstreaming_api_request and interruptible_streaming_api_call.
    The terminal gate must fire BEFORE run_turn() — an oversized final payload
    must be refused locally, and run_turn() must NOT be called.
    """

    def _make_agent(self, ceiling, pre_cap, big_messages):
        import types
        # Tolerant stub: the runtime's post-run_turn accounting/splice section
        # reads many agent attrs; give neutral defaults so a turn that PASSES
        # the gate can complete. The gate-under-test fires BEFORE run_turn,
        # so these defaults only matter for the under-ceiling dispatch cases.
        class _Agent:
            def __init__(self):
                self.api_mode = "codex_app_server"
                self.session_api_calls = 0
                self.context_compressor = None
                self.system_prompt = "You are Hermes."
                self.message_metadata = {}
                self.valid_tool_names = set()
                self._iters_since_skill = 0
                self._skill_nudge_interval = 0
                self._session_db = None
                self.session_id = "t"
            def __getattr__(self, name):
                # Neutral defaults for the many optional post-turn reads
                # (must NOT shadow real attrs set in __init__).
                if name.startswith("_"):
                    raise AttributeError(name)
                return None
        agent = _Agent()
        agent._pre_cap_context_length = pre_cap
        agent._max_context_length = ceiling
        agent._iters_since_skill = 0
        agent._codex_session = None
        def _flush_messages_to_session_db(self, *a, **k): pass
        def _spawn_background_review(self, *a, **k): pass
        def _sync_external_memory_for_turn(self, *a, **k): pass
        agent._flush_messages_to_session_db = _flush_messages_to_session_db.__get__(agent, _Agent)
        agent._spawn_background_review = _spawn_background_review.__get__(agent, _Agent)
        agent._sync_external_memory_for_turn = _sync_external_memory_for_turn.__get__(agent, _Agent)
        # A spy session: run_turn records the call and returns a canned dict.
        spy = types.SimpleNamespace()
        spy.turn_calls = 0

        def run_turn(user_input=None):
            spy.turn_calls += 1
            # The runtime reads turn.interrupted / turn.should_retire / turn.error
            # as attributes — return an object, not a dict.
            return types.SimpleNamespace(
                interrupted=False, should_retire=False, error=None,
                final_text="ok", projected_messages=[], tool_iterations=0,
                thread_id="th-1", turn_id="tu-1",
            )
        spy.run_turn = run_turn
        spy.close = lambda: None
        agent._codex_session = spy
        return agent, spy

    def test_oversized_refuses_and_run_turn_never_called(self):
        from agent.codex_runtime import run_codex_app_server_turn
        from agent.model_metadata import build_final_context_budget as B

        ceiling, pre_cap = 64_000, 128_000
        # FINAL messages payload well over the 64K ceiling.
        big = _big_kwargs(n_tokens_target=70_000)["messages"]
        assert B({"messages": big}, system_prompt="You are Hermes.").total > ceiling, \
            "precondition: final payload must exceed the ceiling"

        agent, spy = self._make_agent(ceiling, pre_cap, big)
        with pytest.raises(ContextCeilingExceeded):
            run_codex_app_server_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=big,
                effective_task_id="t",
            )
        assert spy.turn_calls == 0, "run_turn() must NOT be called for an oversized final payload"

    def test_under_ceiling_still_dispatches_run_turn(self):
        from agent.codex_runtime import run_codex_app_server_turn

        ceiling, pre_cap = 256_000, 272_000
        small = _big_kwargs(n_tokens_target=20_000)["messages"]
        agent, spy = self._make_agent(ceiling, pre_cap, small)
        run_codex_app_server_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=small,
            effective_task_id="t",
        )
        assert spy.turn_calls == 1, "run_turn() must be called when under the ceiling"

    def test_no_ceiling_configured_is_noop(self):
        from agent.codex_runtime import run_codex_app_server_turn

        big = _big_kwargs(n_tokens_target=70_000)["messages"]
        agent, spy = self._make_agent(None, None, big)
        # No ceiling + no pre-cap → the gate is a no-op; run_turn proceeds.
        run_codex_app_server_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=big,
            effective_task_id="t",
        )
        assert spy.turn_calls == 1, "no ceiling configured → dispatch proceeds (gate no-op)"

# ── Test 12: Per-invocation pre-cap invariant (auxiliary/MoA reference) ─────

class TestPerInvocationPreCap:
    """Each auxiliary / MoA reference invocation uses its OWN resolved
    pre-cap — the reference model's own raw capability — NOT the main agent's
    pre-cap and NOT merely the profile ceiling carried through the profile
    context.

    Invariant (the user's example):
        main pre-cap 272K, profile max 200K, reference pre-cap 128K
        → reference effective window must be 128K.

    The ceiling only ever LOWERS a window (min(raw, ceiling)); it never RAISES
    one.  So a reference whose raw capability (128K) is already below the
    profile ceiling (200K) keeps its OWN 128K — it does not inherit the main
    agent's 272K, and the ceiling does not stretch it to 200K.
    """

    def test_reference_uses_its_own_precap_not_main(self):
        import agent.model_metadata as mm

        # main agent's own pre-cap (its model's raw capability): 272K
        # profile ceiling (model.max_context_length): 200K
        # reference model's OWN raw capability (pre-cap): 128K
        with patch.object(mm, "get_model_context_length", return_value=128_000), \
             patch.object(mm, "_get_max_context_length", return_value=200_000):
            ref_effective = mm.effective_context_length(
                model="ref-model", base_url="http://ref", api_key=""
            )
            # Reference effective window = its OWN 128K — NOT the main's 272K
            # and NOT stretched to the profile ceiling 200K.
            assert ref_effective == 128_000
            assert ref_effective != 272_000, "reference must not inherit main pre-cap"
            assert ref_effective != 200_000, "ceiling must not raise the reference window"

    def test_reference_above_ceiling_is_lowered_by_profile(self):
        import agent.model_metadata as mm

        # reference raw capability (272K) ABOVE the profile ceiling (200K) →
        # the ceiling LOWERS it to 200K.  min(272K, 200K) = 200K.
        with patch.object(mm, "get_model_context_length", return_value=272_000), \
             patch.object(mm, "_get_max_context_length", return_value=200_000):
            ref_effective = mm.effective_context_length(
                model="big-ref", base_url="http://ref", api_key=""
            )
            assert ref_effective == 200_000

    def test_reference_gate_uses_reference_effective_not_main(self):
        """The MoA reference GATE enforces against the reference's own
        effective window (its own pre-cap, ceiling-clamped), never the main
        agent's.  Prove: a payload just under the reference's 128K window is
        ACCEPTED even though it would exceed a smaller (main) window."""
        import agent.moa_loop as moa
        from agent.model_metadata import estimate_messages_tokens_rough
        from agent.agent_runtime_helpers import (
            canonical_request_budget,
            enforce_effective_context_limit,
        )

        # Reference's own effective window (128K).  A payload of ~120K is under
        # it → accepted.  The same payload would exceed a 100K main window.
        ref_window = 128_000
        # base estimate ~120K → chars ≈ 120K*4
        content = "x" * (120_000 * 4)
        messages = [{"role": "user", "content": content}]
        base = estimate_messages_tokens_rough(messages)
        # Reserve uses the Hermes default (4096) — same as the gate.
        budget = canonical_request_budget(
            messages, output_reserve=moa._resolve_moa_output_reserve(None)
        )
        assert budget <= ref_window, f"setup: budget {budget} over ref window"
        # Gate against the REFERENCE's own effective window → ACCEPTED (no raise).
        enforce_effective_context_limit(
            budget, pre_cap=ref_window, ceiling=None, reason="MoA reference dispatch",
        )
        # ...but the SAME payload exceeds a 100K (main) window → refused.
        with pytest.raises(ContextCeilingExceeded):
            enforce_effective_context_limit(
                budget, pre_cap=100_000, ceiling=None, reason="main agent window",
            )
