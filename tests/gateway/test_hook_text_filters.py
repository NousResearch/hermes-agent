"""Contract tests for the gateway's collectable text-filter hooks.

A subscribed hook may replace the user text the agent sees
(``agent:message:filter``) or the assistant text that goes out
(``agent:response:filter``). Both call sites funnel through the shared
``apply_collectable_text_filter`` helper, so these tests pin its contract:
the first valid replacement wins, malformed returns are ignored, and a
subscriber that raises never breaks the turn (fail-open).
"""

import contextlib
from typing import Any, cast

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.run_turn import apply_collectable_text_filter, filter_model_history
from gateway.session import SessionSource

# Reused harness: it drives the REAL in-band queued (/queue) drain through
# ``GatewayRunner._run_agent`` with a fake AIAgent — the exact path whose coverage the review
# asked to confirm.
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
    _TwoTurnAgent,
)


class _FakeHooks:
    """Minimal stand-in for the hook registry: only ``emit_collect`` is used."""

    def __init__(self, results=(), raises=None):
        self._results = results
        self._raises = raises
        self.calls = []

    async def emit_collect(self, event_type, context):
        self.calls.append((event_type, context))
        if self._raises is not None:
            raise self._raises
        return self._results


@pytest.mark.asyncio
async def test_no_hooks_leaves_text_untouched():
    """No subscribers (empty result list) must change neither text path."""
    hooks = _FakeHooks(results=[])

    message = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"platform": "telegram"}, "message", "original message",
    )
    response = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"platform": "telegram"}, "response", "original response",
    )

    assert message == "original message"
    assert response == "original response"


@pytest.mark.asyncio
async def test_message_filter_replacement_is_applied():
    """A ``{"message": ...}`` result replaces the text the agent sees."""
    hooks = _FakeHooks(results=[{"message": "redacted message"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"session_id": "s1"}, "message", "secret message",
    )

    assert result == "redacted message"


@pytest.mark.asyncio
async def test_first_valid_result_wins():
    """With several subscribers, the first valid replacement is the one applied."""
    hooks = _FakeHooks(results=[{"response": "first"}, {"response": "second"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"session_id": "s1"}, "response", "original response",
    )

    assert result == "first"


@pytest.mark.asyncio
async def test_malformed_hook_results_are_ignored():
    """Non-dict returns, wrong value types, and missing keys are skipped silently."""
    hooks = _FakeHooks(results=[
        "not a dict",
        {"response": 123},          # present but not a string
        {"other_key": "value"},     # key absent
        None,
    ])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {}, "response", "untouched",
    )

    assert result == "untouched"


@pytest.mark.asyncio
async def test_raising_hook_is_fail_open():
    """A subscriber blowing up returns the original text instead of propagating."""
    hooks = _FakeHooks(raises=RuntimeError("plugin exploded"))

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {}, "message", "untouched",
    )

    assert result == "untouched"


# ── Where the filters are emitted: the single turn funnel ────────────────────────────────
# Both turn entry points pass through ``GatewayRunner._run_agent``: the idle-message handler
# (``_handle_message_with_agent``) and the in-band drain of a message that arrived while a turn
# was still running (``_run_agent_queued_followup``). An emission placed in one caller silently
# skips the other — and the /queue message is precisely the one a user typed while the agent was
# busy, which is the PII case this hook pair exists for.


class _RecordingFilterHooks:
    """Hook registry stand-in: records every emission and rewrites both texts."""

    loaded_hooks = True

    def __init__(self, message_replacement=None, response_prefix=None, raises=None):
        self.events: list = []
        self._message = message_replacement
        self._prefix = response_prefix
        self._raises = raises

    async def emit(self, event_type, context):  # only recorded; nothing subscribes to it here
        self.events.append((event_type, dict(context)))

    async def emit_collect(self, event_type, context):
        self.events.append((event_type, dict(context)))
        if self._raises is not None:
            raise self._raises
        if event_type == "agent:message:filter" and self._message is not None:
            return [{"message": self._message}]
        if event_type == "agent:response:filter" and self._prefix is not None:
            return [{"response": self._prefix + (context.get("response") or "")}]
        return []


# The funnel is called unbound so a stub can stand in for the runner: cast keeps the type
# checker out of the way without weakening the production signature.
_run_agent_unbound = cast(Any, GatewayRunner._run_agent)


class _FunnelStub:
    """Just enough runner for the real ``GatewayRunner._run_agent`` (called unbound) to run."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.seen: list = []
        self.model_histories: list = []

    def _profile_scope_for_source(self, source):
        return contextlib.nullcontext()

    async def _run_agent_inner(self, message, context_prompt, history, source, session_id, **turn_kwargs):
        self.seen.append(message)
        self.model_histories.append(history)
        return {"final_response": f"done:{message}"}


@pytest.mark.asyncio
async def test_the_turn_funnel_filters_both_directions():
    """The model reads the inbound filter's output; the funnel returns the outbound one's."""
    hooks = _RecordingFilterHooks(message_replacement="[redactado]", response_prefix="revelado:")
    stub = _FunnelStub(hooks)

    result = await _run_agent_unbound(
        stub, message="mi dni es 12345678Z", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-filtros",
    )

    assert stub.seen == ["[redactado]"]
    assert result["final_response"] == "revelado:done:[redactado]"

    # One emission per direction, per turn: the move must not double-filter.
    assert [event for event, _ in hooks.events] == [
        "agent:message:filter", "agent:response:filter",
    ]
    context = hooks.events[0][1]
    assert context["session_id"] == "s-filtros"
    assert context["platform"] == "telegram"


@pytest.mark.asyncio
async def test_the_turn_funnel_tolerates_a_runner_without_a_hook_registry():
    """Proxy dispatch builds a runner with no hook registry at all: the turn must still run.

    Regression: emitting the filters on the funnel put a ``self.hooks`` read on the path every
    turn takes, including runners that never carry a registry (``test_proxy_mode``).
    """
    stub = _FunnelStub(_RecordingFilterHooks())
    del stub.hooks

    result = await _run_agent_unbound(
        stub, message="hola", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-sin-registro",
    )

    assert stub.seen == ["hola"]
    assert result["final_response"] == "done:hola"


@pytest.mark.asyncio
async def test_the_turn_funnel_is_fail_open():
    """A raising subscriber leaves both texts untouched instead of breaking the turn."""
    hooks = _RecordingFilterHooks(raises=RuntimeError("plugin exploded"))
    stub = _FunnelStub(hooks)

    result = await _run_agent_unbound(
        stub, message="intacto", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-fail-open",
    )

    assert stub.seen == ["intacto"]
    assert result["final_response"] == "done:intacto"


@pytest.mark.asyncio
async def test_a_queued_followup_is_filtered_too(monkeypatch, tmp_path):
    """The review's ask: a /queue message (drained in-band, mid-turn) must be filtered as well.

    Driven through the real drain — the follow-up re-enters ``_run_agent`` and never touches the
    hook block in ``_handle_message_with_agent``, so before the filters moved to the funnel this
    second message reached the model raw.
    """
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    runner.hooks = _RecordingFilterHooks(message_replacement="[ofuscado]", response_prefix="salida:")

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="el seguimiento", message_type=MessageType.TEXT, source=_source(), message_id="queued-f",
    )

    result = await runner._run_agent(
        message="el primer turno", context_prompt="", history=[], source=_source(),
        session_id="s-cola", session_key=SESSION_KEY,
    )

    # Both the opening turn AND the queued follow-up reached the model filtered.
    assert _TwoTurnAgent.calls == ["[ofuscado]", "[ofuscado]"]
    assert result["final_response"] == "salida:done-2"

    events = [event for event, _ in runner.hooks.events]
    # One inbound filter per turn (two turns ran)...
    assert events.count("agent:message:filter") == 2
    # Every delivered turn in the chain is filtered exactly once: first reply before recursion,
    # terminal reply before normal completion. Neither response is filtered twice by a parent frame.
    assert events.count("agent:response:filter") == 2




class _SentinelRedactor(_RecordingFilterHooks):
    """Redact a test sentinel in both live messages and persisted model history."""

    SECRET = "SENTINEL-PII-938475"

    async def emit_collect(self, event_type, context):
        self.events.append((event_type, dict(context)))
        value = context.get("message")
        if event_type == "agent:message:filter" and isinstance(value, str) and self.SECRET in value:
            return [{"message": value.replace(self.SECRET, "[REDACTED]")}]
        return []


@pytest.mark.asyncio
async def test_filtered_user_text_never_reenters_model_history_on_turn_two():
    """Two-turn PII regression: the transcript may retain raw text, but model history may not.

    Turn one filters the live user message. Turn two is given the exact raw transcript row a store
    can replay (including api_content, the byte-fidelity sidecar) and must hand only redacted text
    to the model. Filtering operates on a view/copy: it does not rewrite the caller's persisted rows.
    """
    hooks = _SentinelRedactor()
    stub = _FunnelStub(hooks)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm")
    raw = "my secret is " + hooks.SECRET

    await _run_agent_unbound(
        stub, message=raw, context_prompt="", history=[], source=source, session_id="s-two-turn-pii",
    )
    persisted_history = [{"role": "user", "content": raw, "api_content": raw, "timestamp": 1}]
    snapshot = [dict(row) for row in persisted_history]
    await _run_agent_unbound(
        stub, message="continue", context_prompt="", history=persisted_history,
        source=source, session_id="s-two-turn-pii",
    )

    model_history_turn_two = stub.model_histories[1]
    assert hooks.SECRET not in repr(model_history_turn_two)
    assert model_history_turn_two[0]["content"] == "my secret is [REDACTED]"
    assert model_history_turn_two[0].get("api_content", "").find(hooks.SECRET) == -1
    assert persisted_history == snapshot, "model-view filtering must not mutate stored transcript rows"
    assert stub.seen == ["my secret is [REDACTED]", "continue"]
    assert any(ctx.get("history") is True for event, ctx in hooks.events if event == "agent:message:filter")


@pytest.mark.asyncio
async def test_history_api_content_sidecar_is_filtered_even_when_display_content_is_clean():
    """A stale/raw API sidecar must not bypass a clean-looking display content field."""
    hooks = _SentinelRedactor()
    secret = hooks.SECRET
    history = [{
        "role": "user", "content": "already safe", "api_content": f"hidden {secret}",
    }]
    filtered = await filter_model_history(hooks, "agent:message:filter", {}, history)
    assert secret not in repr(filtered)
    assert filtered[0]["content"] == "already safe"
    assert filtered[0]["api_content"] == "hidden [REDACTED]"
    assert history[0]["api_content"] == f"hidden {secret}"


@pytest.mark.asyncio
async def test_model_history_filter_is_identity_without_a_hook_registry():
    history = [{"role": "user", "content": "keep exactly"}]
    assert await filter_model_history(None, "agent:message:filter", {}, history) is history


@pytest.mark.asyncio
async def test_queued_first_response_is_filtered_before_its_send(monkeypatch, tmp_path):
    """The first response of a queued chain is a real delivery and must not escape raw."""
    class _CaptureAdapter(HookRecordingAdapter):
        def __init__(self):
            super().__init__()
            self.sent_texts = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent_texts.append(content)
            return SendResult(success=True, message_id=f"sent-{len(self.sent_texts)}")

    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)
    adapter = _CaptureAdapter()
    runner = _make_runner(adapter)
    runner.hooks = _RecordingFilterHooks(response_prefix="filtered:")
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="queued next", message_type=MessageType.TEXT, source=_source(), message_id="queued-first-filter",
    )

    result = await runner._run_agent(
        message="opening", context_prompt="", history=[], source=_source(),
        session_id="s-queued-first-response", session_key=SESSION_KEY,
    )

    assert result["final_response"] == "filtered:done-2"
    assert "filtered:done-1" in adapter.sent_texts
    assert not any(text == "done-1" for text in adapter.sent_texts)
    assert adapter.sent_texts.count("filtered:done-1") == 1
    assert [event for event, _ in runner.hooks.events].count("agent:response:filter") == 2


# ── Puntos de la revisión del mantenedor (30-09-2026) ────────────────────────────────────
# (1) los carriles de texto «en caliente» (steer / redirect) también deben filtrarse;
# (2) en un turno con streaming, el reemplazo de salida NO puede perderse;
# (3) las emisiones deben correr dentro del ámbito del perfil del source, no del llamante.

_busy_unbound = cast(Any, GatewayRunner._try_agent_verb)


class _ScopeTrackingStub:
    """Runner de mentira que registra cuándo está dentro del ámbito de perfil."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.en_scope = False
        self.momentos: list = []          # [(evento, ¿estaba en scope?)]
        self.hooks.momentos = self.momentos
        self.seen: list = []

    def _profile_scope_for_source(self, source):
        # El ámbito real es un context manager SÍNCRONO (``with``), con el trabajo async dentro.
        @contextlib.contextmanager
        def _scope():
            self.en_scope = True
            try:
                yield
            finally:
                self.en_scope = False
        return _scope()

    async def _run_agent_inner(self, message, context_prompt, history, source, session_id, **kw):
        self.seen.append(message)
        return {"final_response": f"done:{message}", "already_sent": True}

    # Lo que usa el camino de steer/redirect
    def _steer_text_with_origin(self, text, event):
        return text

    def _steer_running_agent(self, running_agent, text):
        running_agent.recibido.append(text)
        return True


class _ScopeAware(_RecordingFilterHooks):
    """Hooks que anotan, en cada emisión, si el stub estaba dentro del ámbito del perfil."""

    def __init__(self, stub, message_replacement=None, response_prefix=None):
        super().__init__(message_replacement=message_replacement, response_prefix=response_prefix)
        self._stub = stub

    async def emit_collect(self, event_type, context):
        self._stub.momentos.append((event_type, self._stub.en_scope))
        return await super().emit_collect(event_type, context)


class _ScopeHooks(_RecordingFilterHooks):
    """Registra, en cada emisión, si estaba dentro del ámbito del perfil."""

    momentos: list = []

    async def emit_collect(self, event_type, context):
        self.momentos.append((event_type, getattr(self, "_en_scope", None)))
        return await super().emit_collect(event_type, context)


@pytest.mark.asyncio
async def test_streamed_turn_still_delivers_the_replacement():
    """Punto 2: si el cuerpo ya se envió por streaming, el reemplazo no puede perderse.

    ``already_sent`` se calcula con el texto ANTERIOR al filtro, así que sin reactivar la costura
    de post-streaming (``response_transformed``) la entrega normal se suprime y el texto filtrado
    no llega a nadie.
    """
    hooks = _RecordingFilterHooks(response_prefix="revelado:")
    stub = _ScopeTrackingStub(hooks)

    result = await _run_agent_unbound(
        stub, message="hola", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        session_id="s-stream",
    )

    assert result["already_sent"] is True
    assert result["final_response"] == "revelado:done:hola"
    # La costura existente debe quedar armada para que el consumidor edite el mensaje ya enviado.
    assert result["response_transformed"] is True


@pytest.mark.asyncio
async def test_a_non_streamed_turn_does_not_arm_the_transform():
    """Sin streaming no hay nada que editar: la entrega normal manda el texto filtrado."""
    hooks = _RecordingFilterHooks(response_prefix="revelado:")
    stub = _ScopeTrackingStub(hooks)
    stub._run_agent_inner = None  # no usado; se redefine abajo

    async def _inner(message, context_prompt, history, source, session_id, **kw):
        stub.seen.append(message)
        return {"final_response": f"done:{message}"}

    stub._run_agent_inner = _inner

    result = await _run_agent_unbound(
        stub, message="hola", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        session_id="s-nostream",
    )

    assert result["final_response"] == "revelado:done:hola"
    assert "response_transformed" not in result


@pytest.mark.asyncio
async def test_both_emissions_run_inside_the_source_profile_scope():
    """Punto 3: el registro de hooks se resuelve por el ámbito ACTIVO en el momento de emitir."""
    hooks = _RecordingFilterHooks(message_replacement="[x]", response_prefix="y:")
    stub = _ScopeTrackingStub(hooks)
    stub.hooks = hooks

    hooks = _ScopeAware(stub, message_replacement="[x]", response_prefix="y:")
    stub.hooks = hooks

    await _run_agent_unbound(
        stub, message="dato", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        session_id="s-scope",
    )

    assert stub.momentos == [
        ("agent:message:filter", True),
        ("agent:response:filter", True),
    ]


class _BusyStub(_ScopeTrackingStub):
    """Stub que EJERCITA los helpers reales del runner en los carriles en caliente."""

    _filter_inbound_text = cast(Any, GatewayRunner._filter_inbound_text)
    _text_filter_context = cast(Any, GatewayRunner._text_filter_context)


class _AgentDeMentira:
    """Doble del agente en marcha: registra lo que le llega por steer/redirect."""

    def __init__(self):
        self.recibido: list = []

    def steer(self, text):
        self.recibido.append(text)
        return True

    def redirect(self, text):
        self.recibido.append(text)
        return True


@pytest.mark.asyncio
async def test_steer_and_redirect_text_is_filtered_before_reaching_the_model():
    """Punto 1: el texto tecleado con el agente ocupado también pasa por el filtro.

    Ambas ramas (steer y redirect) desembocan en ``_try_agent_verb``; antes del arreglo el texto
    llegaba al modelo sin pasar por ``agent:message:filter`` — justo el caso PII.
    """
    for verbo in ("steer", "redirect"):
        stub = _BusyStub(_RecordingFilterHooks())
        hooks = _ScopeAware(stub, message_replacement="[dni oculto]")
        stub.hooks = hooks
        stub.momentos = []
        agente = _AgentDeMentira()

        ok = await _busy_unbound(
            stub, agente, verbo, "mi dni es 12345678Z", "s-key",
            event=MessageEvent(
                text="mi dni es 12345678Z", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
                message_id="m1",
            ),
        )

        assert ok is True
        assert agente.recibido == ["[dni oculto]"], f"el verbo {verbo} no filtró el texto"
        assert [e for e, _ in hooks.events] == ["agent:message:filter"]
        assert hooks.events[0][1]["chat_id"] == "1"


@pytest.mark.asyncio
async def test_the_busy_lanes_filter_inside_the_profile_scope_too():
    """El mismo cuidado de ámbito que en el funnel, para los carriles en caliente."""
    stub = _BusyStub(_RecordingFilterHooks())
    hooks = _ScopeAware(stub, message_replacement="[x]")
    stub.hooks = hooks
    stub.momentos = []
    agente = _AgentDeMentira()

    await _busy_unbound(
        stub, agente, "steer", "texto", "s-key",
        event=MessageEvent(
            text="texto", message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
            message_id="m2",
        ),
    )

    assert stub.momentos == [("agent:message:filter", True)]

