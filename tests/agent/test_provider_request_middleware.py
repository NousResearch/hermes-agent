"""Outbound-only middleware applies to side paths without changing stored history."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest

MARKER = " [display-only-marker]"
MODES = ["chat_completions", "codex_responses", "anthropic_messages"]


def history():
    return [
        {"role": "user", "content": "Keep working."},
        {"role": "assistant", "content": "Progress." + MARKER},
        {"role": "user", "content": "Continue."},
    ]


@pytest.fixture
def middleware(monkeypatch):
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = PluginContext(PluginManifest(name="request-filter-test"), manager)
    observed = []

    def install(behavior):
        def callback(request, **context):
            observed.append((deepcopy(request), context))
            key = "input" if context.get("api_mode") == "codex_responses" and context["request_kind"] != "compression_input" else "messages"
            assert key in request
            for message in request[key]:
                if message.get("role") == "assistant":
                    content = message.get("content")
                    if isinstance(content, str):
                        message["content"] = content.replace(MARKER, "")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block.get("text"), str):
                                block["text"] = block["text"].replace(MARKER, "")
            if behavior == "raise":
                # Even mutation followed by failure must not affect the original.
                raise RuntimeError("broken middleware")
            return {"request": request}

        if behavior != "none":
            ctx.register_middleware("llm_request", callback)
        return observed

    return install


def response(text="Summary."):
    return NS(choices=[NS(message=NS(content=text, tool_calls=[]), finish_reason="stop")], usage=None)


def test_turn_to_iteration_summary_keeps_display_footer_only_in_storage(middleware, monkeypatch, tmp_path):
    """Real agent, transports, plugin registration and SQLite; only the provider is mocked."""
    from unittest.mock import MagicMock, patch
    from hermes_state import SessionDB
    from run_agent import AIAgent

    observed = middleware("strip")
    ctx = PluginContext(PluginManifest(name="display-test"), plugins.get_plugin_manager())
    ctx.register_hook("transform_llm_output", lambda response_text, **kw: response_text + MARKER)
    db = SessionDB(db_path=tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_id="session", session_db=db,
        )
    sent = []
    def create(**kwargs):
        sent.append(deepcopy(kwargs))
        return response("Progress." if len(sent) == 1 else "Summary.")
    client = MagicMock()
    client.chat.completions.create = create
    monkeypatch.setattr(agent, "client", client)
    monkeypatch.setattr(agent, "_ensure_primary_openai_client", lambda **kw: client)
    result = agent.run_conversation("Keep working.")
    assert MARKER in result["final_response"]
    stored = db.get_messages("session")
    assert any(MARKER in (m.get("content") or "") for m in stored if m["role"] == "assistant")
    assert agent._handle_max_iterations(deepcopy(result["messages"]), 1) == "Summary."
    assert len(sent) == 2
    assert MARKER not in str(sent[-1]["messages"])
    assert db.get_messages("session") == stored
    assert [c["request_kind"] for _, c in observed] == ["turn", "iteration_summary"]
    db.close()


@pytest.mark.parametrize("api_mode", MODES)
@pytest.mark.parametrize("behavior", ["strip", "none", "raise"])
def test_iteration_summary_filters_each_attempt_only(api_mode, behavior, middleware, monkeypatch):
    from agent import chat_completion_helpers as helpers

    observed = middleware(behavior)
    sent = []
    def send(request):
        sent.append(deepcopy(request))
        return "" if len(sent) == 1 else "Summary."
    key = "input" if api_mode == "codex_responses" else "messages"
    def build(messages, **kwargs):
        return {key: deepcopy(messages), "model": "test-model", "prompt_cache_key": "stable"}
    transport = NS(build_kwargs=lambda **kw: build(kw["messages"]),
                   normalize_response=lambda text, **kw: NS(content=text, tool_calls=[]))
    client = NS(chat=NS(completions=NS(create=lambda **kw: send(kw))))
    agent = NS(
        api_mode=api_mode, provider="test", model="test-model", base_url="https://example.invalid",
        task_id="task", session_id="session", platform="cli", max_iterations=1, max_tokens=2048,
        suppress_status_output=True, _cached_system_prompt="Stable system.", ephemeral_system_prompt="",
        prefill_messages=[], reasoning_config={}, _is_anthropic_oauth=False, _image_rejecting_models=set(),
        _should_sanitize_tool_calls=lambda: False, _copy_reasoning_content_for_api=lambda *a: None,
        _sanitize_api_messages=lambda m: m, _drop_thinking_only_and_merge_users=lambda m: m,
        _build_api_kwargs=build, _get_transport=lambda: transport, _run_codex_stream=send,
        _anthropic_preserve_dots=lambda: False, _anthropic_messages_create=send,
        _ensure_primary_openai_client=lambda **kw: client, _interruptible_api_call=send,
    )
    monkeypatch.setattr(helpers, "sanitize_outbound_kwargs", lambda *a: None)
    monkeypatch.setattr(helpers, "bypass_chat_sdk_request_transform", lambda request, client: request)
    monkeypatch.setattr(helpers, "_merge_nous_portal_messages_extra_body", lambda a, kw: kw)
    monkeypatch.setattr("agent.relay_llm.execute_current", lambda request, callback, **kw: callback(request))
    messages = history()
    original = deepcopy(messages)
    assert helpers.handle_max_iterations(agent, messages, 1) == "Summary."
    assert len(sent) == 2
    for request in sent:
        assert (MARKER in str(request)) == (behavior != "strip")
        assert request["prompt_cache_key"] == "stable"
    assert messages[:len(original)] == original
    if behavior != "none":
        assert len(observed) == 2
        assert all(c["request_kind"] == "iteration_summary" and c["session_id"] == "session" for _, c in observed)


@pytest.mark.parametrize("api_mode", MODES)
@pytest.mark.parametrize("behavior", ["strip", "none", "raise"])
def test_moa_reference_and_prepared_aggregator_filter_canonical_copies(api_mode, behavior, middleware, monkeypatch):
    from agent import moa_loop

    observed = middleware(behavior)
    sent = []
    def call_llm(**kw):
        sent.append(deepcopy(kw))
        return response("Advice.")
    runtime = {"provider": "test", "model": "slot", "api_mode": api_mode}
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: dict(runtime))
    monkeypatch.setattr(moa_loop, "_trim_messages_for_reference", lambda m, *a, **kw: m)
    monkeypatch.setattr(moa_loop, "_maybe_apply_moa_cache_control", lambda m, *a, **kw: m)
    monkeypatch.setattr(moa_loop, "call_llm", call_llm)
    messages = history()
    original = deepcopy(messages)
    agent = NS(task_id="task", session_id="session", platform="cli")
    slot = {"provider": "test", "model": "slot"}
    label, text, accounting = moa_loop._run_references_parallel([slot], messages, agent=agent)[0]
    assert text == "Advice."
    prepared = {"messages": messages, "aggregator": slot, "aggregator_temperature": None}
    original_prepared = deepcopy(prepared)
    facade = moa_loop.MoAChatCompletions("test", agent=agent)
    monkeypatch.setattr(facade, "_plan_aggregator_cache", lambda m, tools, *a: (m, tools))
    facade._pending_trace = {}
    facade.create(_moa_prepared_request=prepared, max_tokens=2048)
    assert len(sent) == 2
    for request in sent:
        assert (MARKER in str(request["messages"])) == (behavior != "strip")
        assert request["api_mode"] == api_mode
    assert accounting.messages == sent[0]["messages"]
    assert facade._pending_trace["aggregator_input_messages"] == sent[1]["messages"]
    assert messages == original and prepared == original_prepared
    if behavior != "none":
        assert [c["request_kind"] for _, c in observed] == ["moa_reference", "moa_aggregator"]
        assert all(c["session_id"] == "session" for _, c in observed)


@pytest.mark.parametrize("tail_mode", ["lean", "legacy"])
@pytest.mark.parametrize("behavior", ["strip", "none", "raise"])
def test_compression_filters_only_summary_input(tail_mode, behavior, middleware, monkeypatch, tmp_path):
    from agent import context_compressor as cc

    observed = middleware(behavior)
    sent = []
    def call_llm(**kw):
        sent.append(deepcopy(kw))
        return response("## Completed Actions\nProgress recorded.")
    monkeypatch.setattr(cc, "call_llm", call_llm)
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **kw: 100000)
    compressor = cc.ContextCompressor(model="test", quiet_mode=True, tail_mode=tail_mode)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session", "cli")
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"Turn {i}." + (MARKER if i % 2 else "")}
        for i in range(12)
    ]
    db.append_messages_batch("session", messages)
    stored = db.get_messages("session")
    compressor.bind_session_state(db, "session")
    monkeypatch.setattr(compressor, "_compress_window", lambda m: (2, 8))
    original = deepcopy(messages)
    result = compressor.compress(messages, force=True)
    assert len(sent) == 1
    assert (MARKER in sent[0]["messages"][0]["content"]) == (behavior != "strip")
    assert messages == original
    assert [(m["role"], m["content"]) for m in result[-2:]] == [
        (m["role"], m["content"]) for m in original[-2:]
    ]  # Retained display footer is not rewritten.
    assert db.get_messages("session") == stored
    db.close()
    if behavior != "none":
        assert observed[0][1]["request_kind"] == "compression_input"
        assert observed[0][1]["session_id"] == "session"


@pytest.mark.parametrize("behavior", ["strip", "none", "raise"])
def test_micro_compression_filters_before_serialization(behavior, middleware, monkeypatch):
    from agent import context_compressor as cc
    from agent import auxiliary_client

    middleware(behavior)
    sent = []
    def call_llm(**kw):
        sent.append(deepcopy(kw))
        # Failed summary leaves both the input and returned transcript intact.
        return response("")
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    compressor = cc.ContextCompressor(model="test", quiet_mode=True)
    compressor._micro_compact_enabled = True
    monkeypatch.setattr(compressor, "_next_exchange", lambda messages: (0, 2))
    messages = history() + [{"role": "assistant", "content": "Tail." + MARKER}]
    original = deepcopy(messages)
    assert compressor._micro_compact(messages) == original
    assert len(sent) == 1
    assert (MARKER in str(sent[0]["messages"])) == (behavior != "strip")
    assert messages == original


@pytest.mark.parametrize("api_mode", MODES)
@pytest.mark.parametrize("behavior", ["none", "raise", "raise_after_rewrite", "infrastructure_failure"])
def test_middleware_failure_or_absence_preserves_payload_identity(api_mode, behavior, middleware, monkeypatch):
    from agent.moa_loop import _moa_request_middleware
    from hermes_cli import middleware as mw

    if behavior == "raise_after_rewrite":
        middleware("strip")
        middleware("raise")
    else:
        middleware(behavior if behavior != "infrastructure_failure" else "none")
    if behavior == "infrastructure_failure":
        def broken(*args, **kwargs):
            raise RuntimeError("manager unavailable")
        monkeypatch.setattr(mw, "apply_llm_request_middleware", broken)
    request = {"messages": history(), "api_mode": api_mode}
    original = deepcopy(request)
    assert mw.apply_request_middleware_or_original(request, request_kind="compression_input") is request
    assert _moa_request_middleware(request, "moa_reference") is request
    assert request == original
