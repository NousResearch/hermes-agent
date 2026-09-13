"""Behavior contracts for the pre-compression memory-context handoff."""

import json
from unittest.mock import MagicMock

import pytest


def _nested_alternate_json_spelling(layers: int) -> str:
    spelling = r"\u0053\/K\u0065y"
    for _ in range(layers - 1):
        spelling = json.dumps(spelling)[1:-1]
    return spelling


ALTERNATE_JSON_SPELLINGS = tuple(
    pytest.param(
        _nested_alternate_json_spelling(layers),
        id=f"json-escape-layers-{layers}",
    )
    for layers in (1, 2, 5, 10, 13)
)


def _make_agent(memory_manager, compressor):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=None,
        session_id="test-session",
        skip_context_files=True,
        skip_memory=True,
    )

    agent._memory_manager = memory_manager
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "new-system-prompt"
    return agent


def _messages():
    return [{"role": "user", "content": f"message {i}"} for i in range(6)]


def _configure_engine_state(engine):
    engine.compression_count = 1
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    engine._last_summary_error = None
    engine._last_compress_aborted = False
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None


def test_on_pre_compress_result_reaches_compressor_with_existing_options():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-orchestrator"
    received = {}
    compressor = MagicMock()

    def capture_compress(
        incoming,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        received.update(
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        return [incoming[0], incoming[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    messages = _messages()

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=100_000,
        focus_topic="checkpoint continuity",
        force=True,
    )

    # Positional arg is the raw transcript (the historical v1 provider
    # contract); evidence_messages carries the normalized handoff for
    # checkpoint API v2+ providers. All-user messages normalize to themselves.
    manager.on_pre_compress.assert_called_once_with(
        messages, evidence_messages=messages
    )
    assert received == {
        "current_tokens": 100_000,
        "focus_topic": "checkpoint continuity",
        "force": True,
        "memory_context": "Checkpoint id: ctx-orchestrator",
    }


def test_legacy_engine_receives_only_supported_compression_arguments():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: unsupported-by-legacy"
    calls = []

    class StrictLegacyEngine:
        def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return [messages[0], messages[-1]]

    engine = StrictLegacyEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)

    compressed, _prompt = agent._compress_context(
        _messages(),
        "sys",
        approx_tokens=100_000,
        focus_topic="unsupported focus",
        force=True,
    )

    assert len(compressed) == 2
    assert calls == [100_000]


def test_provider_context_is_strictly_sanitized_before_plugin_engine(monkeypatch):
    prefix_secret = "sk-" + "a" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    fragment_secret = "FRAG_SECRET"
    relative_secret = "REL_SECRET"
    encoded_key_secret = "ENC_SECRET"
    hyphen_client_secret = "HYPHEN_CLIENT_SECRET"
    hyphen_access_secret = "HYPHEN_ACCESS_SECRET"
    hyphen_api_secret = "HYPHEN_API_SECRET"
    encoded_hyphen_secret = "ENCODED_HYPHEN_SECRET"
    network_userinfo_secret = "NET_SECRET"
    manager = MagicMock()
    manager.on_pre_compress.return_value = (
        f"api key: {prefix_secret}\n"
        f"callback: https://example.test/cb?access_token={query_secret}&state=ok\n"
        f"endpoint: https://user:{userinfo_value}@example.test/private\n"
        f"fragment: https://x.test/#access_token={fragment_secret}&view=public\n"
        f"relative: /resume?token={relative_secret}&view=public\n"
        f"encoded: https://x.test/cb?client%5Fsecret={encoded_key_secret}&view=public\n"
        f"hyphen-client: /resume?client-secret={hyphen_client_secret}&view=public\n"
        f"hyphen-access: /resume?Access-Token={hyphen_access_secret}&view=public\n"
        f"hyphen-api: /resume?api-key={hyphen_api_secret}&view=public\n"
        f"encoded-hyphen: /resume?client%2Dsecret={encoded_hyphen_secret}&view=public\n"
        f"network: //user:{network_userinfo_secret}@x.test/path"
    )
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)

    # Provider-to-engine handoff is an external-LLM egress boundary, so it
    # remains strict even when display/log redaction was explicitly disabled.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    context = received[0]
    assert prefix_secret not in context
    assert query_secret not in context
    assert userinfo_value not in context
    assert fragment_secret not in context
    assert relative_secret not in context
    assert encoded_key_secret not in context
    assert hyphen_client_secret not in context
    assert hyphen_access_secret not in context
    assert hyphen_api_secret not in context
    assert encoded_hyphen_secret not in context
    assert network_userinfo_secret not in context
    assert "access_token=***" in context
    assert "https://user:***@example.test/private" in context
    assert "https://x.test/#access_token=***&view=public" in context
    assert "/resume?token=***&view=public" in context
    assert "client%5Fsecret=***&view=public" in context
    assert "client-secret=***&view=public" in context
    assert "Access-Token=***&view=public" in context
    assert "api-key=***&view=public" in context
    assert "client%2Dsecret=***&view=public" in context
    assert "//user:***@x.test/path" in context


def test_provider_context_exact_masks_active_short_secret_before_engine(monkeypatch):
    """Arbitrary provider text crosses the engine handoff only after masking."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    secret = "r8$!"
    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"ARBITRARY_MEMORY_PROVIDER_SECRET": secret},
    )
    manager = MagicMock()
    manager.on_pre_compress.return_value = f"provider insight contains {secret}"
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert received == ["provider insight contains ***"]


@pytest.mark.parametrize(
    "alternate_spelling",
    ALTERNATE_JSON_SPELLINGS,
)
def test_provider_context_masks_alternate_json_escapes_before_engine(
    monkeypatch, alternate_spelling
):
    """Memory-provider JSON spellings are decoded on the disposable handoff."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    secret = "S/Key"
    provider_context = f'{{"credential":"{alternate_spelling}"}}'
    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"ARBITRARY_MEMORY_PROVIDER_SECRET": secret},
    )
    manager = MagicMock()
    manager.on_pre_compress.return_value = provider_context
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    assert json.loads(received[0]) == {"credential": "***"}
    assert secret not in received[0]
    assert alternate_spelling not in received[0]
    assert manager.on_pre_compress.return_value == provider_context


@pytest.mark.parametrize(
    "fragment_kind",
    (
        "unterminated",
        "invalid_escape",
        "quote_parity",
        "unquoted",
        "single_quoted",
    ),
)
def test_provider_context_masks_malformed_alternate_json_fragments_before_engine(
    monkeypatch, fragment_kind
):
    """Memory handoff tolerates incomplete and invalid quoted JSON fragments."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    secret = "S/Key"
    alternate_spelling = r"\u0053\/K\u0065y"
    if fragment_kind == "unterminated":
        provider_context = f'{{"credential":"{alternate_spelling}'
    elif fragment_kind == "invalid_escape":
        provider_context = f'{{"credential":"\\q{alternate_spelling}"}}'
    elif fragment_kind == "quote_parity":
        provider_context = (
            f'{{"broken":"prefix "credential":"{alternate_spelling}"}}'
        )
    elif fragment_kind == "unquoted":
        provider_context = f"{{credential:{alternate_spelling}}}"
    else:
        provider_context = f"{{'credential':'{alternate_spelling}'}}"
    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MALFORMED_MEMORY_PROVIDER_SECRET": secret},
    )
    manager = MagicMock()
    manager.on_pre_compress.return_value = provider_context
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    assert "***" in received[0]
    assert alternate_spelling not in received[0]
    assert manager.on_pre_compress.return_value == provider_context


def test_provider_context_is_bounded_before_plugin_engine():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "HEAD-SENTINEL" + "x" * 8_000 + "TAIL-SENTINEL"
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    context = received[0]
    assert len(context) <= 6_000
    assert context.startswith("HEAD-SENTINEL")
    assert context.endswith("TAIL-SENTINEL")
    assert "[memory provider context truncated]" in context


def test_internal_engine_type_error_propagates_after_one_call():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-typeerror"
    calls = []

    class BrokenEngine:
        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            calls.append(memory_context)
            raise TypeError("engine implementation bug")

    engine = BrokenEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)

    with pytest.raises(TypeError, match="engine implementation bug"):
        agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert calls == ["Checkpoint id: ctx-typeerror"]
