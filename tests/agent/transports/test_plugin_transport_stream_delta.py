"""A plugin dialect's streamed tool call reaches the assembler (#53054).

``register_transport(api_mode, cls)`` lets a provider plugin speak its own wire dialect, and the
previous fix made its ``api_mode`` survive every gate. What still did not survive was the CALL:
the streaming assembler reads ``delta.tool_calls`` only, so a dialect that streams on the legacy
OpenAI ``delta.function_call`` pair (one call per chunk, carrying ``name``/``arguments``/``id``)
had its call discarded — the turn ended as "empty response" with the tokens already spent.

Measured live against such a provider: ``in=178 out=105`` on the wire folded to
``response_len=7``, ``tool_turns=0``, and three retries of the same silent loss.

``ProviderTransport.normalize_stream_delta`` is the seam. Both tests drive the REAL assembler
(``_StreamingCall._call_chat_completions`` — the actual chunk loop) through a real plugin
transport registered from an isolated HERMES_HOME, so they fail if either half of the contract
regresses: the transport must translate, and the assembler must ask.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

# A transport that follows the legacy dialect: the call arrives on ``delta.function_call``,
# so ``normalize_stream_delta`` promotes it to the indexed ``tool_calls`` shape the assembler
# reads. Arguments arrive as a DICT (the SDK parses them) and must be re-serialized to the
# string the assembler feeds into its argument accumulator.
_LEGACY_TRANSPORT = '''\
from types import SimpleNamespace

from agent.transports import register_transport
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import register_provider
from providers.base import ProviderProfile


def _as_tool_calls(delta):
    call = getattr(delta, "function_call", None)
    if call is None:
        return None
    args = getattr(call, "arguments", "")
    if not isinstance(args, str):
        import json
        args = json.dumps(args or {}, ensure_ascii=False)
    # The dialect sends the call once, whole: the id must stay stable across the
    # chunks that carry it, or the accumulator files each fragment as a new call
    # (Ollama-style index-0 reuse is keyed on the id).
    call_id = getattr(call, "id", None) or "call_legacy"
    return [SimpleNamespace(index=0, id=call_id,
                            function=SimpleNamespace(name=getattr(call, "name", ""), arguments=args))]


class LegacyTransport(ChatCompletionsTransport):
    api_mode = "__MODE__"

    def normalize_stream_delta(self, delta):
        if getattr(delta, "tool_calls", None):
            return delta
        calls = _as_tool_calls(delta)
        return delta if calls is None else SimpleNamespace(tool_calls=calls)

    def normalize_message_tool_calls(self, message):
        """Same dialect, non-streaming: promote ``message.function_call``."""
        if getattr(message, "tool_calls", None):
            return message
        calls = _as_tool_calls(message)
        if not calls:
            return message
        return SimpleNamespace(**{**vars(message), "tool_calls": calls})


register_transport("__MODE__", LegacyTransport)
register_provider(ProviderProfile(name="__NAME__", auth_type="api_key", env_vars=("__ENV__",),
    base_url="https://relay.example.test/v1", api_mode="__MODE__", fallback_models=("example-model",)))
'''


@pytest.fixture
def install_legacy_plugin(tmp_path, monkeypatch):
    """Install a real model-provider plugin whose transport speaks the legacy stream dialect."""
    name, mode = "example-legacy", "example_legacy"
    env = f"{name.upper().replace('-', '_')}_API_KEY"
    plugin_dir = tmp_path / "hermes" / "plugins" / "model-providers" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: legacy stream fixture\n",
        encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        _LEGACY_TRANSPORT.replace("__ENV__", env).replace("__NAME__", name).replace("__MODE__", mode),
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv(env, "sk-fixture")

    import providers as _pkg
    _pkg._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]

    # Load the plugin the way production does, so its ``register_transport`` actually runs
    # under this HERMES_HOME (the profile list alone is not enough: the transport registry is
    # what the seam is keyed on).
    _pkg.list_providers()

    yield name, mode

    from agent.transports import _REGISTRY
    _pkg._REGISTRY.pop(name, None)
    _REGISTRY.pop(mode, None)
    for alias, canonical in list(_pkg._ALIASES.items()):
        if canonical == name:
            _pkg._ALIASES.pop(alias, None)
    _pkg._PROVIDER_LIST_CACHE = None


def _legacy_delta(name: str, args: dict, call_id: str = "call_1"):
    """One chunk as the SDK hands it to the assembler for the legacy dialect."""
    return SimpleNamespace(
        content=None,
        tool_calls=None,
        function_call=SimpleNamespace(name=name, arguments=args, id=call_id),
    )


def test_legacy_stream_delta_is_promoted_to_tool_calls(install_legacy_plugin):
    """The transport's own delta shape is translated, and the default seam stays an identity."""
    from agent.transports import get_transport
    from agent.transports.base import ProviderTransport

    _name, mode = install_legacy_plugin
    transport = get_transport(mode)

    raw = _legacy_delta("example_tool", {"query": "x"})
    normalized = transport.normalize_stream_delta(raw)

    # Contract: the assembler reads ``tool_calls``; a legacy delta must expose one.
    calls = getattr(normalized, "tool_calls", None)
    assert calls and len(calls) == 1, "legacy delta was not promoted to tool_calls"
    call = calls[0]
    assert call.function.name == "example_tool"
    # Arguments must reach the accumulator as a string — it concatenates, never parses.
    assert isinstance(call.function.arguments, str), "arguments must be serialized to str"
    assert '"query"' in call.function.arguments and "x" in call.function.arguments

    # A delta that already speaks the modern shape is passed through untouched.
    modern = SimpleNamespace(tool_calls=[call], function_call=None, content=None)
    assert transport.normalize_stream_delta(modern) is modern

    # The default seam is an identity: an OpenAI-shaped provider pays nothing and is not mutated.
    default = ProviderTransport.__dict__["normalize_stream_delta"]
    plain = SimpleNamespace(tool_calls=None, function_call=None, content="hi")
    assert default(object(), plain) is plain


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(
        id="chatcmpl-fixture", model="m",
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _drive_real_assembler(mode: str, chunks):
    """Run the REAL streaming chunk loop over ``chunks``; return the assembled message.

    The agent is built with every subsystem that would reach the network or the user's config
    switched off, and its transport is pinned to the plugin's own — which is what the assembler
    asks for through ``_get_transport()``.
    """
    import run_agent
    from agent import chat_completion_helpers as helpers
    from agent import relay_llm
    from agent.transports import get_transport

    agent = run_agent.AIAgent(
        api_key="sk-fixture", base_url="http://127.0.0.1:1/v1", model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=1,
    )
    agent._get_transport = lambda api_mode=None: get_transport(mode)

    call = helpers._StreamingCall(
        agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, None)

    class _Stream:
        final_response = None

        def __iter__(self):
            return iter(chunks)

        def close(self):
            pass

    # Only the provider stream is faked; the chunk loop, the accumulator and the dispatch
    # under test are the real ones.
    original = relay_llm.stream
    relay_llm.stream = lambda *a, **kw: _Stream()
    try:
        attempt_id = call._start_stream_attempt()
        response = call._call_chat_completions(attempt_id)
    finally:
        relay_llm.stream = original
    return response.choices[0].message


def test_assembler_asks_the_transport_for_a_legacy_delta(install_legacy_plugin):
    """Without the transport hook the call is dropped; with it the call is assembled.

    Drives the real ``_call_chat_completions`` chunk loop over a split legacy stream, so the
    assertion is about the assembler's behaviour: remove the ``normalize_stream_delta`` dispatch
    from ``chat_completion_helpers`` and no transport is ever asked, ``tool_calls`` stays
    ``None`` and this test fails.
    """
    import json

    _name, mode = install_legacy_plugin

    message = _drive_real_assembler(mode, [
        # The dialect streams the whole call once, on the legacy field pair, with
        # ``arguments`` already decoded as a dict (what the real plugin observes).
        _chunk(SimpleNamespace(role="assistant", tool_calls=None, content=None,
                               function_call=SimpleNamespace(name="example_tool",
                                                             arguments={"path": "a"},
                                                             id="call_1"))),
        _chunk(SimpleNamespace(tool_calls=None, content=None), finish_reason="tool_calls"),
    ])

    assert message.tool_calls, "the legacy call was discarded by the real assembler"
    call = message.tool_calls[0]
    assert call.function.name == "example_tool"
    # The dict must have reached the accumulator as a JSON string it can join and parse.
    assert isinstance(call.function.arguments, str)
    assert json.loads(call.function.arguments) == {"path": "a"}


def test_non_streaming_response_keeps_the_legacy_call(install_legacy_plugin):
    """The non-streaming path reads the same dialect, so it must ask the same seam.

    Regression: only the streaming assembler got the hook, so a legacy provider that answered
    without streaming (or answered with a final response for ``stream=True``, which flips
    ``_disable_streaming``) still normalized to ``tool_calls=None``.
    """
    _name, mode = install_legacy_plugin
    from agent.transports import get_transport

    transport = get_transport(mode)
    message = SimpleNamespace(role="assistant", content=None, tool_calls=None,
                              function_call=SimpleNamespace(name="example_tool",
                                                            arguments={"path": "a"},
                                                            id="call_1"))
    response = SimpleNamespace(
        id="chatcmpl-fixture", model="m", usage=None,
        choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")])

    normalized = transport.normalize_response(response)

    assert normalized.tool_calls, "the non-streaming legacy call was discarded"
    call = normalized.tool_calls[0]
    assert call.name == "example_tool"
    # The normalized ToolCall carries arguments as the JSON string the reader models.
    import json as _json
    assert _json.loads(call.arguments) == {"path": "a"}
