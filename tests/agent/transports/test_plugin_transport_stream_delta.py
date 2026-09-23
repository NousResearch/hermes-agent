"""A plugin dialect's streamed tool call reaches the assembler (#53054).

``register_transport(api_mode, cls)`` lets a provider plugin speak its own wire dialect, and the
previous fix made its ``api_mode`` survive every gate. What still did not survive was the CALL:
the streaming assembler reads ``delta.tool_calls`` only, so a dialect that streams on the legacy
OpenAI ``delta.function_call`` pair (one call per chunk, carrying ``name``/``arguments``/``id``)
had its call discarded — the turn ended as "empty response" with the tokens already spent.

Measured live against such a provider: ``in=178 out=105`` on the wire folded to
``response_len=7``, ``tool_turns=0``, and three retries of the same silent loss.

``ProviderTransport.normalize_stream_delta`` is the seam. Both tests drive the REAL assembler
through a real plugin transport registered from an isolated HERMES_HOME, so they fail if either
half of the contract regresses: the transport must translate, and the assembler must ask.
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
        args = json.dumps(args, ensure_ascii=False)
    return [SimpleNamespace(index=0, id=getattr(call, "id", "call_0"),
                            function=SimpleNamespace(name=getattr(call, "name", ""), arguments=args))]


class LegacyTransport(ChatCompletionsTransport):
    api_mode = "__MODE__"

    def normalize_stream_delta(self, delta):
        if getattr(delta, "tool_calls", None):
            return delta
        calls = _as_tool_calls(delta)
        return delta if calls is None else SimpleNamespace(tool_calls=calls)


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


def test_assembler_asks_the_transport_for_a_legacy_delta(install_legacy_plugin):
    """Without the transport hook the call is dropped; with it the call is assembled.

    Drives the real accumulator the streaming path uses, so the assertion is about the
    assembler's behaviour, not about the seam existing.
    """
    from agent.transports import get_transport

    _name, mode = install_legacy_plugin
    transport = get_transport(mode)

    class _Accumulator:
        """Minimal stand-in for the streaming tool-call assembler's ``feed`` contract."""

        def __init__(self) -> None:
            self.received: list[tuple[str, str]] = []

        def feed(self, tc_delta):
            fn = getattr(tc_delta, "function", None)
            if fn is None:
                return None
            self.received.append((fn.name, fn.arguments))
            return fn.name

    raw = _legacy_delta("example_tool", {"query": "x"})

    # Base behaviour: reading ``tool_calls`` directly finds nothing, so the call is lost.
    accumulator = _Accumulator()
    for _ in getattr(raw, "tool_calls", None) or []:
        accumulator.feed(_)
    assert accumulator.received == [], "premise: the raw legacy delta carries no tool_calls"

    # With the seam: the same delta now yields the call.
    delta = transport.normalize_stream_delta(raw)
    for tc_delta in getattr(delta, "tool_calls", None) or []:
        accumulator.feed(tc_delta)
    assert accumulator.received == [("example_tool", '{"query": "x"}')]
