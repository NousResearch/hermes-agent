"""Behavior contracts for Anthropic-native web search and fetch."""

import json
import logging
import textwrap
from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import (
    _is_third_party_anthropic_endpoint,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
)
from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.codex import ResponsesApiTransport


def _tool(name: str, server_spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
            "_hermes_server_tool": {
                "api_mode": "anthropic_messages",
                "definition": server_spec,
            },
        },
    }


def test_native_endpoint_replaces_web_functions_with_server_tools():
    tools = [
        _tool("web_search", {
            "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
        }),
        _tool("web_extract", {
            "type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5,
        }),
    ]

    converted = convert_tools_to_anthropic(tools, base_url="https://api.anthropic.com")

    assert converted == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5},
    ]


def test_compatible_third_party_endpoint_omits_server_only_tools():
    tools = [_tool("web_search", {
        "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
    })]

    converted = convert_tools_to_anthropic(
        tools, base_url="https://api.minimax.io/anthropic"
    )

    assert converted == []


def test_anthropic_native_endpoint_detection_uses_hostname_boundaries():
    assert not _is_third_party_anthropic_endpoint("https://api.anthropic.com/v1")
    assert _is_third_party_anthropic_endpoint(
        "https://api.anthropic.com.example/v1"
    )
    assert _is_third_party_anthropic_endpoint(
        "https://proxy.example/v1/api.anthropic.com"
    )


@pytest.mark.parametrize(
    "transport",
    [ChatCompletionsTransport(), BedrockTransport(), ResponsesApiTransport()],
)
def test_server_only_tools_are_omitted_from_foreign_transports(transport):
    ordinary = {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    server_only = _tool("web_search", {
        "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
    })

    projected = transport.project_tools([ordinary, server_only])
    converted = transport.convert_tools([ordinary, server_only])

    assert projected == [ordinary]
    assert "_hermes_server_tool" not in str(projected)
    assert "web_search" not in str(converted)
    assert "_hermes_server_tool" not in str(converted)


def test_chat_completions_kwargs_never_leak_server_tool_metadata():
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="example/model",
        messages=[{"role": "user", "content": "hello"}],
        tools=[_tool("web_search", {
            "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
        })],
    )

    assert "tools" not in kwargs
    assert "_hermes_server_tool" not in str(kwargs)


def test_server_blocks_are_captured_and_replayed_in_original_order():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="server_tool_use",
                id="srvtoolu_1",
                name="web_search",
                input={"query": "Hermes Agent"},
            ),
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id="srvtoolu_1",
                content=[{
                    "type": "web_search_result",
                    "title": "Hermes",
                    "url": "https://example.com/hermes",
                }],
            ),
            SimpleNamespace(
                type="text",
                text="Hermes is an agent.",
                citations=[{
                    "type": "web_search_result_location",
                    "title": "Hermes",
                    "url": "https://example.com/hermes",
                }],
            ),
        ],
    )

    normalized = AnthropicTransport().normalize_response(response)
    stored = {
        "role": "assistant",
        "content": normalized.content,
        "anthropic_content_blocks": normalized.anthropic_content_blocks,
    }
    _, replayed = convert_messages_to_anthropic([
        {"role": "user", "content": "Find Hermes"},
        stored,
    ])

    blocks = replayed[-1]["content"]
    assert [block["type"] for block in blocks] == [
        "server_tool_use", "web_search_tool_result", "text",
    ]
    assert blocks[-1]["citations"][0]["url"] == "https://example.com/hermes"
    assert normalized.content.endswith(
        "Sources:\n- Hermes: https://example.com/hermes"
    )


def test_web_fetch_blocks_use_the_same_round_trip_channel():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="server_tool_use",
                id="srvtoolu_fetch",
                name="web_fetch",
                input={"url": "https://example.com"},
            ),
            SimpleNamespace(
                type="web_fetch_tool_result",
                tool_use_id="srvtoolu_fetch",
                content={
                    "type": "web_fetch_result",
                    "url": "https://example.com",
                    "content": {"type": "document", "source": {"type": "text", "data": "ok"}},
                },
            ),
            SimpleNamespace(type="text", text="Fetched."),
        ],
    )

    normalized = AnthropicTransport().normalize_response(response)

    assert [block["type"] for block in normalized.anthropic_content_blocks] == [
        "server_tool_use", "web_fetch_tool_result", "text",
    ]
    assert normalized.content.endswith("Sources:\n- https://example.com")


def test_pause_turn_remains_distinct_for_the_conversation_loop():
    transport = AnthropicTransport()
    response = SimpleNamespace(
        stop_reason="pause_turn",
        content=[SimpleNamespace(
            type="server_tool_use",
            id="srvtoolu_1",
            name="web_search",
            input={"query": "long research"},
        )],
    )

    normalized = transport.normalize_response(response)

    assert transport.map_finish_reason("pause_turn") == "pause_turn"
    assert normalized.finish_reason == "pause_turn"
    assert normalized.anthropic_content_blocks[0]["id"] == "srvtoolu_1"


def test_model_key_does_not_implicitly_replace_the_web_backend(monkeypatch):
    from agent import web_search_registry
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
    monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [])
    keys = {"ANTHROPIC_API_KEY": "sk-ant-api-test"}
    monkeypatch.setattr(web_tools, "_has_env", lambda name: bool(keys.get(name)))
    # Disable the keyless free tier. With it on, a ring provider reports ready
    # on zero credentials, so ``check_web_api_key()`` returns True no matter
    # what the Anthropic key does — masking exactly what this test asserts
    # (see test_web_keyless_fallback.py for the tier's own coverage). Patching
    # the registry module is enough: every reader late-imports the symbol from
    # it inside the function body, so the rebound name is what runs. Turning
    # the tier off also removes the per-process ring rotation, which would
    # otherwise make ``_get_backend()`` return a different vendor per run.
    monkeypatch.setattr(web_search_registry, "_keyless_tier_enabled", lambda: False)

    assert web_tools._get_backend() != "anthropic"
    assert not web_tools.check_web_api_key()

    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"backend": "anthropic"}
    )
    assert web_tools._get_backend() == "anthropic"
    assert web_tools.check_web_api_key()


def test_dynamic_markers_follow_per_capability_selection(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "anthropic")
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "brave-free")

    search = web_tools._anthropic_web_search_schema_overrides()
    extract = web_tools._anthropic_web_fetch_schema_overrides()

    binding = search["_hermes_server_tool"]
    assert binding["api_mode"] == "anthropic_messages"
    assert binding["definition"]["name"] == "web_search"
    assert extract == {}


def test_tools_picker_exposes_anthropic_without_requesting_a_second_key():
    from hermes_cli.tools_config import TOOL_CATEGORIES

    providers = TOOL_CATEGORIES["web"]["providers"]
    anthropic = next(p for p in providers if p.get("web_backend") == "anthropic")

    assert anthropic["env_vars"] == []
    assert "already configured" in anthropic["tag"]


# ---------------------------------------------------------------------------
# The selected backend must match what the active model can actually execute.
#
# These drive the real config file under the per-test HERMES_HOME rather than
# patching the loaders, because the bug they cover was a resolution bug: the
# backend was "available" on evidence (an API key) that says nothing about
# whether the model can run the tool.
# ---------------------------------------------------------------------------

def _write_hermes_config(body: str) -> None:
    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture()
def anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")


def test_anthropic_backend_is_unavailable_on_a_non_anthropic_model(anthropic_key):
    """A model credential must not advertise web access the model cannot use.

    The tools only execute inside Anthropic's Messages API, so on any other
    transport every request drops them.  Reporting the backend as available
    would light the whole ``web`` toolset up — banner, ``hermes tools``,
    ``valid_tool_names`` — for an agent with no web capability at all.
    """
    from tools import web_tools

    _write_hermes_config("""
        model:
          provider: openrouter
          base_url: https://openrouter.ai/api/v1
        web:
          backend: anthropic
    """)

    assert web_tools._is_backend_available("anthropic") is False
    assert web_tools.check_web_api_key() is False


def test_anthropic_backend_stays_available_on_anthropic_models(anthropic_key):
    from tools import web_tools

    _write_hermes_config("""
        model:
          provider: anthropic
        web:
          backend: anthropic
    """)

    assert web_tools._is_backend_available("anthropic") is True
    assert web_tools.check_web_api_key() is True


def test_compatible_third_party_endpoint_does_not_advertise_web_tools(anthropic_key):
    """A ``…/anthropic`` proxy speaks the protocol but does not host the tools."""
    from tools import web_tools

    _write_hermes_config("""
        model:
          provider: minimax
          base_url: https://api.minimax.io/anthropic
          api_mode: anthropic_messages
        web:
          backend: anthropic
    """)

    assert web_tools._is_backend_available("anthropic") is False
    assert web_tools.check_web_api_key() is False


def test_unclassifiable_model_config_keeps_web_available(anthropic_key):
    """Never strip a capability on a config this cannot read as non-Anthropic."""
    from tools import web_tools

    _write_hermes_config("""
        web:
          backend: anthropic
    """)

    assert web_tools._is_backend_available("anthropic") is True
    assert web_tools.check_web_api_key() is True


def test_unexecutable_selection_keeps_the_tool_shaped_so_it_can_explain(
    anthropic_key, monkeypatch
):
    """When another backend keeps ``web`` alive, the tool must stay callable.

    Attaching the server-only binding here would strip ``web_search`` from
    every request while ``hermes tools`` still lists it — the handler's own
    "cannot run locally" error is unreachable for a tool nobody can call.
    """
    from tools import web_tools

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    _write_hermes_config("""
        model:
          provider: openrouter
          base_url: https://openrouter.ai/api/v1
        web:
          backend: anthropic
          extract_backend: tavily
    """)

    assert web_tools.check_web_api_key() is True
    assert web_tools._get_search_backend() == "anthropic"
    assert web_tools._anthropic_web_search_schema_overrides() == {}
    assert web_tools._anthropic_web_fetch_schema_overrides() == {}
    assert "hermes tools" in web_tools.web_search_tool("anything")


def test_per_capability_anthropic_selection_hides_web_on_a_foreign_model(
    anthropic_key, monkeypatch
):
    """``web.backend`` is not the only way to select an unrunnable backend.

    ``web.search_backend``/``web.extract_backend`` are resolved strictly — the
    stored name is returned with no availability probe and no fallback — so a
    per-capability selection routes exactly like the shared key.  With BOTH
    capabilities pinned to ``anthropic`` on a transport that cannot carry the
    server-side tools, nothing can serve either one, and an unrelated web
    credential must not light the toolset up on their behalf.

    ``TAVILY_API_KEY`` is that unrelated credential: it is what makes this
    case return True if the readiness gate looks only at ``web.backend``.
    ``keyless_fallback: false`` keeps Tavily the sole witness, so a failure
    here can never be blamed on the free-tier ring.
    """
    from tools import web_tools

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    _write_hermes_config("""
        model:
          provider: openrouter
          base_url: https://openrouter.ai/api/v1
        web:
          search_backend: anthropic
          extract_backend: anthropic
          keyless_fallback: false
    """)

    assert web_tools._get_search_backend() == "anthropic"
    assert web_tools._get_extract_backend() == "anthropic"
    assert web_tools._is_backend_available("anthropic") is False
    assert web_tools._is_backend_available("tavily") is True
    assert web_tools.check_web_api_key() is False


def test_one_capability_stuck_on_anthropic_leaves_the_other_serving(
    anthropic_key, monkeypatch
):
    """Only a total loss of web hides the toolset; a half-broken split stays.

    Extract still routes to a backend that runs locally, so ``web`` must stay
    available — hiding it would take away a capability the agent really has.
    The stuck capability is not silently dropped either: ``web_search`` keeps
    its ordinary shape and answers with the typed ``tool_error`` that says why
    it cannot run and what to do about it.
    """
    from tools import web_tools

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    _write_hermes_config("""
        model:
          provider: openrouter
          base_url: https://openrouter.ai/api/v1
        web:
          search_backend: anthropic
          extract_backend: tavily
          keyless_fallback: false
    """)

    assert web_tools._get_search_backend() == "anthropic"
    assert web_tools._get_extract_backend() == "tavily"
    assert web_tools.check_web_api_key() is True

    payload = json.loads(web_tools.web_search_tool("anything"))
    assert "results" not in payload
    assert "cannot be executed by Hermes locally" in payload["error"]
    assert "hermes tools" in payload["error"]


def test_a_keyless_capability_still_counts_as_web(anthropic_key):
    """The half-broken split must not be widened to the keyless free tier.

    ``_is_backend_available`` is keyless-blind, so the credential-free half of
    this config looks unavailable to every probe in the readiness gate — yet
    the free-tier ring really does serve it.  A gate that hides ``web`` as
    soon as *either* capability lands on ``anthropic`` would take that away.

    The Anthropic key IS present here (``anthropic_key``), so the False below
    is about the transport, not a missing credential — which is exactly the
    situation a user lands in after selecting ``anthropic`` for search.

    Deliberately asserts no backend name: ``_keyless_preference()`` is seeded
    per process, so which ring vendor answers differs from run to run.
    """
    from tools import web_tools

    _write_hermes_config("""
        model:
          provider: openrouter
          base_url: https://openrouter.ai/api/v1
        web:
          search_backend: anthropic
          extract_backend: exa
    """)

    assert web_tools._is_backend_available("anthropic") is False
    assert web_tools._is_backend_available("exa") is False
    assert web_tools.check_web_api_key() is True


def test_dropping_a_server_only_tool_is_reported_once(caplog):
    """The projection must not remove a user-visible capability in silence.

    The tool name is unique to this test so the report's once-per-process
    suppression is exercised here rather than pre-consumed by another case.
    """
    tools = [_tool("projection_probe_tool", {
        "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
    })]

    with caplog.at_level(logging.WARNING, logger="agent.transports.base"):
        assert ChatCompletionsTransport().project_tools(tools) == []
        assert ChatCompletionsTransport().project_tools(tools) == []

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "projection_probe_tool" in message
    assert "anthropic_messages" in message
    assert "chat_completions" in message
    assert "hermes tools" in message


def test_third_party_endpoint_drop_is_reported_once(caplog):
    tools = [_tool("third_party_probe_tool", {
        "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
    })]

    with caplog.at_level(logging.WARNING, logger="agent.anthropic_adapter"):
        for _ in range(2):
            assert convert_tools_to_anthropic(
                tools, base_url="https://api.minimax.io/anthropic"
            ) == []

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "third_party_probe_tool" in message
    assert "api.minimax.io" in message
    assert "hermes tools" in message


def test_native_web_fetch_bounds_the_content_it_injects(anthropic_key):
    """A server-side fetch must carry a content ceiling of its own.

    The local ``web_extract`` path is bounded before a result reaches the model
    (auxiliary summariser, then ``max_result_size_chars`` on the registry
    entry).  The native fetch runs inside Anthropic's request, so neither guard
    ever sees it: without ``max_content_tokens`` one large page is injected
    whole and — because the block is preserved for replay — resent on every
    later turn of the session.
    """
    from tools import registry, web_tools

    _write_hermes_config("""
        model:
          provider: anthropic
        web:
          backend: anthropic
    """)

    definition = web_tools._anthropic_web_fetch_schema_overrides()
    definition = definition["_hermes_server_tool"]["definition"]

    cap = definition.get("max_content_tokens")
    assert isinstance(cap, int) and cap > 0, definition

    # Tie the ceiling to the local cap rather than freezing a number: the two
    # bound the same thing on two paths, and a reader who raises one should be
    # told to look at the other. ~4 chars/token, so the native ceiling stays
    # within an order of magnitude of the local one.
    local_chars = registry.registry.get_entry("web_extract").max_result_size_chars
    assert local_chars is not None
    assert local_chars / 40 <= cap <= local_chars, (cap, local_chars)
