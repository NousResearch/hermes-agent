"""Payload-level proof that ``--no-tools`` reaches the wire with zero tools.

``tests/hermes_cli/test_safe_mode.py`` only proves the *flag* survives
subcommand dispatch.  That is not evidence: a flag can be accepted, plumbed,
and then silently undone further down (it was — see the kanban re-entry
below).  These tests follow the real chain instead, hop by hop, and assert on
the dict that is handed to the provider SDK:

    argv
      → hermes_cli._parser.build_top_level_parser()
      → hermes_cli.main.cmd_chat            (kwargs["no_tools"])
      → cli.main                            (toolsets_list = [], HERMES_NO_TOOLS=1)
      → HermesCLI(toolsets=[])              (captured here)
      → AIAgent(enabled_toolsets=[])        (real model_tools.get_tool_definitions)
      → agent._build_api_kwargs()           (agent/chat_completion_helpers.build_api_kwargs)
      → <Transport>.build_kwargs()          (spied, real implementation runs)

**Wire representation of "no tools offered" differs per transport, and both
forms are asserted for exactly what they are:**

* ``ChatCompletionsTransport`` — for a *registered* provider (the profile
  path, ``_build_kwargs_from_profile``) the ``tools`` key is **omitted
  entirely**; ``tools: []`` is only emitted on the legacy/unknown-provider
  path.  Either way the request offers no callable tool.
* ``ResponsesApiTransport`` (codex) — emits an explicit ``tools: []`` and
  omits ``tool_choice`` / ``parallel_tool_calls``.  It must not omit the key,
  because ``tools=None`` crashes the OpenAI SDK before the request is sent
  (see ``tests/run_agent/test_codex_no_tools_nonetype.py``).

So the assertion used throughout is ``kwargs.get("tools", []) == []`` — "no
tools key, or an empty one" — plus a check that the transport was *handed*
``[]`` and never a populated list.

No network call is possible: ``run_agent.OpenAI`` is patched out and only
kwargs assembly is exercised.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest


_BOUNDARY_VARS = ("HERMES_NO_TOOLS", "HERMES_KANBAN_TASK", "HERMES_INTERACTIVE")

_MESSAGES = [
    {"role": "system", "content": "You are Hermes."},
    {"role": "user", "content": "hello"},
]


@pytest.fixture(autouse=True)
def _clean_boundary_env(monkeypatch):
    """No boundary/kanban env leaks in or out, and no memoized tool lists."""
    for var in _BOUNDARY_VARS:
        monkeypatch.delenv(var, raising=False)
    import model_tools

    model_tools._clear_tool_defs_cache()
    yield
    for var in _BOUNDARY_VARS:
        os.environ.pop(var, None)
    model_tools._clear_tool_defs_cache()


# ---------------------------------------------------------------------------
# Hop 1: argv → cmd_chat → cli.main → HermesCLI(toolsets=...)
# ---------------------------------------------------------------------------


class _StopAtAgentConstruction(Exception):
    """Raised by the HermesCLI stand-in once its kwargs are captured."""


def _run_cli_main_capturing_hermescli(monkeypatch, argv: list[str]) -> dict[str, Any]:
    """Drive the real argv → ``cmd_chat`` → ``cli.main`` path.

    Everything up to (and including) toolset resolution is the production
    implementation; only ``cli.HermesCLI`` is replaced, so the test can read
    the toolset list the CLI would really have built the agent with.
    """
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(argv)

    # Startup side effects that are irrelevant to tool resolution (and would
    # touch the network / the user's install). Mirrors test_safe_mode.py.
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)

    import cli as cli_mod

    captured: dict[str, Any] = {}

    class _CapturingHermesCLI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            captured["_constructed"] = True
            raise _StopAtAgentConstruction

    monkeypatch.setattr(cli_mod, "HermesCLI", _CapturingHermesCLI)

    try:
        main_mod.cmd_chat(args)
    except _StopAtAgentConstruction:
        pass
    return captured


def test_cli_resolves_no_tools_to_an_empty_toolset_list(monkeypatch):
    """``hermes chat --no-tools`` builds the agent with ``toolsets=[]``."""
    captured = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )

    assert captured["_constructed"] is True
    assert captured["toolsets"] == [], (
        "cli.main must hand HermesCLI an explicitly empty toolset list, "
        f"got {captured['toolsets']!r}"
    )
    assert captured["max_turns"] == 1
    # The capability boundary is published process-wide so downstream tool
    # resolution can tell "explicitly none" from "config resolved to none".
    assert os.environ.get("HERMES_NO_TOOLS") == "1"


def test_cli_without_no_tools_still_resolves_a_real_toolset_list(monkeypatch):
    """Inverse control: the boundary is opt-in, not the default."""
    captured = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "-q", "hello"]
    )

    assert captured["toolsets"] != []
    assert "HERMES_NO_TOOLS" not in os.environ


# ---------------------------------------------------------------------------
# Hop 2: toolsets=[] → AIAgent → transport.build_kwargs → wire payload
# ---------------------------------------------------------------------------


_TRANSPORTS = {
    # api_mode: transport module, class name, base_url, provider, model
    "chat_completions": {
        "module": "agent.transports.chat_completions",
        "cls": "ChatCompletionsTransport",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "model": None,
    },
    "codex_responses": {
        "module": "agent.transports.codex",
        "cls": "ResponsesApiTransport",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "model": None,
    },
    "anthropic_messages": {
        "module": "agent.transports.anthropic",
        "cls": "AnthropicTransport",
        "base_url": "https://api.anthropic.com",
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    "bedrock_converse": {
        "module": "agent.transports.bedrock",
        "cls": "BedrockTransport",
        "base_url": None,
        "provider": "bedrock",
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
    },
}

# Every wire spelling of "here is a tool surface", across all four transports:
# OpenAI chat/responses (`tools`), Bedrock Converse (`toolConfig`), plus the
# steering keys and the deprecated OpenAI function-calling pair.
_TOOL_WIRE_KEYS = (
    "tools",
    "toolConfig",
    "tool_choice",
    "toolChoice",
    "parallel_tool_calls",
    "functions",
    "function_call",
)


def _spy_on_transport(monkeypatch, api_mode: str) -> list[dict[str, Any]]:
    """Wrap the real ``build_kwargs`` of one transport and record every call.

    The production implementation still runs — this is a spy, not a stub, so
    the recorded payload is the genuine one.
    """
    import importlib

    spec = _TRANSPORTS[api_mode]
    cls = getattr(importlib.import_module(spec["module"]), spec["cls"])
    original = cls.build_kwargs
    calls: list[dict[str, Any]] = []

    def _spy(self, model, messages, tools=None, **params):
        result = original(self, model, messages, tools=tools, **params)
        calls.append({"tools_argument": tools, "payload": result})
        return result

    monkeypatch.setattr(cls, "build_kwargs", _spy)
    return calls


def _build_agent(api_mode: str, toolsets, request_overrides=None):
    """A real ``AIAgent`` using the real ``get_tool_definitions``.

    Only the OpenAI client class is patched out, so nothing can reach the
    network — exactly the pattern used by ``tests/run_agent/test_run_agent.py``.
    """
    from run_agent import AIAgent

    spec = _TRANSPORTS[api_mode]
    kwargs: dict[str, Any] = dict(
        api_key="test-key-1234567890",
        provider=spec["provider"],
        api_mode=api_mode,
        enabled_toolsets=toolsets,
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    if spec["base_url"]:
        kwargs["base_url"] = spec["base_url"]
    if spec["model"]:
        kwargs["model"] = spec["model"]
    if request_overrides is not None:
        kwargs["request_overrides"] = request_overrides
    with patch("run_agent.OpenAI"):
        return AIAgent(**kwargs)


def _wire_payload(
    monkeypatch, api_mode: str, toolsets, request_overrides=None
) -> dict[str, Any]:
    """Return the kwargs the provider SDK would be called with."""
    calls = _spy_on_transport(monkeypatch, api_mode)
    agent = _build_agent(api_mode, toolsets, request_overrides=request_overrides)
    assert agent.api_mode == api_mode, (
        f"agent resolved api_mode={agent.api_mode!r}; the {api_mode!r} transport "
        "would not be exercised and the assertion below would be vacuous"
    )
    assert agent.tools == [], f"agent tool snapshot must be empty, got {agent.tools!r}"

    payload = agent._build_api_kwargs(list(_MESSAGES))

    assert calls, f"{api_mode} transport build_kwargs was never reached"
    assert calls[-1]["tools_argument"] == [], (
        "the transport must be handed an explicitly empty tool list, got "
        f"{calls[-1]['tools_argument']!r}"
    )
    assert calls[-1]["payload"] is payload
    return payload


def _assert_offers_no_tools(payload: dict[str, Any], api_mode: str) -> None:
    """No callable tool on the wire — key absent, or present and empty."""
    assert payload.get("tools", []) == [], (
        f"{api_mode}: outgoing request must offer no tools, got "
        f"tools={payload.get('tools')!r}"
    )
    for key in _TOOL_WIRE_KEYS:
        if key == "tools":
            continue
        assert key not in payload, (
            f"{api_mode}: {key!r} must not appear on a no-tools request, got "
            f"{payload[key]!r}"
        )


@pytest.mark.parametrize("api_mode", sorted(_TRANSPORTS))
def test_no_tools_yields_empty_wire_tool_list(monkeypatch, api_mode):
    """End to end: ``--no-tools`` → ``tools == []`` at the model-request boundary."""
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]
    assert toolsets == []

    payload = _wire_payload(monkeypatch, api_mode, toolsets)
    _assert_offers_no_tools(payload, api_mode)


@pytest.mark.parametrize("api_mode", sorted(_TRANSPORTS))
def test_kanban_env_cannot_reenter_tools(monkeypatch, api_mode):
    """Negative control for the kanban re-entry hole.

    ``model_tools._compute_tool_definitions`` force-appends the ``kanban``
    toolset whenever ``HERMES_KANBAN_TASK`` is set — even to an
    ``enabled_toolsets == []``.  A Captain subprocess that inherits that env
    var would therefore have been handed the kanban tool surface despite
    ``--no-tools``.  With the boundary published, the append is refused.
    """
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]
    monkeypatch.setenv("HERMES_KANBAN_TASK", "ntc-g2-proof")

    payload = _wire_payload(monkeypatch, api_mode, toolsets)
    _assert_offers_no_tools(payload, api_mode)


def test_kanban_reentry_is_otherwise_live(monkeypatch):
    """Guard against a vacuous negative control.

    If the kanban toolset were simply unregistered in this process, the test
    above would pass for the wrong reason.  Here the *same* empty toolset list
    is resolved with ``HERMES_KANBAN_TASK`` set but the ``--no-tools`` boundary
    absent: the kanban tools must come back.  If this ever stops holding, the
    negative control has lost its teeth and needs rewriting.
    """
    import model_tools
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "ntc-g2-proof")
    model_tools._clear_tool_defs_cache()

    reentered = model_tools.get_tool_definitions(enabled_toolsets=[], quiet_mode=True)
    names = {t["function"]["name"] for t in reentered}
    if not any(n.startswith("kanban") for n in names):
        pytest.skip("kanban toolset not registered in this environment")

    monkeypatch.setenv("HERMES_NO_TOOLS", "1")
    model_tools._clear_tool_defs_cache()

    assert model_tools.get_tool_definitions(enabled_toolsets=[], quiet_mode=True) == []


# ---------------------------------------------------------------------------
# request_overrides: the config-file bypass
# ---------------------------------------------------------------------------
#
# ``request_overrides`` is a raw user-config dict merged into the outgoing
# kwargs *after* the tools decision, at three sites:
#   chat_completions.py  legacy path   — api_kwargs.update(overrides)
#   chat_completions.py  profile path  — per-key assignment
#   codex.py                           — kwargs.update(request_overrides)
# Under the boundary all tool-offering keys must be stripped before the merge.


_HOSTILE_OVERRIDES = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    "tool_choice": "required",
    "parallel_tool_calls": True,
    "functions": [{"name": "terminal", "parameters": {}}],
    "function_call": "auto",
    # A benign key alongside them: sanitization must be surgical, not a
    # blanket drop of the whole override dict.
    "service_tier": "priority",
}


def test_request_overrides_cannot_reintroduce_tools_codex(monkeypatch):
    """codex.py: ``kwargs.update(request_overrides)`` merge site."""
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]

    payload = _wire_payload(
        monkeypatch,
        "codex_responses",
        toolsets,
        request_overrides=dict(_HOSTILE_OVERRIDES),
    )
    _assert_offers_no_tools(payload, "codex_responses")
    assert payload["service_tier"] == "priority", (
        "sanitization must drop only the tool keys, not the whole override dict"
    )


def test_request_overrides_cannot_reintroduce_tools_chat_profile_path(monkeypatch):
    """chat_completions.py ``_build_kwargs_from_profile`` merge site.

    Reached for any *registered* provider — the common case (openrouter here).
    """
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]

    import providers

    assert providers.get_provider_profile("openrouter") is not None, (
        "this test must exercise the profile path; openrouter has no profile"
    )

    payload = _wire_payload(
        monkeypatch,
        "chat_completions",
        toolsets,
        request_overrides=dict(_HOSTILE_OVERRIDES),
    )
    _assert_offers_no_tools(payload, "chat_completions")
    assert payload["service_tier"] == "priority"


def test_request_overrides_cannot_reintroduce_tools_chat_legacy_path(monkeypatch):
    """chat_completions.py legacy ``api_kwargs.update(overrides)`` merge site.

    Reached only when ``get_provider_profile()`` returns None — an entirely
    unregistered provider.  Simulated by patching the lookup, which is what a
    user-defined ``providers:`` entry in config.yaml produces.
    """
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]

    import providers

    monkeypatch.setattr(providers, "get_provider_profile", lambda *_a, **_k: None)

    payload = _wire_payload(
        monkeypatch,
        "chat_completions",
        toolsets,
        request_overrides=dict(_HOSTILE_OVERRIDES),
    )
    _assert_offers_no_tools(payload, "chat_completions")
    assert payload["service_tier"] == "priority"


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "bedrock_converse"])
def test_request_overrides_never_reach_anthropic_or_bedrock(monkeypatch, api_mode):
    """These two transports are not handed ``request_overrides`` at all.

    ``build_api_kwargs`` forwards overrides only on the chat-completions and
    codex paths.  Pinning that here means a future refactor that starts
    forwarding them has to come back and add sanitization.
    """
    toolsets = _run_cli_main_capturing_hermescli(
        monkeypatch, ["chat", "-Q", "--max-turns", "1", "--no-tools", "-q", "hello"]
    )["toolsets"]

    payload = _wire_payload(
        monkeypatch, api_mode, toolsets, request_overrides=dict(_HOSTILE_OVERRIDES)
    )
    _assert_offers_no_tools(payload, api_mode)
    assert "service_tier" not in payload


def test_sanitizer_is_a_noop_without_the_boundary():
    """Normal runs are untouched: overrides pass through byte-for-byte."""
    from agent.transports.base import ProviderTransport

    overrides = dict(_HOSTILE_OVERRIDES)
    assert ProviderTransport.sanitize_request_overrides(overrides) is overrides


def test_sanitizer_logs_every_dropped_key(monkeypatch, caplog):
    """A stripped key must be visible to the operator, named, at WARNING."""
    import logging

    from agent.transports.base import ProviderTransport

    monkeypatch.setenv("HERMES_NO_TOOLS", "1")
    with caplog.at_level(logging.WARNING, logger="agent.transports.base"):
        cleaned = ProviderTransport.sanitize_request_overrides(
            dict(_HOSTILE_OVERRIDES), context="unit test"
        )

    assert cleaned == {"service_tier": "priority"}
    logged = caplog.text
    for key in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"):
        assert repr(key) in logged, f"dropped key {key!r} was not logged"
    assert "unit test" in logged


# ---------------------------------------------------------------------------
# Route parity: -z/--oneshot and --tui must not accept-and-ignore the flag
# ---------------------------------------------------------------------------


def test_oneshot_honors_no_tools(monkeypatch):
    """``hermes -z --no-tools`` builds the agent with ``toolsets=[]``.

    Before this fix ``run_oneshot`` never saw ``no_tools`` and fell back to the
    configured platform toolsets — the flag was accepted and ignored.
    """
    import hermes_cli.main as main_mod

    captured: dict[str, Any] = {}

    def _fake_run_oneshot(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return 0

    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod, "run_oneshot", _fake_run_oneshot)
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", lambda rc: (_ for _ in ()).throw(SystemExit(rc)))

    with pytest.raises(SystemExit) as exc:
        main_mod._run_and_exit_oneshot("hello", no_tools=True)

    assert exc.value.code == 0
    assert captured["no_tools"] is True


def test_run_oneshot_publishes_boundary_and_forwards_no_tools(monkeypatch):
    """``run_oneshot`` publishes the boundary and hands it to the agent builder."""
    import hermes_cli.oneshot as oneshot_mod

    captured: dict[str, Any] = {}

    def _fake_run_agent(prompt, **kwargs):
        captured.update(kwargs)
        assert os.environ.get("HERMES_NO_TOOLS") == "1", (
            "the boundary must already be published before the agent is built"
        )
        return "ok", {}

    monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent)

    oneshot_mod.run_oneshot("hello", no_tools=True)

    assert captured["no_tools"] is True
    assert captured["use_config_toolsets"] is False
    assert os.environ.get("HERMES_NO_TOOLS") == "1"


def test_oneshot_agent_builder_resolves_empty_toolsets(monkeypatch):
    """The real ``_run_agent`` toolset resolution under ``no_tools=True``.

    This is the hop the original bug lived in: ``use_config_toolsets=False``
    alone leaves ``toolsets_list`` at ``None``, which means *every* toolset.
    """
    import hermes_cli.oneshot as oneshot_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import run_agent as run_agent_mod

    captured: dict[str, Any] = {}

    def _fake_agent(**kwargs):
        captured.update(kwargs)
        raise _StopAtAgentConstruction

    # The hermetic conftest strips every credential env var, so provider
    # resolution would raise before the toolset decision is reached.
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kw: {
            "api_key": "test-key-1234567890",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "requested_provider": "openrouter",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(run_agent_mod, "AIAgent", _fake_agent)

    with pytest.raises(_StopAtAgentConstruction):
        oneshot_mod._run_agent("hello", no_tools=True, use_config_toolsets=False)

    assert captured["enabled_toolsets"] == []


def test_run_oneshot_rejects_no_tools_with_toolsets(monkeypatch, capsys):
    """Fail closed on the contradictory combination, same as ``chat``."""
    import hermes_cli.oneshot as oneshot_mod

    def _never(prompt, **kwargs):
        raise AssertionError("no agent may be built on a rejected flag combo")

    monkeypatch.setattr(oneshot_mod, "_run_agent", _never)

    rc = oneshot_mod.run_oneshot("hello", no_tools=True, toolsets="web")

    assert rc == 2
    assert "--no-tools cannot be combined with --toolsets" in capsys.readouterr().err
    assert "HERMES_NO_TOOLS" not in os.environ


def test_tui_with_no_tools_fails_closed(monkeypatch, capsys):
    """``--tui --no-tools`` exits 2 instead of silently running with tools.

    The TUI's Node gateway resolves its own toolsets and has no channel for an
    explicit empty surface, so the combination is refused at the point the
    interface is chosen — never accepted and ignored.
    """
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--tui", "--no-tools", "-q", "hello"])

    launched: list[object] = []
    monkeypatch.setattr(main_mod, "_launch_tui", lambda *a, **k: launched.append(k))

    with pytest.raises(SystemExit) as exc:
        main_mod._resolve_use_tui(args)

    assert exc.value.code == 2
    assert launched == [], "the TUI must not be launched under --no-tools"
    assert "--no-tools is not supported by the TUI" in capsys.readouterr().err


def test_tui_without_no_tools_still_resolves(monkeypatch):
    """The guard is scoped to the flag; ordinary ``--tui`` is unaffected."""
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)

    assert main_mod._resolve_use_tui(parser.parse_args(["chat", "--tui"])) is True
    assert (
        main_mod._resolve_use_tui(parser.parse_args(["chat", "--cli", "--no-tools"]))
        is False
    )


# ---------------------------------------------------------------------------
# Fail-closed: --no-tools is mutually exclusive with --toolsets
# ---------------------------------------------------------------------------


def test_no_tools_with_toolsets_raises_in_cli_main():
    """``cli.main`` refuses the contradictory combination outright."""
    import cli as cli_mod

    with pytest.raises(ValueError, match="--no-tools cannot be combined with --toolsets"):
        cli_mod.main(query="hello", no_tools=True, toolsets="web")

    # The boundary env var must not be published by a rejected invocation.
    assert "HERMES_NO_TOOLS" not in os.environ


def test_no_tools_with_toolsets_fails_closed_through_cmd_chat(monkeypatch, capsys):
    """``hermes chat --no-tools -t web`` exits non-zero and builds no agent.

    Fail *closed*: the run must abort, not silently drop one of the two flags
    and proceed with a tool surface the operator did not ask for.
    """
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--no-tools", "-t", "web", "-q", "hello"])

    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)

    # cmd_chat calls the real ``cli.main``; only agent construction is stubbed,
    # so the mutual-exclusion guard under test actually runs.
    import cli as cli_mod

    constructed: list[object] = []
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: constructed.append(kw))

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 1
    assert constructed == [], "no agent may be constructed on a rejected flag combo"
    assert "--no-tools cannot be combined with --toolsets" in capsys.readouterr().out
    assert "HERMES_NO_TOOLS" not in os.environ


def test_parser_accepts_no_tools_on_root_and_chat():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()

    assert parser.parse_args(["--no-tools"]).no_tools is True
    assert parser.parse_args(["chat", "--no-tools"]).no_tools is True
    assert parser.parse_args(["chat"]).no_tools is False
