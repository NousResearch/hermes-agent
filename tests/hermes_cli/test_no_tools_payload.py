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
    # api_mode: (transport module, class name, base_url, provider)
    "chat_completions": (
        "agent.transports.chat_completions",
        "ChatCompletionsTransport",
        "https://openrouter.ai/api/v1",
        "openrouter",
    ),
    "codex_responses": (
        "agent.transports.codex",
        "ResponsesApiTransport",
        "https://chatgpt.com/backend-api/codex",
        "openai-codex",
    ),
}


def _spy_on_transport(monkeypatch, api_mode: str) -> list[dict[str, Any]]:
    """Wrap the real ``build_kwargs`` of one transport and record every call.

    The production implementation still runs — this is a spy, not a stub, so
    the recorded payload is the genuine one.
    """
    import importlib

    mod_name, cls_name, _base_url, _provider = _TRANSPORTS[api_mode]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    original = cls.build_kwargs
    calls: list[dict[str, Any]] = []

    def _spy(self, model, messages, tools=None, **params):
        result = original(self, model, messages, tools=tools, **params)
        calls.append({"tools_argument": tools, "payload": result})
        return result

    monkeypatch.setattr(cls, "build_kwargs", _spy)
    return calls


def _build_agent(api_mode: str, toolsets):
    """A real ``AIAgent`` using the real ``get_tool_definitions``.

    Only the OpenAI client class is patched out, so nothing can reach the
    network — exactly the pattern used by ``tests/run_agent/test_run_agent.py``.
    """
    from run_agent import AIAgent

    _mod, _cls, base_url, provider = _TRANSPORTS[api_mode]
    with patch("run_agent.OpenAI"):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url=base_url,
            provider=provider,
            api_mode=api_mode,
            enabled_toolsets=toolsets,
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _wire_payload(monkeypatch, api_mode: str, toolsets) -> dict[str, Any]:
    """Return the kwargs the provider SDK would be called with."""
    calls = _spy_on_transport(monkeypatch, api_mode)
    agent = _build_agent(api_mode, toolsets)
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
    assert "tool_choice" not in payload
    assert "parallel_tool_calls" not in payload


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
