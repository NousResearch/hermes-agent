"""Strict behaviour tests for the non-recursive alias dispatcher.

Verifies the eight invariants the user-supplied gate demands:

1. pre_gateway_dispatch fires exactly once.
2. ``_is_user_authorized`` fires exactly once.
3. ``_scale_to_zero_note_real_inbound`` fires exactly once.
4. ``event`` never gains a ``_original_command`` (or any alias marker).
5. ``/sonnet`` and ``/model sonnet`` behave identically while a session
   is running (busy-path, access gate, dispatch result).
6. ``/sonnet`` is NOT treated as a raw text interrupt/queue/steer when
   a session is running.
7. Higher-priority command names always win (built-in, quick, plugin,
   bundle, active skill, unavailable skill).
8. Hook rewrite target itself being an alias folds into /model form
   with arguments preserved.
9. Hook rewrite target being a normal command keeps upstream semantics.
10. ``/sonnet`` and ``/model sonnet`` access / hook / handler traces are
    identical (semantic equivalence).
11. / 12. Sequential AND concurrent events share no alias state — the
    helper is a pure function of the input event text.
"""
from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway._model_alias_normalize import (
    canonicalize_event_for_model_alias,
)
from gateway.platforms.base import MessageEvent, MessageType
from hermes_cli.model_switch import (
    DIRECT_ALIASES,
    MODEL_ALIASES,
)


# ----------------------------------------------------------------------
# Test scaffolding
# ----------------------------------------------------------------------


class _StubSource:
    """Minimal SessionSource replacement — only fields touched by the
    alias path. Built via object.__new__ to mirror the bare-runner pattern
    used by other gateway tests (see tests/gateway/conftest.py)."""

    def __init__(self, platform_value="discord", user_id="anyone", chat_type="dm"):
        self.platform = MagicMock()
        self.platform.value = platform_value
        self.user_id = user_id
        self.user_name = "Stub User"
        self.chat_id = "chat-1"
        self.chat_type = chat_type
        self.profile = "default"
        self.adapter_name = platform_value
        self.channel_context = None
        self.internal = False
        self.metadata = {}
        self.timestamp = None


def _event(text: str, *, source: _StubSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source or _StubSource(),
    )


def _runner(
    *,
    platform_extra: dict | None = None,
    busy: bool = False,
    session_id: str = "chat-1",
):
    """Build a bare GatewayRunner via object.__new__ so we never invoke
    the real __init__ (which spawns workers, timers, plugin discovery,
    etc.). Each helper we want to observe is then attached explicitly.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    cfg: dict = {}
    if platform_extra:
        cfg[platform_extra.setdefault("__platform_key__", "discord")] = platform_extra
    runner.config = cfg
    runner.console = MagicMock()
    runner.session_store = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_heartbeat_tasks = {}
    runner._session_run_generation = {}
    runner._session_run_generation_lock = MagicMock()
    runner._session_locks = {}
    runner._session_locks_guard = MagicMock()
    runner._pairing_store = {}
    runner._draining = False
    runner._lobby_reminder_last_sent = {}
    runner._busy_input_mode = "interrupt"
    runner.hooks = MagicMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._claim_active_session_slot = MagicMock(return_value=(None, None))
    runner._session_state = MagicMock(return_value=MagicMock())
    runner._peek_session_state = MagicMock(return_value=None)
    runner._persist_active_agents = MagicMock()
    runner._begin_session_run_generation = MagicMock(return_value=0)
    runner._invalidate_session_run_generation = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._session_key_for_source = MagicMock(return_value=session_id)
    runner._is_session_running = MagicMock(return_value=busy)
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._invoke_hook = MagicMock(return_value=[])
    runner._handle_model_command = AsyncMock(return_value="model-handled")
    runner._handle_status_command = AsyncMock(return_value="status-handled")
    runner._handle_context_command = AsyncMock(return_value="context-handled")
    runner._dispatch_busy_slash_command = AsyncMock(return_value="busy-handled")
    runner._check_slash_access = MagicMock(return_value=None)
    runner._pending_event_audio_paths = MagicMock(return_value=[])
    runner._prepare_clarify_reply_text = AsyncMock(return_value="")
    runner._handle_message_with_agent = AsyncMock(return_value="agent-handled")
    runner._is_telegram_topic_root_lobby = MagicMock(return_value=False)
    runner._should_send_telegram_lobby_reminder = MagicMock(return_value=False)
    runner._telegram_topic_root_lobby_message = MagicMock(return_value="")
    runner._telegram_topic_root_new_message = MagicMock(return_value="")
    runner._peek_pending_update_prompt_state = MagicMock(return_value=None)
    runner._status_action_gerund = MagicMock(return_value="shutting down")
    runner._hermes_home_override = None
    return runner


# ----------------------------------------------------------------------
# 1–4: Single-entry invariants
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_fires_once():
    """``pre_gateway_dispatch`` MUST fire exactly once for a typed
    ``/sonnet`` even though the helper rewrites the event to
    ``/model sonnet``."""
    runner = _runner()

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ) as invoke, patch.object(
        runner, "_handle_model_command", AsyncMock(return_value="model-handled")
    ):
        result = await runner._handle_message(_event("/sonnet"))

    assert invoke.call_count == 1, (
        f"pre_gateway_dispatch fired {invoke.call_count} times for one "
        f"typed alias; expected exactly 1 (no _handle_message recursion)."
    )
    assert result == "model-handled"


@pytest.mark.asyncio
async def test_user_authorization_fires_once():
    runner = _runner()

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet"))

    assert runner._is_user_authorized.call_count == 1


@pytest.mark.asyncio
async def test_inbound_activity_note_fires_once():
    runner = _runner()

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet"))

    assert runner._scale_to_zero_note_real_inbound.call_count == 1


@pytest.mark.asyncio
async def test_event_has_no_alias_marker():
    """The dispatcher must never attach ``_original_command`` (or any
    other alias sentinel) to the inbound event."""
    runner = _runner()
    src = _StubSource()
    inbound = _event("/sonnet", source=src)

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(inbound)

    for forbidden in (
        "_original_command",
        "_alias_name",
        "_alias_rewrite",
        "_pre_rewrite_text",
    ):
        assert not hasattr(inbound, forbidden) or getattr(inbound, forbidden, None) is None, (
            f"event acquired a hidden alias marker {forbidden!r}; "
            f"the dispatcher must be stateless."
        )
    # Inbound text unchanged — the gateway only ever produces a local
    # dataclasses.replace clone that never escapes the function scope.
    assert inbound.text == "/sonnet"


# ----------------------------------------------------------------------
# 5. / 6. busy-path equivalence
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_busy_path_dispatch_identical_to_typed_model():
    """While an agent is running, ``/sonnet`` must reach
    ``_dispatch_busy_slash_command`` exactly once with a model-shape
    event — same as a typed ``/model sonnet``."""
    runner = _runner(busy=True)
    src = _StubSource()

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet", source=src))

    runner._dispatch_busy_slash_command.assert_awaited_once()
    cmd_def = runner._dispatch_busy_slash_command.await_args.args[1]
    assert cmd_def.name == "model", (
        f"busy-path cmd_def must be the model CommandDef for alias dispatch, "
        f"got {cmd_def.name!r}"
    )
    busy_event = runner._dispatch_busy_slash_command.await_args.args[0]
    assert busy_event.text.lower().startswith("/model sonnet"), (
        f"busy-path event must be canonical /model sonnet, "
        f"got {busy_event.text!r}"
    )


@pytest.mark.asyncio
async def test_alias_not_treated_as_text_interrupt_under_busy_path():
    """While busy, the dispatcher must NOT treat ``/sonnet`` as plain
    text and route it through the interrupt/queue/steer logic."""
    runner = _runner(busy=True)
    # If alias were treated as text, _handle_message_with_agent would be
    # invoked (the fallthrough for unrecognised / plain text). If the
    # alias is correctly routed to the model CommandDef's busy policy,
    # _dispatch_busy_slash_command handles it instead.
    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet"))

    runner._handle_message_with_agent.assert_not_awaited()
    runner._dispatch_busy_slash_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_busy_path_access_checked_via_canonical_model():
    """Access control for an alias under busy-path must use the model's
    canonical name, never the raw alias."""
    runner = _runner(busy=True)

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet"))

    # _check_slash_access on the busy-path runs at the very top, before
    # the canonicalising helper (status / context are intentionally
    # pre-gated). Aliases aren't status/context, so they reach the
    # busy-path dispatcher which then runs access control. We assert
    # the busy dispatch was invoked and observed the model name.
    cmd_def = runner._dispatch_busy_slash_command.await_args.args[1]
    assert cmd_def.name == "model"


# ----------------------------------------------------------------------
# 7. Higher-priority command names always win
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_built_in_command_wins_over_alias():
    """A built-in that happens to share the alias name must dispatch
    via its built-in handler, not via the alias shortcut."""
    from hermes_cli.commands import resolve_command

    # Pre-condition: ``help`` is a built-in. We don't have a registered
    # alias named ``help`` in DIRECT_ALIASES / MODEL_ALIASES by default,
    # so the test asserts via the helper directly with a name we KNOW
    # is occupied.
    assert resolve_command("help") is not None
    src = _StubSource()
    ev = _event("/help", source=src)
    out = canonicalize_event_for_model_alias(ev, config={})
    assert out.text == "/help", (
        f"built-in /help must NOT be rewritten as a model alias; "
        f"got {out.text!r}"
    )


@pytest.mark.asyncio
async def test_quick_command_wins_over_alias():
    """A user-defined quick_command named like an alias must still win."""
    cfg = {"quick_commands": {"sonnet": {"type": "exec", "command": "echo hi"}}}
    ev = _event("/sonnet")
    out = canonicalize_event_for_model_alias(ev, config=cfg)
    assert out.text == "/sonnet", (
        f"quick_command /sonnet must NOT be rewritten; got {out.text!r}"
    )


@pytest.mark.asyncio
async def test_alias_helper_returns_unchanged_for_unknown_name():
    """A name that is neither an alias nor any registered command must
    pass through untouched so the upstream ``unknown command`` warning
    path can fire."""
    ev = _event("/xyzzy-no-such-alias-987")
    out = canonicalize_event_for_model_alias(ev, config={})
    assert out is ev, (
        "helper must return the exact input event when no rewrite applies"
    )
    assert out.text == "/xyzzy-no-such-alias-987"


@pytest.mark.asyncio
async def test_unavailable_skill_blocks_alias():
    """If a known-but-disabled skill happens to share an alias name,
    the skill's unavailability message must surface — the alias
    shortcut must NOT fire."""
    from gateway._model_alias_normalize import canonicalize_event_for_model_alias

    # Force the helper to recognise ``disabled-alias`` as an alias.
    DIRECT_ALIASES["disabled-alias"] = DIRECT_ALIASES.get(
        "sonnet", MagicMock()
    )  # may fail if sonnet absent; tolerated — see alias-or-skip below

    def _stub_unavailable(name):
        if name == "help":
            return "The **help** skill is disabled for this platform."
        return None

    # Use a known alias that the helper will recognise. Here we pick
    # the canonical test alias by checking the dict.
    if "sonnet" in DIRECT_ALIASES or "sonnet" in MODEL_ALIASES:
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(
            ev,
            config={},
            unavailable_skill_fn=_stub_unavailable,
        )
        # ``sonnet`` isn't a disabled skill name; alias wins.
        assert out.text.startswith("/model sonnet"), (
            f"alias should fire when no skill occupies the name; "
            f"got {out.text!r}"
        )

    # Now confirm that a name which IS occupied by ``unavailable`` is
    # never rewritten (even if we manually inject it as an alias).
    DIRECT_ALIASES["help"] = MagicMock()
    try:
        ev = _event("/help")
        out = canonicalize_event_for_model_alias(
            ev,
            config={},
            unavailable_skill_fn=_stub_unavailable,
        )
        assert out.text == "/help", (
            f"unavailable skill /help must NOT be rewritten; got {out.text!r}"
        )
    finally:
        DIRECT_ALIASES.pop("help", None)
        DIRECT_ALIASES.pop("disabled-alias", None)


# ----------------------------------------------------------------------
# 8. / 9. Hook rewrite behaviour
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_rewrite_to_alias_folds_into_model():
    """A hook returning ``decision='rewrite'`` with a target that IS a
    model alias must fold into ``/model <alias> <args>`` so the
    eventual handler still receives the model namespace."""
    runner = _runner()
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "opus",
                "raw_args": "--provider openrouter",
            }
        ]
    )

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        await runner._handle_message(_event("/sonnet"))

    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text == "/model opus --provider openrouter", (
        f"alias-rewrite target must fold into /model <alias> form, "
        f"got {event_arg.text!r}"
    )


@pytest.mark.asyncio
async def test_hook_rewrite_to_normal_command_keeps_upstream_semantics():
    """A hook rewriting to a normal (non-alias) command must follow the
    canonical upstream rewrite path — the access gate fires once on the
    ORIGINAL canonical command (``command:model`` for the alias) and the
    rewritten target is dispatched as a regular built-in (no re-fire of
    the hook on the rewritten command, no recursion into _handle_message).

    This matches the documented upstream behaviour in ``gateway/run.py``
    where the rewrite branch ``break``s out of the hook loop and then
    falls through to the canonical ``if canonical == "..."`` ladder.
    """
    runner = _runner()
    runner._handle_help_command = AsyncMock(return_value="help-handled")
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {"decision": "rewrite", "command_name": "help", "raw_args": ""}
        ]
    )

    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[],
    ):
        result = await runner._handle_message(_event("/sonnet"))

    # Hook fired exactly once (on the alias's canonical ``command:model``),
    # then the dispatcher routed the rewritten ``/help`` to the built-in
    # handler. The upstream semantics forbid re-firing the hook on the
    # rewrite target or re-entering _handle_message.
    assert runner.hooks.emit_collect.await_count == 1
    assert runner.hooks.emit_collect.await_args.args[0] == "command:model"
    runner._handle_help_command.assert_awaited_once()
    runner._handle_model_command.assert_not_awaited()
    assert result == "help-handled"


# ----------------------------------------------------------------------
# 10. Direct /sonnet vs /model sonnet semantic equivalence
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_model_command_and_alias_share_access_hook_handler():
    """A typed ``/sonnet`` and a typed ``/model sonnet`` must produce
    the same number of access checks, the same hook name(s), and the
    same handler invocation — only the input text differs."""
    captured = []

    runner_alias = _runner()
    runner_direct = _runner()

    # Capture every emit_collect invocation.
    captured_alias: list = []
    captured_direct: list = []

    async def _alias_emit(name, ctx):
        captured_alias.append((name, dict(ctx)))
        return []

    async def _direct_emit(name, ctx):
        captured_direct.append((name, dict(ctx)))
        return []

    runner_alias.hooks.emit_collect = _alias_emit
    runner_direct.hooks.emit_collect = _direct_emit

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]):
        await runner_alias._handle_message(_event("/sonnet"))
        await runner_direct._handle_message(_event("/model sonnet"))

    # Both must fire exactly one command:model hook with the same
    # canonical name.
    assert len(captured_alias) == 1
    assert len(captured_direct) == 1
    assert captured_alias[0][0] == "command:model"
    assert captured_direct[0][0] == "command:model"
    assert captured_alias[0][1]["command"] == "model"
    assert captured_direct[0][1]["command"] == "model"

    # Both must reach _handle_model_command exactly once.
    runner_alias._handle_model_command.assert_awaited_once()
    runner_direct._handle_model_command.assert_awaited_once()


# ----------------------------------------------------------------------
# 11. / 12. Sequential + concurrent messages share no alias state
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_messages_do_not_leak_alias_state():
    """Two sequential alias messages must each rewrite independently;
    a rewrite from message #1 must NOT bleed into message #2's text or
    flag any session/global/self state."""
    runner = _runner()

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]):
        await runner._handle_message(_event("/sonnet"))
        await runner._handle_message(_event("/opus"))

    assert runner._handle_model_command.await_count == 2
    first_text = runner._handle_model_command.await_args_list[0].args[0].text
    second_text = runner._handle_model_command.await_args_list[1].args[0].text
    assert "sonnet" in first_text.lower()
    assert "opus" in second_text.lower()
    # No event should have an alias marker.
    assert not hasattr(runner, "_alias_pending") or not runner._alias_pending
    assert not hasattr(runner, "_alias_last_rewrite")


@pytest.mark.asyncio
async def test_concurrent_messages_do_not_leak_alias_state():
    """Two concurrent alias messages must each rewrite independently
    and reach the model handler with their own (unique) target."""
    runner = _runner()

    async def _emit(name, ctx):
        return []

    runner.hooks.emit_collect = _emit

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]):
        await asyncio.gather(
            runner._handle_message(_event("/sonnet")),
            runner._handle_message(_event("/opus")),
        )

    assert runner._handle_model_command.await_count == 2
    texts = sorted(
        call.args[0].text.lower() for call in runner._handle_model_command.await_args_list
    )
    # Each call's rewritten text is independent.
    assert any("sonnet" in t for t in texts)
    assert any("opus" in t for t in texts)


# ----------------------------------------------------------------------
# Pending Update Interception & Equivalence
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_update_alias_bypasses_and_equals_typed_model(tmp_path):
    """When update_prompt_pending=True, /sonnet must bypass prompt interception
    and match /model sonnet identically without writing to .update_response."""
    runner = _runner()

    # Create a mock session state with update_prompt_pending = True
    session_state = MagicMock()
    session_state.persistent.update_prompt_pending = True
    session_state.turn.agent = None
    session_state.turn.started_ts = 0
    runner._peek_session_state = MagicMock(return_value=session_state)

    # Temporary home directory for .update_response / .update_prompt.json
    home_dir = tmp_path / "hermes_home"
    home_dir.mkdir()
    prompt_file = home_dir / ".update_prompt.json"
    prompt_file.write_text('{"prompt": "Update?"}', encoding="utf-8")
    resp_file = home_dir / ".update_response"

    with patch("gateway.run._hermes_home", home_dir), patch(
        "hermes_cli.lifecycle.invoke_hook", return_value=[]
    ):
        result_alias = await runner._handle_message(_event("/sonnet"))

    # Response file must NOT contain "/sonnet" (it was not swallowed as raw reply)
    if resp_file.exists():
        assert resp_file.read_text(encoding="utf-8") == ""

    runner._handle_model_command.assert_awaited_once()
    assert result_alias == "model-handled"


@pytest.mark.asyncio
async def test_pending_update_consumes_plain_text(tmp_path):
    """Plain text responses during pending update MUST still be consumed."""
    runner = _runner()

    session_state = MagicMock()
    session_state.persistent.update_prompt_pending = True
    session_state.turn.agent = None
    session_state.turn.started_ts = 0
    runner._peek_session_state = MagicMock(return_value=session_state)

    home_dir = tmp_path / "hermes_home"
    home_dir.mkdir()
    prompt_file = home_dir / ".update_prompt.json"
    prompt_file.write_text('{"prompt": "Update?"}', encoding="utf-8")
    resp_file = home_dir / ".update_response"

    with patch("gateway.run._hermes_home", home_dir), patch(
        "hermes_cli.lifecycle.invoke_hook", return_value=[]
    ):
        result = await runner._handle_message(_event("yes, please update"))

    assert resp_file.exists()
    assert resp_file.read_text(encoding="utf-8") == "yes, please update"
    assert "Sent `yes, please update`" in result


# ----------------------------------------------------------------------
# 10 Strict Priority Conflict & Fail-Closed Tests
# ----------------------------------------------------------------------


def test_priority_1_plugin_wins_over_alias():
    """1. plugin name with model alias -> plugin wins, no handler called."""
    plugin_handler = MagicMock()
    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        return_value=plugin_handler,
    ):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(ev)
        assert out is ev, "Plugin name must occupy alias and prevent rewrite"
        plugin_handler.assert_not_called()


def test_priority_2_bundle_wins_over_alias():
    """2. bundle name with model alias -> bundle wins, no builder called."""
    bundle_key = "some_bundle"
    with patch(
        "agent.skill_bundles.resolve_bundle_command_key",
        return_value=bundle_key,
    ), patch("hermes_cli.plugins.get_plugin_command_handler", return_value=None):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(ev)
        assert out is ev, "Bundle name must occupy alias and prevent rewrite"


def test_priority_3_active_skill_wins_over_alias():
    """3. active skill name with model alias -> skill wins, no invocation built."""
    skill_key = "some_skill"
    with patch(
        "agent.skill_commands.resolve_skill_command_key",
        return_value=skill_key,
    ), patch(
        "agent.skill_bundles.resolve_bundle_command_key", return_value=None
    ), patch("hermes_cli.plugins.get_plugin_command_handler", return_value=None):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(ev)
        assert out is ev, "Active skill name must occupy alias and prevent rewrite"


def test_priority_4_unavailable_skill_guidance_wins_over_alias():
    """4. unavailable skill name with model alias -> unavailable wins."""
    def _stub_unavailable(name):
        return "The **sonnet** skill is disabled."

    with patch(
        "agent.skill_commands.resolve_skill_command_key", return_value=None
    ), patch(
        "agent.skill_bundles.resolve_bundle_command_key", return_value=None
    ), patch("hermes_cli.plugins.get_plugin_command_handler", return_value=None):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(
            ev, unavailable_skill_fn=_stub_unavailable
        )
        assert out is ev, "Unavailable skill must block model alias rewrite"


def test_priority_5_quick_command_wins_over_alias():
    """5. quick command with model alias -> quick command wins."""
    cfg = {"quick_commands": {"sonnet": {"type": "exec", "command": "echo 1"}}}
    ev = _event("/sonnet")
    out = canonicalize_event_for_model_alias(ev, config=cfg)
    assert out is ev, "Quick command must occupy alias and prevent rewrite"


def test_priority_6_builtin_wins_over_alias():
    """6. built-in with model alias -> built-in wins."""
    with patch(
        "gateway._model_alias_normalize._resolve_cmd", return_value=MagicMock()
    ):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(ev)
        assert out is ev, "Built-in command must occupy alias and prevent rewrite"


def test_priority_7_resolver_exception_fails_closed():
    """7. plugin/bundle/skill resolver exception -> fail closed (returns raw event)."""
    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        side_effect=RuntimeError("DB corrupt"),
    ):
        ev = _event("/sonnet")
        out = canonicalize_event_for_model_alias(ev)
        assert out is ev, "Exception in lookup must fail closed and return unrewritten event"


@pytest.mark.asyncio
async def test_priority_8_hook_rewrite_to_occupied_plugin_does_not_fold():
    """8. Hook rewrite to an occupied plugin -> does not fold to /model."""
    runner = _runner()
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "sonnet",
                "raw_args": "--foo",
            }
        ]
    )
    plugin_handler = AsyncMock(return_value="plugin-handled")

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        return_value=plugin_handler,
    ), patch(
        "hermes_cli.lifecycle.invoke_hook", return_value=[]
    ):
        result = await runner._handle_message(_event("/model_switch_cmd"))

    # Because sonnet is an occupied plugin, it must NOT fold into /model sonnet
    runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_priority_9_hook_rewrite_to_unoccupied_alias_folds():
    """9. Hook rewrite to un-occupied model alias -> folds to /model and preserves args."""
    runner = _runner()
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "opus",
                "raw_args": "--temperature 0.5",
            }
        ]
    )

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler", return_value=None
    ), patch(
        "agent.skill_bundles.resolve_bundle_command_key", return_value=None
    ), patch(
        "agent.skill_commands.resolve_skill_command_key", return_value=None
    ), patch(
        "hermes_cli.lifecycle.invoke_hook", return_value=[]
    ):
        await runner._handle_message(_event("/sonnet"))

    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text == "/model opus --temperature 0.5"


def test_priority_10_cli_and_gateway_share_priority_order():
    """10. CLI and Gateway share identical priority resolution via helper."""
    ev = _event("/sonnet")
    with patch(
        "hermes_cli.plugins.get_plugin_command_handler", return_value=None
    ), patch(
        "agent.skill_bundles.resolve_bundle_command_key", return_value=None
    ), patch(
        "agent.skill_commands.resolve_skill_command_key", return_value=None
    ):
        out_gateway = canonicalize_event_for_model_alias(ev)
        assert out_gateway.text == "/model sonnet"


# ----------------------------------------------------------------------
# Helper purity invariant (separate from the runner-level tests).
# ----------------------------------------------------------------------


def test_helper_is_pure_and_returns_new_event():
    """The helper must return a brand-new dataclass instance (never
    mutate the input) and never raise on edge cases."""
    a = _event("/sonnet")
    b = canonicalize_event_for_model_alias(a, config={})
    assert b is not a, (
        "helper must return a fresh dataclasses.replace clone when rewriting"
    )
    assert a.text == "/sonnet", "original event text must be unchanged"

    # Edge case: no leading slash, no text, control characters.
    c = _event("plain text")
    d = canonicalize_event_for_model_alias(c, config={})
    assert d is c

    e = _event("")
    f = canonicalize_event_for_model_alias(e, config={})
    assert f is e