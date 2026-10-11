"""Discord /loop is a field form: each field renders into text the shared loop parser reads back."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_slash_commands import FakeTree, _ensure_discord_mock

_ensure_discord_mock()

from hermes_cli.loops import _CONTROL_COMMANDS, parse_loop_args  # noqa: E402
from plugins.platforms.discord.adapter import (  # noqa: E402
    _NATIVE_SLASH_COMMAND_SPECS, DiscordAdapter, _native_slash_commands,
)
from plugins.platforms.discord.adapter_slash_forms import (  # noqa: E402
    LOOP_SLASH_ARGS, LoopSlashInputError, render_loop_command,
)


@pytest.fixture
def adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(tree=FakeTree(), get_channel=lambda _id: None,
                                      user=SimpleNamespace(id=99999, name="HermesBot"))
    adapter._check_slash_authorization = AsyncMock(return_value=True)  # auth: test_discord_slash_auth.py
    return adapter


def _args(text):
    assert text == "/loop" or text.startswith("/loop ")
    return text[len("/loop"):].strip()


@pytest.mark.parametrize("fields, expected", [
    (dict(prompt="check the deploy", every="5m"),
     {"interval_seconds": 300, "prompt": "check the deploy", "times": 0, "until": ""}),
    (dict(prompt="poll CI", every="1h30m", times=30),
     {"interval_seconds": 5400, "prompt": "poll CI", "times": 30, "until": ""}),
    (dict(prompt="watch the queue", every="2m", until="queue is empty"),
     {"interval_seconds": 120, "prompt": "watch the queue", "times": 0, "until": "queue is empty"}),
    (dict(prompt="check --verbose output", every="5m", until="the --timestamp flag shows"),
     {"interval_seconds": 300, "prompt": "check --verbose output", "times": 0, "until": "the --timestamp flag shows"}),
    (dict(prompt="keep fixing tests"),
     {"interval_seconds": None, "prompt": "keep fixing tests", "times": 0, "until": ""}),
    (dict(prompt="  /recap  ", every=" 10m ", times=2, until="done"),
     {"interval_seconds": 600, "prompt": "/recap", "times": 2, "until": "done"}),
])
def test_fields_round_trip_through_the_loop_parser(fields, expected):
    parsed = parse_loop_args(_args(render_loop_command(**fields)))
    assert parsed["error"] is None
    assert {k: parsed[k] for k in expected} == expected


def test_every_action_choice_is_a_loop_control_word():
    choices = {a[0]: a[4] for a in LOOP_SLASH_ARGS}["action"]
    for _label, value in choices:
        assert render_loop_command(action=value) == f"/loop {value}"
        assert value in _CONTROL_COMMANDS


def test_empty_form_is_status():
    assert _args(render_loop_command()) in _CONTROL_COMMANDS


@pytest.mark.parametrize("fields, key", [
    (dict(action="stop", prompt="x"), "error_action_with_fields"),
    (dict(action="pause", times=2), "error_action_with_fields"),
    (dict(every="5m"), "error_missing_prompt"),
    (dict(until="done"), "error_missing_prompt"),
    (dict(prompt="poll CI", every="5"), "error_bad_interval"),
    (dict(prompt="poll CI", until="queue is empty --times soon"), "error_flag_in_text"),
    (dict(prompt="poll CI --until green"), "error_flag_in_text"),
    (dict(prompt="poll CI --times"), "error_flag_in_text"),
])
def test_ambiguous_forms_are_refused_not_guessed(fields, key):
    with pytest.raises(LoopSlashInputError) as exc:
        render_loop_command(**fields)
    assert exc.value.key.endswith(key)


def test_loop_is_a_native_field_form_not_the_generic_args_box(adapter):
    spec = {row[0]: row for row in _NATIVE_SLASH_COMMAND_SPECS}["loop"]
    assert spec[2] is LOOP_SLASH_ARGS and spec[3] is render_loop_command
    assert [a[0] for a in {r[0]: r for r in _native_slash_commands()}["loop"][2]] == \
        [a[0] for a in LOOP_SLASH_ARGS]
    adapter._register_slash_commands()
    # The registered callback is the field form, not the auto `args` proxy.
    assert adapter._client.tree.commands["loop"].__name__ == "slash_loop"


@pytest.mark.asyncio
async def test_registered_callback_dispatches_rendered_text(adapter):
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await adapter._client.tree.commands["loop"](interaction, prompt="check deploy", every="5m", times=3,
                                                 until="it is green", action="")
    adapter._run_simple_slash.assert_awaited_once_with(
        interaction, "/loop 5m check deploy --times 3 --until it is green")
    interaction.response.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_bad_form_replies_ephemerally_and_dispatches_nothing(adapter):
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await adapter._client.tree.commands["loop"](interaction, prompt="poll CI", every="5", times=0,
                                                 until="", action="")
    adapter._run_simple_slash.assert_not_awaited()
    text = interaction.response.send_message.await_args.args[0]
    assert '"5"' in text and interaction.response.send_message.await_args.kwargs == {"ephemeral": True}


@pytest.mark.asyncio
async def test_bad_form_from_unauthorized_user_gets_no_hint(adapter):
    adapter._run_simple_slash = AsyncMock()
    adapter._check_slash_authorization = AsyncMock(return_value=False)
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await adapter._client.tree.commands["loop"](interaction, prompt="", every="5m", times=0,
                                                 until="", action="")
    adapter._check_slash_authorization.assert_awaited_once()
    interaction.response.send_message.assert_not_awaited()
    adapter._run_simple_slash.assert_not_awaited()


@pytest.mark.asyncio
async def test_string_template_commands_still_render_by_format(adapter):
    """The callable branch is opt-in: a ``str`` template command dispatches exactly as before."""
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await adapter._client.tree.commands["steer"](interaction, prompt="look at the logs")
    adapter._run_simple_slash.assert_awaited_once_with(interaction, "/steer look at the logs")
