"""Home guidance uses the receiving adapter's typed command prefix."""

from dataclasses import replace

import pytest

from agent import i18n
from gateway import group_home_consent as consent
from gateway.config import Platform
from hermes_cli.config import save_config
from tests.gateway.test_group_chat_selected_home_owner import selected_home
from tests.gateway.test_group_home_consent import command, home


@pytest.fixture(params=[Platform.MATRIX, Platform.SLACK, Platform.TELEGRAM])
def destination(home, monkeypatch, request):
    home.runner, home.event = selected_home(monkeypatch, request.param, accepted=False)
    home.prefix = home.runner._typed_command_prefix_for(home.event.source)
    save_config({
        "platforms": {
            request.param.value: home.runner.config.platforms[request.param].to_dict()
        }
    })
    return home


def assert_commands(value, prefix, *commands):
    for command_name in commands:
        assert prefix + command_name in value
    assert "{command_prefix}" not in value
    if prefix != "/":
        assert "/group" not in value
        assert "/sethome" not in value
        assert "/whoami" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
@pytest.mark.parametrize("verb", ["help", "usage", "?"])
async def test_help_prefix_without_private_lookup(destination, monkeypatch, authorized, verb):
    runner = destination.runner
    monkeypatch.setattr(
        runner, "_can_control_group_chats",
        lambda event, *, require_audience=True: authorized,
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: pytest.fail("generic help fetched private rooms"),
    )
    prefix = destination.prefix
    result = await command(destination, f"{prefix}group {verb}")
    assert_commands(result, prefix, "group")
    assert "sethome" not in result and "command admin" not in result
    assert "Learn more about Group Chats" in result
    assert "https://hermes-agent.nousresearch.com/docs/user-guide/bot-mode/" in result
    assert "stop" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["setup", "binding", "slash-access"])
async def test_setup_denials_use_receiving_prefix(destination, monkeypatch, kind):
    runner, event = destination.runner, destination.event
    if kind == "binding":
        event.source.thread_id = "other-topic"
    else:
        event.source.user_id = "user-2"
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: pytest.fail("denial fetched private rooms"),
    )
    result = (
        runner._check_slash_access(event.source, "group")
        if kind == "slash-access"
        else await command(destination, f"{destination.prefix}group")
    )
    if kind == "binding":
        assert_commands(result, destination.prefix, "sethome", "group")
    else:
        assert_commands(result, destination.prefix, "whoami")
        assert "sethome" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "expired", "failed", "saving", "cancel_late"])
async def test_consent_retry_copy(destination, monkeypatch, outcome):
    prefix, runner = destination.prefix, destination.runner
    warning = await command(destination, f"{prefix}group")
    assert_commands(warning, prefix, "group confirm", "group cancel")
    assert "/approve" not in warning
    pending = next(iter(runner._group_home_confirmations.values()))
    if outcome == "expired":
        pending.deadline = 0
    elif outcome == "failed":
        monkeypatch.setattr("hermes_cli.config.save_config", lambda *args, **kwargs: None)
    elif outcome in {"saving", "cancel_late"}:
        pending.state, pending.commit_started = "committing", True
    verb = "cancel" if outcome in {"cancel", "cancel_late"} else "confirm"
    if outcome == "saving":
        verb = ""
    result = await command(destination, f"{prefix}group {verb}")
    expected_command = "sethome" if outcome == "cancel_late" else "group"
    assert_commands(result, prefix, expected_command)
    assert result == consent.text(outcome, command_prefix=prefix)


@pytest.mark.asyncio
async def test_sethome_save_failure_copy(destination, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.save_config", lambda *args, **kwargs: None)
    result = await destination.runner._handle_set_home_command(
        replace(destination.event, text=f"{destination.prefix}sethome")
    )
    assert_commands(result, destination.prefix, "sethome")


@pytest.mark.asyncio
async def test_sethome_success_explains_destination_and_next_action(destination):
    result = await destination.runner._handle_set_home_command(
        replace(destination.event, text=f"{destination.prefix}sethome")
    )
    assert_commands(result, destination.prefix, "group")
    assert "scheduled updates and cross-chat messages" in result
    assert "separate authorization" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["call", "callback", "result"])
async def test_disclosure_retry_copy(destination, boundary):
    runner, event = destination.runner, destination.event
    selected = runner.config.get_home_channel(event.source.platform)
    selected.group_audience_ack = consent.acknowledgement(selected)
    stamp = consent._disclosure_stamp(runner, event)
    assert stamp is not None

    async def revoked_result(runner, event):
        selected.group_audience_ack = None
        return "private sentinel"

    if boundary == "result":
        result = await consent.protect_group_result(revoked_result)(runner, event)
    elif boundary == "callback":
        callback = consent.protect_group_callback(runner, event)(
            lambda: pytest.fail("stale callback accessed private data")
        )
        selected.group_audience_ack = None
        result = await callback()
    else:
        selected.group_audience_ack = None
        with pytest.raises(consent.DisclosureChanged) as caught:
            await consent.disclosed_call(
                runner, event, stamp, lambda: pytest.fail("stale private fetch")
            )
        result = str(caught.value)
    assert "private sentinel" not in result
    assert_commands(result, destination.prefix, "group")


@pytest.mark.parametrize("lang", i18n.SUPPORTED_LANGUAGES)
@pytest.mark.parametrize("prefix", ["/", "!"])
def test_all_home_command_translations_format(lang, prefix, monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    for key, commands in {
        "help": ("group", "sethome"),
        "cancel": ("group",),
        "expired": ("group",),
        "failed": ("group",),
        "setup": ("group", "sethome", "whoami"),
        "binding": ("group", "sethome"),
        "home_failed": ("sethome",),
        "cancel_late": ("sethome",),
        "saving": ("group",),
    }.items():
        if key == "setup" and lang == "en":
            commands = ("whoami",)
        result = consent.text(key, command_prefix=prefix)
        assert_commands(result, prefix, *commands)
        if prefix == "/":
            assert result == consent.text(key)


def test_unrelated_slash_denial_unchanged():
    assert consent.slash_denial("status") is None
