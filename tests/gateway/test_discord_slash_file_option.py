"""Long text into a Discord slash command via an attached file.

Discord's composer only offers the ">2000 chars becomes a .txt attachment" conversion to Nitro
subscribers, and a slash-command STRING option cannot be typed past that either, so without a file
option a non-Nitro user cannot give ``/goal`` (or ``/queue``, ``/plan``, ``/bg`` …) a long brief.

These are behaviour contracts on the dispatched command text and on the guards around the upload,
not a snapshot of which commands exist.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_slash_commands import FakeTree, _ensure_discord_mock

_ensure_discord_mock()

from plugins.platforms.discord.adapter import (  # noqa: E402
    _DISCORD_SLASH_FILE_MAX_BYTES,
    _DISCORD_SLASH_FILE_OPTION,
    DiscordAdapter,
)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._register_slash_commands()
    return adapter


def _command(adapter, name: str):
    """The registered callback for ``/name``, whether native (@tree.command stores the function)
    or auto-registered from COMMAND_REGISTRY (stores an app_commands.Command)."""
    entry = adapter._client.tree.commands[name]
    return getattr(entry, "callback", entry)


def _interaction():
    return SimpleNamespace(
        channel=SimpleNamespace(id=123, name="general", guild=SimpleNamespace(id=1, name="G"), topic=None),
        channel_id=123,
        guild_id=1,
        user=SimpleNamespace(id=42, name="Jezza", display_name="Jezza"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock(), is_done=lambda: False),
        followup=SimpleNamespace(send=AsyncMock()),
        edit_original_response=AsyncMock(),
        delete_original_response=AsyncMock(),
    )


def _attachment(data: bytes, *, filename="brief.txt", content_type="text/plain; charset=utf-8"):
    return SimpleNamespace(
        filename=filename,
        content_type=content_type,
        size=len(data),
        read=AsyncMock(return_value=data),
        url=f"https://cdn.discordapp.com/attachments/1/2/{filename}",
    )


# ------------------------------------------------------------------
# The option exists exactly where a free-text argument does
# ------------------------------------------------------------------


def test_free_text_command_gains_a_file_option(adapter):
    """/goal takes free text, so it must expose the file option."""
    params = _command(adapter, "goal").__signature__.parameters
    assert _DISCORD_SLASH_FILE_OPTION in params
    assert params[_DISCORD_SLASH_FILE_OPTION].default is None, "the file option must be optional"


def test_choice_bound_and_argless_commands_gain_nothing(adapter):
    """A choice-bound option (/reasoning) or no option at all (/status) cannot come from a file."""
    assert _DISCORD_SLASH_FILE_OPTION not in _command(adapter, "reasoning").__signature__.parameters
    assert _DISCORD_SLASH_FILE_OPTION not in _command(adapter, "status").__signature__.parameters


def test_required_text_argument_becomes_optional_so_a_file_can_supply_it(adapter):
    """/queue's prompt is required; Discord rejects a command whose required option is omitted,
    so the file alternative only works if the text option stops being required."""
    params = _command(adapter, "queue").__signature__.parameters
    assert params["prompt"].default == "", "prompt must be optional once a file can supply it"


# ------------------------------------------------------------------
# The dispatched command text
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_contents_become_the_command_argument(adapter):
    """The whole point: text far past Discord's paste limit reaches /goal intact."""
    brief = "Ship the thing. " * 500  # 8000 chars — unreachable by typing or by a non-Nitro paste
    adapter._run_simple_slash = AsyncMock()
    interaction = _interaction()

    await _command(adapter, "goal")(
        interaction, args="", **{_DISCORD_SLASH_FILE_OPTION: _attachment(brief.encode())},
    )

    adapter._run_simple_slash.assert_awaited_once()
    dispatched = adapter._run_simple_slash.await_args.args[1]
    assert dispatched == f"/goal {brief}".strip()
    assert len(dispatched) > 2000


@pytest.mark.asyncio
async def test_typed_text_is_kept_in_front_of_the_file(adapter):
    """`/goal draft` + a brief file must still parse as the `draft` verb."""
    adapter._run_simple_slash = AsyncMock()

    await _command(adapter, "goal")(
        _interaction(), args="draft", **{_DISCORD_SLASH_FILE_OPTION: _attachment(b"Long brief body")},
    )

    dispatched = adapter._run_simple_slash.await_args.args[1]
    assert dispatched.startswith("/goal draft\n")
    assert dispatched.endswith("Long brief body")


@pytest.mark.asyncio
async def test_no_file_dispatches_exactly_as_before(adapter):
    """The typed path must be byte-identical to the pre-change behaviour."""
    adapter._run_simple_slash = AsyncMock()
    interaction = _interaction()

    await _command(adapter, "goal")(interaction, args="ship it")

    adapter._run_simple_slash.assert_awaited_once_with(interaction, "/goal ship it")


# ------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_user_never_causes_an_attachment_fetch(adapter):
    """The auth gate must run before the bot downloads attacker-supplied bytes."""
    adapter._check_slash_authorization = AsyncMock(return_value=False)
    adapter._run_simple_slash = AsyncMock()
    attachment = _attachment(b"payload")

    await _command(adapter, "goal")(_interaction(), args="", **{_DISCORD_SLASH_FILE_OPTION: attachment})

    attachment.read.assert_not_awaited()
    adapter._run_simple_slash.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_text_file_is_refused_by_type_not_by_luck(adapter):
    """The extension/MIME gate must do the refusing.

    Use bytes that decode cleanly as UTF-8 so the decode check cannot mask a missing type gate —
    a .png whose bytes happen to be ASCII would otherwise sail through as a goal.
    """
    adapter._run_simple_slash = AsyncMock()
    interaction = _interaction()
    interaction.response.is_done = lambda: True

    await _command(adapter, "goal")(
        interaction, args="",
        **{_DISCORD_SLASH_FILE_OPTION: _attachment(b"PNG-but-valid-utf8", filename="x.png", content_type="image/png")},
    )

    adapter._run_simple_slash.assert_not_awaited()
    interaction.followup.send.assert_awaited_once()
    assert interaction.followup.send.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_a_text_extension_with_binary_bytes_is_also_refused(adapter):
    """Belt and braces: a .txt that is not UTF-8 must not become a half-decoded goal."""
    adapter._run_simple_slash = AsyncMock()

    await _command(adapter, "goal")(
        _interaction(), args="", **{_DISCORD_SLASH_FILE_OPTION: _attachment(b"\xff\xfe\x00bad")},
    )

    adapter._run_simple_slash.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_file_is_refused_before_download(adapter):
    """Size is checked from the attachment metadata, so a huge upload is never pulled."""
    adapter._run_simple_slash = AsyncMock()
    attachment = _attachment(b"x")
    attachment.size = _DISCORD_SLASH_FILE_MAX_BYTES + 1

    await _command(adapter, "goal")(_interaction(), args="", **{_DISCORD_SLASH_FILE_OPTION: attachment})

    attachment.read.assert_not_awaited()
    adapter._run_simple_slash.assert_not_awaited()


@pytest.mark.asyncio
async def test_neither_typed_nor_attached_is_refused_for_a_required_argument(adapter):
    """Making /queue's prompt optional must not let an empty /queue through."""
    adapter._run_simple_slash = AsyncMock()
    interaction = _interaction()

    await _command(adapter, "queue")(interaction, prompt="")

    adapter._run_simple_slash.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_interaction_is_deferred_before_the_download(adapter):
    """Reading an attachment can outlive Discord's 3 s initial-response deadline; ack first, and
    tell _run_simple_slash not to defer twice."""
    adapter._run_simple_slash = AsyncMock()
    interaction = _interaction()

    await _command(adapter, "goal")(
        interaction, args="", **{_DISCORD_SLASH_FILE_OPTION: _attachment(b"brief")},
    )

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    assert adapter._run_simple_slash.await_args.kwargs.get("already_deferred") is True


@pytest.mark.asyncio
async def test_already_deferred_run_does_not_defer_again(adapter):
    """The real _run_simple_slash must skip its own defer() when the caller already acked."""
    interaction = _interaction()
    adapter.handle_message = AsyncMock()

    await adapter._run_simple_slash(interaction, "/goal x", already_deferred=True)

    interaction.response.defer.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "/goal x"
    interaction.delete_original_response.assert_awaited_once()
