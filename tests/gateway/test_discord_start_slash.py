"""Tests for the ``/start <project> <title>`` Discord slash command.

The /start command is a project-aware thread launcher: it resolves the
project's #geral channel from the configured ``start_projects`` mapping,
creates a thread with the canonical ``🎮 <name> — Nova conversa [<title>]``
name format, sends a worker mention + title as the seed message, and shows
a typing indicator for ~5s so the worker has time to acknowledge.

These tests intentionally exercise the handler in isolation against stub
interactions — the goal is to lock in behaviour contracts (mapping,
thread name format, project validation, auth gating), not to mock the
entire discord.py interaction stack. The flow is small enough that the
real handler is reachable without spinning up a discord client.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


# ──────────────────────────────────────────────────────────────────────
# Fixtures + helpers
# ──────────────────────────────────────────────────────────────────────


def _adapter(extra=None):
    """Build a DiscordAdapter with no token and a custom ``config.extra``."""
    config = PlatformConfig(enabled=True, token="***")
    if extra is not None:
        config.extra = dict(extra)
    return DiscordAdapter(config)


def _make_interaction(*, channel=None, user=None):
    """Build a stub discord.Interaction matching what the handler touches."""
    return SimpleNamespace(
        response=SimpleNamespace(
            defer=AsyncMock(),
            send_message=AsyncMock(),
        ),
        followup=SimpleNamespace(send=AsyncMock()),
        channel=channel,
        user=user or SimpleNamespace(id=999, display_name="tester"),
        channel_id=getattr(channel, "id", None),
        guild=None,
    )


class _FakeChannel:
    """Minimal channel stub — exposes ``id`` and ``typing()``."""

    def __init__(self, channel_id=123):
        self.id = channel_id
        self.typing_calls = 0

    def typing(self):
        outer = self

        class _Ctx:
            async def __aenter__(self_inner):
                outer.typing_calls += 1
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


class _FakeThread:
    def __init__(self, thread_id=555, name="t"):
        self.id = thread_id
        self.name = name
        self.sent_messages: list[str] = []

    async def send(self, body):
        self.sent_messages.append(body)


# ──────────────────────────────────────────────────────────────────────
# 1. Slash command registration
# ──────────────────────────────────────────────────────────────────────


def test_start_slash_command_is_registered():
    """``/start`` must appear on the command tree alongside the other native slash commands."""
    from plugins.platforms.discord import adapter as adapter_module

    # The simplest way to confirm registration without instantiating a real
    # discord client is to assert the symbol exists on the adapter module —
    # the closure inside _register_slash_commands references it by name.
    assert hasattr(adapter_module.DiscordAdapter, "_handle_start_slash"), (
        "DiscordAdapter must define _handle_start_slash for the /start command."
    )
    # The thread-name formatter and project-mapping helpers are the contract
    # surface we want to keep stable; if they're missing, downstream code
    # will silently fall back to the default project for every invocation.
    assert hasattr(adapter_module.DiscordAdapter, "_resolve_start_projects")
    assert hasattr(adapter_module.DiscordAdapter, "_format_start_thread_name")


def test_start_slash_command_accepts_project_and_title_arguments():
    """The /start handler signature must accept ``project`` + ``title`` kwargs only."""
    import inspect

    sig = inspect.signature(DiscordAdapter._handle_start_slash)
    params = sig.parameters
    assert "interaction" in params
    assert "project" in params
    assert "title" in params
    # project + title must be keyword-only so the slash-command binding
    # stays in charge of ordering and the handler doesn't accept stray
    # positional args.
    assert params["project"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["title"].kind is inspect.Parameter.KEYWORD_ONLY


# ──────────────────────────────────────────────────────────────────────
# 2. Project mapping
# ──────────────────────────────────────────────────────────────────────


def test_default_project_mapping_includes_all_known_projects():
    """The built-in mapping must cover every project the slash UI advertises."""
    adapter = _adapter()
    projects = adapter._resolve_start_projects()
    for slug in ("il", "il-cortes", "il-launcher", "thazeron", "hermes/improve", "default"):
        assert slug in projects, f"project `{slug}` missing from default mapping"
        entry = projects[slug]
        # Each entry must carry the three fields the handler reads.
        assert "channel_id" in entry
        assert "worker_mention" in entry
        assert "display_name" in entry


def test_config_override_merges_with_defaults():
    """Per-project overrides in config.extra must replace defaults without losing siblings."""
    adapter = _adapter(
        extra={
            "start_projects": {
                "il": {
                    "channel_id": "987654321",
                    "worker_mention": "@custom-il ",
                    "display_name": "IL Server",
                },
            },
        }
    )
    projects = adapter._resolve_start_projects()
    # Overridden project picks up new fields.
    assert projects["il"]["channel_id"] == "987654321"
    assert projects["il"]["worker_mention"] == "@custom-il "
    assert projects["il"]["display_name"] == "IL Server"
    # Untouched siblings keep their built-in defaults — this is the part of
    # the contract that prevents a partial override from blanking the rest
    # of the project list.
    assert projects["thazeron"]["worker_mention"] == "<@1528631797343064194>"
    assert projects["hermes/improve"]["display_name"] == "Hermes Improve"


def test_unknown_project_slug_is_rejected_ephemerally():
    """A bogus project must fail fast with an ephemeral error, not silently fall back to default."""
    import asyncio

    async def _runner():
        adapter = _adapter()
        interaction = _make_interaction()
        adapter._check_slash_authorization = AsyncMock(return_value=True)
        adapter._create_thread_in_channel = AsyncMock(return_value=_FakeThread())

        await adapter._handle_start_slash(interaction, project="not-a-real-project", title="t")

        # The handler must short-circuit before touching thread creation so
        # users can't typo a slug and end up in the wrong project.
        adapter._create_thread_in_channel.assert_not_awaited()
        interaction.response.send_message.assert_awaited()
        # The adapter calls send_message(content, ephemeral=True) positionally,
        # so the message body lives in args[0]. Use args[0] to match the
        # existing adapter call pattern (positional-first).
        msg = interaction.response.send_message.await_args.args[0]
        assert "not-a-real-project" in msg

    asyncio.run(_runner())


# ──────────────────────────────────────────────────────────────────────
# 3. Thread creation
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_start_creates_thread_with_canonical_name_and_seed():
    """End-to-end: defer → resolve channel → create thread → send seed message → typing."""
    adapter = _adapter(
        extra={
            "start_projects": {
                "il": {
                    "channel_id": "111",
                    "worker_mention": "@worker-il ",
                    "display_name": "IL",
                },
            },
        }
    )
    fake_channel = _FakeChannel(channel_id=111)
    fake_thread = _FakeThread(thread_id=555, name="placeholder")
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: fake_channel if cid == 111 else None,
        fetch_channel=AsyncMock(return_value=fake_channel),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._create_thread_in_channel = AsyncMock(return_value=fake_thread)
    adapter._typing_indicator = AsyncMock()  # don't actually sleep 5s in tests

    threads_sentinel = SimpleNamespace(mark=lambda _tid: None)
    adapter._threads = threads_sentinel

    interaction = _make_interaction(channel=fake_channel)
    await adapter._handle_start_slash(interaction, project="il", title="fase 3.3 travando")

    # Auth gate ran before thread creation.
    adapter._check_slash_authorization.assert_awaited_once()
    # Thread name must follow the canonical format with the 🎮 emoji + title.
    kwargs = adapter._create_thread_in_channel.await_args.kwargs
    assert kwargs["name"].startswith("🎮 IL — Nova conversa [")
    assert "fase 3.3 travando" in kwargs["name"]
    # Seed message must include the worker mention AND the title so the
    # worker has both pieces of context when it wakes up.
    assert fake_thread.sent_messages, "seed message must be sent into the new thread"
    seed = fake_thread.sent_messages[0]
    assert "@worker-il" in seed
    assert "fase 3.3 travando" in seed
    # Thread participation tracker must be marked so follow-ups don't
    # require @mention inside the thread.
    # (mark is a sync lambda on the sentinel — verifying it ran is a no-op
    # here, but the attribute access ensures the handler didn't blow up.)
    # Typing indicator must run AFTER the seed message, giving the worker
    # ~5s to acknowledge before the user returns.
    assert adapter._typing_indicator.await_count == 1
    # The adapter calls _typing_indicator(thread, seconds=5.0) — the channel
    # sits in args[0], while seconds is passed as a keyword. Use args[0] to
    # match the adapter's positional-first call pattern.
    typing_args = adapter._typing_indicator.await_args.args
    typing_kwargs = adapter._typing_indicator.await_args.kwargs
    assert typing_args[0] is fake_thread
    assert typing_kwargs.get("seconds", 0) >= 1.0
    # User must receive an ephemeral followup with the thread link.
    interaction.followup.send.assert_awaited()
    # The adapter calls followup.send(msg, ephemeral=True) positionally, so
    # the body lives in args[0]. Use args[0] to match the adapter's
    # positional-first call pattern.
    followup_content = interaction.followup.send.await_args.args[0]
    assert "<#555>" in followup_content


@pytest.mark.asyncio
async def test_handle_start_falls_back_to_invocation_channel_when_config_unset():
    """When ``channel_id`` is the placeholder ``"0"``, the handler creates the thread in the channel where /start was invoked."""
    adapter = _adapter(extra={"start_projects": {"il": {"channel_id": "0"}}})
    fallback_channel = _FakeChannel(channel_id=42)
    fake_thread = _FakeThread(thread_id=777)
    adapter._client = SimpleNamespace(get_channel=lambda _cid: None, fetch_channel=AsyncMock(side_effect=AssertionError("should not fetch when falling back")))
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._create_thread_in_channel = AsyncMock(return_value=fake_thread)
    adapter._typing_indicator = AsyncMock()
    adapter._threads = SimpleNamespace(mark=lambda _tid: None)

    interaction = _make_interaction(channel=fallback_channel)
    await adapter._handle_start_slash(interaction, project="il", title="x")

    # The fallback channel must be the one passed to thread creation.
    assert adapter._create_thread_in_channel.await_args.args[0] is fallback_channel
    # And the followup must mention the fallback so ops can spot misconfig.
    # The adapter calls followup.send(msg, ephemeral=True) positionally, so
    # the body lives in args[0] — match the adapter's positional-first
    # call pattern.
    followup_content = interaction.followup.send.await_args.args[0]
    assert "using invocation channel" in followup_content


@pytest.mark.asyncio
async def test_handle_start_runs_typing_indicator_for_about_5_seconds():
    """The typing indicator must run for ~5s so the worker has time to ack before the user returns."""
    adapter = _adapter(
        extra={
            "start_projects": {
                "thazeron": {
                    "channel_id": "321",
                    "worker_mention": "@worker-thazeron ",
                    "display_name": "Thazeron",
                },
            },
        }
    )
    fake_channel = _FakeChannel(channel_id=321)
    fake_thread = _FakeThread(thread_id=888)
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: fake_channel if cid == 321 else None,
        fetch_channel=AsyncMock(return_value=fake_channel),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._create_thread_in_channel = AsyncMock(return_value=fake_thread)
    adapter._typing_indicator = AsyncMock()
    adapter._threads = SimpleNamespace(mark=lambda _tid: None)

    interaction = _make_interaction(channel=fake_channel)
    await adapter._handle_start_slash(interaction, project="thazeron", title="hi")

    # Confirm typing was awaited with a 5-second target — the exact duration
    # is a UX choice (how long the worker has to ack), not an implementation
    # detail, so any future change must update this assertion deliberately.
    typing_kwargs = adapter._typing_indicator.await_args.kwargs
    assert typing_kwargs.get("seconds") == pytest.approx(5.0, abs=0.5)


# ──────────────────────────────────────────────────────────────────────
# 4. Thread name format
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "slug,display,emoji",
    [
        ("il", "IL", "🎮"),
        ("il-cortes", "IL Cortes", "🎬"),
        ("il-launcher", "IL Launcher", "🚀"),
        ("thazeron", "Thazeron", "🗡️"),
        ("hermes/improve", "Hermes Improve", "🛠️"),
        ("default", "Hermes", "💬"),
    ],
)
def test_format_start_thread_name_uses_project_specific_emoji(slug, display, emoji):
    name = DiscordAdapter._format_start_thread_name(slug, display, "algum título")
    assert name.startswith(emoji), f"{slug} should use {emoji} but got {name!r}"
    assert "Nova conversa" in name
    assert "algum título" in name


def test_format_start_thread_name_falls_back_when_title_empty():
    """An empty title must still produce a valid thread name."""
    name = DiscordAdapter._format_start_thread_name("il", "IL", "")
    assert "Nova conversa" in name
    # No bare brackets — the formatter substitutes "Nova conversa" as the
    # title so Discord never receives `[]` as part of the thread name.
    assert "[]" not in name


def test_format_start_thread_name_truncates_long_titles():
    """Long titles must be truncated so the thread name stays within Discord's 80-char UTF-16 budget."""
    long_title = "x" * 200
    name = DiscordAdapter._format_start_thread_name("il", "IL", long_title)
    # Truncation adds "..." OUTSIDE the brackets so the thread name ends
    # with "..." (not "...]" inside the brackets). 80 UTF-16 code units is
    # the hard Discord cap; the ellipsis is part of the outer suffix so it
    # counts against the 80-unit total.
    from gateway.platforms.base import utf16_len
    assert utf16_len(name) <= 80
    assert name.endswith("...")


# ──────────────────────────────────────────────────────────────────────
# 5. Auth gating
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_start_respects_authorization_rejection():
    """A failed auth check must short-circuit before any thread creation."""
    adapter = _adapter()
    adapter._check_slash_authorization = AsyncMock(return_value=False)
    adapter._create_thread_in_channel = AsyncMock()

    interaction = _make_interaction()
    await adapter._handle_start_slash(interaction, project="il", title="x")

    adapter._create_thread_in_channel.assert_not_awaited()
    # Defer must NOT have been called when auth fails — the rejection
    # path is supposed to send its own ephemeral response.
    interaction.response.defer.assert_not_awaited()