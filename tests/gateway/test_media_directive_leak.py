"""Regression tests for issue #86122: undeliverable ``MEDIA:`` directives must
never leak as raw text to Discord (or any platform).

A ``MEDIA:<path>`` directive that cannot be delivered — placeholder, hallucinated
path, missing file, or denied prefix — must be replaced with a sanitized
``<attachment unavailable>`` token instead of surfacing an absolute filesystem
path to the user.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    neutralize_undeliverable_media_tags,
)


# ---------------------------------------------------------------------------
# Sync unit tests — neutralize_undeliverable_media_tags
# ---------------------------------------------------------------------------

def test_placeholder_tag_neutralized():
    assert (
        neutralize_undeliverable_media_tags("MEDIA:<local-file-path>")
        == "<attachment unavailable>"
    )


def test_system_prompt_example_neutralized():
    # The system prompt teaches ``MEDIA:/absolute/path/to/file`` — a model that
    # echoes it verbatim must not leak the literal text.
    out = neutralize_undeliverable_media_tags("MEDIA:/absolute/path/to/file")
    assert out == "<attachment unavailable>"


def test_denied_prefix_neutralized():
    assert neutralize_undeliverable_media_tags("MEDIA:/etc/passwd") == "<attachment unavailable>"
    assert neutralize_undeliverable_media_tags("MEDIA:~/.ssh/id_rsa") == "<attachment unavailable>"


def test_missing_known_extension_stripped():
    # A known-extension tag whose file is missing is stripped (no token, no leak).
    out = neutralize_undeliverable_media_tags("See MEDIA:/tmp/definitely/missing.png here")
    assert "MEDIA:" not in out
    assert "/tmp/definitely/missing.png" not in out


def test_text_only_unchanged():
    assert neutralize_undeliverable_media_tags("Hello, world!") == "Hello, world!"


def test_prose_mention_of_media_keyword_preserved():
    # A bare ``MEDIA:`` keyword not followed by a path/placeholder is ordinary
    # prose and must be left untouched.
    text = "use the MEDIA: syntax to attach files"
    assert neutralize_undeliverable_media_tags(text) == text


def test_split_tag_prefix_neutralized():
    # A MEDIA: tag split across stream chunks: the prefix chunk's MEDIA: keyword
    # and its partial anchored path are neutralized, so no absolute path leaks.
    prefix = "Here is your file: MEDIA:/tmp/a/very/long/path/fi"
    out = neutralize_undeliverable_media_tags(prefix)
    assert "MEDIA:" not in out
    assert "/tmp/a/very/long/path/fi" not in out


def test_emphasis_wrapped_neutralized():
    assert (
        neutralize_undeliverable_media_tags("**MEDIA:<local-file-path>**")
        == "<attachment unavailable>"
    )
    assert neutralize_undeliverable_media_tags("*MEDIA:/etc/passwd*") == "<attachment unavailable>"


def test_protected_code_block_preserved():
    text = "```\nMEDIA:/etc/passwd\n```"
    assert neutralize_undeliverable_media_tags(text) == text


def test_protected_inline_code_preserved():
    text = "see `MEDIA:/etc/passwd` in docs"
    assert neutralize_undeliverable_media_tags(text) == text


def test_protected_json_string_preserved():
    text = '{"result": "MEDIA:/tmp/stale.png"}'
    assert neutralize_undeliverable_media_tags(text) == text


def test_absolute_path_never_leaks():
    for path in ("/etc/passwd", "~/.ssh/id_rsa", "/etc/shadow", "~/.aws/credentials"):
        out = neutralize_undeliverable_media_tags(f"MEDIA:{path}")
        assert path not in out
        assert "MEDIA:" not in out


def test_display_path_neutralizes():
    out = BasePlatformAdapter.strip_media_directives_for_display("got MEDIA:<local-file-path>")
    assert "MEDIA:" not in out
    assert "<attachment unavailable>" in out


def test_existing_extensionless_file_stripped(tmp_path):
    # An existing, non-denied extension-less file validates and is stripped
    # (delivered upstream), not tokenized.
    f = tmp_path / "Caddyfile"
    f.write_text("x")
    out = neutralize_undeliverable_media_tags(f"MEDIA:{f}")
    assert "MEDIA:" not in out
    assert str(f) not in out
    assert "<attachment unavailable>" not in out


# ---------------------------------------------------------------------------
# Discord adapter backstop (async)
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


@pytest.mark.asyncio
async def test_discord_send_neutralizes_undeliverable_media(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    captured = {}

    async def fake_send(*, content, reference=None):
        captured["content"] = content
        return SimpleNamespace(id=888)

    channel = SimpleNamespace(send=fake_send)
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    await adapter.send("555", "Here MEDIA:<local-file-path> is your file.")

    assert "local-file-path" not in captured["content"]
    assert "<attachment unavailable>" in captured["content"]


@pytest.mark.asyncio
async def test_discord_forum_thread_name_neutralized(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    captured = {}

    async def fake_create_thread(*, name, content, **kw):
        captured["name"] = name
        captured["content"] = content
        thread = SimpleNamespace(id=999)
        thread.message = SimpleNamespace(id=999)
        return thread

    forum = SimpleNamespace(type=15, create_thread=fake_create_thread, send=AsyncMock())
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=forum),
        fetch_channel=AsyncMock(),
    )

    await adapter.send("555", "MEDIA:<local-file-path>\nbody text")

    assert "local-file-path" not in captured["name"]
    assert "<attachment unavailable>" in captured["name"]
