"""Tests for local Discord thread label registry support."""

from pathlib import Path
from types import SimpleNamespace
import tomllib

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.thread_labels import (
    DISCORD_THREAD_LABELS,
    extract_thread_labels,
)


def test_discord_thread_label_registry_includes_wt_labels():
    assert "wtupdate" in DISCORD_THREAD_LABELS
    assert "wtsignoff" in DISCORD_THREAD_LABELS
    assert DISCORD_THREAD_LABELS["wtupdate"]["scope"] == "worktree"
    assert DISCORD_THREAD_LABELS["wtsignoff"]["scope"] == "worktree"


def test_discord_thread_label_registry_json_is_packaged():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        "platforms/discord/thread_labels.json"
        in pyproject["tool"]["setuptools"]["package-data"]["plugins"]
    )


def test_discord_thread_labels_extracted_from_thread_name_prefixes():
    labels = extract_thread_labels("[wtupdate] [wtsignoff] Ship the Discord patch")

    assert [label["id"] for label in labels] == ["wtupdate", "wtsignoff"]
    assert labels[0]["display"] == "WT Update"


def test_discord_thread_labels_are_added_to_formatted_chat_name():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    parent = SimpleNamespace(id=123, name="dev", guild=SimpleNamespace(name="Hermes Server"))
    thread = SimpleNamespace(
        id=456,
        name="[wtupdate] Routing status",
        parent=parent,
        parent_id=123,
        guild=parent.guild,
    )

    assert (
        adapter._format_thread_chat_name(thread)
        == "Hermes Server / #dev / [wtupdate] Routing status [labels: wtupdate]"
    )
