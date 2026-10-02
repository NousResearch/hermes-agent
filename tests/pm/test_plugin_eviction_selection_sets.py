"""Eviction publishes ONE selection, not two disagreeing lists (#131216).

``plugins.enabled`` and ``plugins.disabled`` are opposite halves of the same
choice, and every activation surface (``_activate_key``) writes both halves in
one edit. Eviction used to append to the deny-list alone, which could leave an
evicted key in both lists: the deny-list renders it disabled while the
allow-list still counts it, so the only route back is a manual re-enable.

The contract exercised here: for every evicted key the published config has
that key (and its bare leaf) out of ``plugins.enabled`` and into
``plugins.disabled``, and nothing else moves.
"""
from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml


def _write_selection(home: Path, selection: list[str], provider: str | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config: dict = {"plugins": {"enabled": selection}}
    if provider is not None:
        config["memory"] = {"provider": provider}
    with (home / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream)


def _evict(home: Path, selection: list[str], evicted: dict[str, str]) -> dict:
    """Publish the real PluginEviction edit for *selection*; return the parsed config."""
    from pm.plugin_eviction import PluginEviction

    entries = [(home / "plugins", name, home / "plugins" / name) for name in selection]
    reasons = {(home / "plugins" / name).resolve(): reason for name, reason in evicted.items()}
    PluginEviction(entries, reasons).publish(home)
    with (home / "config.yaml").open(encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def test_evicted_key_leaves_the_allow_list_it_was_published_into(tmp_path, monkeypatch):
    """An evicted key cannot stay in plugins.enabled beside its new deny entry."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    selection = ["hindsight", "anysearch", "observability/langfuse"]
    _write_selection(home, selection)

    plugins = _evict(home, selection, {
        "hindsight": "the dependency environment no longer builds with it: pilk==0.2.4",
    })["plugins"]

    assert "hindsight" in plugins["disabled"]
    assert not set(plugins["enabled"]) & set(plugins["disabled"]), (
        "an evicted key left in both lists reads as disabled while the allow-list still "
        f"counts it: enabled={plugins['enabled']} disabled={plugins['disabled']}"
    )
    # The rest of the selection survives, in config order.
    assert plugins["enabled"] == ["anysearch", "observability/langfuse"]


def test_eviction_purges_a_legacy_leaf_spelling_alongside_the_canonical_key(tmp_path, monkeypatch):
    """A bare-leaf entry names the same plugin: it cannot outlive the canonical key."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    selection = ["observability/langfuse", "langfuse", "anysearch"]
    _write_selection(home, selection)

    plugins = _evict(home, selection, {
        "observability/langfuse": "its dependency environment no longer builds with it",
    })["plugins"]

    assert plugins["disabled"] == ["observability/langfuse"]
    assert plugins["enabled"] == ["anysearch"], (
        "the legacy bare-leaf spelling survived eviction, so the plugin stays a member "
        "of the allow-list its canonical key just left"
    )