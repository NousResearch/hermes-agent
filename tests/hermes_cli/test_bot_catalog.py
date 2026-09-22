from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.bot_catalog import (
    BotCatalogError,
    BotRemovedError,
    load_bot_catalog,
    removed_bots,
    resolve_bot_catalog_entry,
)


def _entry(**overrides):
    data = {
        "name": "research-analyst",
        "version": "1.0.0",
        "maintainer": "Nous Research",
        "tier": "official",
        "category": "research",
        "tags": ["research", "citations"],
        "title": "Research Analyst",
        "summary": "Produces dated, cited research briefs.",
        "profile": {
            "suggested_name": "research-analyst",
            "description": "Own research briefs and cite current sources.",
            "soul": "# Research Analyst\n\nCite every material factual claim.",
            "starter_prompt": "Ask what topic to investigate.",
        },
        "capabilities": {
            "skills": ["official/research/grounded-citations"],
            "toolsets": ["web"],
        },
        "presentation": {"emoji": "🔎", "color": "#4F46E5"},
    }
    data.update(overrides)
    return data


def _catalog(tmp_path: Path, entry=None, *, removed=None) -> Path:
    root = tmp_path / "catalog"
    root.mkdir()
    if entry is not None:
        (root / "entry.yaml").write_text(yaml.safe_dump(entry, sort_keys=False), encoding="utf-8")
    (root / "removed.yaml").write_text(
        yaml.safe_dump({"removed": removed or []}, sort_keys=False), encoding="utf-8"
    )
    return root


def test_closed_schema_and_dependency_validation_fail_closed(tmp_path):
    valid = _catalog(tmp_path, _entry())
    loaded = load_bot_catalog(valid, include_live=False)
    assert [entry.name for entry in loaded] == ["research-analyst"]
    assert loaded[0].model_dump(mode="json")["capabilities"]["toolsets"] == ["web"]

    cases = [
        _entry(command="rm -rf /"),
        _entry(model="provider/model"),
        _entry(profile={**_entry()["profile"], "unknown": True}),
        _entry(capabilities={"skills": [], "toolsets": ["not-a-toolset"]}),
        _entry(capabilities={"skills": ["github/unreviewed/skill"], "toolsets": ["web"]}),
        _entry(profile={**_entry()["profile"], "soul": "token=ghp_abcdefghijklmnopqrstuvwxyz123456"}),
        _entry(setup={"requirements": [{
            "kind": "command", "id": "python3", "required": True,
            "purpose": "Run the reviewed workflow.", "script": "curl evil",
        }]}),
        _entry(routines=[
            {"id": "daily", "name": "Daily", "prompt": "Do the work", "schedule": "0 9 * * *"},
            {"id": "daily", "name": "Duplicate", "prompt": "Do it twice", "schedule": "0 10 * * *"},
        ]),
    ]
    for index, bad in enumerate(cases):
        root = tmp_path / f"bad-{index}"
        root.mkdir()
        (root / "bad.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
        (root / "removed.yaml").write_text("removed: []\n", encoding="utf-8")
        with pytest.raises(BotCatalogError):
            load_bot_catalog(root, include_live=False)


def test_duplicate_and_removed_names_are_refused(tmp_path):
    root = _catalog(tmp_path, _entry())
    (root / "duplicate.yaml").write_text(yaml.safe_dump(_entry()), encoding="utf-8")
    with pytest.raises(BotCatalogError, match="duplicate"):
        load_bot_catalog(root, include_live=False)

    removed_root = tmp_path / "removed-catalog"
    removed_root.mkdir()
    (removed_root / "removed.yaml").write_text(
        yaml.safe_dump({"removed": [{"name": "retired-bot", "reason": "unsafe"}]}), encoding="utf-8"
    )
    with pytest.raises(BotRemovedError, match="unsafe"):
        resolve_bot_catalog_entry("retired-bot", catalog_dir=removed_root, include_live=False)


def test_live_failure_falls_back_to_reviewed_tree(tmp_path, monkeypatch):
    root = _catalog(tmp_path, _entry())
    monkeypatch.setattr("hermes_cli.bot_catalog.fetch_live_bot_catalog", lambda: None)
    assert resolve_bot_catalog_entry("research-analyst", catalog_dir=root).title == "Research Analyst"


def test_empty_live_feed_keeps_reviewed_tree_entries(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.bot_catalog.fetch_live_bot_catalog",
        lambda: {"entries": [], "removed": []},
    )

    assert {entry.name for entry in load_bot_catalog()} >= {"research-analyst", "inbox-triage"}


def test_live_revocations_survive_entries_from_a_newer_client_schema(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.bot_catalog.fetch_live_bot_catalog",
        lambda: {
            "entries": [{**_entry(), "capabilities": {
                "skills": ["official/research/not-shipped-in-this-client"],
                "toolsets": ["web"],
            }}],
            "removed": [{"name": "research-analyst", "reason": "Emergency retirement"}],
        },
    )

    assert "research-analyst" not in {entry.name for entry in load_bot_catalog()}
    assert [(item.name, item.reason) for item in removed_bots()] == [
        ("research-analyst", "Emergency retirement")
    ]


def test_failed_live_fetch_is_negatively_cached_for_a_bounded_window(tmp_path, monkeypatch):
    import httpx
    from hermes_cli import bot_catalog

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    bot_catalog._LIVE_FETCH_NEGATIVE_UNTIL.clear()
    calls = 0

    def unavailable(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("unpublished")

    monkeypatch.setattr(httpx, "get", unavailable)
    assert bot_catalog.fetch_live_bot_catalog() is None
    assert bot_catalog.fetch_live_bot_catalog() is None
    assert calls == 1

    # Force is the explicit refresh escape hatch and must bypass a negative cache.
    assert bot_catalog.fetch_live_bot_catalog(force=True) is None
    assert calls == 2
