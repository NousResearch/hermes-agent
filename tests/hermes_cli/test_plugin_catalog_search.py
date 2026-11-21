"""Catalog search matching (hermes_cli/plugin_catalog.py::filter_entries): the query is split into
words, so a space-separated phrase like "make plain" finds the hyphenated catalog name
``hermes-make-plain``, and a name hit ranks ahead of a description-only hit."""

from __future__ import annotations

from hermes_cli import plugin_catalog as pc

SHA = "38fe0fb53eff98d477f807432e965429e665ca33"


def _entry(name: str, description: str = "d", tools: tuple[str, ...] = ()):
    data = {"name": name, "repo": f"https://github.com/owner/{name}", "sha": SHA,
            "description": description, "maintainer": "owner", "tier": "community",
            "capabilities": {"provides_tools": list(tools)}}
    entry = pc.entry_from_mapping(data, name)
    assert entry is not None
    return entry


def test_space_separated_words_match_a_hyphenated_name():
    entries = [_entry("hermes-make-plain"), _entry("unrelated")]
    assert [e.name for e in pc.filter_entries(entries, "make plain")] == ["hermes-make-plain"]
    assert [e.name for e in pc.filter_entries(entries, "hermes make plain")] == ["hermes-make-plain"]


def test_name_hit_ranks_before_description_hit():
    entries = [_entry("alpha", description="make plain things"), _entry("hermes-make-plain")]
    assert next(e.name for e in pc.filter_entries(entries, "plain")) == "hermes-make-plain"


def test_description_and_tool_hits_still_match():
    entries = [_entry("alpha", description="rewrites a draft"), _entry("beta", tools=("make_plain",))]
    assert [e.name for e in pc.filter_entries(entries, "rewrites")] == ["alpha"]
    assert [e.name for e in pc.filter_entries(entries, "make plain")] == ["beta"]


def test_empty_query_returns_every_entry():
    entries = [_entry("a"), _entry("b")]
    assert [e.name for e in pc.filter_entries(entries, "  ")] == ["a", "b"]
