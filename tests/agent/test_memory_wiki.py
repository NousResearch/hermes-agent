"""Behavior contracts for the local memory-wiki index."""

from __future__ import annotations

import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _with_home(home, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_build_memory_wiki_index_splits_and_classifies_entries(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text(
        "Project uses pytest with xdist\n§\nDecision: keep memory wiki stdlib-only",
        encoding="utf-8",
    )
    (mem_dir / "USER.md").write_text(
        "User prefers concise responses\n§\nConstraint: only show SF events",
        encoding="utf-8",
    )

    def run():
        from agent.memory_wiki import build_memory_wiki_index

        return build_memory_wiki_index()

    index = _with_home(home, run)

    assert index["version"] == 1
    assert index["stats"] == {"entries": 4, "sources": {"memory": 2, "user": 2}}
    entries = index["entries"]
    assert [entry["source"] for entry in entries] == ["memory", "memory", "user", "user"]
    assert {entry["category"] for entry in entries} >= {"environment", "decision", "preference", "constraint"}
    assert all(entry["id"] for entry in entries)
    assert all(entry["keywords"] for entry in entries)
    assert entries[0]["provenance"]["path"].endswith("MEMORY.md")


def test_select_memory_context_is_query_relevant_and_budgeted(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text(
        "Cloudflare credentials live in a secured file and should stay redacted\n§\nHermes code uses pytest and ruff",
        encoding="utf-8",
    )
    (mem_dir / "USER.md").write_text(
        "User prefers beautiful UI polish",
        encoding="utf-8",
    )

    def run():
        from agent.memory_wiki import build_memory_wiki_index, select_memory_context

        index = build_memory_wiki_index()
        return select_memory_context("run pytest for hermes code", index=index, max_chars=160)

    selected = _with_home(home, run)

    assert selected["used_chars"] <= 160
    assert selected["entries"]
    rendered = selected["context"]
    assert rendered.startswith("<memory-wiki-context>\n")
    assert rendered.endswith("\n</memory-wiki-context>")
    assert "pytest" in rendered.lower()
    assert "Cloudflare" not in rendered


def test_memory_wiki_index_json_round_trips(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Stable fact about the environment", encoding="utf-8")

    def run():
        from agent.memory_wiki import build_memory_wiki_index

        return build_memory_wiki_index()

    index = _with_home(home, run)
    decoded = json.loads(json.dumps(index))

    assert decoded["entries"][0]["text"] == "Stable fact about the environment"
    assert decoded["entries"][0]["category"] == "fact"


def test_empty_memory_wiki_index_shape(tmp_path):
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)

    def run():
        from agent.memory_wiki import build_memory_wiki_index, select_memory_context

        index = build_memory_wiki_index()
        selected = select_memory_context("anything", index=index, max_chars=100)
        return index, selected

    index, selected = _with_home(home, run)

    assert index["entries"] == []
    assert index["stats"]["entries"] == 0
    assert selected == {"entries": [], "context": "", "used_chars": 0}
