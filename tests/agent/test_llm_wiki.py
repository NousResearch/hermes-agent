from pathlib import Path

from agent.llm_wiki import (
    DEFAULT_WIKI_PATH,
    classify_memory_entry,
    distill_raw_sources_to_wiki,
    ensure_wiki_scaffold,
    get_configured_wiki_path,
    render_wiki_prefetch,
    retrieve_relevant_wiki_pages,
    sync_memory_store_to_wiki,
)


def _write_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "queries").mkdir(parents=True)
    (wiki / "SCHEMA.md").write_text("# Wiki Schema\n", encoding="utf-8")
    (wiki / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
    (wiki / "log.md").write_text("# Wiki Log\n", encoding="utf-8")
    (wiki / "entities" / "memory-v2.md").write_text(
        "---\n"
        "title: Memory V2\n"
        "created: 2026-04-07\n"
        "updated: 2026-04-07\n"
        "type: entity\n"
        "tags: [memory, sqlite]\n"
        "sources: []\n"
        "---\n\n"
        "Memory v2 uses a local sqlite store as the durable source of truth.\n\n"
        "## Links\n\n"
        "- [[llm-wiki-layer]]\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "llm-wiki-layer.md").write_text(
        "---\n"
        "title: LLM Wiki Layer\n"
        "created: 2026-04-07\n"
        "updated: 2026-04-07\n"
        "type: concept\n"
        "tags: [wiki, memory]\n"
        "sources: []\n"
        "---\n\n"
        "The wiki layer compiles durable markdown knowledge for recall before model calls.\n",
        encoding="utf-8",
    )
    return wiki


def test_get_configured_wiki_path_defaults_without_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert get_configured_wiki_path() == DEFAULT_WIKI_PATH


def test_get_configured_wiki_path_reads_skill_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "custom-wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert get_configured_wiki_path() == wiki


def test_ensure_wiki_scaffold_creates_core_files(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    ensure_wiki_scaffold()

    assert (wiki / "SCHEMA.md").exists()
    assert (wiki / "index.md").exists()
    assert (wiki / "log.md").exists()
    assert (wiki / "concepts" / "memory-v2.md").exists()
    assert (wiki / "concepts" / "llm-wiki-memory-lane.md").exists()


def test_classify_memory_entry_routes_into_first_class_kinds():
    assert classify_memory_entry("User prefers concise answers", target="user") == "preferences"
    assert classify_memory_entry("Never use hidden paid fallbacks", target="user") == "prohibitions"
    assert classify_memory_entry("Use explicit routing. Crew/TARS routing can stay as-is.", target="user") == "project-conventions"
    assert classify_memory_entry("server A runs nginx", target="memory") == "environment-facts"
    assert classify_memory_entry("If user gives a likely path, check it first.", target="user") == "workflow-rules"


def test_sync_memory_store_to_wiki_builds_compiled_pages(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    changed = sync_memory_store_to_wiki(
        ["Memory v2 uses sqlite durability", "Project path lives under Desktop/"],
        ["User prefers concise answers"],
    )

    assert changed
    memory_page = wiki / "concepts" / "persistent-memory-notes.md"
    profile_page = wiki / "entities" / "user-profile.md"
    preferences_page = wiki / "concepts" / "preferences.md"
    env_page = wiki / "concepts" / "environment-facts.md"
    topic_pages = list((wiki / "queries").glob("memory-topic-*.md"))

    assert "Memory v2 uses sqlite durability" in memory_page.read_text(encoding="utf-8")
    assert "User prefers concise answers" in profile_page.read_text(encoding="utf-8")
    assert "[[memory-v2]]" in memory_page.read_text(encoding="utf-8")
    assert "User prefers concise answers" in preferences_page.read_text(encoding="utf-8")
    assert "Project path lives under Desktop/" in env_page.read_text(encoding="utf-8")
    assert topic_pages
    assert any("User prefers concise answers" in page.read_text(encoding="utf-8") for page in topic_pages)


def test_sync_memory_store_to_wiki_removes_stale_topic_pages_on_replace_or_remove(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    sync_memory_store_to_wiki(["Python 3.11 project"], [])
    before = {path.name for path in (wiki / "queries").glob("memory-topic-*.md")}
    assert before

    sync_memory_store_to_wiki(["Python 3.12 project"], [])
    after = {path.name for path in (wiki / "queries").glob("memory-topic-*.md")}

    assert after
    assert before.isdisjoint(after)

    sync_memory_store_to_wiki([], [])
    assert list((wiki / "queries").glob("memory-topic-*.md")) == []


def test_sync_memory_store_to_wiki_renders_provenance_blocks(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    sync_memory_store_to_wiki(
        [],
        ["Never use hidden paid fallbacks"],
        user_records=[{
            "id": 7,
            "content": "Never use hidden paid fallbacks",
            "kind": "project-conventions",
            "strength": "hard_rule",
            "created_in_session_id": "sess-a",
            "replaced_by": 8,
            "forgotten_by": None,
            "status": "superseded",
            "target": "user",
        }],
    )

    conventions = (wiki / "concepts" / "project-conventions.md").read_text(encoding="utf-8")
    topic_page = next((wiki / "queries").glob("memory-topic-*.md")).read_text(encoding="utf-8")

    assert "## Provenance" in conventions
    assert "row_id=7" in conventions
    assert "session=sess-a" in conventions
    assert "strength=hard_rule" in conventions
    assert "replaced_by=8" in conventions
    assert "## Provenance" in topic_page
    assert "status=superseded" in topic_page


def test_retrieve_relevant_wiki_pages_prefers_matching_pages(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = _write_wiki(tmp_path)
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    hits = retrieve_relevant_wiki_pages("how does sqlite memory v2 work", limit=2)

    assert len(hits) == 2
    assert hits[0]["title"] == "Memory V2"
    assert hits[0]["path"] == "entities/memory-v2.md"
    assert "sqlite store" in hits[0]["summary"].lower()


def test_render_wiki_prefetch_returns_compact_block(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = _write_wiki(tmp_path)
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    block = render_wiki_prefetch("wiki recall for sqlite memory")

    assert "## LLM Wiki Recall" in block
    assert "Memory V2 (entities/memory-v2.md)" in block
    assert "LLM Wiki Layer (concepts/llm-wiki-layer.md)" in block


def test_render_wiki_prefetch_can_surface_typed_pages(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    sync_memory_store_to_wiki(
        ["server A runs nginx"],
        ["Never use flattery or padding", "If user gives a likely path, check it first."],
    )

    block = render_wiki_prefetch("what are the workflow rules and prohibitions", limit=4)

    assert "Workflow Rules (concepts/workflow-rules.md)" in block
    assert "Prohibitions (concepts/prohibitions.md)" in block


def test_distill_raw_sources_to_wiki_creates_query_pages(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    ensure_wiki_scaffold(wiki)

    article = wiki / "raw" / "articles" / "open-source-llms-2026.md"
    transcript = wiki / "raw" / "transcripts" / "tim-dillon-skid-ray.md"
    article.write_text(
        "# The Best Open-Source LLMs in 2026\n\n"
        "## Core takeaway\n"
        "Open-weight models are competitive when routing and inference optimization are handled well.\n\n"
        "## Notable points\n"
        "- adaptation matters more than raw model bragging\n"
        "- self-hosting reduces lock-in\n",
        encoding="utf-8",
    )
    transcript.write_text(
        "# The Tim Dillon Show – Skid Ray\n\n"
        "## Why this belongs here\n"
        "Strong Tim/Ray energy reference set: absurd escalation and anti-bullshit cadence.\n\n"
        "## Notes for shell/persona study\n"
        "- concrete detail first, worldview second\n"
        "- absurd specificity without losing the point\n",
        encoding="utf-8",
    )

    changed = distill_raw_sources_to_wiki(wiki)
    assert changed

    pages = list((wiki / "queries").glob("raw-distill-*.md"))
    assert len(pages) == 2
    article_page = next(p for p in pages if "open-source-llms-2026" in p.name)
    text = article_page.read_text(encoding="utf-8")
    assert "## Distilled takeaways" in text
    assert "adaptation matters more than raw model bragging" in text
    assert "## Routing/build implications" in text
    assert "source_file=raw/articles/open-source-llms-2026.md" in text


def test_distill_raw_sources_to_wiki_is_idempotent(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    ensure_wiki_scaffold(wiki)

    article = wiki / "raw" / "articles" / "signal-pack.md"
    article.write_text("# Signal Pack\n\n- one\n- two\n", encoding="utf-8")

    first = distill_raw_sources_to_wiki(wiki)
    second = distill_raw_sources_to_wiki(wiki)

    assert first
    assert second == []


def test_retrieve_relevant_wiki_pages_prefers_hard_rule_over_soft_rule_on_tie(monkeypatch, tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(parents=True)
    wiki = tmp_path / "wiki"
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  config:\n    wiki:\n      path: {wiki}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    ensure_wiki_scaffold(wiki)
    soft = wiki / "concepts" / "zzz-soft-routing.md"
    soft.write_text(
        "---\n"
        "title: Soft Routing Note\n"
        "created: 2026-04-08\n"
        "updated: 2026-04-08\n"
        "type: concept\n"
        "tags: [memory, project, convention]\n"
        "sources: [memories/USER.md]\n"
        "generated_by: memory-mirror-v2\n"
        "memory_kind: project-conventions\n"
        "---\n\n"
        "Use explicit routing for paid fallbacks.\n\n"
        "## Provenance\n\n"
        "- Use explicit routing for paid fallbacks. :: row_id=1 | status=active | strength=soft_rule | session=sess-a | source_file=memories/USER.md\n",
        encoding="utf-8",
    )
    hard = wiki / "concepts" / "aaa-hard-routing.md"
    hard.write_text(
        "---\n"
        "title: Hard Routing Rule\n"
        "created: 2026-04-08\n"
        "updated: 2026-04-08\n"
        "type: concept\n"
        "tags: [memory, project, convention]\n"
        "sources: [memories/USER.md]\n"
        "generated_by: memory-mirror-v2\n"
        "memory_kind: project-conventions\n"
        "---\n\n"
        "Use explicit routing for paid fallbacks.\n\n"
        "## Provenance\n\n"
        "- Use explicit routing for paid fallbacks. :: row_id=2 | status=active | strength=hard_rule | session=sess-b | source_file=memories/USER.md\n",
        encoding="utf-8",
    )

    hits = retrieve_relevant_wiki_pages("explicit routing paid fallbacks", limit=2)
    assert hits
    assert hits[0]["path"] == "concepts/aaa-hard-routing.md"
