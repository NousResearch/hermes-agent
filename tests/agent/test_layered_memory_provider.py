import json
import sqlite3
from pathlib import Path

from plugins.memory import discover_memory_providers, load_memory_provider


class TestLayeredMemoryProviderDiscovery:
    def test_discover_finds_layered_provider(self):
        providers = discover_memory_providers()
        names = [name for name, _, _ in providers]
        assert "layered" in names

    def test_load_provider_by_name(self):
        provider = load_memory_provider("layered")
        assert provider is not None
        assert provider.name == "layered"
        assert provider.is_available()


class TestLayeredMemoryProviderInitialization:
    def test_initialize_creates_local_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")

        provider.initialize(session_id="sess-init", hermes_home=str(tmp_path), platform="cli")

        db_path = tmp_path / "memory" / "layered_memory.db"
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
        finally:
            conn.close()

        assert "memory_items" in tables
        assert "memory_items_fts" in tables

    def test_system_prompt_block_reports_store_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-prompt", hermes_home=str(tmp_path), platform="cli")

        prompt = provider.system_prompt_block()

        assert "Layered Memory" in prompt
        assert "identity" in prompt.lower()
        assert "semantic" in prompt.lower()


class TestLayeredMemoryProviderWrites:
    def test_on_memory_write_mirrors_user_and_memory_targets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-write", hermes_home=str(tmp_path), platform="cli")

        provider.on_memory_write("add", "user", "User prefers control-theory style collaboration")
        provider.on_memory_write("add", "memory", "~/.openclaw is reference-only")

        rows = provider._conn.execute(
            "SELECT layer, content, source, action FROM memory_items ORDER BY id"
        ).fetchall()

        assert rows[0] == (
            "identity_core",
            "User prefers control-theory style collaboration",
            "builtin_memory",
            "add",
        )
        assert rows[1] == (
            "semantic",
            "~/.openclaw is reference-only",
            "builtin_memory",
            "add",
        )

    def test_replace_creates_superseding_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-replace", hermes_home=str(tmp_path), platform="cli")

        provider.on_memory_write("add", "user", "Call the assistant Hermes")
        provider.on_memory_write("replace", "user", "Call the assistant JARVIS")

        rows = provider._conn.execute(
            "SELECT layer, content, action, supersedes_id FROM memory_items ORDER BY id"
        ).fetchall()

        assert len(rows) == 2
        assert rows[0] == ("identity_core", "Call the assistant Hermes", "add", None)
        assert rows[1][0] == "identity_core"
        assert rows[1][1] == "Call the assistant JARVIS"
        assert rows[1][2] == "replace"
        assert rows[1][3] == 1


class TestLayeredMemoryProviderTurnAndSessionHooks:
    def test_sync_turn_persists_archive_records(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-sync", hermes_home=str(tmp_path), platform="cli")

        provider.sync_turn("How does memory work?", "It is layered.")

        rows = provider._conn.execute(
            "SELECT layer, role, content FROM memory_items ORDER BY id"
        ).fetchall()

        assert rows == [
            ("archive", "user", "How does memory work?"),
            ("archive", "assistant", "It is layered."),
        ]

    def test_on_pre_compress_creates_checkpoint_and_returns_note(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-compress", hermes_home=str(tmp_path), platform="cli")

        messages = [
            {"role": "user", "content": "We discussed layered memory."},
            {"role": "assistant", "content": "Use episodic and semantic separation."},
        ]
        note = provider.on_pre_compress(messages)

        row = provider._conn.execute(
            "SELECT layer, role, content FROM memory_items ORDER BY id DESC LIMIT 1"
        ).fetchone()

        assert row[0] == "episodic"
        assert row[1] == "summary"
        assert "layered memory" in row[2].lower()
        assert "Preserve this checkpoint" in note

    def test_on_session_end_creates_session_summary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-end", hermes_home=str(tmp_path), platform="cli")

        messages = [
            {"role": "user", "content": "Implement a layered memory provider."},
            {"role": "assistant", "content": "I will add tests and SQLite persistence."},
            {"role": "user", "content": "Keep built-in memory as the hot layer."},
        ]
        provider.on_session_end(messages)

        row = provider._conn.execute(
            "SELECT layer, role, content FROM memory_items ORDER BY id DESC LIMIT 1"
        ).fetchone()

        assert row[0] == "episodic"
        assert row[1] == "session_summary"
        assert "layered memory provider" in row[2].lower()
        assert "built-in memory" in row[2].lower()


class TestLayeredMemoryProviderPrefetch:
    def test_prefetch_returns_compact_layered_bundle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-prefetch", hermes_home=str(tmp_path), platform="cli")

        provider.on_memory_write("add", "user", "Call the assistant JARVIS")
        provider.on_memory_write("add", "memory", "The active codebase is ~/.hermes/hermes-agent")
        provider.sync_turn("We are implementing the layered provider.", "Understood.")
        provider.on_session_end([
            {"role": "user", "content": "Implement the layered provider."},
            {"role": "assistant", "content": "I will use SQLite and FTS."},
        ])

        bundle = provider.prefetch("layered provider SQLite JARVIS")

        assert "Layered Memory Recall" in bundle
        assert "Identity Core" in bundle
        assert "Semantic Memory" in bundle
        assert "Recent Episodic" in bundle
        assert "JARVIS" in bundle
        assert "SQLite" in bundle
        assert "~/.hermes/hermes-agent" in bundle

    def test_prefetch_ranks_high_score_memory_above_more_recent_low_score_memory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-ranking", hermes_home=str(tmp_path), platform="cli")

        provider._insert_item(
            "semantic",
            "memory",
            "SQLite FTS5 is the preferred retrieval index for layered memory",
            source="builtin_memory",
            action="add",
            score=0.95,
            importance=0.95,
        )
        provider._insert_item(
            "semantic",
            "memory",
            "SQLite is mentioned here too but this fact is low importance",
            source="builtin_memory",
            action="add",
            score=0.10,
            importance=0.10,
        )

        bundle = provider.prefetch("SQLite layered memory")
        semantic_section = bundle.split("## Semantic Memory\n", 1)[1].split("\n\n", 1)[0]
        lines = [line for line in semantic_section.splitlines() if line.startswith("-")]

        assert "preferred retrieval index" in lines[0]


class TestLayeredMemoryProviderPhase2:
    def test_get_config_schema_exposes_local_phase2_fields(self):
        provider = load_memory_provider("layered")
        schema = provider.get_config_schema()
        keys = {field["key"] for field in schema}

        assert "db_path" in keys
        assert "identity_limit" in keys
        assert "semantic_limit" in keys
        assert "episodic_limit" in keys
        assert "reflection_limit" in keys
        assert "enable_reflection" in keys
        assert "enable_consolidation" in keys

    def test_save_config_persists_under_memory_layered(self, tmp_path):
        provider = load_memory_provider("layered")

        provider.save_config(
            {
                "db_path": "$HERMES_HOME/custom-layered.db",
                "identity_limit": "2",
                "semantic_limit": "5",
                "enable_reflection": "true",
            },
            str(tmp_path),
        )

        import yaml

        config = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert config["memory"]["layered"]["db_path"] == "$HERMES_HOME/custom-layered.db"
        assert config["memory"]["layered"]["identity_limit"] == "2"
        assert config["memory"]["layered"]["enable_reflection"] == "true"

    def test_memory_items_store_metadata_and_score(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-meta", hermes_home=str(tmp_path), platform="cli")

        provider._insert_item(
            "semantic",
            "memory",
            "Layered memory facts need metadata",
            source="builtin_memory",
            action="add",
            score=0.88,
            importance=0.91,
            confidence=0.77,
            recurrence=3,
            metadata={"tags": ["memory", "phase2"], "origin": "test"},
        )

        row = provider._conn.execute(
            "SELECT score, importance, confidence, recurrence, metadata_json FROM memory_items ORDER BY id DESC LIMIT 1"
        ).fetchone()

        assert row[0] == 0.88
        assert row[1] == 0.91
        assert row[2] == 0.77
        assert row[3] == 3
        metadata = json.loads(row[4])
        assert metadata["origin"] == "test"
        assert metadata["tags"] == ["memory", "phase2"]

    def test_on_session_end_extracts_reflection_from_success_and_failure_signals(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-reflection", hermes_home=str(tmp_path), platform="cli")

        provider.on_session_end([
            {"role": "user", "content": "Please fix the layered provider test failures."},
            {"role": "assistant", "content": "I found the root cause and fixed the ranking bug."},
            {"role": "assistant", "content": "The previous approach failed because recent-only retrieval buried the important fact."},
        ])

        rows = provider._conn.execute(
            "SELECT layer, role, content FROM memory_items WHERE layer = 'reflection' ORDER BY id"
        ).fetchall()

        assert rows
        assert any("root cause" in row[2].lower() or "failed because" in row[2].lower() for row in rows)

    def test_on_session_end_consolidates_repeated_semantic_fact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-consolidate", hermes_home=str(tmp_path), platform="cli")

        provider.on_memory_write("add", "memory", "The active codebase is ~/.hermes/hermes-agent")
        provider.on_memory_write("add", "memory", "The active codebase is ~/.hermes/hermes-agent")
        provider.on_memory_write("add", "memory", "The active codebase is ~/.hermes/hermes-agent")

        provider.on_session_end([
            {"role": "user", "content": "Remember that the active codebase is ~/.hermes/hermes-agent."},
            {"role": "assistant", "content": "Stored."},
        ])

        row = provider._conn.execute(
            """
            SELECT recurrence, score, metadata_json
            FROM memory_items
            WHERE layer = 'semantic' AND content = 'The active codebase is ~/.hermes/hermes-agent'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        assert row[0] >= 3
        assert row[1] >= 0.9
        metadata = json.loads(row[2])
        assert metadata.get("consolidated") is True


class TestLayeredMemoryProviderPhase3:
    def test_on_delegation_persists_episodic_and_procedural_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-delegation", hermes_home=str(tmp_path), platform="cli")

        provider.on_delegation(
            task="Implement layered memory ranking with SQLite FTS and tests",
            result="Implemented ranking with SQLite FTS, added tests, and verified all targeted tests pass.",
            child_session_id="child-123",
        )

        episodic = provider._conn.execute(
            "SELECT layer, role, content FROM memory_items WHERE layer = 'episodic' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        procedural = provider._conn.execute(
            "SELECT layer, role, content, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        assert episodic[0] == "episodic"
        assert "layered memory ranking" in episodic[2].lower()
        assert procedural[0] == "procedural_index"
        assert "sqlite fts" in procedural[2].lower()
        metadata = json.loads(procedural[3])
        assert metadata["source"] == "delegation"
        assert metadata["child_session_id"] == "child-123"

    def test_on_session_end_promotes_repeated_archive_fact_to_semantic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-promote-semantic", hermes_home=str(tmp_path), platform="cli")

        provider.sync_turn("The deployment target is staging-cluster-1.", "Stored.")
        provider.sync_turn("Remember the deployment target is staging-cluster-1.", "Stored again.")
        provider.sync_turn("Deployment target remains staging-cluster-1.", "Acknowledged.")

        provider.on_session_end([
            {"role": "user", "content": "The deployment target is staging-cluster-1."},
            {"role": "assistant", "content": "Understood."},
        ])

        rows = provider._conn.execute(
            "SELECT layer, content, metadata_json FROM memory_items WHERE layer = 'semantic' ORDER BY id DESC"
        ).fetchall()

        assert rows
        assert any("staging-cluster-1" in row[1] for row in rows)
        promoted = next(row for row in rows if "staging-cluster-1" in row[1])
        metadata = json.loads(promoted[2])
        assert metadata["promoted_from"] == "archive"

    def test_on_session_end_promotes_repeated_success_pattern_to_procedural_index(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-promote-procedural", hermes_home=str(tmp_path), platform="cli")

        provider.on_session_end([
            {"role": "user", "content": "Please implement a bugfix with tests."},
            {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
        ])
        provider.on_session_end([
            {"role": "user", "content": "Please implement another fix with tests."},
            {"role": "assistant", "content": "I again wrote failing tests first, fixed the bug, and all tests passed."},
        ])

        rows = provider._conn.execute(
            "SELECT layer, content, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC"
        ).fetchall()

        assert rows
        assert any("failing tests first" in row[1].lower() for row in rows)
        promoted = next(row for row in rows if "failing tests first" in row[1].lower())
        metadata = json.loads(promoted[2])
        assert metadata["promoted_from"] == "successful_pattern"

    def test_prefetch_uses_recency_to_break_near_score_ties(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-recency", hermes_home=str(tmp_path), platform="cli")

        provider._insert_item(
            "semantic",
            "memory",
            "Older fact about release window alpha",
            source="builtin_memory",
            action="add",
            score=0.80,
            importance=0.80,
        )
        provider._insert_item(
            "semantic",
            "memory",
            "Newer fact about release window alpha",
            source="builtin_memory",
            action="add",
            score=0.80,
            importance=0.80,
        )

        bundle = provider.prefetch("release window alpha")
        semantic_section = bundle.split("## Semantic Memory\n", 1)[1].split("\n\n", 1)[0]
        lines = [line for line in semantic_section.splitlines() if line.startswith("-")]

        assert "Newer fact" in lines[0]


class TestLayeredMemoryProviderPhase4:
    def test_procedural_pattern_crosses_threshold_and_generates_skill_draft_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-skill-threshold", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        rows = provider._conn.execute(
            "SELECT content, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC"
        ).fetchall()
        assert rows
        promoted = next(row for row in rows if "failing tests first" in row[0].lower())
        metadata = json.loads(promoted[1])

        assert metadata["skill_candidate"] is True
        assert metadata["skill_draft_path"].endswith("write-failing-tests-first-then-verify-tests-pass.md")

        draft_path = Path(metadata["skill_draft_path"])
        assert draft_path.exists()
        draft = draft_path.read_text()
        assert draft.startswith("---\n")
        assert "name: write-failing-tests-first-then-verify-tests-pass" in draft
        assert "## Trigger" in draft
        assert "## Steps" in draft
        assert "## Verification" in draft

    def test_delegation_generated_skill_draft_contains_context_and_source_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-skill-delegation", hermes_home=str(tmp_path), platform="cli")

        provider.on_delegation(
            task="Implement layered memory ranking with SQLite FTS and tests",
            result="Implemented ranking with SQLite FTS, added targeted tests, and verified they pass.",
            child_session_id="child-skill-1",
        )
        provider.on_delegation(
            task="Implement layered memory ranking with SQLite FTS and tests",
            result="Implemented ranking with SQLite FTS again, added targeted tests, and verified they pass.",
            child_session_id="child-skill-2",
        )
        provider.on_delegation(
            task="Implement layered memory ranking with SQLite FTS and tests",
            result="Implemented ranking with SQLite FTS a third time, added targeted tests, and verified they pass.",
            child_session_id="child-skill-3",
        )

        row = provider._conn.execute(
            "SELECT content, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metadata = json.loads(row[1])
        draft_path = Path(metadata["skill_draft_path"])

        assert metadata["skill_candidate"] is True
        assert metadata["source"] in {"delegation", "promotion"}
        assert draft_path.exists()

        draft = draft_path.read_text()
        assert "SQLite FTS" in draft
        assert "targeted tests" in draft
        assert "delegation" in draft.lower() or "child session" in draft.lower()


class TestLayeredMemoryProviderPhase5:
    def test_skill_candidate_generates_publish_ready_package_and_review_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-phase5-package", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        row = provider._conn.execute(
            "SELECT content, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metadata = json.loads(row[1])

        assert metadata["review_status"] == "pending"
        assert metadata["review_gate_reason"]
        assert metadata["publish_ready_dir"].endswith("write-failing-tests-first-then-verify-tests-pass")
        assert metadata["candidate_index_path"].endswith("skill_candidates.json")
        assert metadata["evidence"]["source"] in {"promotion", "delegation"}
        assert metadata["evidence"]["promotion_rationale"]
        assert metadata["evidence"]["sample_evidence"]
        assert metadata["evidence"]["verification_hints"]

        package_dir = Path(metadata["publish_ready_dir"])
        assert package_dir.exists()
        assert (package_dir / "SKILL.md").exists()
        assert (package_dir / "candidate.json").exists()

        skill_md = (package_dir / "SKILL.md").read_text()
        candidate_json = json.loads((package_dir / "candidate.json").read_text())
        index_json = json.loads(Path(metadata["candidate_index_path"]).read_text())

        assert "name: write-failing-tests-first-then-verify-tests-pass" in skill_md
        assert candidate_json["review_status"] == "pending"
        assert candidate_json["skill_name"] == "write-failing-tests-first-then-verify-tests-pass"
        assert candidate_json["evidence"]["promotion_rationale"]
        assert candidate_json["evidence"]["sample_evidence"]
        assert any(entry["skill_name"] == "write-failing-tests-first-then-verify-tests-pass" for entry in index_json)

    def test_low_confidence_candidate_is_not_packaged_for_publish(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-phase5-low-confidence", hermes_home=str(tmp_path), platform="cli")

        provider._insert_item(
            "procedural_index",
            "procedure",
            "Write failing tests first, then implement the fix, then verify tests pass.",
            source="promotion",
            action="promoted_success_pattern",
            score=0.55,
            importance=0.55,
            confidence=0.40,
            recurrence=5,
            metadata={"occurrences": 5, "source": "promotion"},
        )
        provider._promote_skill_candidates()

        row = provider._conn.execute(
            "SELECT metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metadata = json.loads(row[0])

        assert metadata["review_status"] == "rejected"
        assert "low_confidence" in metadata["review_gate_reason"]
        assert "publish_ready_dir" not in metadata


class TestLayeredMemoryProviderPhase6:
    def test_approve_candidate_installs_skill_into_local_skills_dir_and_updates_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-phase6-approve", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        installed_path = provider.approve_skill_candidate("write-failing-tests-first-then-verify-tests-pass")

        assert installed_path.endswith("skills/write-failing-tests-first-then-verify-tests-pass/SKILL.md")
        skill_file = Path(installed_path)
        assert skill_file.exists()
        assert "name: write-failing-tests-first-then-verify-tests-pass" in skill_file.read_text()

        row = provider._conn.execute(
            "SELECT metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metadata = json.loads(row[0])
        assert metadata["review_status"] == "approved"
        assert metadata["installed_skill_path"] == installed_path

        index_path = Path(metadata["candidate_index_path"])
        index_json = json.loads(index_path.read_text())
        entry = next(item for item in index_json if item["skill_name"] == "write-failing-tests-first-then-verify-tests-pass")
        assert entry["review_status"] == "approved"
        assert entry["installed_skill_path"] == installed_path

    def test_approve_candidate_with_existing_skill_updates_in_place(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-phase6-update", hermes_home=str(tmp_path), platform="cli")

        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        existing = skills_dir / "SKILL.md"
        existing.write_text("---\nname: write-failing-tests-first-then-verify-tests-pass\ndescription: old\nversion: 0.0.1\n---\n\nold body\n")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        installed_path = provider.approve_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
        text = Path(installed_path).read_text()

        assert "Auto-generated skill draft" in text
        assert "old body" not in text

    def test_reject_candidate_updates_index_without_installing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-phase6-reject", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        provider.reject_skill_candidate("write-failing-tests-first-then-verify-tests-pass", reason="manual_reject")

        row = provider._conn.execute(
            "SELECT metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        metadata = json.loads(row[0])
        assert metadata["review_status"] == "rejected"
        assert metadata["review_gate_reason"] == "manual_reject"

        assert not (Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass" / "SKILL.md").exists()


class TestLayeredMemoryProviderProductization:
    def test_list_skill_candidates_returns_summary_view(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-list", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        candidates = provider.list_skill_candidates()

        assert candidates
        candidate = next(item for item in candidates if item["skill_name"] == "write-failing-tests-first-then-verify-tests-pass")
        assert candidate["review_status"] == "pending"
        assert candidate["review_gate_reason"]
        assert candidate["effective_recurrence"] >= 3

    def test_inspect_skill_candidate_returns_index_and_metadata_details(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-inspect", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        details = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")

        assert details["skill_name"] == "write-failing-tests-first-then-verify-tests-pass"
        assert details["review_status"] == "pending"
        assert details["skill_draft_path"].endswith("write-failing-tests-first-then-verify-tests-pass.md")
        assert details["publish_ready_dir"].endswith("write-failing-tests-first-then-verify-tests-pass")
        assert "candidate_json" in details
        assert details["candidate_json"]["skill_name"] == "write-failing-tests-first-then-verify-tests-pass"

    def test_decide_install_strategy_returns_patch_existing_when_name_exists_and_content_differs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-strategy-patch", hermes_home=str(tmp_path), platform="cli")

        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("existing skill")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        strategy = provider.decide_install_strategy("write-failing-tests-first-then-verify-tests-pass")

        assert strategy == "patch_existing"

    def test_decide_install_strategy_returns_duplicate_skip_when_existing_skill_matches_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-strategy-duplicate", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        details = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
        package_skill = Path(details["publish_ready_dir"]) / "SKILL.md"
        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(package_skill.read_text())

        strategy = provider.decide_install_strategy("write-failing-tests-first-then-verify-tests-pass")

        assert strategy == "duplicate_skip"

    def test_decide_install_strategy_returns_create_variant_for_reserved_name_collision(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-strategy-variant", hermes_home=str(tmp_path), platform="cli")

        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("---\nname: write-failing-tests-first-then-verify-tests-pass\ndescription: canonical approved skill\nversion: 1.0.0\n---\n\n## Trigger\nUse when: canonical\n")
        (skills_dir / "LOCKED").write_text("1")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        strategy = provider.decide_install_strategy("write-failing-tests-first-then-verify-tests-pass")

        assert strategy == "create_variant"

    def test_decide_install_strategy_returns_create_when_skill_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-create", hermes_home=str(tmp_path), platform="cli")

        strategy = provider.decide_install_strategy("write-failing-tests-first-then-verify-tests-pass")

        assert strategy == "create"

    def test_approve_candidate_duplicate_skip_does_not_overwrite_existing_skill(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-approve-duplicate", hermes_home=str(tmp_path), platform="cli")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        details = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
        package_skill = Path(details["publish_ready_dir"]) / "SKILL.md"
        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        existing = skills_dir / "SKILL.md"
        existing.write_text(package_skill.read_text())

        result = provider.approve_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
        metadata = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")

        assert result.endswith("skills/write-failing-tests-first-then-verify-tests-pass/SKILL.md")
        assert metadata["approval_strategy"] == "duplicate_skip"
        assert metadata["review_gate_reason"] == "manual_approve:duplicate_skip"

    def test_approve_candidate_create_variant_installs_variant_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = load_memory_provider("layered")
        provider.initialize(session_id="sess-product-approve-variant", hermes_home=str(tmp_path), platform="cli")

        skills_dir = Path(tmp_path) / "skills" / "write-failing-tests-first-then-verify-tests-pass"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text("---\nname: write-failing-tests-first-then-verify-tests-pass\ndescription: canonical approved skill\nversion: 1.0.0\n---\n\n## Trigger\nUse when: canonical\n")
        (skills_dir / "LOCKED").write_text("1")

        for _ in range(3):
            provider.on_session_end([
                {"role": "user", "content": "Please implement a bugfix with tests."},
                {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
            ])

        result = provider.approve_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
        metadata = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")

        assert "@variant-" in result
        assert metadata["approval_strategy"] == "create_variant"
        assert metadata["installed_skill_path"] == result
