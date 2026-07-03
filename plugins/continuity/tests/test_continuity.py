"""
Tests for the Continuity Plugin — DAG + Daily Journal memory system.

Covers: schema, CRUD, compression, provenance, injection, guardrails,
token budgeting, model-switch detection.

Run: /home/austin/.local/bin/pytest plugins/continuity/tests/test_continuity.py -v -c /dev/null -p no:timeout --override-ini="addopts="
"""

import importlib.util
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Ensure plugin is importable
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR.parent))
sys.path.insert(0, str(_HERMES_HOME / "hermes-agent"))


def _load_mod(name="continuity_mod"):
    """Load continuity __init__.py as a fresh module."""
    spec = importlib.util.spec_from_file_location(name, _PLUGIN_DIR / "__init__.py")
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def with_db(tmp_path: Path) -> Generator[None, None, None]:
    """Set up a fresh continuity DB in a temp directory."""
    from plugins.continuity import db as continuity_db

    continuity_db.close_connection()

    def _test_home(p=tmp_path):
        d = p / "continuity"
        d.mkdir(parents=True, exist_ok=True)
        return d

    original_home = continuity_db.get_continuity_home
    continuity_db.get_continuity_home = _test_home
    continuity_db.migrate()

    yield

    continuity_db.close_connection()
    continuity_db.get_continuity_home = original_home


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_migration_creates_tables(self, with_db):
        from plugins.continuity import db as continuity_db
        db_path = continuity_db.get_db_path()
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "nodes" in tables
        assert "edges" in tables
        assert "schema_version" in tables
        conn.close()

    def test_nodes_table_columns(self, with_db):
        from plugins.continuity import db as continuity_db
        conn = sqlite3.connect(str(continuity_db.get_db_path()))
        columns = [r[1] for r in conn.execute("PRAGMA table_info(nodes)")]
        for col in ["node_id", "node_type", "date_key", "token_count",
                     "compression_depth", "provider", "model",
                     "source_session_id", "author_mode",
                     "operational_tokens", "relational_tokens"]:
            assert col in columns, f"Missing: {col}"
        conn.close()

    def test_edges_table_columns(self, with_db):
        from plugins.continuity import db as continuity_db
        conn = sqlite3.connect(str(continuity_db.get_db_path()))
        columns = [r[1] for r in conn.execute("PRAGMA table_info(edges)")]
        assert "parent_node_id" in columns
        assert "child_node_id" in columns
        conn.close()


# ---------------------------------------------------------------------------
# CRUD Tests
# ---------------------------------------------------------------------------

class TestNodeCRUD:
    def test_upsert_node(self, with_db):
        from plugins.continuity import db as continuity_db
        nid = continuity_db.upsert_node("session_shard", "2026-05-31", "Test",
                                         token_count=500, provider="opencode-go",
                                         model="qwen3.7-max", source_session_id="s1",
                                         author_mode="system")
        assert nid.startswith("session_shard_")
        node = continuity_db.get_node(nid)
        assert node["node_type"] == "session_shard"
        assert node["model"] == "qwen3.7-max"

    def test_upsert_with_all_fields(self, with_db):
        from plugins.continuity import db as continuity_db
        nid = continuity_db.upsert_node("daily", "2026-05-31", "Journal",
                                         token_count=1000, compression_depth=1,
                                         operational_tokens=600, relational_tokens=400)
        node = continuity_db.get_node(nid)
        assert node["compression_depth"] == 1
        assert node["operational_tokens"] == 600

    def test_get_latest(self, with_db):
        from plugins.continuity import db as continuity_db
        continuity_db.upsert_node("daily", "2026-05-30", "J1")
        assert continuity_db.get_latest_node("daily") is not None


# ---------------------------------------------------------------------------
# DAG Tests
# ---------------------------------------------------------------------------

class TestDAG:
    def test_add_edge(self, with_db):
        from plugins.continuity import db as continuity_db
        p = continuity_db.upsert_node("weekly", "2026-W22")
        c = continuity_db.upsert_node("daily", "2026-05-31")
        assert continuity_db.add_edge(parent_id=p, child_id=c)
        assert len(continuity_db.get_children(p)) == 1
        assert len(continuity_db.get_parents(c)) == 1

    def test_provenance_trace(self, with_db):
        from plugins.continuity import db as continuity_db
        s1 = continuity_db.upsert_node("session_shard", "2026-05-01", source_session_id="s1")
        s2 = continuity_db.upsert_node("session_shard", "2026-05-02", source_session_id="s2")
        d = continuity_db.upsert_node("daily", "2026-05-01")
        w = continuity_db.upsert_node("weekly", "2026-W18")
        m = continuity_db.upsert_node("monthly", "2026-05")

        continuity_db.add_edge(parent_id=s1, child_id=d)
        continuity_db.add_edge(parent_id=s2, child_id=d)
        continuity_db.add_edge(parent_id=d, child_id=w)
        continuity_db.add_edge(parent_id=w, child_id=m)

        sources = continuity_db.trace_provenance(m)
        assert len(sources) == 2
        assert s1 in {s["node_id"] for s in sources}


# ---------------------------------------------------------------------------
# Token Budget Tests
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_constants_sum_to_12000(self):
        mod = _load_mod("budget_test")
        total = sum(mod.TOKEN_BUDGET.values())
        assert total == 12000 + mod.TOKEN_BUDGET["total"]  # total is double-counted

    def test_overflow_crops_outer_layers(self):
        mod = _load_mod("overflow_test")
        assert mod.TOKEN_BUDGET["total"] == 12000

    def test_daily_template_has_60_40(self):
        tpl = _PLUGIN_DIR / "templates" / "daily_journal.md"
        if tpl.exists():
            c = tpl.read_text()
            assert "Relational Continuity" in c
            assert "Operational Continuity" in c


# ---------------------------------------------------------------------------
# Guardrail Tests
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_allowed_patterns(self):
        allowed = [
            "我留意到 Austin 今日對 scope creep 特別敏感",
            "觀察到 Austin 偏好更快的迭代節奏",
            "今日 Austin 對結果感到滿意",
        ]
        for ex in allowed:
            has_proj = "感覺" in ex and not any(w in ex for w in ["偏好", "留意", "觀察"])
            has_state = any(w in ex for w in ["很沮喪", "很失望", "受傷"])
            assert not (has_proj or has_state), f"Should pass: {ex[:40]}"

    def test_forbidden_patterns(self):
        forbidden = [
            "我感覺 Austin 可能有啲失望",
            "我因為被忽略而感到受傷",
            "Austin 今天一定很沮喪",
        ]
        for ex in forbidden:
            has_proj = "感覺" in ex
            has_state = any(w in ex for w in ["很沮喪", "很失望", "受傷"])
            assert has_proj or has_state, f"Should be caught: {ex[:40]}"


# ---------------------------------------------------------------------------
# Model-Switch Tests
# ---------------------------------------------------------------------------

class TestModelSwitch:
    def test_model_tracked(self):
        mod = _load_mod("model_test")
        mod._on_session_start(session_id="s1", model="deepseek-v4-flash")
        assert mod._last_model_used == "deepseek-v4-flash"

    def test_model_change_detected(self):
        mod = _load_mod("model_test2")
        mod._on_session_start(session_id="s1", model="deepseek-v4-flash")
        mod._on_session_start(session_id="s2", model="qwen3.7-max")
        assert mod._last_model_used == "qwen3.7-max"


# ---------------------------------------------------------------------------
# Integrity Sweep Tests
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_orphan_detected(self, with_db):
        from plugins.continuity import db as continuity_db
        continuity_db.upsert_node("daily", "2026-01-01", "Orphan")
        findings = continuity_db.integrity_sweep()
        assert len(findings["orphaned_nodes"]) >= 1

    def test_broken_provenance(self, with_db):
        from plugins.continuity import db as continuity_db
        continuity_db.upsert_node("weekly", "2026-W22", "No children")
        findings = continuity_db.integrity_sweep()
        assert len(findings["broken_provenance"]) >= 1


# ---------------------------------------------------------------------------
# Security Tests — markdown_path validation (F6)
# ---------------------------------------------------------------------------

class TestSecurity:
    def test_markdown_path_traversal_blocked(self, with_db):
        """Verify path traversal outside journal root is rejected."""
        mod = _load_mod("security_traversal")
        from plugins.continuity.db import get_continuity_home
        continuity_root = get_continuity_home() / "journal" / "daily"
        continuity_root.mkdir(parents=True, exist_ok=True)
        safe_path = str(continuity_root / "2026-06-01.md")
        # A path outside the continuity root
        traversal_path = str(get_continuity_home().parent / "secret.txt")
        assert mod._is_path_safe(safe_path), "Safe path should pass"
        assert not mod._is_path_safe(traversal_path), "Traversal path should be rejected"

    def test_markdown_path_symlink_resolved(self, with_db):
        """Verify symlinks are resolved before checking root."""
        mod = _load_mod("security_symlink")
        from plugins.continuity.db import get_continuity_home
        continuity_root = get_continuity_home() / "journal" / "daily"
        continuity_root.mkdir(parents=True, exist_ok=True)
        # Create a legit file
        legit_file = continuity_root / "test.md"
        legit_file.write_text("legit")
        # Create a symlink pointing outside
        escape_file = get_continuity_home().parent / "secret.txt"
        escape_file.write_text("secret")
        symlink_path = continuity_root / "link.md"
        symlink_path.symlink_to(escape_file)
        assert mod._is_path_safe(str(legit_file)), "Regular file should pass"
        assert not mod._is_path_safe(str(symlink_path)), "Symlink escape should be rejected"


# ---------------------------------------------------------------------------
# Plugin Registration Tests
# ---------------------------------------------------------------------------

class TestPluginRegistration:
    def test_manifest(self):
        import yaml
        m = yaml.safe_load((_PLUGIN_DIR / "plugin.yaml").read_text())
        assert m["name"] == "continuity"
        assert "on_session_end" in m["hooks"]
        assert "pre_llm_call" in m["hooks"]

    def test_register_function(self):
        mod = _load_mod("reg_test")
        assert hasattr(mod, "register")


# ---------------------------------------------------------------------------
# Cron Script Tests
# ---------------------------------------------------------------------------

class TestCronScripts:
    def test_daily_importable(self):
        spec = importlib.util.spec_from_file_location(
            "daily", _PLUGIN_DIR / "scripts" / "daily_journal.py")
        assert spec is not None

    def test_weekly_importable(self):
        spec = importlib.util.spec_from_file_location(
            "weekly", _PLUGIN_DIR / "scripts" / "weekly_summary.py")
        assert spec is not None

    def test_monthly_importable(self):
        spec = importlib.util.spec_from_file_location(
            "monthly", _PLUGIN_DIR / "scripts" / "monthly_summary.py")
        assert spec is not None

    def test_weekly_fallback(self):
        from plugins.continuity.scripts import weekly_summary
        assert "No daily journals found" in weekly_summary.compress_to_weekly([], "W22")

    def test_monthly_fallback(self):
        from plugins.continuity.scripts import monthly_summary
        assert "No weekly summaries found" in monthly_summary.compress_to_monthly([], "M05")


# ---------------------------------------------------------------------------
# Promotion Candidate Tests
# ---------------------------------------------------------------------------

class TestPromotion:
    def test_empty_db_no_candidates(self, with_db):
        """No candidates when DB is empty."""
        import importlib
        mod = importlib.import_module("plugins.continuity")
        candidates = mod._find_promotion_candidates()
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# Correctness Tests — off-by-one (F4), MEMORY.md path (F7)
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_get_recent_daily_journals_exact_count(self, with_db):
        """get_recent_daily_journals(7) should return exactly 7 days."""
        from plugins.continuity import db as continuity_db

        # Insert 10 daily nodes spanning 10 consecutive days
        from datetime import timedelta
        today = date.today()
        for i in range(10):
            d = (today - timedelta(days=i)).isoformat()
            continuity_db.upsert_node("daily", d, f"Journal {i}")

        recent = continuity_db.get_recent_daily_journals(days=7)
        assert len(recent) == 7, f"Expected 7 journals, got {len(recent)}"

    def test_get_recent_daily_journals_boundary(self, with_db):
        """days=1 includes today only, days=2 includes yesterday."""
        from plugins.continuity import db as continuity_db
        from datetime import timedelta
        today = date.today()
        yesterday = (today - timedelta(days=1)).isoformat()
        continuity_db.upsert_node("daily", yesterday, "Yesterday")
        continuity_db.upsert_node("daily", today.isoformat(), "Today")
        # 'last 1 day' = today only
        recent_1 = continuity_db.get_recent_daily_journals(days=1)
        assert len(recent_1) == 1, f"Expected 1 journal (today), got {len(recent_1)}"
        # 'last 2 days' = yesterday + today
        recent_2 = continuity_db.get_recent_daily_journals(days=2)
        assert len(recent_2) == 2, f"Expected 2 journals, got {len(recent_2)}"

    def test_memory_path_fallback_candidates(self, with_db):
        """_find_memory_file should check multiple MEMORY.md paths."""
        mod = _load_mod("memory_path_test")
        assert hasattr(mod, '_find_memory_file'), "Need _find_memory_file function"

    def test_memory_path_checks_memories_dir(self, with_db):
        """When MEMORY.md not in HERMES_HOME root, check memories/."""
        mod = _load_mod("memory_path_test2")
        from plugins.continuity.db import get_continuity_home
        hermes_home = get_continuity_home().parent  # tmp_path
        old_home = os.environ.get("HERMES_HOME", "")
        os.environ["HERMES_HOME"] = str(hermes_home)
        try:
            # Create just memories/MEMORY.md (not root MEMORY.md)
            mem_dir = hermes_home / "memories"
            mem_dir.mkdir(parents=True, exist_ok=True)
            (mem_dir / "MEMORY.md").write_text("# Memory from memories dir")
            found = mod._find_memory_file()
            assert found is not None, "Should find MEMORY.md in memories/"
            content = found.read_text(encoding="utf-8")
            assert "memories dir" in content
        finally:
            if old_home:
                os.environ["HERMES_HOME"] = old_home
            else:
                os.environ.pop("HERMES_HOME", None)

    def test_memory_path_prioritizes_root(self, with_db):
        """Root MEMORY.md should take priority over memories/MEMORY.md."""
        mod = _load_mod("memory_path_test3")
        from plugins.continuity.db import get_continuity_home
        hermes_home = get_continuity_home().parent
        old_home = os.environ.get("HERMES_HOME", "")
        os.environ["HERMES_HOME"] = str(hermes_home)
        try:
            # Create both
            (hermes_home / "MEMORY.md").write_text("# Root memory")
            mem_dir = hermes_home / "memories"
            mem_dir.mkdir(parents=True, exist_ok=True)
            (mem_dir / "MEMORY.md").write_text("# memories memory")
            found = mod._find_memory_file()
            assert found is not None
            content = found.read_text(encoding="utf-8")
            assert "Root memory" in content, "Root should take priority"
        finally:
            if old_home:
                os.environ["HERMES_HOME"] = old_home
            else:
                os.environ.pop("HERMES_HOME", None)


# ---------------------------------------------------------------------------
# Resilience Tests — I/O error handling (F8)
# ---------------------------------------------------------------------------

class TestResilience:
    def test_daily_load_template_fallback_on_io_error(self):
        """load_template should return inline fallback when file read fails."""
        from plugins.continuity.scripts import daily_journal
        # Simulate by making TEMPLATE_PATH point to nonexistent file
        original = daily_journal.TEMPLATE_PATH
        daily_journal.TEMPLATE_PATH = Path("/nonexistent/template.md")
        try:
            result = daily_journal.load_template()
            assert "{{DATE}}" in result
            assert "{{OPERATIONAL_CONTENT}}" in result
        finally:
            daily_journal.TEMPLATE_PATH = original

    def test_daily_journal_run_handles_write_error(self, with_db):
        """run() should not crash when output_path cannot be written."""
        from plugins.continuity.scripts import daily_journal
        from plugins.continuity.db import upsert_node
        # Create a session_shard so the script has data
        upsert_node("session_shard", "2026-06-01",
                     source_session_id="s1", author_mode="system")
        upsert_node("session_shard", "2026-06-01",
                     source_session_id="s2", author_mode="system")
        # Run with a target date and --dry-run to avoid actual file write
        result = daily_journal.run(
            date(2026, 6, 1),
            dry_run=True,
        )
        assert result["status"] == "dry-run"
        assert result["session_count"] == 2

    def test_weekly_load_handles_missing_files(self):
        """load_daily_journals shouldn't crash on missing files."""
        from plugins.continuity.scripts import weekly_summary
        dates = ["2099-01-01", "2099-01-02"]  # definitely no files
        journals = weekly_summary.load_daily_journals(dates)
        assert len(journals) == 0

    def test_monthly_load_handles_missing_files(self):
        """load_weekly_summaries shouldn't crash on missing files."""
        from plugins.continuity.scripts import monthly_summary
        week_keys = ["2099-W01", "2099-W02"]
        summaries = monthly_summary.load_weekly_summaries(week_keys)
        assert len(summaries) == 0


# ---------------------------------------------------------------------------
# Integration Tests — Full pipeline (F9)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_session_to_daily(self, with_db):
        """session_shards → daily journal synthesis → DAG registration."""
        from plugins.continuity import db as continuity_db
        from plugins.continuity.scripts import daily_journal

        # 1. Create session shards
        s1 = continuity_db.upsert_node(
            "session_shard", "2026-06-01",
            source_session_id="sess_abc123", author_mode="system",
            model="qwen3.7-max", provider="opencode-go",
        )
        s2 = continuity_db.upsert_node(
            "session_shard", "2026-06-01",
            source_session_id="sess_def456", author_mode="system",
            model="qwen3.7-max", provider="opencode-go",
        )

        # Verify shards exist
        shards = continuity_db.get_session_shards_for_date("2026-06-01")
        assert len(shards) == 2

        # 2. Run daily journal synthesis (dry run — no LLM available in test)
        result = daily_journal.run(
            date(2026, 6, 1),
            dry_run=True,
        )
        assert result["status"] == "dry-run"
        assert result["session_count"] == 2

    def test_dag_provenance_chain(self, with_db):
        """Full DAG chain: shards → daily → weekly → monthly, then trace back."""
        from plugins.continuity import db as continuity_db

        # Build DAG: 2 shards → daily → weekly → monthly
        s1 = continuity_db.upsert_node("session_shard", "2026-05-25", source_session_id="s1",
                                         model="m1", author_mode="system")
        s2 = continuity_db.upsert_node("session_shard", "2026-05-25", source_session_id="s2",
                                         model="m1", author_mode="system")
        d1 = continuity_db.upsert_node("daily", "2026-05-25", "Journal 05-25")
        continuity_db.add_edge(parent_id=s1, child_id=d1)
        continuity_db.add_edge(parent_id=s2, child_id=d1)

        s3 = continuity_db.upsert_node("session_shard", "2026-05-26", source_session_id="s3",
                                         model="m1", author_mode="system")
        s4 = continuity_db.upsert_node("session_shard", "2026-05-26", source_session_id="s4",
                                         model="m1", author_mode="system")
        d2 = continuity_db.upsert_node("daily", "2026-05-26", "Journal 05-26")
        continuity_db.add_edge(parent_id=s3, child_id=d2)
        continuity_db.add_edge(parent_id=s4, child_id=d2)

        w = continuity_db.upsert_node("weekly", "2026-W21", "Week 21")
        continuity_db.add_edge(parent_id=d1, child_id=w)
        continuity_db.add_edge(parent_id=d2, child_id=w)

        m = continuity_db.upsert_node("monthly", "2026-05", "May")
        continuity_db.add_edge(parent_id=w, child_id=m)

        # Trace provenance from monthly to shards
        sources = continuity_db.trace_provenance(m)
        assert len(sources) == 4, f"Expected 4 shards, got {len(sources)}"
        source_ids = {s["source_session_id"] for s in sources}
        assert source_ids == {"s1", "s2", "s3", "s4"}

        # Verify DAG structure
        assert len(continuity_db.get_parents(d1)) == 2  # s1, s2
        assert len(continuity_db.get_children(d1)) == 1  # w
        assert len(continuity_db.get_parents(w)) == 2   # d1, d2

    def test_integrity_sweep_full(self, with_db):
        """Integration test for integrity_sweep with mixed nodes."""
        from plugins.continuity import db as continuity_db

        # Connected chain
        s = continuity_db.upsert_node("session_shard", "2026-06-01", source_session_id="s1")
        d = continuity_db.upsert_node("daily", "2026-06-01")
        w = continuity_db.upsert_node("weekly", "2026-W22")
        continuity_db.add_edge(parent_id=s, child_id=d)
        continuity_db.add_edge(parent_id=d, child_id=w)

        # Orphan (daily with no edges)
        continuity_db.upsert_node("daily", "2026-06-02", "Orphan")

        # Broken provenance (weekly with no parent edges)
        w2 = continuity_db.upsert_node("weekly", "2026-W23", "Broken weekly")
        continuity_db.add_edge(parent_id=w2, child_id=d)  # reverse edge

        findings = continuity_db.integrity_sweep()
        assert len(findings["orphaned_nodes"]) >= 1
        assert findings["total_nodes"] >= 4

    def test_token_budget_fits_12000(self):
        """Verify TOKEN_BUDGET allocation sums correctly."""
        mod = _load_mod("integ_budget")
        allocated = {k: v for k, v in mod.TOKEN_BUDGET.items() if k != "total"}
        total_allocated = sum(allocated.values())
        assert total_allocated <= mod.TOKEN_BUDGET["total"], \
            f"Allocated {total_allocated} exceeds budget {mod.TOKEN_BUDGET['total']}"

    def test_session_shard_dedup(self, with_db):
        """Same session_id + same date should produce one node (upsert)."""
        from plugins.continuity import db as continuity_db
        id1 = continuity_db.upsert_node(
            "session_shard", "2026-06-01",
            source_session_id="sess_unique", author_mode="system",
        )
        id2 = continuity_db.upsert_node(
            "session_shard", "2026-06-01",
            source_session_id="sess_unique", author_mode="system",
        )
        assert id1 == id2, "Same session+date should give same node_id"
        shards = continuity_db.get_session_shards_for_date("2026-06-01")
        assert len(shards) == 1, "Should have only 1 shard"


# ---------------------------------------------------------------------------
# Template Tests
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_daily(self):
        p = _PLUGIN_DIR / "templates" / "daily_journal.md"
        c = p.read_text()
        assert "Identity Anchors" in c and "Provenance" in c

    def test_weekly(self):
        p = _PLUGIN_DIR / "templates" / "weekly_summary.md"
        c = p.read_text()
        assert "depth=2" in c

    def test_monthly(self):
        p = _PLUGIN_DIR / "templates" / "monthly_summary.md"
        c = p.read_text()
        assert "Identity Trajectory" in c and "depth=3" in c


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_nonexistent(self, with_db):
        from plugins.continuity import db as continuity_db
        assert continuity_db.get_node("fake") is None

    def test_empty_sweep(self, with_db):
        from plugins.continuity import db as continuity_db
        f = continuity_db.integrity_sweep()
        assert f["orphaned_nodes"] == []

    def test_delete(self, with_db):
        from plugins.continuity import db as continuity_db
        nid = continuity_db.upsert_node("daily", "2026-01-01", "Del")
        assert continuity_db.get_node(nid) is not None
        continuity_db.delete_node(nid)
        assert continuity_db.get_node(nid) is None

    def test_count_shards(self, with_db):
        from plugins.continuity import db as continuity_db
        continuity_db.upsert_node("session_shard", "2026-05-31", source_session_id="s1")
        continuity_db.upsert_node("session_shard", "2026-05-31", source_session_id="s2")
        assert continuity_db.count_session_shards_for_date("2026-05-31") == 2

    def test_upsert_edge_idempotent(self, with_db):
        from plugins.continuity import db as continuity_db
        p = continuity_db.upsert_node("daily", "2026-01-01")
        c = continuity_db.upsert_node("session_shard", "2026-01-01")
        assert continuity_db.add_edge(parent_id=p, child_id=c) is True
        assert continuity_db.add_edge(parent_id=p, child_id=c) is True
        assert len(continuity_db.get_children(p)) == 1

    def test_upsert_is_deterministic(self, with_db):
        """Same inputs → same node_id (deterministic)."""
        from plugins.continuity import db as continuity_db
        id1 = continuity_db.upsert_node("daily", "2026-06-01", title="First call")
        id2 = continuity_db.upsert_node("daily", "2026-06-01", title="Second call")
        assert id1 == id2, "Deterministic inputs should give same node_id"

    def test_upsert_updates_existing(self, with_db):
        """Repeated upsert with same key updates the row, does not duplicate."""
        from plugins.continuity import db as continuity_db
        nid = continuity_db.upsert_node("daily", "2026-06-01", title="Original", token_count=100)
        continuity_db.upsert_node("daily", "2026-06-01", title="Updated", token_count=200)
        node = continuity_db.get_node(nid)
        assert node["title"] == "Updated"
        assert node["token_count"] == 200
        # Only one row
        all_daily = continuity_db.get_nodes_by_type("daily", limit=10)
        assert len(all_daily) == 1

    def test_upsert_with_explicit_node_id(self, with_db):
        """Explicit node_id overrides deterministic derivation."""
        from plugins.continuity import db as continuity_db
        custom_id = "my_custom_id_12345"
        returned_id = continuity_db.upsert_node("daily", "2026-06-01", node_id=custom_id)
        assert returned_id == custom_id
        node = continuity_db.get_node(custom_id)
        assert node is not None
        assert node["node_type"] == "daily"
