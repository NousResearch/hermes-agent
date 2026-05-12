"""Tests for CJK entity extraction in holographic memory plugin.

Regression tests for issue #24416: _extract_entities was ASCII-only,
producing zero entities for CJK/non-English facts.
"""

from pathlib import Path

import pytest

from plugins.memory.holographic.store import MemoryStore


class TestCJKEntityExtraction:
    """Verify _extract_entities produces entities for CJK content."""

    def _make_store(self, tmp_path: Path) -> MemoryStore:
        return MemoryStore(db_path=tmp_path / "test.db")

    # --- Rule 5: CJK brackets and quotes ---

    def test_cjk_corner_brackets(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("用户公司日常用「白兔」/「白兔控股」")
        assert "白兔" in entities
        assert "白兔控股" in entities

    def test_cjk_double_corner_brackets(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("参考『红楼梦』的写法")
        assert "红楼梦" in entities

    def test_cjk_angle_brackets(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("推荐阅读《资治通鉴》")
        assert "资治通鉴" in entities

    def test_cjk_smart_double_quotes(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("\u201c重要项目\u201d已启动")
        assert "重要项目" in entities

    def test_cjk_smart_single_quotes(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("\u2018代号\u2019已完成")
        assert "代号" in entities

    # --- Rule 6: Mixed-script identifiers ---

    def test_mixed_script_latin_identifier(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("飞书白兔 App 已于 2026-5-10 接入完成")
        assert "App" in entities

    def test_mixed_script_versioned(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("使用 GPT-5.5 进行推理")
        assert "GPT-5.5" in entities

    def test_mixed_script_dot_separated(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("域名是 baitugroup.com")
        assert "baitugroup.com" in entities

    def test_mixed_script_kebab_case(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("安装 lark-cli 工具")
        assert "lark-cli" in entities

    def test_mixed_script_ihms(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("IHMS 系统已部署")
        assert "IHMS" in entities

    # --- Integration: full add_fact round-trip ---

    def test_add_fact_creates_entities_for_cjk(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_fact("飞书白兔 App 已于 2026-5-10 接入完成")

        entity_count = store._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        assert entity_count > 0, "CJK fact should produce at least one entity"

    def test_add_fact_creates_fact_entity_links(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_fact("飞书白兔 App 已于 2026-5-10 接入完成")

        link_count = store._conn.execute(
            "SELECT COUNT(*) FROM fact_entities"
        ).fetchone()[0]
        assert link_count > 0, "CJK fact should have fact_entity links"

    def test_add_fact_cjk_quoted_entities(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_fact("用户公司日常用「白兔」/「白兔控股」，不要用工商执照名「成都抖咖」")

        entities = [
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM entities"
            ).fetchall()
        ]
        assert "白兔" in entities
        assert "白兔控股" in entities
        assert "成都抖咖" in entities

    def test_add_fact_mixed_chinese_english(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_fact("Coco 香港插班项目计划")

        entities = [
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM entities"
            ).fetchall()
        ]
        # "Coco" is a Latin identifier — rule 6
        assert "Coco" in entities

    def test_add_fact_three_cjk_facts(self, tmp_path):
        """Reproduce the exact scenario from issue #24416."""
        store = self._make_store(tmp_path)
        store.add_fact("飞书白兔 App 已于 2026-5-10 接入完成")
        store.add_fact("Coco 香港插班项目计划")
        store.add_fact("用户公司日常用「白兔」/「白兔控股」，不要用工商执照名「成都抖咖」")

        fact_count = store._conn.execute(
            "SELECT COUNT(*) FROM facts"
        ).fetchone()[0]
        entity_count = store._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        link_count = store._conn.execute(
            "SELECT COUNT(*) FROM fact_entities"
        ).fetchone()[0]

        assert fact_count == 3
        assert entity_count > 0, "Before fix: 0 entities. After fix: should be > 0"
        assert link_count > 0, "Before fix: 0 links. After fix: should be > 0"

    # --- English behavior unchanged ---

    def test_english_capitalized_unchanged(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("John Doe works at Acme Corp")
        assert "John Doe" in entities
        assert any("Acme" in e for e in entities)

    def test_english_double_quotes_unchanged(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities('Use "Python" for scripting')
        assert "Python" in entities

    def test_english_single_quotes_unchanged(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("Run 'pytest' to test")
        assert "pytest" in entities

    def test_english_aka_unchanged(self, tmp_path):
        store = self._make_store(tmp_path)
        entities = store._extract_entities("Guido aka BDFL")
        assert "Guido" in entities
        assert "BDFL" in entities

    def test_english_only_fact_still_works(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add_fact("John Doe prefers concise PR comments")

        entity_count = store._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        assert entity_count > 0
