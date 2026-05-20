"""kanban-paper-nexus skill contract checks (no network)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "research" / "kanban-paper-nexus" / "SKILL.md"
SCRIPT = REPO / "skills" / "research" / "kanban-paper-nexus" / "scripts" / "paper_nexus_metadata.py"


def test_skill_md_exists_and_description_length():
    text = SKILL.read_text(encoding="utf-8")
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m, "missing description"
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, len(desc)


def test_bilingual_builder_reads_meta():
    import importlib.util
    import json
    import tempfile

    meta_path = SKILL.parent / "scripts" / "paper_nexus_metadata.py"
    build_path = SKILL.parent / "scripts" / "build_bilingual_doc_md.py"
    spec = importlib.util.spec_from_file_location("build_bilingual", build_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    meta = {"paper_id": "2402.03300", "title": "Test Paper", "summary": "中文摘要。", "published": "2024-01-01", "authors": ["A"], "arxiv_abs": "https://arxiv.org/abs/2402.03300", "arxiv_pdf": "https://arxiv.org/pdf/2402.03300"}
    text = mod.build(meta)
    assert "核心总结" in text and "Executive Summary" in text
    assert "参考方向" in text
    assert "主张–证据–局限" in text
    assert "【待填" in text  # skeleton for workers
    assert "DeepSeekMath" not in text or "Test Paper" in meta["title"]


def test_canonical_paper_id_strips_version():
    import importlib.util

    path = SKILL.parent / "scripts" / "paper_doc_registry.py"
    spec = importlib.util.spec_from_file_location("paper_doc_registry", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.canonical_paper_id("2402.03300v3") == "2402.03300"
    assert mod.canonical_paper_id("2402.03300") == "2402.03300"


def test_registry_resolve_create_then_update(tmp_path, monkeypatch):
    import importlib.util

    board_dir = tmp_path / "kanban" / "boards" / "paper-nexus"
    board_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    path = SKILL.parent / "scripts" / "paper_doc_registry.py"
    spec = importlib.util.spec_from_file_location("paper_doc_registry", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod.resolve("2402.03300")["action"] == "create"
    mod.register("2402.03300v3", "https://my.feishu.cn/docx/TESTDOC")
    r = mod.resolve("2402.03300")
    assert r["action"] == "update"
    assert r["doc_url"] == "https://my.feishu.cn/docx/TESTDOC"


def test_metadata_script_parses_arxiv_id():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.normalize_paper_id("https://arxiv.org/abs/2402.03300") == "2402.03300"
    assert mod.normalize_paper_id("2402.03300v3") == "2402.03300v3"
