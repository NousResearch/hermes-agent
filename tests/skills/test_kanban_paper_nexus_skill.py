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


def test_memory_search_query_id_only_and_with_title():
    import importlib.util

    path = SKILL.parent / "scripts" / "paper_memory_search_query.py"
    spec = importlib.util.spec_from_file_location("paper_memory_search_query", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod.build_search_query("2402.03300") == "2402.03300"
    q = mod.build_search_query("2402.03300v3", "DeepSeekMath: Pushing the Limits of Mathematical Reasoning")
    assert q.startswith("2402.03300 ")
    assert len(q) <= mod.MAX_QUERY_CHARS
    mod.validate_query(q)

    try:
        mod.validate_query("2402.03300 kanban-feishu-design full doc")
    except ValueError as exc:
        assert "forbidden" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for forbidden substring")


def test_feishu_stage_notify_render():
    import importlib.util

    live = SKILL.parents[2] / "devops" / "kanban-feishu-live" / "scripts" / "kanban_feishu_stage_notify.py"
    spec = importlib.util.spec_from_file_location("kanban_feishu_stage_notify", live)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    session = {
        "canonical_id": "2402.03300",
        "title_zh": "深度求索数学推理",
        "feishu_doc_url": "https://my.feishu.cn/docx/TEST",
        "tasks": {"T0": "t_a", "T1": "t_b"},
    }
    msg = mod.render_message(
        session,
        board="paper-nexus",
        event="stage_done",
        stage="T1",
        summary="测试摘要",
        kb=None,
        conn=None,
    )
    assert "2402.03300" in msg
    assert "深度求索" in msg
    assert "T1" in msg and "完成" in msg
    assert "TEST" in msg


def test_paper_doc_title_zh():
    import importlib.util

    path = SKILL.parent / "scripts" / "paper_doc_title.py"
    spec = importlib.util.spec_from_file_location("paper_doc_title", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    zh = mod.resolve_title_zh(
        {"title": "DeepSeekMath English"},
        handoff={"title_zh": "深度求索数学"},
    )
    assert mod.feishu_doc_title("2402.03300", zh) == "[2402.03300] 深度求索数学"


def test_memory_markdown_entry():
    import importlib.util
    import json
    import tempfile

    path = SKILL.parent / "scripts" / "paper_memory_markdown.py"
    spec = importlib.util.spec_from_file_location("paper_memory_markdown", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    handoff = {
        "paper_id": "2402.03300v3",
        "canonical_id": "2402.03300",
        "thesis_one_liner": "测试论点",
        "feishu_doc_url": "https://my.feishu.cn/docx/TEST",
        "claims": [{"id": "C1", "claim_zh": "x", "strength": "weak"}],
    }
    entry = mod.build_entry("T1", handoff, session_id="sess-1", task_id="t_abcd")
    assert "workflow_id: paper-nexus:2402.03300" in entry
    assert "store" not in entry  # raw entry for MCP, not instruction
    assert "importance_score: 0.75" in entry
    assert "测试论点" in entry


def test_resolve_canonical_id_from_s2_url_without_network():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    cid = mod.resolve_canonical_id(
        "https://www.semanticscholar.org/paper/ceced53f349f7e425352ecf4813b307667cd8aa6"
    )
    assert cid == "s2:ceced53f349f7e425352ecf4813b307667cd8aa6"


def test_metadata_parses_semanticscholar_url():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    cid = mod._s2_corpus_from_raw(
        "https://www.semanticscholar.org/paper/ceced53f349f7e425352ecf4813b307667cd8aa6"
    )
    assert cid == "ceced53f349f7e425352ecf4813b307667cd8aa6"


def test_metadata_script_parses_arxiv_id():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.resolve_canonical_id("https://arxiv.org/abs/2402.03300") == "2402.03300"
    assert mod.resolve_canonical_id("2402.03300v3") == "2402.03300"
