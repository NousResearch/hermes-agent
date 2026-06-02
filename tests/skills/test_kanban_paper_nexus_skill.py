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
    assert "核心公式" in text
    assert "运行时策略细节" in text
    assert "边界分析" in text
    assert "软硬件 Delta 对照表" in text
    assert "【待填" in text  # skeleton for workers
    assert "DeepSeekMath" not in text or "Test Paper" in meta["title"]


def test_bilingual_builder_prefers_stage_handoffs():
    import importlib.util

    build_path = SKILL.parent / "scripts" / "build_bilingual_doc_md.py"
    spec = importlib.util.spec_from_file_location("build_bilingual", build_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    meta = {
        "paper_id": "2403.04652v3",
        "canonical_id": "2403.04652",
        "title": "Yi: Open Foundation Models by 01.AI",
        "summary": "中文摘要。",
        "published": "2024-03-07",
        "authors": ["01.AI Team"],
        "arxiv_abs": "https://arxiv.org/abs/2403.04652",
        "arxiv_pdf": "https://arxiv.org/pdf/2403.04652",
    }
    handoffs = {
        "T0": {"thesis": "Yi 依靠数据工程而非架构魔改提升性能。", "reading_map_sections": ["data_pipeline", "model_architecture"]},
        "T1": {"claims_summary": [{"id": 1, "topic": "data_quality_over_scale", "key_evidence": "Table 2 MMLU 76.3", "key_limit": "contamination risk"}]},
        "T2": {
            "task": "method-and-reproduction",
            "key_numbers": {
                "pretrain_tokens": "3.1T",
                "sft_data_size": "<10K",
                "context_extension_tokens": "5B-10B",
                "context_extension_steps": 100,
                "rope_base_frequency": 10000000.0,
                "base_context": 4096,
                "extended_context": 200000,
                "vit_resolution_stage1": "224x224",
                "vit_resolution_stage2_3": "448x448",
            },
            "github_repo": "https://github.com/01-ai/Yi",
            "license": "Apache 2.0",
            "reproduction_bottlenecks": ["web-scale data curation", "manual SFT verification"],
        },
        "T3": {
            "deliverable": "benchmark-and-open-source-map.md",
            "key_benchmarks": {"yi_34b_mmlu": 76.3, "yi_34b_cmmlu": 82.3, "yi_34b_chat_alpacaeval_winrate": 94.08},
            "model_family": [{"name": "Yi-34B", "type": "Base LLM", "date": "2023.11"}],
        },
        "T4": {
            "audit_type": "experiment_audit_and_limits",
            "dimension_scores": {"data_contamination_check": {"score": 2, "summary": "not checked"}},
            "key_findings": ["No contamination check performed"],
            "applicable_boundaries": {"credible": ["relative ranking among open models"], "not_credible": ["absolute GPT-4 equivalence"]},
        },
    }
    text = mod.build(meta, handoffs=handoffs)
    assert "Table 2 MMLU 76.3" in text
    assert "RoPE / RoPE-ABF" in text
    assert "5B-10B" in text
    assert "No contamination check performed" in text
    assert "relative ranking among open models" in text
    assert "训练规模" in text
    assert "【待填" not in text


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


def test_feishu_stage_notify_thread_reply_uses_messages_reply(monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    live = SKILL.parents[2] / "devops" / "kanban-feishu-live" / "scripts" / "kanban_feishu_stage_notify.py"
    spec = importlib.util.spec_from_file_location("kanban_feishu_stage_notify", live)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    captured = {}

    def _fake_run(args, capture_output=True, text=True, check=False):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    mod.send_feishu_text(
        "oc_test",
        "hello",
        thread_id="om_thread",
        as_identity="bot",
    )
    assert captured["args"][:3] == ["lark-cli", "im", "+messages-reply"]
    assert "--message-id" in captured["args"]
    assert "--reply-in-thread" in captured["args"]
    assert "--thread-id" not in captured["args"]


def test_paper_feishu_live_init_runs_three_steps(monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    path = SKILL.parent / "scripts" / "paper_feishu_live_init.py"
    spec = importlib.util.spec_from_file_location("paper_feishu_live_init", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    calls = []

    def _fake_run(args, capture_output=True, text=True, check=False):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    out = mod.initialize_live(
        "2402.03300",
        '{"T0":"t_a","T1":"t_b"}',
        title_zh="DeepSeekMath",
    )
    assert out["ok"] is True
    assert len(calls) == 3
    assert calls[0][2:].count("init") == 1
    assert "kanban_feishu_subscribe.py" in calls[1][1]
    assert calls[2][-2:] == ["--event", "pipeline_started"]


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


def test_collect_stage_handoffs_scans_workspace(tmp_path, monkeypatch):
    import importlib.util
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    workspaces = tmp_path / "kanban" / "boards" / "paper-nexus" / "workspaces"
    for name, payload in {
        "t0": {"paper_id": "2403.04652v3", "thesis": "x", "reading_map_sections": ["a"]},
        "t1": {"paper_id": "2403.04652v3", "claims_summary": [{"id": 1, "topic": "x", "key_evidence": "Table 2", "key_limit": "y"}]},
        "t2": {"paper_id": "2403.04652v3", "task": "method-and-reproduction", "reproduction_bottlenecks": ["z"]},
        "t3": {"paper_id": "2403.04652v3", "deliverable": "benchmark-and-open-source-map.md", "model_family": [{"name": "Yi-34B"}]},
        "t4": {"paper_id": "2403.04652v3", "audit_type": "experiment_audit_and_limits"},
    }.items():
        d = workspaces / name
        d.mkdir(parents=True)
        (d / "handoff.json").write_text(json.dumps(payload), encoding="utf-8")

    path = SKILL.parent / "scripts" / "paper_feishu_doc_sync.py"
    spec = importlib.util.spec_from_file_location("paper_feishu_doc_sync", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out = mod.collect_stage_handoffs(
        {"paper_id": "2403.04652v3", "canonical_id": "2403.04652"},
        board="paper-nexus",
    )
    assert sorted(out.keys()) == ["T0", "T1", "T2", "T3", "T4"]


def test_collect_stage_handoffs_matches_runtime_worker_shapes(tmp_path, monkeypatch):
    import importlib.util
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    workspaces = tmp_path / "kanban" / "boards" / "paper-nexus" / "workspaces"
    payloads = {
        "t0": {"paper_id": "2410.24164", "title": "pi0", "thesis": "x", "reading_map_sections": ["a"]},
        "t1": {"paper_id": "2410.24164", "stage": "T1", "claims": [{"id": "C1", "claim": "x", "evidence_refs": ["§1"], "strength": "强"}]},
        "t2": {"paper_id": "2410.24164", "artifact": "method-and-reproduction.md", "architecture": {"vlm_backbone": "PaliGemma"}, "training": {"pretraining_data": "10,000+ hours"}, "inference": {"gpu": "RTX 4090"}, "key_formulas": ["L(x)"]},
        "t3": {"canonical_id": "2410.24164", "stage": "T3", "benchmark_map": {"compared_models": ["OpenVLA"]}, "top_3_recommendations": [{"model": "OpenVLA"}]},
        "t4": {"paper_id": "arXiv:2410.24164", "stage": "T3", "audit_scores": {"Q1_baseline_fairness": 3, "overall": 4.5}, "key_findings": ["baseline unfair"], "recommended_actions": ["do x"], "verdict": "needs caution"},
    }
    for name, payload in payloads.items():
        d = workspaces / name
        d.mkdir(parents=True)
        (d / "handoff.json").write_text(json.dumps(payload), encoding="utf-8")

    path = SKILL.parent / "scripts" / "paper_feishu_doc_sync.py"
    spec = importlib.util.spec_from_file_location("paper_feishu_doc_sync", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out = mod.collect_stage_handoffs(
        {"paper_id": "2410.24164", "canonical_id": "2410.24164"},
        board="paper-nexus",
    )
    assert sorted(out.keys()) == ["T0", "T1", "T2", "T3", "T4"]


def test_bilingual_builder_supports_pi05_runtime_handoffs():
    import importlib.util

    build_path = SKILL.parent / "scripts" / "build_bilingual_doc_md.py"
    spec = importlib.util.spec_from_file_location("build_bilingual", build_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    meta = {
        "paper_id": "2504.16054",
        "canonical_id": "2504.16054",
        "title": "π0.5: a Vision-Language-Action Model with Open-World Generalization",
        "title_zh": "π₀.₅：面向开放世界泛化的视觉-语言-动作模型",
        "summary": "中文摘要。",
        "published": "2025-04-01",
        "authors": ["Physical Intelligence"],
        "arxiv_abs": "https://arxiv.org/abs/2504.16054",
        "arxiv_pdf": "https://arxiv.org/pdf/2504.16054",
    }
    handoffs = {
        "T0": {"thesis": "π0.5 通过异构数据共训练实现开放世界泛化。", "sections": ["problem_formulation", "architecture"]},
        "T1": {"claims_covered": ["Co-training improves generalization"], "key_figures_cited": ["Fig. 7"]},
        "T2": {
            "key_findings": {
                "architecture": "300M action expert + FAST token pretraining",
                "data": "6 heterogeneous data sources",
                "training": "280k pretraining → 80k post-training",
                "inference": "hierarchical subtask prediction → continuous action generation",
            },
            "reproduction_requirements": {"gpu_inference": ">8GB", "gpu_finetune_lora": ">22.5GB", "gpu_finetune_full": ">70GB"},
            "code_repo": "https://github.com/Physical-Intelligence/openpi",
            "model_checkpoints": {"base": "gs://openpi-assets/checkpoints/pi05_base"},
        },
        "T3": {
            "comparison_models": ["OpenVLA", "Octo", "π₀.₅"],
            "open_source_status": {"pi0_5_weights": "openpi + HuggingFace"},
            "top_3_relevant_works": [{"work": "OpenVLA", "reason": "baseline", "action": "compare compute-matched"}],
        },
        "T4": {
            "audit_scores": {"Q1_baseline_fairness": 6, "overall": 6.5},
            "key_findings": ["Main figures still have zero error bars"],
            "recommended_actions": ["Short-term: fine-tune open weights", "Mid-term: collect 20-50 environments", "Long-term: replace zero-padding alignment"],
            "verdict": "Direction worth following with caution",
        },
    }
    text = mod.build(meta, handoffs=handoffs)
    assert "6 heterogeneous data sources" in text
    assert "OpenVLA" in text
    assert "zero error bars" in text
    assert "Short-term: fine-tune open weights" in text
    assert "【待填" not in text


def test_sync_paper_doc_create_uses_title_and_markdown(tmp_path, monkeypatch):
    import importlib.util
    import json
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    path = SKILL.parent / "scripts" / "paper_feishu_doc_sync.py"
    spec = importlib.util.spec_from_file_location("paper_feishu_doc_sync", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    meta = {
        "paper_id": "2410.24164",
        "canonical_id": "2410.24164",
        "title": "pi0",
        "summary": "中文摘要。",
        "published": "2024-10-01",
        "authors": ["A"],
        "arxiv_abs": "https://arxiv.org/abs/2410.24164",
        "arxiv_pdf": "https://arxiv.org/pdf/2410.24164",
    }
    monkeypatch.setattr(mod, "resolve_and_fetch", lambda _: dict(meta))
    monkeypatch.setattr(mod, "load_handoff", lambda _: None)
    monkeypatch.setattr(mod, "collect_stage_handoffs", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "resolve_title_zh", lambda *args, **kwargs: "测试中文标题")
    monkeypatch.setattr(mod, "resolve", lambda *args, **kwargs: {"action": "create", "canonical_id": "2410.24164"})

    calls = []

    def _fake_run(args, input=None, capture_output=True, text=True, timeout=120):
        calls.append(args)
        if args[1:3] == ["docs", "+create"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"ok": True, "data": {"doc_url": "https://my.feishu.cn/docx/TESTDOC", "doc_id": "TESTDOC"}}),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    out = mod.sync_paper_doc("2410.24164")
    create_call = next(args for args in calls if args[1:3] == ["docs", "+create"])
    assert "--title" in create_call
    assert "--markdown" in create_call
    assert "--content" not in create_call
    assert "--doc-format" not in create_call
    assert out["doc_url"] == "https://my.feishu.cn/docx/TESTDOC"

    reg = json.loads((tmp_path / "kanban" / "boards" / "paper-nexus" / "paper_doc_registry.json").read_text())
    assert reg["papers"]["2410.24164"]["doc_url"] == "https://my.feishu.cn/docx/TESTDOC"


def test_paper_nexus_status_reports_missing_when_no_tasks(tmp_path, monkeypatch):
    import importlib.util
    import sqlite3

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    board = tmp_path / "kanban" / "boards" / "paper-nexus"
    board.mkdir(parents=True)
    conn = sqlite3.connect(board / "kanban.db")
    conn.execute(
        "create table tasks (id text primary key, title text not null, status text not null, created_at integer not null)"
    )
    conn.commit()
    conn.close()

    path = SKILL.parent / "scripts" / "paper_nexus_status.py"
    spec = importlib.util.spec_from_file_location("paper_nexus_status", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out = mod.inspect_status("2410.24164")
    assert out["exists"] is False
    assert out["next_stage"] == "T0"
    assert out["tasks"]["T0"]["status"] == "missing"


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


def test_metadata_script_parses_openalex_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    expected = "openalex:w2741809807"
    assert mod.resolve_canonical_id("W2741809807") == expected
    assert mod.resolve_canonical_id("https://openalex.org/W2741809807") == expected


def test_metadata_script_parses_doi_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    expected = "doi:10.1126/scirobotics.aau4984"
    assert mod.resolve_canonical_id("10.1126/scirobotics.aau4984") == expected
    assert mod.resolve_canonical_id("https://doi.org/10.1126/scirobotics.aau4984") == expected


def test_metadata_doi_falls_back_to_crossref(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    def _boom(_paper_ref: str):
        raise RuntimeError("s2 unavailable")

    def _fake_crossref(doi: str):
        return {
            "paper_id": f"doi:{doi}",
            "canonical_id": f"doi:{doi}",
            "source": "crossref",
            "title": "Learning ambidextrous robot grasping policies",
            "summary": "",
            "published": "2019-01-30",
            "authors": ["A", "B"],
            "categories": [],
            "arxiv_abs": "",
            "arxiv_pdf": "",
            "s2_url": "",
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}",
            "venue": "Science Robotics",
            "citation_count": 0,
            "influential_citation_count": 0,
        }

    monkeypatch.setattr(mod, "fetch_s2_entry", _boom)
    monkeypatch.setattr(mod, "fetch_crossref_entry", _fake_crossref)
    meta = mod.resolve_and_fetch("10.1126/scirobotics.aau4984")
    assert meta["canonical_id"] == "doi:10.1126/scirobotics.aau4984"
    assert meta["source"] == "crossref"


def test_metadata_openalex_fetch_maps_result(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    def _fake_get_json(_url: str, timeout: int = 45, headers=None):
        return {
            "id": "https://openalex.org/W2741809807",
            "display_name": "Attention Is All You Need",
            "publication_year": 2017,
            "cited_by_count": 123456,
            "abstract_inverted_index": {"Attention": [0], "wins": [1]},
            "ids": {"doi": "https://doi.org/10.48550/arXiv.1706.03762"},
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
            "primary_location": {
                "landing_page_url": "https://openalex.org/W2741809807",
                "source": {"display_name": "NeurIPS"},
            },
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762"},
        }

    monkeypatch.setattr(mod, "_get_json", _fake_get_json)
    meta = mod.fetch_openalex_entry("W2741809807")
    assert meta["canonical_id"] == "doi:10.48550/arxiv.1706.03762"
    assert meta["source"] == "openalex"
    assert meta["venue"] == "NeurIPS"


def test_crossref_url_includes_mailto(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_nexus_metadata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    monkeypatch.setenv("CROSSREF_MAILTO", "user@example.com")
    url = mod._crossref_url("10.1126/scirobotics.aau4984")
    assert "mailto=user%40example.com" in url
