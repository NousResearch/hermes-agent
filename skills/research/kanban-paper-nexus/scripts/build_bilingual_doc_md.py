#!/usr/bin/env python3
"""Build bilingual Feishu markdown, preferring upstream handoffs when available."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _canonical_id(paper_id: str) -> str:
    pid = (paper_id or "").strip()
    if pid.lower().startswith("s2:"):
        return pid.lower()
    return re.sub(r"v\d+$", "", pid, flags=re.I)


def _zh_title_short(title: str, max_len: int = 48) -> str:
    t = title.replace("\n", " ").strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    m = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)
    return m[0] if m else text[:200]


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = "|" + "|".join(rows[0]) + "|"
    sep = "|" + "|".join("-" for _ in rows[0]) + "|"
    body = ["|" + "|".join(str(c) for c in row) + "|" for row in rows[1:]]
    return "\n".join([head, sep, *body])


def _stage(handoffs: dict | None, key: str) -> dict:
    return (handoffs or {}).get(key) or {}


def _render_exec_summary(meta: dict, handoffs: dict | None, thesis_seed: str) -> str:
    t0, t2, t3, t4 = (_stage(handoffs, k) for k in ("T0", "T2", "T3", "T4"))
    if not any((t0, t2, t3, t4)):
        return (
            "**中文（工人 T1/T5 填写，此处为种子句）：**\n"
            f"1. **论点：** {thesis_seed}（T0 应精炼为 ≤40 字）\n"
            "2. **方法：** 【待填：数据 / 模型 / 训练或推理机制】\n"
            "3. **结果：** 【待填：主基准 + 数字 + 设定】\n"
            "4. **局限：** 【待填：证据弱点或外推风险】\n"
            "5. **参考方向：** 【待填：复现 / 产品 / 对标 各 1 条】\n\n"
            "**English:** 【待填：mirror bullets 2–4 lines each】"
        )
    thesis = t0.get("thesis") or thesis_seed
    key_numbers = t2.get("key_numbers") or {}
    methods = (
        f"3.1T bilingual 预训练 + <10K 人工校验 SFT + RoPE ABF 长上下文扩展"
        if key_numbers
        else "经典 Transformer 主干 + 更强数据工程与后训练配方"
    )
    results = []
    kb = t3.get("key_benchmarks") or {}
    if kb.get("yi_34b_mmlu"):
        results.append(f"MMLU {kb['yi_34b_mmlu']}")
    if kb.get("yi_34b_cmmlu"):
        results.append(f"CMMLU {kb['yi_34b_cmmlu']}")
    if kb.get("yi_34b_chat_alpacaeval_winrate"):
        results.append(f"AlpacaEval {kb['yi_34b_chat_alpacaeval_winrate']}%")
    limits = (t4.get("key_findings") or ["缺 contamination 检查，且数学/代码与 GPT-4 有明显差距"])[0]
    return (
        "**中文：**\n"
        f"1. **论点：** {thesis}\n"
        f"2. **方法：** {methods}\n"
        f"3. **结果：** {'；'.join(results) if results else '开源榜单表现强，但结论需锚定表格'}\n"
        f"4. **局限：** {limits}\n"
        "5. **参考方向：** 复现优先验证量化与长上下文；产品化需补 agent / tool-use 内测；对标建议加测 Qwen / Llama 新代模型\n\n"
        "**English:** Thesis / method / result / limits mirrored from the Chinese bullets above."
    )


def _render_cel(handoffs: dict | None) -> str:
    t1 = _stage(handoffs, "T1")
    claims = t1.get("claims_summary") or []
    if not claims and t1.get("claims"):
        claims = [
            {
                "id": str(claim.get("id", "")).removeprefix("C") or str(i + 1),
                "topic": claim.get("claim") or claim.get("claim_zh") or claim.get("topic") or "",
                "key_evidence": "; ".join(claim.get("evidence_refs") or []) or "—",
                "key_limit": claim.get("limit") or claim.get("key_limit") or "—",
            }
            for i, claim in enumerate(t1.get("claims") or [])
        ]
    if not claims and t1.get("claims_covered"):
        figures = t1.get("key_figures_cited") or []
        claims = [
            {
                "id": str(i + 1),
                "topic": claim,
                "key_evidence": figures[min(i, len(figures) - 1)] if figures else "—",
                "key_limit": "—",
            }
            for i, claim in enumerate(t1.get("claims_covered") or [])
        ]
    if not claims:
        return (
            "| ID | 主张 (ZH) | 证据 Evidence | 强度 | 局限 |\n"
            "|----|-----------|---------------|------|------|\n"
            "| C1 | 【T1 填写】 | § / Fig / Table | 强/中/弱 | 【T1】 |\n\n"
            "*English: One row summary after table is finalized.*"
        )
    rows = [["ID", "主张 (ZH)", "证据 Evidence", "强度", "局限"]]
    for claim in claims:
        rows.append(
            [
                f"C{claim['id']}",
                str(claim.get("topic", "")).replace("_", " "),
                claim.get("key_evidence", "—"),
                "中",
                claim.get("key_limit", "—"),
            ]
        )
    return _md_table(rows) + "\n\n*English: CEL table is evidence-anchored and should be mirrored only after final QA.*"


def _render_reading_map(handoffs: dict | None) -> str:
    t0 = _stage(handoffs, "T0")
    sections = t0.get("reading_map_sections") or t0.get("sections") or []
    if not sections:
        return "**中文（T0）：** 【待填：Abstract → 图1 → Method § → 主实验表 → Appendix 要点】\n\n*English: Section walk order.*"
    seq = " → ".join(str(s).replace("_", " ") for s in sections)
    return f"**中文（T0）：** {seq}\n\n*English: Read data pipeline first, then architecture, finetuning, evaluation, and infrastructure.*"


def _render_problem(meta: dict, handoffs: dict | None) -> str:
    t0 = _stage(handoffs, "T0")
    thesis = t0.get("thesis") or _first_sentence(meta["summary"])
    return f"**中文：** {thesis}\n\n*English: The paper argues that stronger data engineering and post-training can push a classical Transformer to a much better efficiency frontier.*"


def _render_method(handoffs: dict | None) -> str:
    t2 = _stage(handoffs, "T2")
    if not t2:
        return (
            "**中文：**\n"
            "- 数据：【待填】\n"
            "- 模型/算法：【待填】\n"
            "- 复现清单：代码 / 权重 / 算力 / 关键超参 【待填】\n\n"
            "*English: 【待填】*"
        )
    if t2.get("architecture") or t2.get("training") or t2.get("reproduction_feasibility"):
        arch = t2.get("architecture") or {}
        training = t2.get("training") or {}
        repro = t2.get("reproduction_feasibility") or {}
        return (
            "**中文：**\n"
            f"- 数据：{training.get('pretraining_data', '—')}；post-training {training.get('posttraining_data_range', '—')}\n"
            f"- 模型/算法：{arch.get('vlm_backbone', '—')} + {arch.get('action_expert', '—')}；总参数 {arch.get('total_params', '—')}；action horizon {arch.get('action_horizon', '—')}；flow steps {arch.get('flow_steps', '—')}\n"
            f"- 复现清单：公开权重 {repro.get('model_weights', '—')}；开源数据 {repro.get('open_data_only', '—')}；完整复现 {repro.get('full_reproduction_requires', '—')}\n\n"
            "*English: Data, architecture, and reproduction constraints are mirrored from the Chinese bullets above.*"
        )
    if t2.get("key_findings") or t2.get("reproduction_requirements"):
        findings = t2.get("key_findings") or {}
        repro = t2.get("reproduction_requirements") or {}
        checkpoints = ", ".join((t2.get("model_checkpoints") or {}).keys()) or "—"
        return (
            "**中文：**\n"
            f"- 数据：{findings.get('data', '—')}\n"
            f"- 模型/算法：{findings.get('architecture', '—')}\n"
            f"- 训练/推理：{findings.get('training', '—')}；{findings.get('inference', '—')}\n"
            f"- 复现清单：代码 {t2.get('code_repo', '—')}；checkpoint {checkpoints}；硬件要求 推理 {repro.get('gpu_inference', '—')} / LoRA {repro.get('gpu_finetune_lora', '—')} / 全参 {repro.get('gpu_finetune_full', '—')}\n\n"
            "*English: Method, training, and reproduction requirements are mirrored from the Chinese bullets above.*"
        )
    kn = t2.get("key_numbers") or {}
    bottlenecks = t2.get("reproduction_bottlenecks") or []
    return (
        "**中文：**\n"
        f"- 数据：预训练 {kn.get('pretrain_tokens', '—')}；SFT {kn.get('sft_data_size', '—')}；长上下文继续预训练 {kn.get('context_extension_tokens', '—')}\n"
        f"- 模型/算法：RoPE base {kn.get('rope_base_frequency', '—')}；长上下文约 {kn.get('context_extension_steps', '—')} steps；ViT 分辨率 {kn.get('vit_resolution_stage1', '—')} → {kn.get('vit_resolution_stage2_3', '—')}\n"
        f"- 复现清单：代码 {t2.get('github_repo', '—')}；许可证 {t2.get('license', '—')}；瓶颈 {'；'.join(bottlenecks[:3]) if bottlenecks else '【待填】'}\n\n"
        "*English: Data / algorithm / reproduction checklist mirrored from the Chinese bullets above.*"
    )


def _render_core_math(handoffs: dict | None) -> str:
    t2 = _stage(handoffs, "T2")
    if not t2:
        return (
            "**中文：**\n"
            "- 目标函数 / 奖励 / 损失：【待填】\n"
            "- 符号释义：【待填】\n"
            "- 为什么它决定系统行为：【待填】\n\n"
            "*English: 【待填】*"
        )
    if t2.get("key_formulas"):
        bullets = "\n".join(f"- {formula}" for formula in (t2.get("key_formulas") or [])[:4])
        return (
            "**中文：**\n"
            "- 本文核心是 **flow matching 连续动作生成**，并非提出全新的闭式优化目标；公式重点在损失、Euler 积分、动作嵌入与时间步采样。\n"
            f"{bullets}\n"
            "- 为什么它决定系统行为：10 步 Euler 积分 + 50 步 action chunk 共同决定了精度/延迟折中。\n\n"
            "*English: The mathematical core is conditional flow matching for continuous action generation, not a brand-new closed-form objective.*"
        )
    if t2.get("key_findings"):
        findings = t2.get("key_findings") or {}
        return (
            "**中文：**\n"
            "- 本文核心不是新闭式目标函数，而是 **FAST 离散 token 预训练 + flow matching 连续动作推理** 的混合动作表示。\n"
            f"- 架构要点：{findings.get('architecture', '—')}\n"
            f"- 推理要点：{findings.get('inference', '—')}\n"
            "- 为什么它决定系统行为：高层语义子任务预测与低层连续动作生成的分层链路，决定了开放世界长程操作的泛化与实时性折中。\n\n"
            "*English: The math-bearing novelty is the hybrid FAST-token pretraining plus flow-matching continuous control stack, not a new closed-form loss family.*"
        )
    base = t2.get("key_numbers", {}).get("rope_base_frequency")
    return (
        "**中文：**\n"
        "- 本文**没有新的闭式目标函数**；核心仍是标准 decoder-only Transformer attention，而不是提出新 loss / reward。\n"
        "- 关键机制 1：`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`，决定主干推理行为。\n"
        f"- 关键机制 2：RoPE / RoPE-ABF（base ≈ {base if base is not None else '【待填】'}），决定位置编码在长上下文里的行为边界。\n"
        "- 为什么它决定系统行为：Yi 的贡献更多在数据与训练策略，因此公式解释的重点应放在“哪些是经典机制、哪些不是论文原创”。\n\n"
        "*English: No new closed-form objective is introduced; the paper mainly improves data engineering and training recipe on top of a classical Transformer backbone.*"
    )


def _render_runtime(handoffs: dict | None) -> str:
    t2 = _stage(handoffs, "T2")
    if not t2:
        return (
            "**中文：**\n"
            "- 输入表征：【待填】\n"
            "- 候选生成 / 搜索 / 解码：【待填】\n"
            "- 关键超参 / 近似：【待填】\n"
            "- 部署坑点：【待填】\n\n"
            "*English: 【待填】*"
        )
    if t2.get("inference") or t2.get("architecture"):
        arch = t2.get("architecture") or {}
        inf = t2.get("inference") or {}
        return (
            "**中文：**\n"
            "- 输入表征：视觉 + 语言 + 本体状态共同进入 VLM/backbone，再接 flow matching action expert。\n"
            f"- 候选生成 / 搜索 / 解码：{arch.get('flow_steps', '—')} 步 Euler 积分生成 {arch.get('action_horizon', '—')} 步动作 chunk。\n"
            f"- 关键超参 / 近似：20Hz = {inf.get('control_rate_20hz', '—')}；50Hz = {inf.get('control_rate_50hz', '—')}。\n"
            f"- 部署坑点：{inf.get('gpu', 'GPU')} 上延迟 onboard {inf.get('onboard_latency_ms', '—')} ms / offboard {inf.get('offboard_latency_ms', '—')} ms，边缘部署需量化验证。\n\n"
            "*English: Runtime behavior is dominated by multi-step Euler integration, chunked action generation, and GPU latency constraints.*"
        )
    if t2.get("key_findings") or t2.get("reproduction_requirements"):
        findings = t2.get("key_findings") or {}
        repro = t2.get("reproduction_requirements") or {}
        return (
            "**中文：**\n"
            f"- 输入表征：{findings.get('architecture', '—')}\n"
            f"- 候选生成 / 搜索 / 解码：{findings.get('inference', '—')}\n"
            f"- 关键超参 / 近似：训练 {findings.get('training', '—')}\n"
            f"- 部署坑点：推理显存 {repro.get('gpu_inference', '—')}；LoRA 微调 {repro.get('gpu_finetune_lora', '—')}；全参 {repro.get('gpu_finetune_full', '—')}；OS {repro.get('os', '—')}\n\n"
            "*English: Runtime behavior depends on the hierarchical subtask-to-action chain and substantial GPU memory requirements.*"
        )
    kn = t2.get("key_numbers") or {}
    return (
        "**中文：**\n"
        f"- 输入表征：基座 seq len {kn.get('base_context', '—')}，长上下文扩展到 {kn.get('extended_context', '—')}。\n"
        f"- 候选生成 / 搜索 / 解码：长上下文依赖 continual pretraining（{kn.get('context_extension_tokens', '—')}，约 {kn.get('context_extension_steps', '—')} steps），而不是结构性 sparse attention 改写。\n"
        f"- 关键超参 / 近似：RoPE base {kn.get('rope_base_frequency', '—')}；ViT stage1 {kn.get('vit_resolution_stage1', '—')}，stage2/3 {kn.get('vit_resolution_stage2_3', '—')}。\n"
        f"- 部署坑点：{'；'.join((t2.get('reproduction_bottlenecks') or [])[:3]) if t2.get('reproduction_bottlenecks') else '长上下文与量化版本需分开验证'}。\n\n"
        "*English: Runtime behavior is dominated by continual pretraining, long-context adaptation, and serving tradeoffs rather than by a novel inference objective.*"
    )


def _render_related(handoffs: dict | None) -> str:
    t3 = _stage(handoffs, "T3")
    fam = t3.get("model_family") or []
    if not fam and t3.get("benchmark_map"):
        compared = t3.get("benchmark_map", {}).get("compared_models") or []
        fam = [{"name": name, "type": "benchmark peer", "date": "—"} for name in compared]
    if not fam and t3.get("comparison_models"):
        fam = [{"name": name, "type": "comparison model", "date": "—"} for name in (t3.get("comparison_models") or [])]
    if not fam and t3.get("top_3_relevant_works"):
        fam = [
            {"name": item.get("work", "—"), "type": item.get("action", "related work"), "date": item.get("reason", "—")}
            for item in (t3.get("top_3_relevant_works") or [])
        ]
    if not fam:
        return (
            "| 工作/项目 | 关系 | 为何相关 |\n"
            "|-----------|------|----------|\n"
            "| 【T3】 | 竞争/工具 | … |"
        )
    rows = [["工作/项目", "关系", "为何相关"]]
    for item in fam[:6]:
        rows.append([item["name"], item.get("type", "—"), item.get("date", "—")])
    return _md_table(rows)


def _render_audit(handoffs: dict | None) -> str:
    t4 = _stage(handoffs, "T4")
    dims = t4.get("dimension_scores") or {}
    if not dims and t4.get("audit_scores"):
        labels = {
            "Q1_baseline_fairness": "baseline fairness",
            "Q2_metrics_sufficiency": "metrics sufficiency",
            "Q3_variance_stability": "variance stability",
            "Q4_ablation_sufficiency": "ablation sufficiency",
            "Q5_reproducibility": "reproducibility",
        }
        dims = {
            labels[key]: {"score": value, "summary": "see key findings"}
            for key, value in (t4.get("audit_scores") or {}).items()
            if key != "overall"
        }
    if not dims:
        return (
            "| 检查项 | 结论 (ZH) |\n"
            "|--------|-----------|\n"
            "| Baseline 公平性 | 【T4】 |\n"
            "| 指标对口 | 【T4】 |\n"
            "| 方差/多次运行 | 【T4】 |\n"
            "| 消融 | 【T4】 |\n"
            "| 可复现性 | 【T4】 |"
        )
    rows = [["检查项", "结论 (ZH)"]]
    for label, data in dims.items():
        rows.append([label.replace("_", " "), f"{data.get('score', '—')}/10 · {data.get('summary', '—')}"])
    return _md_table(rows)


def _render_boundary(handoffs: dict | None) -> str:
    t4 = _stage(handoffs, "T4")
    if not t4:
        return (
            "**中文：**\n"
            "- 今天仍成立的假设：【待填】\n"
            "- 今天已失效的假设：【待填】\n"
            "- 可替换的新范式（如端侧 VLM / VLA）：【待填】\n\n"
            "*English: 【待填】*"
        )
    app = t4.get("applicable_boundaries") or {}
    credible = "；".join(app.get("credible") or ["flow matching + VLM 组合值得继续跟踪"])
    not_credible = "；".join(app.get("not_credible") or [t4.get("verdict") or "【待填】"])
    findings = "；".join((t4.get("key_findings") or [])[:2]) or "【待填】"
    return (
        "**中文：**\n"
        f"- 今天仍成立的假设：{credible}\n"
        f"- 今天已失效的假设：{not_credible}\n"
        f"- 可替换的新范式（如端侧 VLM / VLA）：优先替换后训练、agent/tool-use 评测与多模态交互层；原因：{findings}\n\n"
        "*English: The paper remains useful as an engineering reference, but not as a direct proxy for 2026 production-model selection.*"
    )


def _render_delta(handoffs: dict | None) -> str:
    t2, t3 = _stage(handoffs, "T2"), _stage(handoffs, "T3")
    if not any((t2, t3)):
        return (
            "| 维度 | 论文设定 | 我们当前设定 | Delta / 影响 |\n"
            "|------|----------|--------------|--------------|\n"
            "| 感知 | 【待填】 | 【待填】 | 【待填】 |\n"
            "| 末端执行器 | 【待填】 | 【待填】 | 【待填】 |\n"
            "| 训练数据 | 【待填】 | 【待填】 | 【待填】 |\n"
            "| 推理预算 | 【待填】 | 【待填】 | 【待填】 |"
        )
    if t2.get("architecture") or t3.get("benchmark_map"):
        arch = t2.get("architecture") or {}
        training = t2.get("training") or {}
        pi0 = (t3.get("benchmark_map") or {}).get("pi0_positioning") or {}
        rows = [
            ["维度", "论文设定", "我们当前设定", "Delta / 影响"],
            ["模型架构", f"{arch.get('vlm_backbone', '—')} + {arch.get('action_expert', '—')}", "通常消费现成开源权重", "更适合作为架构借鉴，不是原样复刻"],
            ["训练数据", training.get("pretraining_data", "—"), "多为 <100 小时自有数据", "数据规模差距大，需缩成 single-embodiment MVP"],
            ["动作表示", pi0.get("action_representation", "连续控制"), "依具体栈而定", "需实测高频控制收益"],
            ["部署形态", pi0.get("open_source_status", "闭源"), "更偏本地/边缘", "量化与蒸馏是必经之路"],
        ]
        return _md_table(rows)
    if t2.get("reproduction_requirements") or t3.get("open_source_status"):
        repro = t2.get("reproduction_requirements") or {}
        oss = t3.get("open_source_status") or {}
        rows = [
            ["维度", "论文设定", "我们当前设定", "Delta / 影响"],
            ["训练数据", (t2.get("key_findings") or {}).get("data", "—"), "通常远小于论文规模", "需收缩到最小闭环验证"],
            ["推理硬件", repro.get("gpu_inference", "—"), "本地单机/少量 GPU", "先验证推理延迟和显存"],
            ["微调成本", repro.get("gpu_finetune_lora", "—"), "更偏 LoRA/QLoRA", "优先小样本适配"],
            ["开源状态", oss.get("pi0_5_weights", "—"), "依赖开源 checkpoint", "可落地性显著强于 π0"],
        ]
        return _md_table(rows)
    kn = t2.get("key_numbers") or {}
    rows = [
        ["维度", "论文设定", "我们当前设定", "Delta / 影响"],
        ["训练规模", f"{kn.get('pretrain_tokens', '—')} + {kn.get('sft_data_size', '—')} SFT", "通常只能消费现成权重", "更多是范式启发，不是原样复刻"],
        ["长上下文", f"{kn.get('base_context', '—')} → {kn.get('extended_context', '—')}", "更常见是直接评估现成 checkpoint", "验证推理效果，不验证原始扩展工艺"],
        ["部署硬件", "论文强调消费级量化部署", "本地单机/少量 GPU", "200K 版本需单独评估 KV cache / 吞吐"],
        ["评测目标", "academic benchmarks + Arena/AlpacaEval", "自有任务 / 工具调用 / agent workflow", "论文分数不能直接替代上线内测"],
    ]
    return _md_table(rows)


def _render_next_steps(handoffs: dict | None) -> str:
    t2, t3, t4 = (_stage(handoffs, k) for k in ("T2", "T3", "T4"))
    if not any((t2, t3, t4)):
        return (
            "**中文（必填，可执行）：**\n"
            "- **复现：** 【读哪一节、要什么资源】\n"
            "- **产品化：** 【场景 + 风险】\n"
            "- **对标：** 【2–3 篇后续工作或 repo】\n\n"
            "**English:** 【mirror】"
        )
    if t4.get("recommended_actions"):
        actions = t4.get("recommended_actions") or []
        compare = [item.get("model") for item in (t3.get("top_3_recommendations") or []) if item.get("model")]
        return (
            "**中文（必填，可执行）：**\n"
            f"- **复现：** {actions[0] if len(actions) > 0 else '【待填】'}\n"
            f"- **产品化：** {actions[1] if len(actions) > 1 else '【待填】'}\n"
            f"- **对标：** {actions[2] if len(actions) > 2 else (' / '.join(compare) if compare else '【待填】')}\n\n"
            "**English:** Reproduction, productization, and benchmarking directions are mirrored from the Chinese bullets above."
        )
    compare = []
    for item in (t3.get("model_family") or [])[-3:]:
        compare.append(item["name"])
    product_risk = (t4.get("key_findings") or ["缺 contamination 审计"])[0]
    return (
        "**中文（必填，可执行）：**\n"
        f"- **复现：** 从数据流水线、长上下文 continual pretraining、量化 serving 三块入手；主要瓶颈：{'；'.join((t2.get('reproduction_bottlenecks') or [])[:2]) if t2.get('reproduction_bottlenecks') else '【待填】'}\n"
        f"- **产品化：** 先做本地 4-bit / 长上下文 / 工具调用三组内测；核心风险：{product_risk}\n"
        f"- **对标：** 建议与 {' / '.join(compare) if compare else 'Qwen / Llama 新代模型'} 做同任务横评\n\n"
        "**English:** Reproduction, productization, and benchmarking directions are mirrored from the Chinese bullets above."
    )


def build(
    meta: dict,
    marker: str = "",
    *,
    stage_notes: str = "",
    handoffs: dict | None = None,
) -> str:
    """Build doc body, filling sections from upstream handoffs when available."""
    title = meta.get("title_zh") or meta["title"]
    pid = meta.get("paper_id") or meta.get("canonical_id") or ""
    cid = meta.get("canonical_id") or _canonical_id(pid)
    link = meta.get("arxiv_abs") or meta.get("s2_url") or "—"
    pdf = meta.get("arxiv_pdf") or "—"
    doi = meta.get("doi") or "—"
    venue = meta.get("venue") or "—"
    authors = ", ".join(meta["authors"][:6])
    if len(meta["authors"]) > 6:
        authors += " et al."
    cats = ", ".join(meta.get("categories") or [])[:120]
    summary_zh = meta["summary"].strip()
    thesis_seed = _first_sentence(summary_zh)
    summary_en = summary_zh[:500] + ("…" if len(summary_zh) > 500 else "")

    notes_block = f"\n\n> {stage_notes}\n" if stage_notes else ""
    marker_row = f"| 批次 Run | {marker} |\n" if marker else ""

    executive_summary = _render_exec_summary(meta, handoffs, thesis_seed)
    cel_block = _render_cel(handoffs)
    reading_map = _render_reading_map(handoffs)
    problem_block = _render_problem(meta, handoffs)
    method_block = _render_method(handoffs)
    core_math = _render_core_math(handoffs)
    runtime_details = _render_runtime(handoffs)
    related_work = _render_related(handoffs)
    audit_block = _render_audit(handoffs)
    boundary_block = _render_boundary(handoffs)
    delta_block = _render_delta(handoffs)
    next_steps = _render_next_steps(handoffs)

    # Use ## here: create path prepends a single `# {feishu_doc_title}` as the online doc name.
    return f"""## 论文精读 / Paper Brief：{_zh_title_short(title)}

| 项 Item | 内容 |
|---------|------|
| canonical_id | `{cid}` |
| 链接 Link | {link} |
| DOI | {doi} |
| 期刊 Venue | {venue} |
| 发表 Published | {meta['published']} |
| 领域 Categories | {cats or '—'} |
| 作者 Authors | {authors} |
| PDF | {pdf} |
| 流水线 Pipeline | `paper-nexus` · `/kanban-paper-nexus` |
{marker_row}{notes_block}
---

## 核心总结（30 秒读懂）/ Executive Summary

{executive_summary}

---

## 主张–证据–局限 / Claims–Evidence–Limits

{cel_block}

---

## 阅读地图 / Reading Map

{reading_map}

---

## 问题与动机 / Problem & Motivation

{problem_block}

---

## 方法与复现要点 / Method & Reproducibility

{method_block}

---

## 核心公式 / Core Math

{core_math}

---

## 运行时策略细节 / Runtime Strategy Details

{runtime_details}

---

## 对标与开源地图 / Related Work & Open Source

{related_work}

---

## 实验审计 / Experiment Audit

{audit_block}

---

## 边界分析 / Boundary Analysis

{boundary_block}

---

## 软硬件 Delta 对照表 / HW-SW Delta Table

{delta_block}

---

## 参考方向 / Where to Go Next

{next_steps}

---

## 开源与工具 / Tools

| 项目 | 用途 | 链接 |
|------|------|------|
| PaperQA | PDF RAG | https://github.com/Future-House/paper-qa |
| Hermes arxiv | 元数据 | `skills/research/arxiv` |
| GROBID | PDF 结构 | https://github.com/kermitt2/grobid |

---

## 摘要摘录 / Abstract Excerpt

**中文：** {summary_zh[:700]}{'…' if len(summary_zh) > 700 else ''}

**English (placeholder):** {summary_en}
"""


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: build_bilingual_doc_md.py <metadata.json> [marker]", file=sys.stderr)
        return 2
    meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    marker = sys.argv[2] if len(sys.argv) > 2 else ""
    sys.stdout.write(build(meta, marker))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
