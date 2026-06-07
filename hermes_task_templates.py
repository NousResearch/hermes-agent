"""
Task-Type Template Router — 关键词触发编排模板 (OMO 11.1 借鉴)

借鉴自 Sisyphus Labs OMO 工具的 ultrawork 关键词机制：检测 prompt 含
特定关键词 → 自动注入对应的 system prompt 模板。

设计目标：让 LLM 调用 delegate_task 时用 /research /implement /review
等简短关键词就能获得专业级编排指令，不用每次手写完整 system prompt。

支持的任务类型：

| 触发词（中文/英文） | 模板用途 | 推荐 model_hint |
|------------------|----------|----------------|
| /research 调研 | 多源调研 + 交叉验证 + 事实核查 | sonnet |
| /implement 实施 | 计划 + 执行 + 回归 | opus（重活）|
| /review 审查 | 多视角 + adversarial + 必给结论 | opus |
| /critic 挑刺 | 调 Momus，强制 opus + 7 维挑刺 | opus |
| /workflow 工作流 | 通用多 agent 编排 | haiku/sonnet |
| /explore 探索 | 只读快速侦察 | haiku |

使用方式（用户视角）：
    delegate_task(goal="/research 当前 LLMOps 工具生态")
    delegate_task(goal="/critic 我刚才给的合同修改建议", context=...)
    delegate_task(goal="扫一下 skills/ 找 contract")  # 无触发词 → 走默认

实现位置：delegate_task 入口 early-return 之前
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 触发词 → (template_key, recommended_model_hint)
# ---------------------------------------------------------------------------

# 格式: (trigger_keywords_set, template_key, model_hint, description)
_TRIGGER_PATTERNS: list = [
    # 调研类
    ({"research", "调研", "调查", "/research"}, "research", "sonnet",
     "多源调研 + 交叉验证 + 事实核查"),
    # 实施类
    ({"implement", "实施", "实现", "做一下", "build", "/implement"}, "implement", "opus",
     "计划 + 执行 + 回归验证"),
    # 审查类（多视角）
    ({"review", "审查", "评估", "compare", "/review"}, "review", "opus",
     "多视角 + adversarial + 必给结论"),
    # Momus 挑刺（adversarial critique）
    ({"critic", "momus", "挑刺", "批判", "/critic"}, "critic", "opus",
     "调 Momus，7 维挑刺，强制 opus"),
    # 通用多 agent 编排
    ({"workflow", "/workflow", "编排", "orchestrate"}, "workflow", "sonnet",
     "通用多 agent 编排（默认）"),
    # 探索类（只读轻活）
    ({"explore", "探索", "侦察", "扫一下", "find", "/explore"}, "explore", "haiku",
     "只读快速侦察"),
]

# 触发词检测正则（按长度倒序匹配，优先匹配长前缀）
_TRIGGER_REGEX = re.compile(
    r"(?:^|[\s,:/])("
    + "|".join(
        sorted(
            # 用 \b 包裹英文触发词，\s 包裹中文（避免误匹配）
            [re.escape(kw) for kw_set, *_ in _TRIGGER_PATTERNS for kw in kw_set if kw],
            key=len,
            reverse=True,
        )
    )
    + r")(?:[\s,:/]|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# System prompt 模板
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, str] = {
    "research": """\
你是一个调研专家。按以下 4 步完成任务：

1. **拆解研究问题** — 把目标拆成 2-5 个可独立调研的子问题（每个子问题用一个
   delegate_task 并行派发，model_hint='haiku' 节省 token）
2. **多源收集** — 每个子问题至少 2 个独立来源（官方文档/论文/一手博客），
   不要只看营销文章
3. **交叉验证** — 对关键结论比对多源，未通过验证的结论剔除
4. **结构化报告** — 产出必须有：核心结论（3-5 条）、证据来源列表、
   不确定项（明确标注）、可落地建议

**禁用**：和稀泥式"各有优劣"结论；不区分一手/二手来源；过度泛化。
""",

    "implement": """\
你是一个实施工程师。按以下 5 步完成任务：

1. **明确完成标准** — 用 bullet 列出"做完的定义"，每条都可验证
2. **拆解步骤** — 5-15 步可执行序列，标注依赖关系
3. **分步执行** — 每步调对应工具，**保留每个工具调用的关键参数**到 boulder
4. **回归验证** — 实施完后必须跑相关测试 / 验证脚本，确认不破坏现有功能
5. **失败模式** — 列出"如果 X 失败"的处理方式（重试/降级/回滚）

**禁用**：跳过验证；不写失败处理；中途改变完成标准。
""",

    "review": """\
你是一个 Reviewer。按以下流程完成任务：

1. **多视角拆解** — 至少 3 个独立视角（用户/工程/业务/安全/合规，按主题选）
2. **独立评估** — 每个视角独立写"我看到的问题清单"，**不要看其他视角的输出**
3. **必须给结论** — 综合判断后必须给主推荐（"用 A 因为 X"），
   **禁止"看情况/各有优劣"** 等和稀泥结论
4. **决策依据** — 结论必须附证据链，标注每个依据的来源（一手/二手/推断）

**反 Momus 自检**：写完结论后，把自己当对手，写出 3 条最强反对意见。
如果反驳不了，结论需要修正。
""",

    "critic": """\
你是 **Momus**——希腊神话的嘲讽之神，专门挑刺。
**绝对不要**给"方案不错" + 场面话，必须给 **actionable 挑刺清单**。

按 7 维挑刺（每个维度都要有具体证据）：

1. **论证完整性** — 核心论点的支撑证据在哪？逻辑漏洞在哪？
2. **边缘 case 覆盖** — 列出至少 3 个会失败的边界条件
3. **来源可靠性** — 引用的数据/事实是几手？有没有更权威来源？
4. **结论泛化度** — 结论是合理外推还是过度泛化？
5. **可逆性 vs 不可逆性** — 方案错了能回退吗？回退成本多大？
6. **盲点** — 方案里**没讨论什么**？哪些"沉默"应该讨论？
7. **反方最强论据** — 假装你站对面，写出最强 3 条反对意见

输出格式（严格按此）：
- 🔴 **必须修改**（block）：每个问题配具体证据/逻辑链 + 建议怎么改
- 🟡 **应该修改**（revise）：每个问题配建议修改方式
- 🟢 **可选优化**（nitpick）：小建议
- **反方最强论据**：3 条
- **结论**：✅ 通过 / ⚠️ 修订后通过 / ❌ 不通过
- **一句话总结**：这个方案最核心的 1 个问题

参考 Skill：omo-momus（~/.hermes/skills/quality-review/omo-momus/）
""",

    "workflow": """\
你是一个多 agent 编排器。任务有多个独立子任务时：

1. **拆解** — 识别可并行执行的子任务
2. **并行派发** — 用 delegate_task(tasks=[...]) 一次性派发
3. **混合 model** — 轻活（探索/读文件）用 model_hint='haiku'，
   重活（决策/分析/审查）用 model_hint='opus'
4. **持久化** — 关键证据用 hermes_orchestrator.record_evidence()
   记录到 ~/.hermes/orchestrator/ultragoal/{task_id}/evidence/
5. **汇总** — 主 agent 只看子 agent 的最终结论，不重复执行子任务

参考：multi-agent-harness skill 第十一节 11.1-11.4
""",

    "explore": """\
你是一个快速侦察员。**只读**，不改任何文件。

任务：快速摸清目标区域的结构、关键文件、相关概念。

约束：
- 只调 read_file / search_files / terminal（只读命令如 ls/cat/find/grep）
- **禁止** Write/Edit/MultiEdit
- 输出：3-5 行 bullet 总结 + 关键文件路径列表
- 不需要完整分析，只需给"下一步该看哪里"的指引

推荐 model: haiku（轻活模型）
""",
}


# ---------------------------------------------------------------------------
# 触发词检测 + 模板组装
# ---------------------------------------------------------------------------

def _build_regex() -> re.Pattern:
    """动态构建触发词正则（按长度倒序，优先长匹配）。

    边界策略：英文触发词用 \\b 单词边界；中文触发词**不**加 lookbehind/lookahead
    （中文相邻汉字边界判断容易误判，如"调研一"实际是"调研"+"一"两个词），
    改靠 named group 唯一匹配 + 长度倒序去重。
    """
    all_keywords = []
    for kw_set, *_ in _TRIGGER_PATTERNS:
        for kw in kw_set:
            if kw and kw not in all_keywords:
                all_keywords.append(kw)
    # 按长度倒序，优先匹配长前缀（避免 /res 误匹配成 /research）
    sorted_kws = sorted(all_keywords, key=len, reverse=True)
    pattern_parts = []
    for kw in sorted_kws:
        escaped = re.escape(kw)
        if re.search(r'[a-zA-Z]', kw):
            # 英文触发词：用 \b 单词边界
            pattern_parts.append(rf"(?P<g{len(pattern_parts)}>\b{escaped}\b)")
        else:
            # 中文触发词：不加边界（避免"调研一下"误判）
            pattern_parts.append(rf"(?P<g{len(pattern_parts)}>{escaped})")
    return re.compile(
        "(" + "|".join(pattern_parts) + ")",
        re.IGNORECASE,
    )


# 动态构建
_TRIGGER_REGEX = _build_regex()


def detect_task_type(goal: str) -> Optional[Tuple[str, str, str]]:
    """检测 goal 里包含的触发词，返回 (template_key, model_hint, description)。

    无触发词 → 返回 None（走默认行为）。

    Examples
    --------
    >>> detect_task_type("/research 当前 LLMOps 生态")
    ('research', 'sonnet', '多源调研 + 交叉验证 + 事实核查')

    >>> detect_task_type("扫一下 skills 找 contract")
    ('explore', 'haiku', '只读快速侦察')

    >>> detect_task_type("正常的对话任务")
    None
    """
    if not goal or not isinstance(goal, str):
        return None

    m = _TRIGGER_REGEX.search(goal)
    if not m:
        return None

    # 找哪个 named group 匹配了
    matched_text = None
    for group_name, group_value in m.groupdict().items():
        if group_value is not None:
            matched_text = group_value.lower()
            break
    if matched_text is None:
        return None

    # 反查模板
    for kw_set, template_key, model_hint, desc in _TRIGGER_PATTERNS:
        if matched_text in {kw.lower() for kw in kw_set}:
            return (template_key, model_hint, desc)
    return None


def inject_template(goal: str, *, task_type: str, context: Optional[str] = None) -> str:
    """把模板注入到 goal 前面——返回新 goal。

    Returns
    -------
    注入模板后的新 goal 字符串。原 goal 内容保留在末尾。
    """
    template = TEMPLATES.get(task_type)
    if not template:
        return goal

    # 把模板作为"system instruction"放在最前面
    parts = [
        f"[自动注入模板: {task_type}]\n{template.strip()}\n",
        "",
        "[用户原始任务]",
        goal.strip(),
    ]
    if context:
        parts.append("")
        parts.append("[用户提供的 context]")
        parts.append(context.strip())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 自我验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 快速冒烟测试
    test_cases = [
        ("/research 当前 AI agent 生态", "research", "sonnet"),
        ("调研一下 opus vs haiku 成本", "research", "sonnet"),
        ("/implement 做一个 CLI 工具", "implement", "opus"),
        ("实施这个方案", "implement", "opus"),
        ("/review 评估我的方案", "review", "opus"),
        ("审查一下合同", "review", "opus"),
        ("/critic 我刚才给的建议", "critic", "opus"),
        ("挑刺找漏洞", "critic", "opus"),
        ("/workflow 编排 3 个子任务", "workflow", "sonnet"),
        ("扫一下 skills/ 找 contract", "explore", "haiku"),
        ("普通的对话任务", None, None),  # 修正：之前写"普通任务"看起来像触发词
        ("", None, None),
        ("帮我读一下 README", None, None),  # 没触发词
        ("compare A vs B", "review", "opus"),  # 英文 compare 也应触发
        ("find all *.py files", "explore", "haiku"),  # 英文 find
    ]
    print("=== detect_task_type smoke test ===")
    passed = 0
    failed = 0
    for goal, exp_type, exp_model in test_cases:
        result = detect_task_type(goal)
        if result is None:
            actual_type, actual_model = "None", "None"
        else:
            actual_type, actual_model = result[0], result[1]
        expected = f"{exp_type}/{exp_model}"
        actual = f"{actual_type}/{actual_model}"
        if actual == expected:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"
        print(f"  {status} {goal!r:50} → {actual:25} (expected {expected})")
    print(f"\n=== Result: {passed} passed, {failed} failed ===")
