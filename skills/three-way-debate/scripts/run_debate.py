#!/usr/bin/env python3
"""
三方辩论 — 子智能体 prompt 生成器 v2.1
用法: python run_debate.py "你的议题" [--mode quick|standard|deep] [--lang zh|en]
      python run_debate.py "议题" --mode standard --round1-radical "..." --round1-conservative "..." --round1-neutral "..."
输出: JSON 格式的各轮次 delegate_task 参数（round_1 / round_2 / verdict_prompt）
"""

import sys
import json
import argparse

# ── 复杂度配置 ──

DEPTH_MODES = {
    "quick":    {"round1": "120-220字", "round2": "120-220字", "rounds": 2},
    "standard": {"round1": "200-400字", "round2": "200-300字", "rounds": 3},
    "deep":     {"round1": "400-700字", "round2": "300-600字", "rounds": 3},
}

# Depth modes in English (word counts)
DEPTH_MODES_EN = {
    "quick":    {"round1": "60-110 words", "round2": "60-110 words", "rounds": 2},
    "standard": {"round1": "100-200 words", "round2": "100-150 words", "rounds": 3},
    "deep":     {"round1": "200-350 words", "round2": "150-300 words", "rounds": 3},
}

# ── 结构化输出格式 ──

STRUCTURED_FORMAT = """输出格式（严格遵守）：
立场：[支持/反对/折中/暂缓]
关键理由：
  1. ...
  2. ...
  3. ...
最大风险：[1条]
对其他方的质疑：
  - 对[某方]：[质疑内容]
决策阈值：[满足什么条件会改变判断]"""

STRUCTURED_FORMAT_R2 = """输出格式（严格遵守）：
立场：[支持/反对/折中/暂缓]
回应质疑：[针对对方的反驳，1段]
自我修正：[我的假设弱点 + 修正，1段]
对其他方的追问：[1-2条]
决策阈值：[更新后的判断条件]"""

STRUCTURED_FORMAT_EN = """Output format (follow strictly):
Position: [Support / Oppose / Compromise / Defer]
Key Reasons:
  1. ...
  2. ...
  3. ...
Biggest Risk: [1 item]
Challenges to other roles:
  - To [role]: [challenge]
Decision Threshold: [what evidence would change your mind]"""

STRUCTURED_FORMAT_R2_EN = """Output format (follow strictly):
Position: [Support / Oppose / Compromise / Defer]
Rebuttal: [counter-argument to the opposing role, 1 paragraph]
Self-Correction: [weakest assumption in your round 1 + correction, 1 paragraph]
Follow-up Questions: [1-2 items]
Decision Threshold: [updated conditions for changing judgment]"""

# ── 中文角色配置 ──

ROLES_ZH = {
    "radical": {
        "name": "🔴激进派",
        "stance": "破局者 / 高风险偏好",
        "persona": """你是激进派——破局者、颠覆者。

评估函数：优先评估突破性收益、机会窗口、竞争优势、速度、非线性回报。必须指出"不行动的代价"。

核心信条：不破不立。存量市场是零和博弈，增量市场才是星辰大海。
决策偏好：高风险高回报、弯道超车、先做了再说。
你擅长：看到被忽视的机会、提出颠覆性方案、打破"不可能"的假设。
你的盲区：容易低估执行难度、忽视系统性风险、过度乐观。
你的风格：直接、犀利、一针见血。用数据和案例支撑，不空洞喊口号。""",
    },
    "conservative": {
        "name": "🔵保守派",
        "stance": "守成者 / 风控优先",
        "persona": """你是保守派——守成者、风控官。

评估函数：优先评估失败概率、尾部风险、资源消耗、合规/声誉/组织阻力。必须指出"最坏情况下如何止损"。

核心信条：先虑败后虑胜。存活比赢更重要，复利比暴利更可靠。
决策偏好：低风险、可验证、渐进式改进、有退路。
你擅长：发现隐藏风险、评估执行可行性、借鉴历史教训。
你的盲区：可能因过度谨慎错失窗口期、低估技术或市场的非线性变化。
你的风格：冷静、务实、有据可查。引用类似案例的失败教训，不煽情。""",
    },
    "neutral": {
        "name": "⚪中立派",
        "stance": "仲裁者 / 数据驱动",
        "persona": """你是中立派——仲裁者、分析师。

评估函数：优先评估证据质量、假设强弱、变量权重、可逆性、试验设计。必须指出"当前最缺的数据是什么"。

核心信条：兼听则明。没有绝对的对错，只有不同约束条件下的最优解。
决策偏好：数据驱动、多维度权衡、寻找帕累托最优。
你擅长：量化利弊、发现双方都没注意到的第三选项、调和矛盾。
你的盲区：可能过于折中失去锋芒、在需要决断时犹豫不决。
你的风格：客观、结构化、引用数据和框架。你的结论必须有明确的"在...条件下建议..."，而不是"各有利弊"。""",
    },
}

# ── English role configuration ──

ROLES_EN = {
    "radical": {
        "name": "🔴 The Radical",
        "stance": "Disruptor / High risk appetite",
        "persona": """You are the Radical — disruptor, visionary.

Evaluation function: Prioritize breakthrough upside, opportunity windows, competitive advantage, speed, and non-linear returns. Always state "the cost of inaction."

Core belief: Break before you build. Incrementalism is death by a thousand cuts.
Decision bias: High risk, high reward. Move fast, fix later.
Strengths: Spotting overlooked opportunities, proposing disruptive solutions, challenging "impossible" assumptions.
Blind spots: Underestimating execution difficulty, ignoring systemic risks, over-optimism.
Style: Direct, sharp, data-backed. Cut through the noise.""",
    },
    "conservative": {
        "name": "🔵 The Conservative",
        "stance": "Guardian / Risk manager",
        "persona": """You are the Conservative — guardian, risk manager.

Evaluation function: Prioritize failure probability, tail risks, resource drain, compliance/reputation/organizational friction. Always state "the worst-case stop-loss."

Core belief: Plan for failure first. Survival > winning. Compound interest > moonshots.
Decision bias: Low risk, verifiable, incremental progress, always have an exit.
Strengths: Spotting hidden risks, assessing execution feasibility, citing historical failures.
Blind spots: Missing windows of opportunity, underestimating non-linear market/tech shifts.
Style: Calm, pragmatic, evidence-based. Cite failure cases. No hype.""",
    },
    "neutral": {
        "name": "⚪ The Neutral",
        "stance": "Arbitrator / Data-driven",
        "persona": """You are the Neutral — arbitrator, analyst.

Evaluation function: Prioritize evidence quality, assumption strength, variable weighting, reversibility, and experiment design. Always state "what data we're missing most."

Core belief: Hear both sides. There's no absolute right — only optimal solutions under constraints.
Decision bias: Data-driven, multi-dimensional trade-offs, seek Pareto optimal.
Strengths: Quantifying trade-offs, finding third options neither side saw, bridging gaps.
Blind spots: Over-compromising, losing edge, hesitating when decisiveness is needed.
Style: Objective, structured, framework-driven. Conclusions must be conditional: "Under X conditions, recommend Y" — never "it depends.\"""",
    },
}

# ── Helpers ──

def get_role(key: str, lang: str = "zh") -> dict:
    if lang == "en":
        return ROLES_EN[key]
    return ROLES_ZH[key]


def _prompt_zh(topic: str, wc: str) -> str:
    return (
        "现在有一个问题需要你给出开篇陈词：\n\n"
        f"【问题】\n{topic}\n\n"
        f"请给出你的开篇陈词（{wc}）：\n"
        "1. 你的核心立场是什么？\n"
        "2. 支撑你立场的 2-3 个关键论据\n"
        "3. 你认为其他立场最大的问题是什么？\n\n"
        + STRUCTURED_FORMAT + "\n\n"
        "直接、犀利、不骑墙。用中文。"
    )


def _prompt_en(topic: str, wc: str) -> str:
    return (
        "You are asked to deliver an opening statement on the following issue:\n\n"
        f"**Issue**\n{topic}\n\n"
        f"Deliver your opening statement ({wc}):\n"
        "1. What is your core position?\n"
        "2. 2-3 key arguments supporting your position\n"
        "3. What is the biggest flaw in the other positions?\n\n"
        + STRUCTURED_FORMAT_EN + "\n\n"
        "Be direct, sharp, and take a clear stance. Respond in English."
    )


def _r2_context_zh(own_op: str, op1_label: str, op1_text: str,
                   op2_label: str, op2_text: str, wc: str,
                   role_instructions: str) -> str:
    return (
        f"【你的第1轮立场】\n{own_op}\n\n"
        f"【{op1_label}的第1轮立场】\n{op1_text}\n\n"
        f"【{op2_label}的第1轮立场】\n{op2_text}\n\n"
        f"请以当前身份回应（{wc}）：\n"
        + role_instructions + "\n\n"
        + STRUCTURED_FORMAT_R2 + "\n\n"
        "用中文。"
    )


def _r2_context_en(own_op: str, op1_label: str, op1_text: str,
                   op2_label: str, op2_text: str, wc: str,
                   role_instructions: str) -> str:
    return (
        f"**Your Round 1 Position**\n{own_op}\n\n"
        f"**{op1_label}'s Round 1 Position**\n{op1_text}\n\n"
        f"**{op2_label}'s Round 1 Position**\n{op2_text}\n\n"
        f"Respond as your current role ({wc}):\n"
        + role_instructions + "\n\n"
        + STRUCTURED_FORMAT_R2_EN + "\n\n"
        "Respond in English."
    )


# ── Prompt generators ──

def generate_round1(topic: str, mode: str = "standard", lang: str = "zh") -> list[dict]:
    """生成第1轮（开篇陈词）的子智能体任务"""
    depths = DEPTH_MODES if lang == "zh" else DEPTH_MODES_EN
    wc = depths[mode]["round1"]

    prompt_fn = _prompt_zh if lang == "zh" else _prompt_en
    prompt_text = prompt_fn(topic, wc)

    tasks = []
    for key in ["radical", "conservative", "neutral"]:
        role = get_role(key, lang)
        tasks.append({
            "goal": f"{role['name']} opening on '{topic[:30]}'",
            "context": role["persona"] + "\n\n" + prompt_text,
        })
    return tasks


def generate_round2(
    radical_op: str, conservative_op: str, neutral_op: str,
    mode: str = "standard", lang: str = "zh"
) -> list[dict]:
    """生成第2轮（交叉质询 + 自我修正）的子智能体任务"""
    depths = DEPTH_MODES if lang == "zh" else DEPTH_MODES_EN
    wc = depths[mode]["round2"]
    r = get_role("radical", lang)
    c = get_role("conservative", lang)
    n = get_role("neutral", lang)

    if lang == "zh":
        # 激进派
        radical_task = {
            "goal": f"{r['name']} cross-examination",
            "context": (
                r["persona"] + "\n\n" +
                _r2_context_zh(
                    radical_op, c["name"], conservative_op,
                    n["name"], neutral_op, wc,
                    "1. 反驳保守派你最不认同的1个核心论据\n"
                    "2. 指出你自己第1轮中最脆弱的假设是什么\n"
                    "3. 修正或强化你的立场"
                )
            ),
        }
        # 保守派
        conservative_task = {
            "goal": f"{c['name']} cross-examination",
            "context": (
                c["persona"] + "\n\n" +
                _r2_context_zh(
                    conservative_op, r["name"], radical_op,
                    n["name"], neutral_op, wc,
                    "1. 反驳激进派你最不认同的1个核心论据\n"
                    "2. 指出在什么条件下你愿意接受更激进的方案\n"
                    "3. 修正或强化你的立场"
                )
            ),
        }
        # 中立派
        neutral_task = {
            "goal": f"{n['name']} cross-examination",
            "context": (
                n["persona"] + "\n\n" +
                _r2_context_zh(
                    neutral_op, r["name"], radical_op,
                    c["name"], conservative_op, wc,
                    "1. 点评双方各自最大的优势和盲区\n"
                    "2. 审查你自己第1轮的判断——听取了双方论据后，原判决是否仍然成立？\n"
                    "3. 当前最缺什么数据导致无法确定？\n"
                    "4. 在什么条件下，哪一派的观点更正确？"
                )
            ),
        }
    else:
        # English
        radical_task = {
            "goal": f"{r['name']} cross-examination",
            "context": (
                r["persona"] + "\n\n" +
                _r2_context_en(
                    radical_op, c["name"], conservative_op,
                    n["name"], neutral_op, wc,
                    "1. Rebut the one core argument from the Conservative you disagree with most\n"
                    "2. Identify your weakest assumption from Round 1\n"
                    "3. Revise or strengthen your position"
                )
            ),
        }
        conservative_task = {
            "goal": f"{c['name']} cross-examination",
            "context": (
                c["persona"] + "\n\n" +
                _r2_context_en(
                    conservative_op, r["name"], radical_op,
                    n["name"], neutral_op, wc,
                    "1. Rebut the one core argument from the Radical you disagree with most\n"
                    "2. Under what conditions would you accept a more aggressive approach?\n"
                    "3. Revise or strengthen your position"
                )
            ),
        }
        neutral_task = {
            "goal": f"{n['name']} cross-examination",
            "context": (
                n["persona"] + "\n\n" +
                _r2_context_en(
                    neutral_op, r["name"], radical_op,
                    c["name"], conservative_op, wc,
                    "1. Identify each side's greatest strength and blind spot\n"
                    "2. Review your own Round 1 judgment — after hearing both sides, does it still hold?\n"
                    "3. What data are we missing most that prevents certainty?\n"
                    "4. Under what conditions would one side be more correct?"
                )
            ),
        }

    return [radical_task, conservative_task, neutral_task]


def generate_verdict_prompt(topic: str, round1: dict, round2: dict,
                            mode: str = "standard", lang: str = "zh") -> str:
    """生成主持人最终裁决的 prompt"""
    r_label = "🔴 激进派" if lang == "zh" else "🔴 The Radical"
    c_label = "🔵 保守派" if lang == "zh" else "🔵 The Conservative"
    n_label = "⚪ 中立派" if lang == "zh" else "⚪ The Neutral"

    if lang == "zh":
        header = f"""你是三方辩论的主持人。请综合以下辩论记录，给出最终裁决。

【议题】
{topic}

【第1轮 — 开篇陈词】
{r_label}：{round1.get('radical', '（缺）')}

{c_label}：{round1.get('conservative', '（缺）')}

{n_label}：{round1.get('neutral', '（缺）')}

【第2轮 — 交叉质询 + 自我修正】
{r_label}回应：{round2.get('radical', '（缺）')}

{c_label}回应：{round2.get('conservative', '（缺）')}

{n_label}回应：{round2.get('neutral', '（缺）')}

请按以下格式输出最终裁决（用中文）：

## 🏛️ 三方辩论 — 最终裁决


"""
        sections = [
            ("辩论摘要", "| 派别 | 立场 | 关键理由 | 最大风险 | 决策阈值 |\n"
                        "|------|------|---------|---------|---------|\n"
                        f"| {r_label} | ... | ... | ... | ... |\n"
                        f"| {c_label} | ... | ... | ... | ... |\n"
                        f"| {n_label} | ... | ... | ... | ... |"),
            ("核心分歧", "最多3条分歧点，标注来自：价值取向 / 事实判断 / 风险偏好。"),
            ('主持人推荐', '明确选择 A/B/C 或组合方案（说明主次关系）。不允许只说"视情况而定"。'),
            ('推荐理由', '按收益、风险、可逆性、执行成本、时间窗口排序。'),
            ('关键风险与止损', '前3个风险 + 每个风险的监控信号 + 止损动作。'),
            ('下一步行动', '1-3个可执行动作，优先低成本验证。给具体时间建议。'),
            ('反转条件', '出现什么新证据/信号会推翻当前建议、采取相反策略。'),
            ('置信度', '高 / 中 / 低。说明哪些信息缺口拉低了置信度。'),
            ('禁止项', '禁止使用："需要综合考虑""各有利弊""应根据实际情况"。若必须保留不确定性，必须说明具体变量和验证方式。'),
        ]
        return header + "\n".join(
            f"### {title}\n{body}\n" for title, body in sections
        )
    else:
        header = f"""You are the moderator of a three-way debate. Synthesize the following debate record and deliver a final verdict.

**Topic**
{topic}

**Round 1 — Opening Statements**
{r_label}: {round1.get('radical', '(missing)')}

{c_label}: {round1.get('conservative', '(missing)')}

{n_label}: {round1.get('neutral', '(missing)')}

**Round 2 — Cross-Examination & Self-Correction**
{r_label} response: {round2.get('radical', '(missing)')}

{c_label} response: {round2.get('conservative', '(missing)')}

{n_label} response: {round2.get('neutral', '(missing)')}

Output the final verdict in the following format (in English):

## 🏛️ Three-Way Debate — Final Verdict

"""
        sections = [
            ("Debate Summary", f"| Role | Position | Key Reason | Biggest Risk | Decision Threshold |\n"
                              "|------|----------|------------|--------------|--------------------|\n"
                              f"| {r_label} | ... | ... | ... | ... |\n"
                              f"| {c_label} | ... | ... | ... | ... |\n"
                              f"| {n_label} | ... | ... | ... | ... |"),
            ("Core Disagreements", "Up to 3 disagreement points. Label source: values / facts / risk appetite."),
            ("Moderator Recommendation", "Pick A/B/C or a combined approach (with primary/secondary roles). Do NOT say 'it depends.'"),
            ("Rationale", "Ordered by upside, risk, reversibility, cost, time window."),
            ("Key Risks & Stop-Loss", "Top 3 risks + monitoring signals + stop-loss actions for each."),
            ("Next Steps", "1-3 executable actions. Prioritize low-cost validation. Give time estimates."),
            ("Reversal Conditions", "What new evidence or signal would overturn the recommendation."),
            ("Confidence", "High / Medium / Low. Note which information gaps lower confidence."),
            ("Forbidden", "Do not use: 'it depends', 'on the other hand', 'in some cases'. If uncertainty remains, specify the exact variable and how to resolve it."),
        ]
        return header + "\n".join(
            f"### {title}\n{body}\n" for title, body in sections
        )


def main():
    parser = argparse.ArgumentParser(description="三方辩论 — 子智能体 prompt 生成器")
    parser.add_argument("topic", nargs="+", help="辩论议题")
    parser.add_argument("--mode", choices=["quick", "standard", "deep"],
                        default="standard", help="复杂度 (默认: standard)")
    parser.add_argument("--lang", choices=["zh", "en"],
                        default="zh", help="语言 (默认: zh)")
    # Round 2 inputs (optional — if provided, only generate round 2)
    parser.add_argument("--round1-radical", help="激进派第1轮输出")
    parser.add_argument("--round1-conservative", help="保守派第1轮输出")
    parser.add_argument("--round1-neutral", help="中立派第1轮输出")
    parser.add_argument("--output", choices=["all", "round1", "round2", "verdict"],
                        default="all", help="输出模式 (默认: all)")
    args = parser.parse_args()

    topic = " ".join(args.topic)
    mode = args.mode
    lang = args.lang
    depths = DEPTH_MODES if lang == "zh" else DEPTH_MODES_EN
    depth = depths[mode]

    output = {
        "topic": topic,
        "mode": mode,
        "lang": lang,
        "rounds": depth["rounds"],
    }

    has_round1 = all([args.round1_radical, args.round1_conservative, args.round1_neutral])

    if has_round1:
        # Generate round 2 only
        if args.output in ("all", "round2"):
            output["round_2"] = generate_round2(
                args.round1_radical, args.round1_conservative, args.round1_neutral,
                mode, lang
            )
        if args.output in ("all", "verdict"):
            round1_dict = {
                "radical": args.round1_radical,
                "conservative": args.round1_conservative,
                "neutral": args.round1_neutral,
            }
            round2_dict = {"radical": "待填入", "conservative": "待填入", "neutral": "待填入"}
            output["verdict_prompt"] = generate_verdict_prompt(
                topic, round1_dict, round2_dict, mode, lang
            )
    else:
        # Generate round 1
        if args.output in ("all", "round1"):
            output["round_1"] = generate_round1(topic, mode, lang)
        output["_usage"] = {
            "round_1": "用 delegate_task(tasks=output['round_1']) 并行调用 3 个子智能体",
            "round_1_async": "delegate_task 是异步的——结果会以消息形式注入回会话。等结果到达后再生成 round_2。",
            "round_2_generation": (
                "python run_debate.py '议题' --mode {} --lang {} "
                "--round1-radical '...' --round1-conservative '...' --round1-neutral '...'"
            ).format(mode, lang),
            "round_2_skip_for_quick": "quick 模式跳过第2轮，直接从 round_1 进入 verdict",
            "verdict": "收集所有结果后，用 generate_verdict_prompt(topic, round1_dict, round2_dict, mode, lang) 的输出作为 prompt 给出最终裁决",
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
