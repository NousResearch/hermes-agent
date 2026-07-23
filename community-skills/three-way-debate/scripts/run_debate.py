#!/usr/bin/env python3
"""
三方辩论 — 子智能体 prompt 生成器 v2.0
用法: python run_debate.py "你的议题" [--mode quick|standard|deep]
输出: JSON 格式的各轮次 delegate_task 参数
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

# ── 角色配置（可扩展） ──

ROLES = {
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


def get_role(key: str) -> dict:
    return ROLES[key]


def generate_round1(topic: str, mode: str = "standard") -> list[dict]:
    """生成第1轮（开篇陈词）的子智能体任务"""
    wc = DEPTH_MODES[mode]["round1"]
    prompt_template = (
        "现在有一个问题需要你给出开篇陈词：\n\n"
        "【问题】\n{topic}\n\n"
        "请给出你的开篇陈词（{wc}）：\n"
        "1. 你的核心立场是什么？\n"
        "2. 支撑你立场的 2-3 个关键论据\n"
        "3. 你认为其他立场最大的问题是什么？\n\n"
        + STRUCTURED_FORMAT + "\n\n"
        "直接、犀利、不骑墙。用中文。"
    )

    tasks = []
    for key in ["radical", "conservative", "neutral"]:
        role = get_role(key)
        tasks.append({
            "goal": f"{role['name']}对「{topic[:30]}」开篇陈词",
            "context": (
                role["persona"] + "\n\n" +
                prompt_template.format(topic=topic, wc=wc)
            ),
        })
    return tasks


def generate_round2(
    radical_op: str, conservative_op: str, neutral_op: str,
    mode: str = "standard"
) -> list[dict]:
    """生成第2轮（交叉质询 + 自我修正）的子智能体任务"""

    wc = DEPTH_MODES[mode]["round2"]
    r = get_role("radical")
    c = get_role("conservative")
    n = get_role("neutral")

    # 激进派：收到保守派和中立派的第1轮，反驳保守派 + 自我修正
    radical_task = {
        "goal": f"{r['name']}交叉质询「{radical_op[:30]}...」",
        "context": (
            r["persona"] + "\n\n"
            f"【你的第1轮立场】\n{radical_op}\n\n"
            f"【保守派的第1轮立场】\n{conservative_op}\n\n"
            f"【中立派的第1轮立场】\n{neutral_op}\n\n"
            f"请以激进派身份回应（{wc}）：\n"
            f"1. 反驳保守派你最不认同的1个核心论据\n"
            f"2. 指出你自己第1轮中最脆弱的假设是什么\n"
            f"3. 修正或强化你的立场\n\n"
            + STRUCTURED_FORMAT_R2 + "\n\n"
            "犀利直接。用中文。"
        ),
    }

    # 保守派：收到激进派和中立派的第1轮，反驳激进派 + 自我修正
    conservative_task = {
        "goal": f"{c['name']}交叉质询「{conservative_op[:30]}...」",
        "context": (
            c["persona"] + "\n\n"
            f"【你的第1轮立场】\n{conservative_op}\n\n"
            f"【激进派的第1轮立场】\n{radical_op}\n\n"
            f"【中立派的第1轮立场】\n{neutral_op}\n\n"
            f"请以保守派身份回应（{wc}）：\n"
            f"1. 反驳激进派你最不认同的1个核心论据\n"
            f"2. 指出在什么条件下你愿意接受更激进的方案\n"
            f"3. 修正或强化你的立场\n\n"
            + STRUCTURED_FORMAT_R2 + "\n\n"
            "务实锋利。用中文。"
        ),
    }

    # 中立派：审查双方 + 审查自己第1轮判断
    neutral_task = {
        "goal": f"{n['name']}交叉质询「{neutral_op[:30]}...」",
        "context": (
            n["persona"] + "\n\n"
            f"【你的第1轮立场】\n{neutral_op}\n\n"
            f"【激进派的第1轮立场】\n{radical_op}\n\n"
            f"【保守派的第1轮立场】\n{conservative_op}\n\n"
            f"请以中立派身份回应（{wc}）：\n"
            f"1. 点评双方各自最大的优势和盲区\n"
            f"2. 审查你自己第1轮的判断——听取了双方论据后，原判决是否仍然成立？\n"
            f"3. 当前最缺什么数据导致无法确定？\n"
            f"4. 在什么条件下，哪一派的观点更正确？\n\n"
            + STRUCTURED_FORMAT_R2 + "\n\n"
            "客观、有倾向。用中文。"
        ),
    }

    return [radical_task, conservative_task, neutral_task]


def generate_verdict_prompt(topic: str, round1: dict, round2: dict) -> str:
    """生成主持人最终裁决的 prompt"""
    return f"""你是三方辩论的主持人。请综合以下辩论记录，给出最终裁决。

【议题】
{topic}

【第1轮 — 开篇陈词】
🔴 激进派：{round1.get('radical', '（缺）')}

🔵 保守派：{round1.get('conservative', '（缺）')}

⚪ 中立派：{round1.get('neutral', '（缺）')}

【第2轮 — 交叉质询 + 自我修正】
🔴 激进派回应：{round2.get('radical', '（缺）')}

🔵 保守派回应：{round2.get('conservative', '（缺）')}

⚪ 中立派回应：{round2.get('neutral', '（缺）')}

请按以下格式输出最终裁决（用中文）：

## 🏛️ 三方辩论 — 最终裁决

### 辩论摘要
| 派别 | 立场 | 关键理由 | 最大风险 | 决策阈值 |
|------|------|---------|---------|---------|
| 🔴 激进派 | ... | ... | ... | ... |
| 🔵 保守派 | ... | ... | ... | ... |
| ⚪ 中立派 | ... | ... | ... | ... |

### 核心分歧
最多3条分歧点，标注来自：价值取向 / 事实判断 / 风险偏好。

### 主持人推荐
明确选择 A/B/C 或组合方案（说明主次关系）。不允许只说"视情况而定"。

### 推荐理由
按收益、风险、可逆性、执行成本、时间窗口排序。

### 关键风险与止损
前3个风险 + 每个风险的监控信号 + 止损动作。

### 下一步行动
1-3个可执行动作，优先低成本验证。给具体时间建议。

### 反转条件
出现什么新证据/信号会推翻当前建议、采取相反策略。

### 置信度
高 / 中 / 低。说明哪些信息缺口拉低了置信度。

### 禁止项
禁止使用："需要综合考虑""各有利弊""应根据实际情况"。若必须保留不确定性，必须说明具体变量和验证方式。"""


def main():
    parser = argparse.ArgumentParser(description="三方辩论 — 子智能体 prompt 生成器")
    parser.add_argument("topic", nargs="+", help="辩论议题")
    parser.add_argument("--mode", choices=["quick", "standard", "deep"],
                        default="standard", help="复杂度 (默认: standard)")
    parser.add_argument("--lang", choices=["zh", "en"],
                        default="zh", help="语言 (默认: zh)")
    args = parser.parse_args()

    topic = " ".join(args.topic)
    mode = args.mode
    depth = DEPTH_MODES[mode]

    output = {
        "topic": topic,
        "mode": mode,
        "lang": args.lang,
        "rounds": depth["rounds"],
        "round_1": generate_round1(topic, mode),
        "_usage": {
            "round_1": "用 delegate_task(tasks=output['round_1']) 并行调用 3 个子智能体",
            "round_2": "收集 round_1 结果后，调用 generate_round2(激进派, 保守派, 中立派, mode)，再用 delegate_task(tasks=...) 并行调用",
            "round_2_skip_for_quick": "quick 模式跳过第2轮",
            "verdict": "收集结果后，用 generate_verdict_prompt(topic, round1_dict, round2_dict) 作为 prompt 输出最终裁决",
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
