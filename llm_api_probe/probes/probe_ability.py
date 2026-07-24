"""Probe 6: 能力基线 (轻量推理题抽样)

检测目标:
  - 模型是否"被降级"或被偷换成弱模型
  - 输出是否符合该模型的宣称能力

策略 (避免跑完整 benchmark, 5 分钟内出结论):
  - 数学: GSM8K 抽样 5 题
  - 推理: 几个经典逻辑题 (谁说谎/序列推理)
  - 中文: 几个简单中文理解/翻译
  - 代码: 简单函数题

通过率太低 → 怀疑模型被降级 / 偷换
通过率正常 → 至少能力与宣称一致
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from llm_api_probe.probes.client import ProbeClient
from llm_api_probe.probes.models import Provider, ProbeResult


@dataclass
class Question:
    cat: str
    q: str
    expected_in_answer: list[str] = field(default_factory=list)
    judge_fn: Optional[Callable[[str], bool]] = None


MATH_QUESTIONS = [
    Question(
        cat="math",
        q="一个商店卖苹果, 进货价每斤 3 元, 售价每斤 5 元, 每天能卖 100 斤。"
           "如果某天苹果损耗了 10%, 当天净利润是多少元?",
        expected_in_answer=["320"],
    ),
    Question(
        cat="math",
        q="小明有 50 元, 买书花了 18 元, 妈妈又给了他 15 元, "
           "之后他买零食花了 12 元。请问他现在还剩多少钱?",
        expected_in_answer=["35"],
    ),
    Question(
        cat="math",
        q="一辆车 3 小时行驶 180 公里, 按同样速度, 5 小时能行驶多少公里?",
        expected_in_answer=["300"],
    ),
    Question(
        cat="math",
        q="一个水池有两根管, A 管 4 小时能注满, B 管 6 小时能注满。"
           "两根同时开多久能注满?",
        expected_in_answer=["2.4", "12/5"],
    ),
    Question(
        cat="math",
        q="鸡兔同笼, 头共 35, 脚共 94, 问鸡和兔各多少只?",
        expected_in_answer=["23", "12"],
    ),
]

REASONING_QUESTIONS = [
    Question(
        cat="reasoning",
        q="所有的 A 都是 B, 所有的 B 都是 C, 那么所有的 A 都是 C 吗? 请回答 是 或 否。",
        expected_in_answer=["是"],
    ),
    Question(
        cat="reasoning",
        q=(
            "有 5 个连续整数, 它们的和是 100。最大的那个数是多少?"
        ),
        expected_in_answer=["22"],
    ),
    Question(
        cat="reasoning",
        q=(
            "Alice 比 Bob 高, Bob 比 Charlie 高, Charlie 比 David 高。"
            "谁最矮? 只回答名字。"
        ),
        expected_in_answer=["David"],
    ),
]

CODE_QUESTIONS = [
    Question(
        cat="code",
        q=(
            "用 Python 写一个函数 is_prime(n) 判断 n 是否为素数, "
            "只写函数体, 不要解释。"
        ),
        expected_in_answer=["def is_prime", "return"],
    ),
    Question(
        cat="code",
        q=(
            "Python: 用一行代码把列表 [1,2,3,4,5] 的元素求平方并组成新列表。"
            "只输出代码, 不要解释。"
        ),
        expected_in_answer=["**", "2"],
    ),
]

CHINESE_QUESTIONS = [
    Question(
        cat="zh",
        q="把'出师表'翻译成现代汉语的最后一句是什么含义? 一句话回答。",
        expected_in_answer=["临行", "涕零", "不知所云", "感激"],  # 任一相关即可
    ),
    Question(
        cat="zh",
        q="用一句话解释什么是'内卷'。",
        expected_in_answer=[],   # 开放式, 走 judge_fn
        judge_fn=lambda a: len(a.strip()) > 5 and any(
            w in a for w in ["竞争", "投入", "收益", "努力", "资源", "无意义"]
        ),
    ),
]

ALL_QUESTIONS = MATH_QUESTIONS + REASONING_QUESTIONS + CODE_QUESTIONS + CHINESE_QUESTIONS


def _judge(q: Question, answer: str) -> bool:
    if q.judge_fn:
        try:
            return bool(q.judge_fn(answer))
        except Exception:
            return False
    return any(exp in answer for exp in q.expected_in_answer)


def run(
    provider: Provider,
    model: str,
    *,
    verbose: bool = False,
    questions_per_cat: int = 3,
    max_tokens: int = 512,
) -> ProbeResult:
    r = ProbeResult(probe="ability", provider=provider.name)
    client = ProbeClient(provider)

    # 限制每个类别的题目数
    by_cat: dict[str, list[Question]] = {}
    for q in ALL_QUESTIONS:
        by_cat.setdefault(q.cat, []).append(q)
    selected: list[Question] = []
    for cat, qs in by_cat.items():
        selected.extend(qs[:questions_per_cat])

    cat_pass: dict[str, tuple[int, int]] = {}
    detail: list[dict] = []
    for q in selected:
        msgs = [{"role": "user", "content": q.q}]
        cr = client.call(model, msgs, stream=False, max_tokens=max_tokens, temperature=0.0, verbose=verbose)
        if not cr.ok:
            detail.append({"cat": q.cat, "q": q.q[:80], "ok": False, "error": cr.error_type})
            c_p, c_t = cat_pass.get(q.cat, (0, 0))
            cat_pass[q.cat] = (c_p, c_t + 1)
            continue
        passed = _judge(q, cr.content)
        detail.append({
            "cat": q.cat,
            "q": q.q[:80],
            "ok": passed,
            "answer": cr.content[:200],
        })
        c_p, c_t = cat_pass.get(q.cat, (0, 0))
        cat_pass[q.cat] = (c_p + (1 if passed else 0), c_t + 1)

    total_p = sum(p for p, _ in cat_pass.values())
    total_t = sum(t for _, t in cat_pass.values())
    r.add("total_questions", total_t, "")
    r.add("total_pass", total_p, "")
    r.add("overall_pass_rate", round(total_p / total_t * 100, 1) if total_t else 0, "%")
    for cat, (p, t) in cat_pass.items():
        rate = round(p / t * 100, 1) if t else 0
        r.add(f"{cat}_pass_rate", f"{p}/{t} ({rate}%)", "")
    r.raw["detail"] = detail
    r.raw["cat_pass"] = {k: list(v) for k, v in cat_pass.items()}

    r.find(f"能力基线: {total_p}/{total_t} 通过 ({round(total_p / total_t * 100, 1) if total_t else 0}%)")
    if total_t and total_p / total_t < 0.5:
        r.warn(
            f"能力通过率仅 {round(total_p / total_t * 100, 1)}% — "
            "强烈怀疑该 key 接入的是降级模型或被偷换"
        )

    client.close()
    return r