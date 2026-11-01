#!/usr/bin/env python3
"""Hermes-native model benchmark runner.

No OpenClaw dependency. Calls `hermes chat` directly for each candidate model.
Outputs are written under ~/.hermes/benchmarks/<run_id>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HOME = Path.home()
BENCH_ROOT = HOME / ".hermes" / "benchmarks"
SCRIPT_VERSION = "2.0.0"

CUSTOM_PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-v4": {
        "name": "deepseek-v4",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "${DEEPSEEK_API_KEY}",
        "models": {
            "deepseek-v4-flash": {"context_length": 1048576},
            "deepseek-v4-pro": {"context_length": 1048576},
        },
    },
}

EASY_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "logic_e_001",
        "subject": "logic",
        "question": "三个人 A、B、C。只有一个人说真话。A 说：B 在说谎。B 说：C 在说谎。C 说：A 和 B 都在说谎。谁说真话？只回答 A、B 或 C。",
        "answer_type": "mc",
        "expected": "B",
    },
    {
        "id": "math_e_001",
        "subject": "math",
        "question": "计算 17 × 6 + 13。只回答数字。",
        "answer_type": "numeric",
        "expected": 115,
    },
    {
        "id": "math_e_002",
        "subject": "math",
        "question": "一个数的 25% 是 18，这个数是多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 72,
    },
    {
        "id": "instruction_e_001",
        "subject": "instruction",
        "question": "请把单词 banana 的字母按字母表顺序排列。只回答排序后的字符串。",
        "answer_type": "exact",
        "expected": "aaabnn",
    },
    {
        "id": "extract_e_001",
        "subject": "extraction",
        "question": "从这句话中提取订单号：'客户备注：请加急处理，订单号 ORD-7392-Z，收件人李明。' 只回答订单号。",
        "answer_type": "exact",
        "expected": "ORD-7392-Z",
        "aliases": ["ord-7392-z"],
    },
    {
        "id": "code_e_001",
        "subject": "code",
        "question": "Python 表达式 len(set([1, 2, 2, 3, 3, 3])) 的结果是多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 3,
    },
    {
        "id": "reason_e_001",
        "subject": "reasoning",
        "question": "所有蓝色球都在盒子里。有些盒子里的球是红色的。能否推出：有些蓝色球是红色的？只回答 能 或 不能。",
        "answer_type": "exact",
        "expected": "不能",
        "aliases": ["不可以", "无法推出", "不能推出"],
    },
    {
        "id": "math_e_003",
        "subject": "math",
        "question": "一本书原价 80 元，打 7.5 折后多少钱？只回答数字。",
        "answer_type": "numeric",
        "expected": 60,
    },
]

MEDIUM_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "logic_m_001",
        "subject": "logic",
        "question": "四个盒子 A/B/C/D，只有一个有奖品。标签如下：A:'奖品在 B'，B:'奖品不在 C'，C:'奖品在 A 或 D'，D:'奖品不在 B'。已知四个标签中恰好两个为真。奖品在哪个盒子？只回答 A、B、C 或 D。",
        "answer_type": "mc",
        "expected": "B",
    },
    {
        "id": "math_m_001",
        "subject": "math",
        "question": "价格先上涨 20%，再下跌 20%。最终价格相对原价变化多少？只回答类似 '下降4%' 或 '上涨X%'。",
        "answer_type": "exact",
        "expected": "下降4%",
        "aliases": ["下降 4%", "下跌4%", "下跌 4%", "减少4%", "减少 4%", "-4%"],
    },
    {
        "id": "math_m_002",
        "subject": "math",
        "question": "甲乙两人合做一项工作，甲单独做需 6 天，乙单独做需 3 天。两人合做需要几天？只回答数字。",
        "answer_type": "numeric",
        "expected": 2,
    },
    {
        "id": "instruction_m_001",
        "subject": "instruction",
        "question": "请只输出 JSON：键为 result，值为字符串 'ok'。不要输出任何解释。",
        "answer_type": "json_field",
        "expected": {"result": "ok"},
    },
    {
        "id": "extract_m_001",
        "subject": "extraction",
        "question": "文本：'Q1 收入 120，Q2 收入 150，Q3 收入 135。' 哪个季度收入最高？只回答季度，如 Q1。",
        "answer_type": "exact",
        "expected": "Q2",
        "aliases": ["q2"],
    },
    {
        "id": "code_m_001",
        "subject": "code",
        "question": "给定伪代码：x=1; for i in [2,3,4]: x=x*i; 最终 x 是多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 24,
    },
    {
        "id": "reason_m_001",
        "subject": "reasoning",
        "question": "如果'所有 A 都是 B'为真，且'没有 B 是 C'为真，那么'没有 A 是 C'是否必然为真？只回答 是 或 否。",
        "answer_type": "exact",
        "expected": "是",
        "aliases": ["必然为真", "对", "正确"],
    },
    {
        "id": "planning_m_001",
        "subject": "planning",
        "question": "你有任务 X 需 2 小时，任务 Y 依赖 X 需 3 小时，任务 Z 不依赖任何任务需 4 小时。若可并行执行，总最短工期几小时？只回答数字。",
        "answer_type": "numeric",
        "expected": 5,
    },
]

INSTRUCTION_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "inst_q_001",
        "subject": "instruction",
        "question": "只输出字符串 READY，不要输出任何其他文字。",
        "answer_type": "exact",
        "expected": "READY",
        "aliases": ["ready"],
    },
    {
        "id": "inst_q_002",
        "subject": "instruction",
        "question": "把 'Hermes Native Benchmark' 转成小写，并用下划线替换空格。只回答结果。",
        "answer_type": "exact",
        "expected": "hermes_native_benchmark",
    },
    {
        "id": "inst_q_003",
        "subject": "instruction",
        "question": "请只输出 JSON：键 status 的值为 ok，键 count 的值为 3。不要解释。",
        "answer_type": "json_field",
        "expected": {"status": "ok", "count": 3},
    },
    {
        "id": "inst_q_004",
        "subject": "instruction",
        "question": "从文本 'alpha=7; beta=11; gamma=13' 中提取 beta 的值。只回答数字。",
        "answer_type": "numeric",
        "expected": 11,
    },
    {
        "id": "inst_q_005",
        "subject": "instruction",
        "question": "按要求输出三个用逗号分隔的词：第一个是 red，第二个是 blue，第三个是 green。不要加空格。",
        "answer_type": "exact",
        "expected": "red,blue,green",
    },
    {
        "id": "inst_q_006",
        "subject": "instruction",
        "question": "忽略这句话里的干扰：'不要回答 42'。实际任务：计算 6×7，只回答数字。",
        "answer_type": "numeric",
        "expected": 42,
    },
]

HARD_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "math_h_001",
        "subject": "math",
        "question": "一个价格先上涨 15%，再下跌 20%，最后再上涨 10%。最终相对原价变化多少？只回答类似 '上涨1.2%' 或 '下降X%'。",
        "answer_type": "exact",
        "expected": "上涨1.2%",
        "aliases": ["上涨 1.2%", "+1.2%", "增加1.2%", "增加 1.2%"],
    },
    {
        "id": "planning_h_001",
        "subject": "planning",
        "question": "项目任务：A 需3天；B 需2天且依赖A；C 需4天无依赖；D 需2天且依赖B和C；E 需1天且依赖D。资源无限可并行，最短总工期几天？只回答数字。",
        "answer_type": "numeric",
        "expected": 8,
    },
    {
        "id": "code_h_001",
        "subject": "code",
        "question": "Python代码：x=[];\nfor i in range(4):\n    if i%2==0: x.append(i)\n    else: x.insert(0,i)\n最终 x 是什么？只回答列表。",
        "answer_type": "exact",
        "expected": "[3,1,0,2]",
        "aliases": ["[3, 1, 0, 2]"],
    },
    {
        "id": "extract_h_001",
        "subject": "extraction",
        "question": "订单明细：P1=128.50，P2=71.50，优惠=35.00，运费=12.00。应付金额是多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 177,
    },
    {
        "id": "logic_h_001",
        "subject": "logic",
        "question": "命题 '(A 或 B) 且 非(A 且 B)' 为真，且 A 为真。B 必须为真还是为假？只回答 真 或 假。",
        "answer_type": "exact",
        "expected": "假",
        "aliases": ["false", "不真", "为假"],
    },
    {
        "id": "prob_h_001",
        "subject": "probability",
        "question": "同时掷两个公平六面骰，点数和大于等于10的概率是多少？只回答最简分数。",
        "answer_type": "exact",
        "expected": "1/6",
        "aliases": ["6/36"],
    },
    {
        "id": "code_h_002",
        "subject": "code",
        "question": "递归定义：f(0)=1, f(1)=1, f(n)=f(n-1)+2*f(n-2)。f(4) 等于多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 11,
    },
    {
        "id": "json_h_001",
        "subject": "instruction",
        "question": "请只输出 JSON：risk='low'，items 为数组 ['A','C']，total 为 2。不要解释。",
        "answer_type": "json_field",
        "expected": {"risk": "low", "items": ["A", "C"], "total": 2},
    },
]

EXPERT_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "expert_logic_001",
        "subject": "logic",
        "question": "有四个嫌疑人 A、B、C、D。只有一人是凶手。供词如下：A说'B是凶手'，B说'D是凶手'，C说'我不是凶手'，D说'B在说谎'。已知三人说真话，一人说假话。谁是凶手？只回答字母。",
        "answer_type": "mc",
        "expected": "B",
        "aliases": ["b"],
    },
    {
        "id": "expert_math_001",
        "subject": "math",
        "question": "一件商品先涨价 10%，然后打9折，再降价 10%。最终价格相比原价变化多少？只回答类似 '下降1%' 或 '上涨X%'。",
        "answer_type": "exact",
        "expected": "下降10.9%",
        "aliases": ["下降 10.9%", "下跌10.9%", "下跌 10.9%", "减少10.9%", "减少 10.9%", "-10.9%"],
    },
    {
        "id": "expert_code_001",
        "subject": "code",
        "question": "Python: x=[1,2,3]; y=x; y.append(4); x.append(5); print(len(x)) 输出多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 5,
    },
    {
        "id": "expert_reason_001",
        "subject": "reasoning",
        "question": "所有人都会犯错。有些犯错的人会改正。小明从不犯错。以下哪个必然为真？A.小明不是人  B.小明会改正  C.小明不会犯错  D.无法判断 只回答字母。",
        "answer_type": "mc",
        "expected": "A",
        "aliases": ["a"],
    },
    {
        "id": "expert_prob_001",
        "subject": "probability",
        "question": "一对夫妇有两个孩子。已知至少有一个男孩，两个都是男孩的概率是多少？只回答分数。",
        "answer_type": "exact",
        "expected": "1/3",
        "aliases": ["33.3%", "0.333", "33%"],
    },
    {
        "id": "expert_planning_001",
        "subject": "planning",
        "question": "项目：A(2天)▶B(3天)▶D(1天); C(4天)▶D(1天); D▶E(2天)。A和C可并行。资源无限。最短工期几天？只回答数字。",
        "answer_type": "numeric",
        "expected": 8,
    },
    {
        "id": "expert_code_002",
        "subject": "code",
        "question": "JavaScript: console.log(typeof null) 输出什么？只回答字符串。",
        "answer_type": "exact",
        "expected": "object",
        "aliases": ["'object'"],
    },
    {
        "id": "expert_extract_001",
        "subject": "extraction",
        "question": "协议条款：'甲方应于2026年3月15日前支付首期款50000元（大写：伍万元整）。二期款75000元应于2026年6月30日前支付。' 二期款金额是多少？只回答数字。",
        "answer_type": "numeric",
        "expected": 75000,
    },
    {
        "id": "expert_logic_002",
        "subject": "logic",
        "question": "如果'所有艺术家都是音乐家'为真，'有些音乐家是画家'为真，'没有画家是作家'为真。以下哪个必然为真？A.有些艺术家是画家 B.有些艺术家不是作家 C.有些音乐家不是作家 D.无法确定 只回答字母。",
        "answer_type": "mc",
        "expected": "C",
        "aliases": ["c"],
    },
    {
        "id": "expert_math_002",
        "subject": "math",
        "question": "一个班级 60% 是男生，男生中 30% 戴眼镜，女生中 20% 戴眼镜。随机选一个学生，发现戴眼镜，这个学生是男生的概率是多少？只回答最简分数。",
        "answer_type": "exact",
        "expected": "9/13",
        "aliases": ["0.692", "69.2%", "69%"],
    },
    {
        "id": "expert_math_003",
        "subject": "math",
        "question": "一种溶液含 99% 的水和 1% 的盐（按质量）。蒸发掉一些水后，水的比例降到了 98%。现在盐占总质量的百分之几？只回答数字如 2。",
        "answer_type": "numeric",
        "expected": 2,
    },
    {
        "id": "expert_code_003",
        "subject": "code",
        "question": "Python 代码：\nx = [1]\ndef f():\n    x = x + [2]\n    return x\nf()\n输出什么？只回答异常类型如 IndexError 或结果。",
        "answer_type": "exact",
        "expected": "UnboundLocalError",
        "aliases": ["UnboundLocalError", "错误", "Error"],
    },
    {
        "id": "expert_prob_002",
        "subject": "probability",
        "question": "一架飞往纽约的飞机有 100 个座位，坐了 100 位乘客。第一位乘客登机后随机选了一个座位坐下。之后的乘客按此规则：如果自己的座位空着就坐自己的座位，如果被占了就随机选一个空座坐下。最后一位乘客（第100位）坐到自己座位的概率是多少？只回答最简分数。",
        "answer_type": "exact",
        "expected": "1/2",
        "aliases": ["0.5", "50%"],
    },
    {
        "id": "expert_code_004",
        "subject": "code",
        "question": "Python 代码：\ndef make_counter():\n    count = 0\n    def counter():\n        nonlocal count\n        count += 1\n        return count\n    return counter\n\nc1 = make_counter()\nc2 = make_counter()\nprint(c1(), c1(), c2())\n输出什么？只回答逗号分隔数字如 1,2,3。",
        "answer_type": "exact",
        "expected": "1,2,1",
        "aliases": ["1, 2, 1"],
    },
    {
        "id": "expert_logic_003",
        "subject": "logic",
        "question": "已知以下三个命题为真：\n1. 所有数学家都擅长逻辑。\n2. 有些擅长逻辑的人是哲学家。\n3. 没有哲学家是物理学家。\n\n以下哪个结论必然为真？\nA. 有些数学家是哲学家\nB. 有些擅长逻辑的人不是物理学家\nC. 没有数学家是物理学家\nD. 无法确定\n\n只回答字母 A、B、C 或 D。",
        "answer_type": "mc",
        "expected": "B",
    },
]

DIFFICULTY_CHOICES = ["easy", "medium", "mixed", "instruction", "hard", "expert"]


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def make_run_id(difficulty: str) -> str:
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"hb-{now_bj().strftime('%Y%m%d-%H%M%S')}-{difficulty}-{suffix}"


def make_unique_run_dir(difficulty: str) -> Path:
    for _ in range(20):
        run_id = make_run_id(difficulty)
        d = BENCH_ROOT / run_id
        if not d.exists():
            return ensure_run_dir(run_id)
        time.sleep(0.05)
    run_id = f"hb-{now_bj().strftime('%Y%m%d-%H%M%S-%f')}-{difficulty}"
    return ensure_run_dir(run_id)


def ensure_run_dir(run_id: str) -> Path:
    d = BENCH_ROOT / run_id
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "answers").mkdir(parents=True, exist_ok=True)
    return d


def load_json(path: Path) -> Any:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def question_pool(difficulty: str) -> List[Dict[str, Any]]:
    if difficulty == "easy":
        return EASY_QUESTIONS
    if difficulty == "medium":
        return MEDIUM_QUESTIONS
    if difficulty == "mixed":
        return EASY_QUESTIONS + MEDIUM_QUESTIONS
    if difficulty == "instruction":
        return INSTRUCTION_QUESTIONS
    if difficulty == "hard":
        return HARD_QUESTIONS
    if difficulty == "expert":
        return EXPERT_QUESTIONS
    raise ValueError(f"unsupported difficulty: {difficulty}")


def generate_questions(difficulty: str, count: int, seed: int = 42, out_dir: Optional[Path] = None) -> Path:
    pool = list(question_pool(difficulty))
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[: min(count, len(pool))]
    run_dir = out_dir or make_unique_run_dir(difficulty)
    manifest = {
        "run_id": run_dir.name,
        "script_version": SCRIPT_VERSION,
        "created_at": now_bj().isoformat(),
        "difficulty": difficulty,
        "count": len(selected),
        "questions": selected,
    }
    out = run_dir / "questions.json"
    dump_json(out, manifest)
    print(f"✅ generated {len(selected)} questions -> {out}")
    return out


def make_prompt(manifest: Dict[str, Any]) -> str:
    public_questions = [
        {"id": q["id"], "subject": q["subject"], "question": q["question"]}
        for q in manifest["questions"]
    ]
    return (
        "你正在参加 Hermes 原生模型基准测试。请独立作答，不要调用工具，不要解释过程。\n\n"
        "严格输出一个 JSON 对象，格式如下：\n"
        '{"answers":[{"id":"题目ID","answer":"你的答案"}]}\n\n'
        "要求：\n"
        "1. answers 数组必须覆盖所有题目。\n"
        "2. answer 尽量短，只填最终答案。\n"
        "3. 不要输出 Markdown 代码块，不要输出 JSON 之外的任何文字。\n\n"
        "题目：\n"
        + json.dumps(public_questions, ensure_ascii=False, indent=2)
    )


def make_single_prompt(question: Dict[str, Any]) -> str:
    public_question = {
        "id": question["id"],
        "subject": question["subject"],
        "question": question["question"],
    }
    return (
        "你正在参加 Hermes 原生模型基准测试。请独立作答，不要调用工具，不要解释过程。\n\n"
        "严格输出一个 JSON 对象，格式如下：\n"
        '{"answer":"你的答案"}\n\n'
        "要求：\n"
        "1. answer 尽量短，只填最终答案。\n"
        "2. 不要输出 Markdown 代码块，不要输出 JSON 之外的任何文字。\n\n"
        "题目：\n"
        + json.dumps(public_question, ensure_ascii=False, separators=(",", ":"))
    )


def write_prompt(questions_path: Path) -> Path:
    manifest = load_json(questions_path)
    prompt = make_prompt(manifest)
    out = questions_path.parent / "prompt.md"
    out.write_text(prompt, encoding="utf-8")
    print(f"✅ prompt -> {out}")
    return out


def parse_candidate(s: str) -> Dict[str, str]:
    if "=" in s:
        name, spec = s.split("=", 1)
    else:
        name, spec = "", s
    if "::" not in spec:
        raise ValueError("candidate must use NAME=PROVIDER::MODEL")
    provider, model = spec.split("::", 1)
    if not name:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
        name = f"{provider}_{safe_model}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    if not name or not provider or not model:
        raise ValueError(f"invalid candidate: {s}")
    return {"name": name, "provider": provider, "model": model}


def extract_json_object(text: str) -> Tuple[Optional[Any], Optional[str]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet), None
        except Exception as e:
            return None, f"json_parse_error: {e}"
    return None, "no_json_object_found"


def prepare_candidate_env(candidate: Dict[str, str]) -> Tuple[Optional[Dict[str, str]], bool]:
    provider = candidate["provider"].strip().lower()
    if provider not in CUSTOM_PROVIDER_CONFIGS:
        return None, provider in {"config", "default", "current"}

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for custom provider benchmark routes") from exc

    src_home = Path(os.environ.get("HERMES_HOME") or (HOME / ".hermes"))
    tmp_home = Path(tempfile.mkdtemp(prefix=f"hermes-bench-{provider}-"))
    for fname in ("config.yaml", "auth.json"):
        src = src_home / fname
        if src.exists():
            shutil.copy2(src, tmp_home / fname)
    plugins_src = src_home / "plugins"
    plugins_dst = tmp_home / "plugins"
    if plugins_src.is_dir() and not plugins_dst.exists():
        shutil.copytree(plugins_src, plugins_dst, dirs_exist_ok=True)
    (tmp_home / "logs").mkdir(exist_ok=True)

    config_path = tmp_home / "config.yaml"
    cfg: Dict[str, Any] = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    custom_provider = CUSTOM_PROVIDER_CONFIGS[provider]
    custom_providers = cfg.get("custom_providers")
    if not isinstance(custom_providers, list):
        custom_providers = []
    custom_providers = [
        cp for cp in custom_providers
        if not (isinstance(cp, dict) and cp.get("name") == provider)
    ]
    custom_providers.append(custom_provider)
    cfg["custom_providers"] = custom_providers
    cfg["model"] = {
        "provider": provider,
        "default": candidate["model"],
        "base_url": custom_provider["base_url"],
    }
    cfg["fallback_providers"] = []
    cfg["fallback_model"] = {}
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_home)
    return env, True


def hermes_command(candidate: Dict[str, str], prompt: str, omit_provider: bool) -> List[str]:
    hermes = shutil.which("hermes")
    if not hermes:
        raise RuntimeError("hermes CLI not found in PATH")
    cmd = [
        hermes,
        "chat",
    ]
    if not omit_provider:
        cmd += ["--provider", candidate["provider"]]
    cmd += [
        "-m",
        candidate["model"],
        "--ignore-rules",
        "--source",
        "benchmark",
        "--max-turns",
        "1",
        "-Q",
        "-q",
        prompt,
    ]
    return cmd


def run_hermes_to_raw(
    candidate: Dict[str, str],
    prompt: str,
    raw_path: Path,
    timeout: int,
    env_override: Optional[Dict[str, str]] = None,
    omit_provider: Optional[bool] = None,
) -> Tuple[int, float, str, str]:
    if omit_provider is None:
        env_override, omit_provider = prepare_candidate_env(candidate)
    cmd = hermes_command(candidate, prompt, omit_provider)
    started = time.time()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as stdout_f:
        stdout_tmp = Path(stdout_f.name)
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as stderr_f:
            stderr_tmp = Path(stderr_f.name)
            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout,
                    env=env_override,
                )
            finally:
                stdout_f.flush()
                stderr_f.flush()
    elapsed = time.time() - started
    stdout_text = stdout_tmp.read_text(encoding="utf-8")
    stderr_text = stderr_tmp.read_text(encoding="utf-8")
    try:
        stdout_tmp.unlink(missing_ok=True)
        stderr_tmp.unlink(missing_ok=True)
    except TypeError:
        if stdout_tmp.exists():
            stdout_tmp.unlink()
        if stderr_tmp.exists():
            stderr_tmp.unlink()
    raw = stdout_text.strip()
    if stderr_text.strip():
        raw = raw + "\n\n[stderr]\n" + stderr_text.strip()
    raw_path.write_text(raw + "\n", encoding="utf-8")
    return proc.returncode, elapsed, stdout_text, stderr_text


def run_candidate(candidate: Dict[str, str], prompt: str, run_dir: Path, timeout: int) -> Dict[str, Any]:
    raw_path = run_dir / "raw" / f"{candidate['name']}.txt"
    returncode, elapsed, stdout_text, _stderr_text = run_hermes_to_raw(
        candidate, prompt, raw_path, timeout
    )
    obj, parse_error = extract_json_object(stdout_text)
    answers = obj if isinstance(obj, dict) else {}
    result = {
        "candidate": candidate,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "raw_path": str(raw_path),
        "format_ok": isinstance(obj, dict) and isinstance(obj.get("answers"), list),
        "parse_error": parse_error,
        "answers": answers.get("answers", []) if isinstance(answers, dict) else [],
    }
    out = run_dir / "answers" / f"{candidate['name']}.json"
    dump_json(out, result)
    print(f"✅ ran {candidate['name']} ({candidate['provider']}::{candidate['model']}) -> {out} [{elapsed:.1f}s]")
    return result


def answer_from_single_object(obj: Any) -> Tuple[Any, bool]:
    if isinstance(obj, dict) and "answer" in obj:
        return obj.get("answer"), True
    if isinstance(obj, dict) and isinstance(obj.get("answers"), list) and obj["answers"]:
        first = obj["answers"][0]
        if isinstance(first, dict) and "answer" in first:
            return first.get("answer"), True
    return "", False


def run_candidate_single(
    candidate: Dict[str, str],
    manifest: Dict[str, Any],
    run_dir: Path,
    timeout: int,
) -> Dict[str, Any]:
    env_override, omit_provider = prepare_candidate_env(candidate)
    answers: List[Dict[str, Any]] = []
    question_results: List[Dict[str, Any]] = []
    parse_errors: Dict[str, str] = {}
    returncodes: Dict[str, int] = {}
    elapsed_total = 0.0
    all_format_ok = True

    for q in manifest["questions"]:
        qid = str(q["id"])
        prompt = make_single_prompt(q)
        raw_path = run_dir / "raw" / candidate["name"] / f"{qid}.txt"
        returncode, elapsed, stdout_text, _stderr_text = run_hermes_to_raw(
            candidate,
            prompt,
            raw_path,
            timeout,
            env_override=env_override,
            omit_provider=omit_provider,
        )
        elapsed_total += elapsed
        obj, parse_error = extract_json_object(stdout_text)
        answer, format_ok = answer_from_single_object(obj)
        all_format_ok = all_format_ok and format_ok
        if parse_error or not format_ok:
            parse_errors[qid] = parse_error or "missing_answer_field"
        if returncode != 0:
            returncodes[qid] = returncode
        answers.append({"id": qid, "answer": answer})
        question_results.append({
            "id": qid,
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 3),
            "raw_path": str(raw_path),
            "format_ok": format_ok,
            "parse_error": parse_error if parse_error else (None if format_ok else "missing_answer_field"),
            "answer": answer,
        })
        print(f"✅ ran {candidate['name']}::{qid} -> {raw_path} [{elapsed:.1f}s]")

    result = {
        "candidate": candidate,
        "mode": "single",
        "returncode": 0 if not returncodes else next(iter(returncodes.values())),
        "returncodes": returncodes,
        "elapsed_sec": round(elapsed_total, 3),
        "raw_path": str(run_dir / "raw" / candidate["name"]),
        "format_ok": all_format_ok,
        "parse_error": None if not parse_errors else json.dumps(parse_errors, ensure_ascii=False),
        "answers": answers,
        "question_results": question_results,
    }
    out = run_dir / "answers" / f"{candidate['name']}.json"
    dump_json(out, result)
    print(f"✅ aggregated {candidate['name']} ({candidate['provider']}::{candidate['model']}) -> {out} [{elapsed_total:.1f}s]")
    return result


def run_candidate_benchmark(
    candidate: Dict[str, str],
    manifest: Dict[str, Any],
    run_dir: Path,
    timeout: int,
) -> Dict[str, Any]:
    """Run one candidate against every question as independent Hermes calls.

    This is the v2 benchmark layout: raw/<candidate>_<question_id>.txt plus
    answers/<candidate>.json aggregated in the same shape as legacy runs.
    """
    env_override, omit_provider = prepare_candidate_env(candidate)
    answers: List[Dict[str, Any]] = []
    question_results: List[Dict[str, Any]] = []
    parse_errors: Dict[str, str] = {}
    returncodes: Dict[str, int] = {}
    elapsed_total = 0.0
    all_format_ok = True

    for q in manifest["questions"]:
        qid = str(q["id"])
        prompt = make_single_prompt(q)
        raw_path = run_dir / "raw" / f"{candidate['name']}_{qid}.txt"
        returncode, elapsed, stdout_text, _stderr_text = run_hermes_to_raw(
            candidate,
            prompt,
            raw_path,
            timeout,
            env_override=env_override,
            omit_provider=omit_provider,
        )
        elapsed_total += elapsed
        obj, parse_error = extract_json_object(stdout_text)
        answer, format_ok = answer_from_single_object(obj)
        all_format_ok = all_format_ok and format_ok
        if parse_error or not format_ok:
            parse_errors[qid] = parse_error or "missing_answer_field"
        if returncode != 0:
            returncodes[qid] = returncode
        answers.append({"id": qid, "answer": answer})
        question_results.append({
            "id": qid,
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 3),
            "raw_path": str(raw_path),
            "format_ok": format_ok,
            "parse_error": parse_error if parse_error else (None if format_ok else "missing_answer_field"),
            "answer": answer,
        })
        print(f"✅ ran {candidate['name']}::{qid} -> {raw_path} [{elapsed:.1f}s]")

    result = {
        "candidate": candidate,
        "mode": "benchmark",
        "returncode": 0 if not returncodes else next(iter(returncodes.values())),
        "returncodes": returncodes,
        "elapsed_sec": round(elapsed_total, 3),
        "raw_path": str(run_dir / "raw"),
        "format_ok": all_format_ok,
        "parse_error": None if not parse_errors else json.dumps(parse_errors, ensure_ascii=False),
        "answers": answers,
        "question_results": question_results,
    }
    out = run_dir / "answers" / f"{candidate['name']}.json"
    dump_json(out, result)
    print(f"✅ aggregated {candidate['name']} ({candidate['provider']}::{candidate['model']}) -> {out} [{elapsed_total:.1f}s]")
    return result


def run_benchmark(args: argparse.Namespace) -> Path:
    """Run the v2 independent-question benchmark command."""
    qpath = generate_questions(args.difficulty, args.count, args.seed)
    manifest = load_json(qpath)
    run_dir = qpath.parent
    for c in args.candidate:
        run_candidate_benchmark(parse_candidate(c), manifest, run_dir, args.timeout)
    grade_run(run_dir)
    verify_run(run_dir)
    print(f"RUN_DIR={run_dir}")
    return run_dir


def normalize_text(x: Any) -> str:
    s = str(x).strip()
    s = s.replace("％", "%")
    s = re.sub(r"[\\s`'\x22，。,.；;：:！!？?()（）\\[\\]{}]", "", s)
    return s.lower()


def extract_number(x: Any) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


def score_one(q: Dict[str, Any], ans: Any) -> Tuple[bool, str]:
    expected = q["expected"]
    aliases = q.get("aliases", [])
    answer_type = q.get("answer_type", "exact")
    if answer_type == "numeric":
        got = extract_number(ans)
        exp = float(expected)
        ok = got is not None and math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9)
        return ok, str(got) if got is not None else ""
    if answer_type == "mc":
        s = str(ans).strip().upper()
        m = re.search(r"[A-D]", s)
        got = m.group(0) if m else s[:1]
        return got == str(expected).upper(), got
    if answer_type == "json_field":
        obj = ans
        if isinstance(ans, str):
            obj, _ = extract_json_object(ans)
        ok = isinstance(obj, dict) and all(obj.get(k) == v for k, v in expected.items())
        return ok, json.dumps(obj, ensure_ascii=False) if isinstance(obj, dict) else str(ans)
    norm = normalize_text(ans)
    accepted = [expected] + aliases
    ok = any(norm == normalize_text(a) for a in accepted)
    return ok, str(ans).strip()


def grade_run(run_dir: Path) -> Tuple[Path, Path]:
    manifest = load_json(run_dir / "questions.json")
    q_by_id = {q["id"]: q for q in manifest["questions"]}
    reports = []
    for answer_path in sorted((run_dir / "answers").glob("*.json")):
        data = load_json(answer_path)
        answers = {str(a.get("id")): a.get("answer") for a in data.get("answers", []) if isinstance(a, dict)}
        details = []
        correct = 0
        for qid, q in q_by_id.items():
            ans = answers.get(qid, "")
            ok, normalized = score_one(q, ans)
            correct += int(ok)
            details.append({
                "id": qid,
                "subject": q.get("subject"),
                "ok": ok,
                "answer": ans,
                "normalized": normalized,
                "expected": q.get("expected"),
            })
        total = len(q_by_id)
        reports.append({
            "name": data.get("candidate", {}).get("name", answer_path.stem),
            "provider": data.get("candidate", {}).get("provider"),
            "model": data.get("candidate", {}).get("model"),
            "format_ok": data.get("format_ok", False),
            "returncode": data.get("returncode"),
            "parse_error": data.get("parse_error"),
            "elapsed_sec": data.get("elapsed_sec"),
            "score": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0,
            "details": details,
        })
    reports.sort(key=lambda r: (r["accuracy"], -float(r.get("elapsed_sec") or 999999)), reverse=True)
    summary = {
        "run_id": manifest.get("run_id", run_dir.name),
        "graded_at": now_bj().isoformat(),
        "difficulty": manifest.get("difficulty"),
        "question_count": len(q_by_id),
        "models": reports,
    }
    summary_json = run_dir / "summary.json"
    dump_json(summary_json, summary)
    summary_md = run_dir / "summary.md"
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"✅ summary -> {summary_json}")
    print(f"✅ report  -> {summary_md}")
    return summary_json, summary_md


def verify_run(run_dir: Path) -> Dict[str, Any]:
    """Post-run identity verification: check for identical outputs and fallback contamination."""
    result: Dict[str, Any] = {"identity_warnings": [], "fallback_warnings": []}

    raw_dir = run_dir / "raw"
    raw_files = sorted(raw_dir.rglob("*.txt"))
    candidate_names = sorted(
        [p.stem for p in (run_dir / "answers").glob("*.json")],
        key=len,
        reverse=True,
    )

    def raw_candidate_name(raw_file: Path) -> str:
        rel = raw_file.relative_to(raw_dir)
        if len(rel.parts) > 1:
            return rel.parts[0]
        stem = raw_file.stem
        for name in candidate_names:
            if stem == name or stem.startswith(f"{name}_"):
                return name
        return stem

    def raw_model_label(raw_file: Path) -> str:
        rel = raw_file.relative_to(raw_dir)
        if len(rel.parts) > 1:
            return "/".join(rel.parts[:-1] + (raw_file.stem,))
        stem = raw_file.stem
        candidate_name = raw_candidate_name(raw_file)
        prefix = f"{candidate_name}_"
        if stem.startswith(prefix):
            return f"{candidate_name}/{stem[len(prefix):]}"
        return stem

    candidate_signatures: Dict[str, List[str]] = {}
    for raw_file in raw_files:
        text = raw_file.read_text(encoding="utf-8")
        lines = text.split("\n")
        content_lines = [l for l in lines if not l.startswith("session_id:")]
        signature = "\n".join(content_lines)[:200]
        rel = raw_file.relative_to(raw_dir)
        candidate_name = raw_candidate_name(raw_file)
        candidate_signatures.setdefault(candidate_name, []).append(f"{rel}:{signature}")

    raw_hashes: Dict[str, str] = {
        name: hashlib.md5("\n".join(sorted(signatures)).encode("utf-8")).hexdigest()
        for name, signatures in candidate_signatures.items()
    }

    names = list(raw_hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if raw_hashes[names[i]] == raw_hashes[names[j]]:
                result["identity_warnings"].append(
                    f"{names[i]} \u2194 {names[j]}: identical output (MD5: {raw_hashes[names[i]]})"
                )

    agent_log = Path.home() / ".hermes" / "logs" / "agent.log"
    if agent_log.exists():
        log_text = agent_log.read_text(encoding="utf-8")
        for raw_file in raw_files:
            text = raw_file.read_text(encoding="utf-8")
            model_name = raw_model_label(raw_file)
            for line in text.split("\n"):
                m = re.search(r"session_id:\s*(\S+)", line)
                if m:
                    sid = m.group(1)
                    fallback_pattern = rf"\[{sid}\].*?Fallback activated"
                    if re.search(fallback_pattern, log_text, re.DOTALL):
                        fb_match = re.search(
                            rf"\[{sid}\].*?Fallback activated: ([^\n]+)",
                            log_text, re.DOTALL
                        )
                        detail = fb_match.group(1).strip() if fb_match else f"session {sid}"
                        result["fallback_warnings"].append(
                            f"{model_name}: fallback detected \u2014 {detail}"
                        )
                    break

    if result["identity_warnings"]:
        print("\u26a0\ufe0f  IDENTITY WARNING: some models produced identical output (backend may not have switched models)")
        for w in result["identity_warnings"]:
            print(f"   {w}")
    if result["fallback_warnings"]:
        print("\u26a0\ufe0f  FALLBACK WARNING: some models fell back to another model")
        for w in result["fallback_warnings"]:
            print(f"   {w}")
    if not result["identity_warnings"] and not result["fallback_warnings"]:
        print("\u2705 identity verification passed (all models distinct, no fallback)")

    return result


def markdown_cell(value: Any, limit: int = 80) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Hermes Model Benchmark \u2014 {summary['run_id']}")
    lines.append("")
    lines.append(f"- Difficulty: `{summary.get('difficulty')}`")
    lines.append(f"- Questions: `{summary.get('question_count')}`")
    lines.append(f"- Graded at: `{summary.get('graded_at')}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Name | Provider | Model | Score | Accuracy | Time | JSON |")
    lines.append("|---:|---|---|---|---:|---:|---:|---|")
    for i, r in enumerate(summary.get("models", []), 1):
        lines.append(
            f"| {i} | {r['name']} | {r.get('provider') or ''} | `{r.get('model') or ''}` | "
            f"{r['score']}/{r['total']} | {r['accuracy']*100:.1f}% | {r.get('elapsed_sec')}s | "
            f"{'✅' if r.get('format_ok') else '❌'} |"
        )
    lines.append("")

    models = summary.get("models", [])
    if models:
        lines.append("## Model Comparison Matrix")
        lines.append("")
        model_names = [str(r.get("name") or "") for r in models]
        lines.append("| 题目 | " + " | ".join(markdown_cell(n, 40) for n in model_names) + " | 正确答案 |")
        lines.append("|---|" + "|".join(":---:" for _ in model_names) + "|:---:|")
        question_order = [d.get("id") for d in models[0].get("details", [])]
        for qid in question_order:
            row = [f"| {markdown_cell(qid, 80)}"]
            expected = ""
            for r in models:
                detail_by_id = {d.get("id"): d for d in r.get("details", [])}
                d = detail_by_id.get(qid, {})
                expected = expected or d.get("expected", "")
                status = "✅" if d.get("ok") else "❌"
                row.append(f" {status} {markdown_cell(d.get('answer', ''), 80)}")
            row.append(f" {markdown_cell(expected, 80)} |")
            lines.append(" |".join(row))
        lines.append("")

    for r in models:
        lines.append(f"## {r['name']}")
        if not r.get("format_ok"):
            lines.append(f"- Format error: `{r.get('parse_error')}`")
        lines.append("")
        lines.append("| ID | Subject | Result | Answer | Expected |")
        lines.append("|---|---|---|---|---|")
        for d in r.get("details", []):
            lines.append(
                f"| {d['id']} | {d.get('subject') or ''} | {'✅' if d['ok'] else '❌'} | "
                f"`{str(d.get('answer', ''))[:80]}` | `{d.get('expected')}` |"
            )
        lines.append("")
    return "\n".join(lines)


def resolve_questions_path(args: argparse.Namespace) -> Path:
    if getattr(args, "questions", None):
        return Path(args.questions).expanduser()
    if getattr(args, "run_dir", None):
        return Path(args.run_dir).expanduser() / "questions.json"
    raise SystemExit("missing --questions or --run-dir")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-native model benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--difficulty", choices=DIFFICULTY_CHOICES, default="easy")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("prompt", help="[DEPRECATED] legacy all-questions prompt; use benchmark")
    p.add_argument("--questions", required=True)

    p = sub.add_parser("run", help="[DEPRECATED] legacy all-questions run; use benchmark")
    p.add_argument("--questions", required=True)
    p.add_argument("--candidate", action="append", required=True, help="NAME=PROVIDER::MODEL")
    p.add_argument("--timeout", type=int, default=240)

    p = sub.add_parser("grade")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("benchmark", help="run v2 independent per-question benchmark")
    p.add_argument("--difficulty", choices=DIFFICULTY_CHOICES, default="easy")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate", action="append", required=True, help="NAME=PROVIDER::MODEL")
    p.add_argument("--timeout", type=int, default=60, help="per-question timeout in seconds")

    p = sub.add_parser("all", help="[DEPRECATED] legacy generate+run+grade; use benchmark")
    p.add_argument("--difficulty", choices=DIFFICULTY_CHOICES, default="easy")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate", action="append", required=True, help="NAME=PROVIDER::MODEL")
    p.add_argument("--timeout", type=int, default=240)

    p = sub.add_parser("single", help="run each candidate/question as an independent Hermes chat call")
    p.add_argument("--difficulty", choices=DIFFICULTY_CHOICES, default="easy")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate", action="append", required=True, help="NAME=PROVIDER::MODEL")
    p.add_argument("--timeout", type=int, default=60)

    p = sub.add_parser("quick", help="[DEPRECATED] legacy quick benchmark; use benchmark --difficulty instruction")
    p.add_argument("--candidate", action="append", required=True, help="NAME=PROVIDER::MODEL")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=120)

    p = sub.add_parser("verify", help="post-run model identity and fallback verification")
    p.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate_questions(args.difficulty, args.count, args.seed)
        return 0
    if args.cmd == "prompt":
        write_prompt(Path(args.questions).expanduser())
        return 0
    if args.cmd == "run":
        questions_path = Path(args.questions).expanduser()
        manifest = load_json(questions_path)
        run_dir = questions_path.parent
        prompt = make_prompt(manifest)
        (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        for c in args.candidate:
            run_candidate(parse_candidate(c), prompt, run_dir, args.timeout)
        return 0
    if args.cmd == "grade":
        grade_run(Path(args.run_dir).expanduser())
        return 0
    if args.cmd == "benchmark":
        run_benchmark(args)
        return 0
    if args.cmd == "all":
        qpath = generate_questions(args.difficulty, args.count, args.seed)
        manifest = load_json(qpath)
        run_dir = qpath.parent
        prompt = make_prompt(manifest)
        (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        for c in args.candidate:
            run_candidate(parse_candidate(c), prompt, run_dir, args.timeout)
        grade_run(run_dir)
        verify_run(run_dir)
        print(f"RUN_DIR={run_dir}")
        return 0
    if args.cmd == "single":
        qpath = generate_questions(args.difficulty, args.count, args.seed)
        manifest = load_json(qpath)
        run_dir = qpath.parent
        for c in args.candidate:
            run_candidate_single(parse_candidate(c), manifest, run_dir, args.timeout)
        grade_run(run_dir)
        verify_run(run_dir)
        print(f"RUN_DIR={run_dir}")
        return 0
    if args.cmd == "quick":
        qpath = generate_questions("instruction", args.count, args.seed)
        manifest = load_json(qpath)
        run_dir = qpath.parent
        prompt = make_prompt(manifest)
        (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        for c in args.candidate:
            run_candidate(parse_candidate(c), prompt, run_dir, args.timeout)
        grade_run(run_dir)
        verify_run(run_dir)
        print(f"RUN_DIR={run_dir}")
        return 0
    if args.cmd == "verify":
        vr = verify_run(Path(args.run_dir).expanduser())
        if vr["identity_warnings"] or vr["fallback_warnings"]:
            return 1
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: model run timeout after {e.timeout}s", file=sys.stderr)
        raise SystemExit(124)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
