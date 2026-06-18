"""Hermes plugin that guards code modification delegation."""

import re
from typing import Any


BLOCK_MESSAGE = (
    'delegate_task 不可用于代码修改任务。请使用 codex exec --sandbox workspace-write "prompt" 完成代码修改。\n'
    "分工原则：Claude 规划，Codex 实现。"
)


_RESEARCH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"研究",
        r"\bresearch\b",
        r"分析",
        r"\banaly[sz]e\b",
        r"\bsummarize\b",
        r"总结",
        r"报告",
        r"\breport\b",
    )
)


_CODE_MODIFICATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"修改",
        r"修复",
        r"\bfix(?:e[ds])?\b",
        r"\bbug(?:s)?\b",
        r"\brefactor(?:ed|ing)?\b",
        r"\bimplement(?:ed|ing|ation)?\b",
        r"实现",
        r"添加",
        r"\badd(?:ed|ing)?\b",
        r"\badd\s+feature\b",
        r"代码",
        r"\bcode\b",
        r"\bfile(?:s)?\b",
        r"文件",
        r"\bfunction(?:s)?\b",
        r"函数",
        r"\bclass(?:es)?\b",
        r"类",
        r"\bedit(?:ed|ing)?\b",
        r"\bchange(?:d|s|ing)?\b",
        r"\bupdate(?:d|s|ing)?\b",
        r"\brewrite(?:s|ing|n)?\b",
        r"写入",
        r"\bcreate\s+file\b",
        r"\btest(?:s|ed|ing)?\b",
        r"测试",
        r"\bdebug(?:ged|ging)?\b",
        r"调试",
        r"\bpatch(?:ed|es|ing)?\b",
        r"\bcommit(?:s|ted|ting)?\b",
        r"\bpr\b",
        r"\bpull\s+request\b",
    )
)


_CODE_FILE_PATH_PATTERN = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|mjs|cjs|go|rs|java|kt|kts|c|cc|cpp|cxx|h|hpp|cs|rb|php|swift|scala|sh|bash|zsh|fish|ps1|sql|html|css|scss|sass|vue|svelte|json|yaml|yml|toml|md|rst|lock)(?![\w.-])",
    re.IGNORECASE,
)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)


def on_pre_tool_call(*, tool_name: str = "", args: dict = None, **kwargs):
    # 只拦截 delegate_task，其他工具一律放行。
    if tool_name != "delegate_task":
        return None

    if not isinstance(args, dict):
        return None

    for goal in _iter_goals(args):
        if _has_code_modification_intent(goal):
            return {"action": "block", "message": BLOCK_MESSAGE}

    return None


def _iter_goals(args: dict):
    # 检查顶层 goal。
    goal = args.get("goal")
    if isinstance(goal, str):
        yield goal

    # 检查 tasks[].goal，忽略格式异常的 task。
    tasks = args.get("tasks")
    if not isinstance(tasks, list):
        return

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_goal = task.get("goal")
        if isinstance(task_goal, str):
            yield task_goal


def _has_code_modification_intent(goal: str) -> bool:
    text = goal.strip()
    if not text:
        return False

    # 研究/分析/总结/报告类任务放行，即使提到代码修改关键词。
    if _matches_any(_RESEARCH_PATTERNS, text):
        return False

    has_explicit_file_path = bool(_CODE_FILE_PATH_PATTERN.search(text))

    # 短目标且没有明确代码文件路径时按简单问题处理。
    if len(text) < 50 and not has_explicit_file_path:
        return False

    return _matches_any(_CODE_MODIFICATION_PATTERNS, text)


def _matches_any(patterns: tuple[Any, ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)
