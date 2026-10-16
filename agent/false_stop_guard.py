"""False-stop guard: empty routing block plus an unkept promise.

Pure decision. ``apply_stop_gates`` is the only caller. A ``finish_reason=stop``
with zero tool calls, a first-person promise or a closed-verb imperative, and
a routing block that is not a substantial delivery, gets one continuation.
The assistant row stays real content; only the nudge is flagged synthetic.

Delivery measurement drops the routing block, then bare marker lines, then
requires the remainder to be both >200 characters and at least two complete
sentences.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

# Extended frozen set (23). Do not shrink back to the plan's original 18,
# and do not add one-off keys such as 说明 / 备注 / browser_actions.
ROUTING_BLOCK_KEYS: Tuple[str, ...] = (
    "分类",
    "流程族",
    "路径",
    "技能",
    "算力包",
    "验证",
    "成本闸",
    "原子步骤",
    "子代理",
    "岗位",
    "推理深度",
    "领域",
    "领域深度",
    "可行性",
    "研究价值",
    "意图",
    "计划强度",
    "执行强度",
    "对抗审查类型",
    "对抗审查角色",
    "用户烤问",
    "计划场景",
    "场景来源",
)

FALSE_STOP_SIGNALS = {
    "first_person_zh": ("我先", "我将", "接下来我会"),
    "first_person_en": ("Let me", "I'll", "Next I will"),
    "imperative_prefix": ("先", "接下来", "开始"),
    "imperative_verbs": ("读", "看", "核实", "加载", "检索", "打开"),
    "exclude_directed": ("你可以", "建议你"),
    "exclude_process_tail": ("再决定", "再执行", "再处理"),
    # 对用户工序收束，不是自己承诺。只压祈使，不压第一人称。
    "exclude_user_close": ("即可",),
}

# One sentence. Do not embed the gold fixture.
FALSE_STOP_NUDGE = "你刚才停在承诺或空路由上，请继续把动作做完。"

_ROUTING_KEY_SET = frozenset(ROUTING_BLOCK_KEYS)


def _alt(items: Sequence[str]) -> str:
    return "|".join(re.escape(item) for item in items)


_ZH_COMMIT = re.compile(
    r"(?<![\w])(?:" + _alt(FALSE_STOP_SIGNALS["first_person_zh"]) + r")"
)
_EN_COMMIT = re.compile(
    r"(?<![A-Za-z])(?:" + _alt(FALSE_STOP_SIGNALS["first_person_en"]) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"(?<![\w])(?:"
    + _alt(FALSE_STOP_SIGNALS["imperative_prefix"])
    + r")(?:"
    + _alt(FALSE_STOP_SIGNALS["imperative_verbs"])
    + r")"
)
_USER_DIRECTED = re.compile(_alt(FALSE_STOP_SIGNALS["exclude_directed"]))
_PROCESS_TAIL = re.compile(_alt(FALSE_STOP_SIGNALS["exclude_process_tail"]))
_ROUTING_HEADER = re.compile(r"^路由\s*[:：]?\s*$")
_KEY_LINE = re.compile(r"^\s*[-*•]\s*([^:：]{1,40}?)\s*[:：]")
_HEADING_LINE = re.compile(r"^\s*#+\s*.*$")
_FENCE_LINE = re.compile(r"^\s*```.*$")
_PURE_QUOTE_LINE = re.compile(r"^\s*>+\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？?!])")
_COMPLETE_SENTENCE = re.compile(r"[^。！？.\s][^。！？.]*[。！？.]", re.DOTALL)


def classify_false_stop_budget(iteration_budget, max_iterations, api_call_count) -> str:
    """Map already-getattr'd budget fields to exhausted / remaining / unknown.

    Missing or unreadable fields are unknown (treated as not exhausted; the
    continuation counter still allows one nudge). The call site must pass
    ``getattr(agent, "iteration_budget", None)`` and
    ``getattr(agent, "max_iterations", None)``. This function does not read
    ``agent`` attributes itself.
    """
    if iteration_budget is None or max_iterations is None:
        return "unknown"
    try:
        remaining = iteration_budget.remaining
        max_i = int(max_iterations)
        used = int(api_call_count)
        remaining_n = float(remaining)
    except Exception:
        return "unknown"
    if remaining_n <= 0 or used >= max_i:
        return "exhausted"
    return "remaining"


def _is_table_sep(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return re.fullmatch(r"[\s|:\-]+", stripped) is not None


def _is_bare_marker(line: str) -> bool:
    if _HEADING_LINE.match(line):
        return True
    if _FENCE_LINE.match(line):
        return True
    if _PURE_QUOTE_LINE.match(line):
        return True
    return _is_table_sep(line)


def _is_frozen_key_line(line: str) -> bool:
    match = _KEY_LINE.match(line)
    if not match:
        return False
    return match.group(1).strip() in _ROUTING_KEY_SET


def _routing_span(lines: Sequence[str]) -> Optional[Tuple[int, int]]:
    """Return [start, end) of the first routing block, or None.

    A block is a line-start 「路由」 header plus at least one following list
    key from the frozen set. It ends at the first non-key line.
    """
    for index, line in enumerate(lines):
        if not _ROUTING_HEADER.match(line):
            continue
        end = index + 1
        found_key = False
        while end < len(lines) and _is_frozen_key_line(lines[end]):
            found_key = True
            end += 1
        if found_key:
            return index, end
    return None


def _without_routing_block(text: str) -> str:
    lines = text.splitlines()
    span = _routing_span(lines)
    if span is None:
        return text
    start, end = span
    return "\n".join(lines[:start] + lines[end:])


def _has_routing_block(text: str) -> bool:
    return _routing_span(text.splitlines()) is not None


def _delivery_remainder(text: str) -> str:
    body = _without_routing_block(text)
    kept = [line for line in body.splitlines() if not _is_bare_marker(line)]
    return "\n".join(kept).strip()


def _complete_sentence_count(text: str) -> int:
    return len(_COMPLETE_SENTENCE.findall(text))


def _is_substantial_delivery(text: str) -> bool:
    remainder = _delivery_remainder(text)
    return len(remainder) > 200 and _complete_sentence_count(remainder) >= 2


def _condition_c(text: str) -> bool:
    """(c) holds only when a routing block exists and delivery is not substantial."""
    if not _has_routing_block(text):
        return False
    return not _is_substantial_delivery(text)


def _signal_flags(text: str) -> Tuple[bool, bool]:
    """Return (first_person, imperative) after exclusions. Body only."""
    body = _without_routing_block(text or "")
    if body.rstrip().endswith(("？", "?")):
        return False, False
    first = False
    imperative = False
    pieces = _SENTENCE_SPLIT.split(body) or [body]
    for sentence in pieces:
        if not sentence.strip():
            continue
        if sentence.rstrip().endswith(("？", "?")):
            continue
        if _PROCESS_TAIL.search(sentence):
            continue
        user_close = any(
            token in sentence for token in FALSE_STOP_SIGNALS["exclude_user_close"]
        ) and not (_ZH_COMMIT.search(sentence) or _EN_COMMIT.search(sentence))
        for matcher, kind in (
            (_EN_COMMIT, "first"),
            (_ZH_COMMIT, "first"),
            (_IMPERATIVE, "imperative"),
        ):
            for match in matcher.finditer(sentence):
                if _USER_DIRECTED.search(sentence[: match.start()]):
                    continue
                if kind == "imperative" and user_close:
                    continue
                if kind == "first":
                    first = True
                else:
                    imperative = True
    return first, imperative


def false_stop_signal_category(text: str) -> str:
    """Signal class for the ``guard_fired`` log line. Empty if none survived."""
    first, imperative = _signal_flags(text)
    labels = []
    if first:
        labels.append("第一人称承诺")
    if imperative:
        labels.append("缺主语祈使")
    return "+".join(labels)


def false_stop_decision(
    text,
    finish_reason,
    tool_call_count,
    continuations,
    budget,
) -> str:
    """Return ``nudge`` or ``release``.

    Conjunction: (a) stop and zero tools, (b) promise or closed-verb
    imperative after exclusions, (c) routing block and no substantial
    delivery. ``exhausted`` and a continuation count already >= 1 release
    unconditionally. ``unknown`` is treated as not exhausted.
    """
    try:
        cont = int(continuations or 0)
    except (TypeError, ValueError):
        cont = 0
    if cont >= 1:
        return "release"
    if budget == "exhausted":
        return "release"
    if finish_reason != "stop" or tool_call_count != 0:
        return "release"
    if budget not in ("remaining", "unknown"):
        return "release"
    first, imperative = _signal_flags(text or "")
    if not (first or imperative):
        return "release"
    if not _condition_c(text or ""):
        return "release"
    return "nudge"


__all__ = [
    "FALSE_STOP_NUDGE",
    "FALSE_STOP_SIGNALS",
    "ROUTING_BLOCK_KEYS",
    "classify_false_stop_budget",
    "false_stop_decision",
    "false_stop_signal_category",
]
