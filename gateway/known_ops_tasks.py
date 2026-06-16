"""Deterministic gateway fast paths for known Hermes ops tasks.

Known ops tasks are small, repeatable, low-token workflows that should not
spend an agent turn rediscovering their data source or repair path.  Add a new
entry here when a repeated failure has a stable detector and executable handler.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


Detector = Callable[[str], bool]
Handler = Callable[[str], str]


@dataclass(frozen=True)
class KnownOpsTask:
    """Registered deterministic task available before agent dispatch."""

    name: str
    platforms: frozenset[str]
    detector: Detector
    handler: Handler
    verification: tuple[str, ...]
    promotion_hint: str
    description: str = ""

    def supports_platform(self, platform: str) -> bool:
        normalized = (platform or "").strip().lower()
        return "all" in self.platforms or normalized in self.platforms


@dataclass(frozen=True)
class KnownOpsTaskResult:
    task: KnownOpsTask
    text: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _looks_like_today_token_usage_request(text: str) -> bool:
    """Detect the repeated Feishu daily token-report request shape."""
    normalized = _normalize_text(text)
    if not normalized:
        return False
    has_token = "token" in normalized or "tokens" in normalized
    has_today = any(marker in normalized for marker in ("今天", "今日", "当日", "截止到现在"))
    has_usage = any(marker in normalized for marker in ("消耗", "统计", "输入", "输出", "用量"))
    return bool(has_token and has_today and has_usage)


def _parse_top_n(text: str, default: int = 3) -> int:
    match = re.search(r"(?:前|top)\s*([0-9一二三四五六七八九十]+)", text or "", re.IGNORECASE)
    if not match:
        return default
    raw = match.group(1)
    chinese_digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    try:
        value = int(raw)
    except ValueError:
        value = chinese_digits.get(raw, default)
    return max(1, min(value, 10))


def _today_token_usage_scope(text: str) -> str:
    normalized = _normalize_text(text)
    if "飞书" in normalized and not any(marker in normalized for marker in ("全部", "所有", "整体")):
        return "feishu"
    return "all"


def _render_today_token_usage_report(text: str) -> str:
    from tools.local_repair_tool import render_today_token_usage_report

    return render_today_token_usage_report(
        scope=_today_token_usage_scope(text),
        top_n=_parse_top_n(text),
    )


KNOWN_OPS_TASKS: tuple[KnownOpsTask, ...] = (
    KnownOpsTask(
        name="today_token_usage_report",
        platforms=frozenset({"feishu"}),
        detector=_looks_like_today_token_usage_request,
        handler=_render_today_token_usage_report,
        verification=(
            "unit: tests/gateway/test_known_ops_tasks.py",
            "unit: tests/tools/test_local_repair_tool.py",
            "runtime: hermes gateway restart && hermes gateway status",
        ),
        promotion_hint=(
            "Repeated Feishu requests for same-day token usage should stay on this "
            "deterministic state.db report path; add adjacent usage reports as new "
            "KnownOpsTask entries instead of adding one-off gateway conditionals."
        ),
        description="Render today's Hermes input/output token usage and top sessions.",
    ),
)


def iter_known_ops_tasks(platform: str) -> tuple[KnownOpsTask, ...]:
    return tuple(task for task in KNOWN_OPS_TASKS if task.supports_platform(platform))


def match_known_ops_task(platform: str, text: str) -> KnownOpsTask | None:
    for task in iter_known_ops_tasks(platform):
        try:
            if task.detector(text):
                return task
        except Exception as exc:
            logger.warning("Known ops detector failed for %s: %s", task.name, exc)
    return None


def render_known_ops_task(platform: str, text: str) -> KnownOpsTaskResult | None:
    task = match_known_ops_task(platform, text)
    if task is None:
        return None
    return KnownOpsTaskResult(task=task, text=task.handler(text))


def known_ops_task_metadata(platform: str | None = None) -> list[dict[str, object]]:
    tasks: Sequence[KnownOpsTask]
    if platform:
        tasks = iter_known_ops_tasks(platform)
    else:
        tasks = KNOWN_OPS_TASKS
    return [
        {
            "name": task.name,
            "platforms": sorted(task.platforms),
            "description": task.description,
            "verification": list(task.verification),
            "promotion_hint": task.promotion_hint,
        }
        for task in tasks
    ]

