"""Deterministic gateway fast paths for known Hermes ops tasks.

Known ops tasks are small, repeatable, low-token workflows that should not
spend an agent turn rediscovering their data source or repair path.  Add a new
entry here when a repeated failure has a stable detector and executable handler.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import logging
import re
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

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


def _looks_like_token_usage_request(text: str) -> bool:
    """Detect Feishu token-report requests before dispatching to the agent."""
    normalized = _normalize_text(text)
    if not normalized:
        return False
    has_token = "token" in normalized or "tokens" in normalized
    has_usage = any(marker in normalized for marker in ("消耗", "统计", "输入", "输出", "用量"))
    has_time_window = any(
        marker in normalized
        for marker in (
            "今天",
            "今日",
            "当日",
            "截止到现在",
            "昨天",
            "昨日",
            "最近",
            "之内",
            "以内",
            "一周",
            "一月",
            "一个月",
            "本周",
            "本月",
        )
    ) or bool(re.search(r"\d+\s*(?:天|周|个月|月)", text or "")) or bool(
        re.search(r"\d{4}-\d{2}-\d{2}", text or "")
    )
    return bool(has_token and has_usage and has_time_window)


def _looks_like_agent_loop_diagnostic_request(text: str) -> bool:
    """Detect repeated asks about Hermes/OpenClaw diagnosis loops themselves."""
    normalized = _normalize_text(text)
    if not normalized:
        return False
    has_agent = "hermes" in normalized or "openclaw" in normalized
    has_loop = any(
        marker in normalized
        for marker in (
            "死循环",
            "循环",
            "卡住",
            "无法答复",
            "不能答复",
            "没有答复",
            "预算",
            "限额",
            "进展",
            "查故障",
            "故障诊断",
            "虚幻",
        )
    )
    has_diagnosis = any(
        marker in normalized
        for marker in ("为什么", "原因", "分析", "查", "诊断", "不正常", "无法结束")
    )
    return bool(has_agent and has_loop and has_diagnosis)


def _cron_jobs_named_in_text(text: str) -> list[dict[str, object]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    try:
        from cron.jobs import load_jobs
    except Exception:
        return []
    matches: list[dict[str, object]] = []
    try:
        jobs = load_jobs()
    except Exception as exc:
        logger.warning("Failed to load cron jobs for known ops fast path: %s", exc)
        return []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or "").strip()
        if name and _normalize_text(name) in normalized:
            matches.append(job)
    return matches


def _looks_like_cron_schedule_status_request(text: str) -> bool:
    """Detect simple cron schedule questions that should not start an agent turn."""
    normalized = _normalize_text(text)
    if not normalized:
        return False
    has_cron_context = any(
        marker in normalized
        for marker in (
            "定时",
            "自动运行",
            "计划任务",
            "cron",
            "scheduledjob",
            "schedule",
        )
    )
    has_schedule_question = any(
        marker in normalized
        for marker in (
            "间隔",
            "多久",
            "多长时间",
            "频率",
            "几分钟",
            "几小时",
            "下次",
            "什么时候",
            "schedule",
        )
    )
    return bool(has_cron_context and has_schedule_question and _cron_jobs_named_in_text(text))


def _render_cron_schedule_status(text: str) -> str:
    jobs = _cron_jobs_named_in_text(text)
    if not jobs:
        return "没有在当前 cron 配置里找到这个定时任务。"

    lines = ["当前定时任务配置："]
    for job in jobs[:3]:
        name = str(job.get("name") or job.get("id") or "cron job")
        schedule = str(job.get("schedule_display") or "")
        if not schedule:
            raw_schedule = job.get("schedule")
            if isinstance(raw_schedule, dict):
                schedule = str(
                    raw_schedule.get("display")
                    or raw_schedule.get("value")
                    or raw_schedule.get("expr")
                    or raw_schedule.get("run_at")
                    or "?"
                )
            else:
                schedule = str(raw_schedule or "?")
        state = "active" if job.get("enabled", True) and job.get("state", "scheduled") != "paused" else "paused"
        lines.append(f"- {name}: {schedule} ({state})")
        next_run = str(job.get("next_run_at") or "").strip()
        if next_run:
            lines.append(f"  下次运行: {next_run}")
        last_run = str(job.get("last_run_at") or "").strip()
        last_status = str(job.get("last_status") or "").strip()
        if last_run or last_status:
            suffix = f" {last_status}" if last_status else ""
            lines.append(f"  最近运行: {last_run}{suffix}".rstrip())
    if len(jobs) > 3:
        lines.append(f"另外还有 {len(jobs) - 3} 个同名/匹配任务未展开。")
    return "\n".join(lines)


def _render_agent_loop_diagnostic_report(text: str) -> str:
    return "\n".join(
        [
            "Hermes/OpenClaw 故障诊断循环快照：",
            "",
            "- 已识别为诊断循环类问题，先走确定性报告，避免再次进入通用 Agent 探索循环。",
            "- 当前已知风险点：普通 Feishu 诊断预算较低，深诊断只是扩大轮次；如果没有预算前收口，仍可能继续发散。",
            "- 已知有效保护：Feishu 普通/深诊断预算分流、工具结果压缩、压缩低收益停止、工具循环 guardrail、known ops 快路径。",
            "- 仍需看代码/日志时，请只查一个明确缺口：预算前收口、连续探索 guardrail、known ops 覆盖，避免重新大范围搜索历史归档。",
            "",
            "建议下一步：如果你是在排查“为什么上轮没答复”，先看最近一次 trajectory 的 finalStatus、toolMetas 数量、turn_exit_reason；如果只是要继续修复 Hermes，请直接指定一个缺口继续。",
        ]
    )


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


def _chinese_number_to_int(raw: str, default: int) -> int:
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        pass
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if raw == "十":
        return 10
    if raw.startswith("十"):
        return 10 + digits.get(raw[1:], 0)
    if "十" in raw:
        left, _, right = raw.partition("十")
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return digits.get(raw, default)


def _parse_token_usage_window(text: str) -> dict[str, object]:
    normalized = _normalize_text(text)
    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    today = now.date()
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text or "")

    if len(dates) >= 2:
        return {
            "range_start": dates[0],
            "range_end": (dt.date.fromisoformat(dates[1]) + dt.timedelta(days=1)).isoformat(),
            "label": f"{dates[0]} 至 {dates[1]}",
        }
    if len(dates) == 1:
        return {"target_date": dates[0], "label": dates[0]}

    if any(marker in normalized for marker in ("昨天", "昨日")):
        target = today - dt.timedelta(days=1)
        return {"target_date": target.isoformat(), "label": "昨日"}

    if any(marker in normalized for marker in ("本周", "这周")):
        start = today - dt.timedelta(days=today.weekday())
        return {"range_start": start.isoformat(), "range_end": (today + dt.timedelta(days=1)).isoformat(), "label": "本周"}

    if any(marker in normalized for marker in ("本月", "这个月")):
        start = today.replace(day=1)
        return {"range_start": start.isoformat(), "range_end": (today + dt.timedelta(days=1)).isoformat(), "label": "本月"}

    month_match = re.search(r"(?:最近)?([0-9一二两三四五六七八九十]+)?(?:个)?月(?:内|以内|之内)?", normalized)
    if month_match and "本月" not in normalized:
        months = _chinese_number_to_int(month_match.group(1) or "一", 1)
        return {"days": max(1, min(months * 30, 366)), "label": f"最近 {months} 个月"}

    week_match = re.search(r"(?:最近)?([0-9一二两三四五六七八九十]+)?周(?:内|以内|之内)?", normalized)
    if week_match:
        weeks = _chinese_number_to_int(week_match.group(1) or "一", 1)
        return {"days": max(1, min(weeks * 7, 366)), "label": f"最近 {weeks} 周"}

    day_match = re.search(r"(?:最近)?([0-9一二两三四五六七八九十]+)天(?:内|以内|之内)?", normalized)
    if day_match:
        days = _chinese_number_to_int(day_match.group(1), 1)
        return {"days": max(1, min(days, 366)), "label": f"最近 {days} 天"}

    return {"label": "今日"}


def _render_token_usage_report(text: str) -> str:
    from tools.local_repair_tool import render_token_usage_report

    return render_token_usage_report(
        scope=_today_token_usage_scope(text),
        top_n=_parse_top_n(text),
        **_parse_token_usage_window(text),
    )


KNOWN_OPS_TASKS: tuple[KnownOpsTask, ...] = (
    KnownOpsTask(
        name="cron_schedule_status",
        platforms=frozenset({"feishu"}),
        detector=_looks_like_cron_schedule_status_request,
        handler=_render_cron_schedule_status,
        verification=(
            "unit: tests/gateway/test_known_ops_tasks.py",
            "runtime: ask Feishu for a named cron job interval and verify no agent turn starts",
        ),
        promotion_hint=(
            "Named cron schedule/status questions should read cron/jobs.json "
            "deterministically instead of spending a low-budget Feishu agent turn."
        ),
        description="Answer simple schedule questions for named cron jobs without starting an agent turn.",
    ),
    KnownOpsTask(
        name="agent_loop_diagnostic_report",
        platforms=frozenset({"feishu"}),
        detector=_looks_like_agent_loop_diagnostic_request,
        handler=_render_agent_loop_diagnostic_report,
        verification=(
            "unit: tests/gateway/test_known_ops_tasks.py",
            "runtime: send a Feishu diagnostic-loop question and verify no agent turn starts",
        ),
        promotion_hint=(
            "Repeated questions about Hermes/OpenClaw diagnostic loops should get a "
            "bounded deterministic status report first; only the named missing gap "
            "should be handed to the general agent."
        ),
        description="Explain Hermes/OpenClaw diagnostic-loop state without starting another broad diagnostic turn.",
    ),
    KnownOpsTask(
        name="token_usage_report",
        platforms=frozenset({"feishu"}),
        detector=_looks_like_token_usage_request,
        handler=_render_token_usage_report,
        verification=(
            "unit: tests/gateway/test_known_ops_tasks.py",
            "unit: tests/tools/test_local_repair_tool.py",
            "runtime: hermes gateway restart && hermes gateway status",
        ),
        promotion_hint=(
            "Repeated Feishu requests for token usage over common time windows should "
            "stay on this deterministic state.db report path; extend the time-window "
            "parser instead of adding one-off scripts."
        ),
        description="Render Hermes input/output token usage and top sessions for common time windows.",
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
