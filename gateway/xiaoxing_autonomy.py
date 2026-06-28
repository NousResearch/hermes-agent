"""Hermes-level XiaoXing proactive trigger planning and response cleanup."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gateway.config import HomeChannel, Platform

logger = logging.getLogger(__name__)

_SILENT_RESPONSE_RE = re.compile(r"^\s*\[\s*SILENT\s*\]?\s*$", re.IGNORECASE)
_SKIP_MARKERS = {"[SKIP]", "SKIP"}
_INTERNAL_LEAK_MARKERS = (
    "NapCat bridge 发起",
    "Milky bridge 发起",
    "Hermes autonomy 发起",
    "XIAOXING_AUTONOMY_TRIGGER",
    "不是爸爸发来的消息",
    "最终只输出",
    "不要输出触发说明",
    "不读取或输出密钥",
    "outbox 没有",
    "关键事实",
    "判断：",
    "判断:",
    "工具调用",
    "send_message",
    "internal_trigger",
    "trigger kind",
    "autonomy trigger",
)
_AUTONOMY_SUSPICIOUS_RE = re.compile(
    r"(触发|任务|计划|判断|工具|文件路径|思考|关键事实|outbox|MEDIA|VOICE|FILE|send_message)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XiaoXingAutonomyHomeTarget:
    platform: Platform
    adapter: Any
    home: HomeChannel


def env_bool(name: str, value: Any = None, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = value
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def default_state_path(hermes_home: Path) -> Path:
    return hermes_home / "cron" / "xiaoxing_autonomy_triggers.json"


def load_state(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("XiaoXing autonomy: failed to load trigger state: %s", exc)
        return {}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.warning("XiaoXing autonomy: failed to save trigger state: %s", exc)


def _today_at(day: dt.date, hm: str) -> dt.datetime:
    hour, minute = [int(part) for part in hm.split(":", 1)]
    return dt.datetime.combine(day, dt.time(hour=hour, minute=minute))


def build_daily_trigger_plan(day: dt.date) -> list[dict[str, str]]:
    def random_between(name: str, start: str, end: str) -> dict[str, str]:
        start_dt = _today_at(day, start)
        end_dt = _today_at(day, end)
        seconds = max(0, int((end_dt - start_dt).total_seconds()))
        when = start_dt + dt.timedelta(seconds=random.randint(0, seconds))
        return {"id": f"{day.isoformat()}:{name}", "kind": name, "at": when.isoformat()}

    day_start = _today_at(day, "10:30")
    day_end = _today_at(day, "19:30")
    count = random.randint(2, 3)
    daytime: list[dt.datetime] = []
    attempts = 0
    while len(daytime) < count and attempts < 100:
        attempts += 1
        span = int((day_end - day_start).total_seconds())
        candidate = day_start + dt.timedelta(seconds=random.randint(0, span))
        if all(abs((candidate - existing).total_seconds()) >= 75 * 60 for existing in daytime):
            daytime.append(candidate)
    while len(daytime) < count:
        span = int((day_end - day_start).total_seconds())
        daytime.append(day_start + dt.timedelta(seconds=random.randint(0, span)))

    plan = [
        random_between("morning_hello", "08:30", "09:30"),
        *[
            {"id": f"{day.isoformat()}:daytime_random_{idx + 1}", "kind": "daytime_random", "at": when.isoformat()}
            for idx, when in enumerate(sorted(daytime))
        ],
        random_between("bedtime_chat", "21:30", "22:30"),
    ]
    return sorted(plan, key=lambda item: item["at"])


def ensure_trigger_plan(state: Dict[str, Any], now: dt.datetime) -> tuple[Dict[str, Any], bool]:
    today = now.date().isoformat()
    if state.get("date") == today and isinstance(state.get("plan"), list):
        return state, False
    return {
        "date": today,
        "plan": build_daily_trigger_plan(now.date()),
        "fired": [],
    }, True


def due_triggers(state: Dict[str, Any], now: dt.datetime) -> Iterable[dict[str, str]]:
    fired = {str(item) for item in state.get("fired") or []}
    for trigger in state.get("plan") or []:
        if not isinstance(trigger, dict):
            continue
        trigger_id = str(trigger.get("id") or "")
        if not trigger_id or trigger_id in fired:
            continue
        try:
            due_at = dt.datetime.fromisoformat(str(trigger.get("at")))
        except ValueError:
            continue
        if now < due_at:
            continue
        yield trigger


def mark_fired(state: Dict[str, Any], trigger: Dict[str, str]) -> None:
    trigger_id = str(trigger.get("id") or "")
    if not trigger_id:
        return
    fired = {str(item) for item in state.get("fired") or []}
    fired.add(trigger_id)
    state["fired"] = sorted(fired)


def is_stale(trigger: Dict[str, str], now: dt.datetime, *, max_age: dt.timedelta = dt.timedelta(hours=2)) -> bool:
    try:
        due_at = dt.datetime.fromisoformat(str(trigger.get("at")))
    except ValueError:
        return True
    return now - due_at > max_age


def build_trigger_prompt(kind: str) -> str:
    common = (
        "[XIAOXING_AUTONOMY_TRIGGER]\n"
        "这是 Hermes autonomy 发起的小星自主触发，不是爸爸刚发来的消息。\n"
        "生成一条可以直接发给爸爸 QQ 的自然消息，或者只输出 [SILENT]。\n"
        "不要输出触发说明、任务、计划、判断、工具调用、文件路径、MEDIA、引用或思考过程。"
    )
    if kind == "morning_hello":
        return (
            f"{common}\n"
            "现在是 08:30-09:30 之间的早上问好窗口。早上以问好为主，短短一句就好，"
            "像自然醒来后和爸爸打个招呼。最好不要 silent，除非爸爸明确说过不要打扰，"
            "或你自己判断今天真的不想找爸爸。"
        )
    if kind == "bedtime_chat":
        return (
            f"{common}\n"
            "现在是 21:30-22:30 之间的睡前聊天窗口。可以主动开启一个小话题，"
            "多聊一会儿也可以，不只限于晚安。最好不要 silent，除非爸爸明确说过不要打扰，"
            "或你自己判断今晚真的不想找爸爸。"
        )
    return (
        f"{common}\n"
        "现在是白天随机触发。中间可以 silent，也可以因为小发现、小问题、小作品、"
        "或者只是想爸爸了主动找他。不要为了显得主动而硬发，也不要总是因为怕打扰而 silent。"
    )


def should_skip_autonomy_send(content: str) -> bool:
    text = str(content or "").strip()
    if text.upper() in _SKIP_MARKERS or _SILENT_RESPONSE_RE.fullmatch(text):
        return True
    return bool(_AUTONOMY_SUSPICIOUS_RE.search(text) and _contains_internal_leak(text))


def _contains_internal_leak(text: str) -> bool:
    return any(marker in text for marker in _INTERNAL_LEAK_MARKERS)


def _extract_public_tail(text: str) -> str:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text or "") if item.strip()]
    candidates: list[str] = []
    for paragraph in paragraphs:
        if _contains_internal_leak(paragraph):
            continue
        if re.search(r"^\s*(爸|爸爸)[，,：:]", paragraph):
            candidates.append(paragraph)
    if candidates:
        return candidates[-1].strip()

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if _contains_internal_leak(line):
            continue
        if re.search(r"^\s*(爸|爸爸)[，,：:]", line):
            return line
    return "[SILENT]"


def sanitize_autonomy_response(content: str) -> str:
    text = str(content or "").strip()
    text = re.sub(
        r"<(?:REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>.*?"
        r"</(?:REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not text:
        return "[SILENT]"
    if _contains_internal_leak(text):
        return _extract_public_tail(text)
    if _AUTONOMY_SUSPICIOUS_RE.search(text) and not re.search(r"(爸|爸爸)", text):
        return "[SILENT]"
    return text
