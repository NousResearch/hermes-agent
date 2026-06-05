from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent, GatewayRunner

logger = logging.getLogger(__name__)

# Channels configuration
CHANNELS = {
    "help": "1509389598923559053",
}

# Resolve real path of the project profile dir dynamically
# Note: Path.home() resolves to /home/ameobius, but in some cron runs it might differ,
# so we use a relative fallback path if it lives under security-workstation.
PRODUCERS_DIR = Path("/home/ameobius/projects/security-workstation/.hermes/profiles/producers")
if not PRODUCERS_DIR.is_dir():
    PRODUCERS_DIR = Path.home() / ".hermes" / "profiles" / "producers"

CONSENT_FILE = PRODUCERS_DIR / "weekly_digest_consent.json"
HELP_INTAKE_FILE = PRODUCERS_DIR / "help_case_intake_state.json"
QUEUE_FILE = PRODUCERS_DIR / "gitdb_tools_queue.json"
POST_STATE_FILE = PRODUCERS_DIR / "gitdb_tools_post_state.json"
SANITY_SCRIPT = PRODUCERS_DIR / "scripts" / "public_text_sanitizer.py"
GROWTH_ACTIONS_FILE = PRODUCERS_DIR / "discord_channel_growth_actions.json"
GROWTH_STATUS_POST_STATE_FILE = PRODUCERS_DIR / "discord_growth_status_post_state.json"
COMMUNITY_POST_GUARD_SCRIPT = PRODUCERS_DIR / "scripts" / "community_post_guard.py"
HUMAN_QUESTIONS_FILE = PRODUCERS_DIR / "discord_human_questions.json"
HUMAN_QUESTIONS_SCRIPT = PRODUCERS_DIR / "scripts" / "discord_human_questions.py"


def run_sanitizer(text: str) -> str:
    """Helper to sanitize output texts via the profile sanitizer logic."""
    if not SANITY_SCRIPT.is_file():
        return text
    try:
        res = subprocess.run(
            [sys.executable, str(SANITY_SCRIPT)],
            input=text,
            text=True,
            capture_output=True,
            check=True
        )
        return res.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to run sanitizer: {e}")
        return text


def _event_text(event: Any) -> str:
    return str(getattr(event, "text", None) or getattr(event, "content", None) or "")


def _event_source(event: Any) -> Any:
    return getattr(event, "source", None)


def _event_author(event: Any) -> str:
    source = _event_source(event)
    return str(
        getattr(event, "author", None)
        or getattr(source, "user_name", None)
        or getattr(source, "user_id", None)
        or "unknown"
    )


def _event_channel_id(event: Any) -> str:
    source = _event_source(event)
    return str(getattr(event, "channel_id", None) or getattr(source, "chat_id", None) or "")


def _event_message_id(event: Any) -> str | None:
    source = _event_source(event)
    value = getattr(event, "message_id", None) or getattr(source, "message_id", None)
    return str(value) if value else None


async def _send_reply(event: Any, gateway: Any | None, text: str) -> None:
    """Reply from a pre_gateway_dispatch hook without relying on adapter-specific helpers."""
    reply_fn = getattr(event, "reply", None)
    if callable(reply_fn):
        maybe = reply_fn(text)
        if hasattr(maybe, "__await__"):
            await maybe
        return

    source = _event_source(event)
    if gateway is None or source is None:
        logger.warning("producers-triage reply dropped: gateway/source unavailable")
        return

    adapter = getattr(gateway, "adapters", {}).get(getattr(source, "platform", None))
    if adapter is None:
        logger.warning("producers-triage reply dropped: adapter unavailable for %s", getattr(source, "platform", None))
        return

    metadata = None
    thread_id = getattr(source, "thread_id", None)
    if thread_id:
        metadata = {"thread_id": str(thread_id)}
    await adapter.send(
        str(getattr(source, "chat_id", _event_channel_id(event))),
        text,
        reply_to=_event_message_id(event),
        metadata=metadata,
    )


_FIELD_ALIASES = {
    "prompt": {"prompt", "промпт"},
    "goal_or_genre": {"goal", "цель", "жанр", "genre", "mood", "вайб", "style", "стиль"},
    "failure_mode": {"failure", "problem", "issue", "проблема", "сломалось", "не нравится", "что не нравится", "что не так"},
}


def _normalize_field_key(key: str) -> str | None:
    norm = re.sub(r"\s+", " ", key.strip().lower())
    for canonical, aliases in _FIELD_ALIASES.items():
        if norm in aliases:
            return canonical
    return None


def _extract_prompt_doctor_fields(text: str) -> dict[str, str]:
    extracted = {"prompt": "", "goal_or_genre": "", "failure_mode": ""}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:=\n]{2,40})\s*[:=]\s*(.*?)\s*$", line)
        if not match:
            continue
        canonical = _normalize_field_key(match.group(1))
        if canonical and match.group(2).strip():
            extracted[canonical] = match.group(2).strip()
    return extracted


def _is_producers_profile() -> bool:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home().name == "producers"
    except Exception as exc:
        logger.warning("producers-triage profile check failed: %s", exc)
        return False


def _should_intercept(event: Any) -> bool:
    if not _is_producers_profile():
        return False
    raw_text = _event_text(event).strip()
    norm_text = raw_text.lower()
    if not norm_text:
        return False
    channel_id = _event_channel_id(event)
    triggers = (
        "кработ согласие",
        "кработ отказ",
        "почини промпт",
        "разбери промпт",
        "разбери prompt",
        "prompt doctor",
        "кработ вопросы",
        "кработ оживи",
        "кработ вовлечение",
        "кработ статус роста",
        "кработ статус канала",
        "кработ метрики",
        "кработ рост",
        "кработ каналы",
        "кработ очередь",
        "кработ кандидаты",
        "кработ некст",
        "кработ следующий",
        "кработ запости",
    )
    return channel_id == CHANNELS["help"] or any(trigger in norm_text for trigger in triggers)


def _pre_gateway_dispatch_hook(**kwargs: Any) -> dict[str, Any] | None:
    event = kwargs.get("event")
    if event is None or not _should_intercept(event):
        return None
    gateway = kwargs.get("gateway") or kwargs.get("runner")
    session_store = kwargs.get("session_store")
    asyncio.create_task(pre_gateway_dispatch(event=event, gateway=gateway, session_store=session_store))
    return {"action": "skip", "reason": "producers-triage-fast-path"}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch_hook)


def load_consent_db() -> dict:
    if not CONSENT_FILE.is_file():
        return {}
    try:
        return json.loads(CONSENT_FILE.read_text())
    except Exception:
        return {}


def save_consent_db(db: dict):
    PRODUCERS_DIR.mkdir(parents=True, exist_ok=True)
    CONSENT_FILE.write_text(json.dumps(db, indent=2, ensure_ascii=False))


def normalize_username(user: str) -> str:
    return re.sub(r"\s+", " ", user.strip().lower())


def get_user_consent(user: str) -> bool:
    db = load_consent_db()
    norm = normalize_username(user)
    return db.get(norm, {}).get("consent", False)


def set_user_consent(user: str, consent: bool, channel: str):
    db = load_consent_db()
    norm = normalize_username(user)
    db[norm] = {
        "consent": consent,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "channel": channel
    }
    save_consent_db(db)


def load_intake_state() -> dict:
    if not HELP_INTAKE_FILE.is_file():
        return {}
    try:
        return json.loads(HELP_INTAKE_FILE.read_text())
    except Exception:
        return {}


def save_intake_state(state: dict):
    PRODUCERS_DIR.mkdir(parents=True, exist_ok=True)
    HELP_INTAKE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def make_message_hash(content: str, channel_id: str, author: str) -> str:
    payload = f"{content.strip()}:{channel_id}:{author.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sync_vikunja() -> dict[str, Any]:
    env = os.environ.copy()
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    cmd = [sys.executable, "scripts/ops/bd_to_vikunja.py"]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return {"ok": True, "output": res.stdout}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"Exit {e.returncode}: {e.stderr or e.stdout}"}


def get_pending_candidates() -> list[dict]:
    if not QUEUE_FILE.is_file():
        return []
    try:
        q = json.loads(QUEUE_FILE.read_text())
    except Exception:
        return []

    posted_set = set()
    hash_set = set()
    if POST_STATE_FILE.is_file():
        try:
            s = json.loads(POST_STATE_FILE.read_text())
            posted_set = set(s.get("posted", []))
            hash_set = set(s.get("content_hashes", []))
        except Exception:
            pass

    norm = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
    def get_hash(t):
        return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()

    pending = []
    for c in q.get("candidates", []):
        name = c.get("full_name")
        card_text = c.get("card", "")
        if name not in posted_set and get_hash(card_text) not in hash_set:
            pending.append(c)
    return pending


def load_growth_actions() -> dict[str, Any]:
    if not GROWTH_ACTIONS_FILE.is_file():
        return {}
    try:
        return json.loads(GROWTH_ACTIONS_FILE.read_text())
    except Exception:
        return {}


def load_growth_status_post_state() -> dict[str, Any]:
    if not GROWTH_STATUS_POST_STATE_FILE.is_file():
        return {}
    try:
        return json.loads(GROWTH_STATUS_POST_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_python_module(path: Path, name: str):
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning(f"Failed to load module {path}: {e}")
        return None


def load_post_guard_decision() -> dict[str, Any]:
    mod = _load_python_module(COMMUNITY_POST_GUARD_SCRIPT, "community_post_guard_runtime")
    if not mod:
        return {"allowed": False, "reason": "community post guard unavailable"}
    try:
        return mod.decide_post_allowed().as_dict()
    except Exception as e:
        return {"allowed": False, "reason": f"community post guard failed: {e}"}


def _format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "unknown"


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return text[:12] if text else "none"


def format_growth_status_reply() -> str:
    metrics = load_growth_actions()
    totals = (metrics.get("totals") or {}) if metrics else {}
    actions = metrics.get("priority_actions", []) if metrics else []
    guard = load_post_guard_decision()
    weekly_state = load_growth_status_post_state()

    allowed = bool(guard.get("allowed"))
    guard_state = "разрешен" if allowed else "заблокирован"
    ratio = _format_ratio(guard.get("human_ratio", totals.get("human_ratio")))
    min_ratio = _format_ratio(guard.get("min_human_ratio", 0.55))
    streak = guard.get("consecutive_healthy", 0)
    min_streak = guard.get("min_consecutive_healthy", 2)
    mode = weekly_state.get("last_mode") or "нет записи"
    posted_at = weekly_state.get("last_posted_at") or "нет записи"
    message_id = weekly_state.get("last_message_id") or "нет message id"
    digest = _short_hash(weekly_state.get("last_report_hash"))

    next_action = "кработ вопросы"
    if allowed:
        next_action = "мягко разморозить один ручной пост и проверить ответ людей"

    lines = [
        "статус роста:",
        f"guard: {guard_state} - {guard.get('reason', 'без причины')}",
        f"human ratio: {ratio} из {min_ratio}",
        f"здоровая серия: {streak} из {min_streak}",
        f"сообщения: {totals.get('recent_messages', 0)} - люди {totals.get('recent_human', 0)} - боты {totals.get('recent_bot', 0)}",
        f"weekly report: {mode} - hash: {digest} - message: {message_id}",
        f"last weekly: {posted_at}",
        f"следующее действие: {next_action}",
    ]
    if not allowed:
        lines.append("автопостинг не трогать, пока ratio не держится выше порога два замера подряд")
    return run_sanitizer("\n".join(lines))


def load_human_questions() -> dict[str, Any]:
    if HUMAN_QUESTIONS_FILE.is_file():
        try:
            return json.loads(HUMAN_QUESTIONS_FILE.read_text())
        except Exception:
            pass

    if not HUMAN_QUESTIONS_SCRIPT.is_file():
        return {}

    try:
        subprocess.run(
            [sys.executable, str(HUMAN_QUESTIONS_SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
        )
        if HUMAN_QUESTIONS_FILE.is_file():
            return json.loads(HUMAN_QUESTIONS_FILE.read_text())
    except Exception as e:
        logger.warning(f"Failed to generate human questions: {e}")
    return {}


def format_human_questions_reply() -> str:
    report = load_human_questions()
    questions = report.get("questions", []) if report else []
    if not questions:
        return "готовых вопросов пока нет - сначала нужен ночной замер метрик роста"

    totals = report.get("totals", {}) or {}
    lines = [
        "вопросы для оживления дискорда:",
        f"human ratio сейчас: {totals.get('human_ratio', 'unknown')}",
        "",
    ]
    for idx, item in enumerate(questions[:5], 1):
        channel = item.get("channel", "канал")
        question = item.get("question", "какой один вопрос стоит задать тут?")
        followup = item.get("followup", "достаточно короткого ответа")
        lines.append(f"{idx}- {channel}: {question}")
        lines.append(f"   добивка: {followup}")

    lines.append("")
    lines.append("важно: это ручные вопросы, не автопостинг - гвард публикаций остается включенным")
    return run_sanitizer("\n".join(lines))


def _format_delta(value: Any) -> str:
    try:
        number = int(value)
    except Exception:
        return "0"
    return f"{number:+d}"


def format_growth_metrics_reply() -> str:
    report = load_growth_actions()
    if not report:
        return "метрики роста пока не собраны - дождись ночного замера или запусти пересчет метрик"

    totals = report.get("totals", {}) or {}
    trend = report.get("trend") or {}
    actions = report.get("priority_actions", []) or []
    healthy = report.get("healthy_lanes", []) or []

    lines = [
        "метрики дискорда:",
        f"сообщения: {totals.get('recent_messages', 0)} - люди {totals.get('recent_human', 0)} - боты {totals.get('recent_bot', 0)}",
        f"human ratio: {totals.get('human_ratio', 'unknown')}",
    ]

    if trend:
        lines.append(
            "тренд: "
            f"люди {_format_delta(trend.get('human_delta'))} - "
            f"боты {_format_delta(trend.get('bot_delta'))} - "
            f"сообщения {_format_delta(trend.get('message_delta'))}"
        )

    if actions:
        lines.append("")
        lines.append("топ действий:")
        for idx, item in enumerate(actions[:5], 1):
            tags = ", ".join(item.get("tags") or ["normal"])
            cadence = item.get("cadence") or "наблюдать"
            first_action = (item.get("actions") or ["наблюдать"])[0]
            lines.append(f"{idx}- {item.get('name', 'канал')}: {tags} - {cadence}")
            lines.append(f"   действие: {first_action}")

    if healthy:
        lanes = ", ".join(item.get("name", "канал") for item in healthy[:4])
        lines.append("")
        lines.append(f"живые каналы, которые не надо забивать ботом: {lanes}")

    lines.append("")
    lines.append("правило: не увеличивать автопостинг, пока human ratio не держится выше 0.55 два ночных замера подряд")
    return run_sanitizer("\n".join(lines))


def trigger_posting(count: int = 1) -> dict[str, Any]:
    env = os.environ.copy()
    # Explicitly clear proxy if targeting local posting script
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"

    script_path = PRODUCERS_DIR / "scripts" / "gitdb_tools_poster.py"
    if not script_path.is_file():
        return {"ok": False, "error": "Poster script not found"}

    cmd = [sys.executable, str(script_path), "--count", str(count)]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return {"ok": True, "output": res.stdout}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"Exit {e.returncode}: {e.stderr or e.stdout}"}


async def pre_gateway_dispatch(
    event: MessageEvent,
    runner: GatewayRunner | None = None,
    gateway: GatewayRunner | None = None,
    session_store: Any | None = None,
) -> dict[str, Any] | None:
    """Fast-path handler for explicit triage commands and intakes.

    Intercepts consent/help/GitDB tool review patterns.
    """
    # Verify we run in producers profile context
    if not _is_producers_profile():
        return None

    active_gateway = gateway or runner
    raw_text = _event_text(event).strip()
    norm_text = raw_text.lower()
    author = _event_author(event)
    channel_id = _event_channel_id(event)

    # 1. User Consent Commands
    if "кработ согласие" in norm_text:
        set_user_consent(author, True, channel_id)
        reply = run_sanitizer(f"запомнил, {author} - теперь твои наработки из #аудио-наработки будут попадать в еженедельный дайджест")
        await _send_reply(event, active_gateway, reply)
        return {"action": "skip"}

    if "кработ отказ" in norm_text:
        set_user_consent(author, False, channel_id)
        reply = run_sanitizer(f"понял, {author} - исключил твои наработки из дайджестов")
        await _send_reply(event, active_gateway, reply)
        return {"action": "skip"}

    # 1.5 Prompt Doctor Command Triage & Fast-Path Fallback
    # Triggers on explicit prompt doctor commands (e.g. "кработ почини промпт", "кработ prompt doctor")
    if "почини промпт" in norm_text or "разбери промпт" in norm_text or "разбери prompt" in norm_text or "prompt doctor" in norm_text:
        # Check if we have prompt doctor specs
        from .prompt_doctor_fallback import run_prompt_doctor_offline

        extracted = _extract_prompt_doctor_fields(raw_text)
        prompt_val = extracted.get("prompt", "")
        goal_val = extracted.get("goal_or_genre", "")
        failure_val = extracted.get("failure_mode", "")

        # If we didn't extract fields through strict key-value notation, try parsing with regex fallback
        if not prompt_val or not goal_val or not failure_val:
            # Simple regex search: look for text block after trigger phrases
            clean_cmd = re.sub(r"(?is)@?krabot\s+(?:почини промпт|разбери промпт|разбери prompt|prompt doctor)\s*[:=]?", "", raw_text).strip()
            # If the user sent a plain text list separated by commas, newlines or semicolons, guess parts:
            parts = [p.strip() for p in re.split(r"[\n;]", clean_cmd) if p.strip()]
            if len(parts) >= 3:
                prompt_val = parts[0]
                goal_val = parts[1]
                failure_val = parts[2]
            elif len(parts) == 2:
                prompt_val = parts[0]
                goal_val = parts[1]
                failure_val = "generic"
            elif len(parts) == 1:
                prompt_val = parts[0]
                goal_val = "ambient electronic"
                failure_val = "muddy sound"

        # If we have some inputs, execute the offline fallback immediately!
        if prompt_val:
            diag_output = run_prompt_doctor_offline(prompt_val, goal_val, failure_val)
            await _send_reply(event, active_gateway, run_sanitizer(diag_output))
            return {"action": "skip"}
        else:
            await _send_reply(event, active_gateway, run_sanitizer("пожалуйста, пришли параметры промпта в формате:\nпромпт: [текст]\nцель: [жанр/вайб]\nпроблема: [что сломалось]"))
            return {"action": "skip"}

    if "кработ вопросы" in norm_text or "кработ оживи" in norm_text or "кработ вовлечение" in norm_text:
        await _send_reply(event, active_gateway, format_human_questions_reply())
        return {"action": "skip"}

    if "кработ статус роста" in norm_text or "кработ статус канала" in norm_text:
        await _send_reply(event, active_gateway, format_growth_status_reply())
        return {"action": "skip"}

    if "кработ метрики" in norm_text or "кработ рост" in norm_text or "кработ каналы" in norm_text:
        await _send_reply(event, active_gateway, format_growth_metrics_reply())
        return {"action": "skip"}

    # 2. GitDB Tools Review Commands (kra queue/next/approve/reject/post)
    is_admin = author in ["ameobius", "a meobius", "a_meobius"]

    if "кработ очередь" in norm_text or "кработ кандидаты" in norm_text:
        pending = get_pending_candidates()
        if not pending:
            await _send_reply(event, active_gateway, "очередь кандидатов пуста")
            return {"action": "skip"}

        reply_lines = [f"в очереди {len(pending)} кандидатов - вот первые 5:"]
        for i, c in enumerate(pending[:5], 1):
            stars = c.get("stars", 0)
            lang = c.get("language", "unknown")
            reply_lines.append(f"{i}- **{c['full_name']}** ({lang}, {stars} stars)")
        reply_lines.append("\nпоказать карточку следующего: `кработ некст`\nзапостить: `кработ запости [число]`")
        await _send_reply(event, active_gateway, run_sanitizer("\n".join(reply_lines)))
        return {"action": "skip"}

    if "кработ некст" in norm_text or "кработ следующий" in norm_text:
        pending = get_pending_candidates()
        if not pending:
            await _send_reply(event, active_gateway, "очередь кандидатов пуста")
            return {"action": "skip"}

        next_c = pending[0]
        card = next_c.get("card", "")
        # Present candidate card directly
        await _send_reply(event, active_gateway, run_sanitizer(f"следующий кандидат в очереди:\n\n{card}"))
        return {"action": "skip"}

    if "кработ запости" in norm_text:
        if not is_admin:
            await _send_reply(event, active_gateway, "команда доступна только администраторам")
            return {"action": "skip"}

        count = 1
        match = re.search(r"кработ запости\s+(\d+)", norm_text)
        if match:
            count = int(match.group(1))

        pending = get_pending_candidates()
        if not pending:
            await _send_reply(event, active_gateway, "нечего постить, очередь пуста")
            return {"action": "skip"}

        await _send_reply(event, active_gateway, run_sanitizer(f"запускаю публикацию кандидатов ({count} шт)-"))

        # Trigger background posting task to not block gateway response
        loop = asyncio.get_running_loop()
        def run_post():
            res = trigger_posting(count)
            # Send notification back
            if res.get("ok"):
                # Run notification safely
                asyncio.run_coroutine_threadsafe(
                    _send_reply(event, active_gateway, run_sanitizer(f"публикация завершена успешно:\n{res['output']}")),
                    loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    _send_reply(event, active_gateway, run_sanitizer(f"ошибка при публикации: {res['error']}")),
                    loop
                )

        asyncio.create_task(asyncio.to_thread(run_post))
        return {"action": "skip"}

    # 3. Help Intake Processing
    if channel_id == CHANNELS["help"]:
        # Skip checking bot's own posts/replies to avoid loops
        if author.strip().lower() in ["hermes", "hermes_agent", "hermes-agent", "krabot", "кработ"]:
            return None

        # Build issue and metadata
        msg_hash = make_message_hash(raw_text, channel_id, author)
        state = load_intake_state()

        if msg_hash in state:
            # Already created or skipped to avoid duplicate posts
            return {"action": "skip"}

        # Extract fields for triage
        title_lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        issue_title = title_lines[0][:100] if title_lines else "не работает бот в канале #помощь"

        # Guard title length and formatting
        if not issue_title.startswith("помощь:"):
            issue_title = f"помощь: {issue_title}"

        # Create issue inside Dolt DB via bd command
        cmd = [
            "bd", "create",
            "--title", issue_title,
            "--type", "task",
            "--priority", "2",
            "--description", f"сообщение от {author} в канале #помощь:\n\n{raw_text}"
        ]

        try:
            # Execute synchronously in subprocess
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = res.stdout or ""

            # Extract new issue ID
            new_id = "unknown"
            match = re.search(r"Created issue:\s+([a-zA-Z0-9\-\.]+)", output)
            if match:
                new_id = match.group(1)

            # Record state
            state[msg_hash] = {
                "issue_id": new_id,
                "author": author,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content_preview": raw_text[:200]
            }
            save_intake_state(state)

            # 4. Trigger one-way bd -> Vikunja sync
            sync_res = _sync_vikunja()

            # Format user acknowledgement
            ack_msg = f"принял запрос в работу, {author} - создал задачу `{new_id}`"
            if not sync_res.get("ok"):
                logger.warning(f"Vikunja sync failed during help intake: {sync_res.get('error')}")

            await _send_reply(event, active_gateway, run_sanitizer(ack_msg))
            return {"action": "skip"}

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create bd issue during help intake: {e.stderr or e.stdout}")
            # Do not throw, fallback to normal flow or skip
            await _send_reply(event, active_gateway, run_sanitizer("возникли проблемы с созданием тикета - я записал ошибку в логи"))
            return {"action": "skip"}

    return None
