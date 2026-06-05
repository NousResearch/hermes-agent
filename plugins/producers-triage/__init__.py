from __future__ import annotations

import hashlib
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
    runner: GatewayRunner,
) -> dict[str, Any] | None:
    """Fast-path handler for explicit triage commands and intakes.

    Intercepts consent/help/GitDB tool review patterns.
    """
    # Verify we run in producers profile context
    from hermes_constants import get_hermes_home
    if get_hermes_home().name != "producers":
        return None

    raw_text = (event.content or "").strip()
    norm_text = raw_text.lower()
    author = event.author or "unknown"
    channel_id = event.channel_id

    # 1. User Consent Commands
    if "кработ согласие" in norm_text:
        set_user_consent(author, True, channel_id)
        reply = run_sanitizer(f"запомнил, {author} - теперь твои наработки из #аудио-наработки будут попадать в еженедельный дайджест")
        await event.reply(reply)
        return {"action": "skip"}

    if "кработ отказ" in norm_text:
        set_user_consent(author, False, channel_id)
        reply = run_sanitizer(f"понял, {author} - исключил твои наработки из дайджестов")
        await event.reply(reply)
        return {"action": "skip"}

    if "кработ метрики" in norm_text or "кработ рост" in norm_text or "кработ каналы" in norm_text:
        await event.reply(format_growth_metrics_reply())
        return {"action": "skip"}

    # 2. GitDB Tools Review Commands (kra queue/next/approve/reject/post)
    is_admin = author in ["ameobius", "a meobius", "a_meobius"]

    if "кработ очередь" in norm_text or "кработ кандидаты" in norm_text:
        pending = get_pending_candidates()
        if not pending:
            await event.reply("очередь кандидатов пуста")
            return {"action": "skip"}

        reply_lines = [f"в очереди {len(pending)} кандидатов - вот первые 5:"]
        for i, c in enumerate(pending[:5], 1):
            stars = c.get("stars", 0)
            lang = c.get("language", "unknown")
            reply_lines.append(f"{i}- **{c['full_name']}** ({lang}, {stars} stars)")
        reply_lines.append("\nпоказать карточку следующего: `кработ некст`\nзапостить: `кработ запости [число]`")
        await event.reply(run_sanitizer("\n".join(reply_lines)))
        return {"action": "skip"}

    if "кработ некст" in norm_text or "кработ следующий" in norm_text:
        pending = get_pending_candidates()
        if not pending:
            await event.reply("очередь кандидатов пуста")
            return {"action": "skip"}

        next_c = pending[0]
        card = next_c.get("card", "")
        # Present candidate card directly
        await event.reply(run_sanitizer(f"следующий кандидат в очереди:\n\n{card}"))
        return {"action": "skip"}

    if "кработ запости" in norm_text:
        if not is_admin:
            await event.reply("команда доступна только администраторам")
            return {"action": "skip"}

        count = 1
        match = re.search(r"кработ запости\s+(\d+)", norm_text)
        if match:
            count = int(match.group(1))

        pending = get_pending_candidates()
        if not pending:
            await event.reply("нечего постить, очередь пуста")
            return {"action": "skip"}

        await event.reply(run_sanitizer(f"запускаю публикацию кандидатов ({count} шт)-"))

        # Trigger background posting task to not block gateway response
        loop = asyncio.get_running_loop()
        def run_post():
            res = trigger_posting(count)
            # Send notification back
            if res.get("ok"):
                # Run notification safely
                asyncio.run_coroutine_threadsafe(
                    event.reply(run_sanitizer(f"публикация завершена успешно:\n{res['output']}")),
                    loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    event.reply(run_sanitizer(f"ошибка при публикации: {res['error']}")),
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

            await event.reply(run_sanitizer(ack_msg))
            return {"action": "skip"}

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create bd issue during help intake: {e.stderr or e.stdout}")
            # Do not throw, fallback to normal flow or skip
            await event.reply(run_sanitizer("возникли проблемы с созданием тикета - я записал ошибку в логи"))
            return {"action": "skip"}

    return None
