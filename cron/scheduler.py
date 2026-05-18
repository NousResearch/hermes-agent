"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports BEFORE repo-level imports.
# Without this, standalone invocations (e.g. after `hermes update` reloads
# the module) fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from hermes_cli.config import load_config
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Valid delivery platforms — used to validate user-supplied platform names
# in cron delivery targets, preventing env var enumeration via crafted names.
_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "mattermost", "homeassistant", "dingtalk", "feishu",
    "wecom", "sms", "email", "webhook",
})

from cron.jobs import get_due_jobs, mark_job_run, save_job_output, advance_next_run

# Sentinel: when a cron agent has nothing new to report, it can start its
# response with this marker to suppress delivery.  Output is still saved
# locally for audit.
SILENT_MARKER = "[SILENT]"

# ---------------------------------------------------------------------------
# Output quality validation — catches degenerate LLM output before delivery
# ---------------------------------------------------------------------------

def _validate_output_quality(text: str) -> tuple:
    """Check if LLM output is coherent enough to deliver to user.

    Free-tier models occasionally produce garbled text (mixed-script fragments,
    random Unicode, path-like concatenations).  This gate runs *before*
    delivery so bad output never reaches the user.

    Returns ``(True, "ok")`` when the text looks reasonable, or
    ``(False, reason)`` when it should be suppressed.
    """
    if not text or not text.strip():
        return False, "empty"

    s = text.strip()

    # 1. Non-ASCII ratio — coherent responses are mostly ASCII.
    #    Threshold 0.15 allows emoji, light formatting, proper nouns.
    if len(s) > 30:
        non_ascii = sum(1 for c in s if ord(c) > 127)
        ratio = non_ascii / len(s)
        if ratio > 0.15:
            return False, f"non-ASCII ratio {ratio:.0%}"

    # 2. Word density — garbled text lacks word boundaries.
    words = s.split()
    if len(s) > 50 and len(words) < 5:
        return False, f"too few words ({len(words)}) for length {len(s)}"

    # 3. Average word length — garbled concatenations produce very long tokens.
    if len(words) >= 3:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 30:
            return False, f"avg word length {avg_len:.0f}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Briefing anti-pattern guard — Artemis B-0510-01 Phase 3
# ---------------------------------------------------------------------------
#
# Coach's daily-briefing output occasionally leaks chain-of-thought into the
# deliverable channel under input contradiction (e.g. strategy field vs.
# profile mismatch).  Three observed sub-shapes:
#   A   — quiet-day total replacement (reasoning instead of brief)
#   A'  — content-path prefix pollution (reasoning then brief)
#   A'' — template bypass (reasoning-only, no brief emitted)
#
# Prefix-level SKILL.md constraints (F template + Phase 2b paired examples)
# enforce on inputs without competing priority but fail open when the model
# treats input contradiction as a reconciliation sub-task.  This scanner
# runs post-LLM, before delivery, and flags content carrying any of the
# anti-pattern markers SKILL.md § Rules forbids.  On hit, the scheduler
# substitutes a deterministic quiet-day fallback rather than delivering the
# leak.  See docs/plans/investigations/coach-briefing-output-fidelity.md
# (Artemis) § Phase 3 for the failure taxonomy and rationale.

import re as _re

# Phrases that are reasoning markers when they lead a sentence/clause.
# Matched at the start of any line (after optional whitespace / list markers).
#
# Note: bare "looking at" is NOT here — briefing skill documents
# "Looking at <topic>" as a legitimate opener pattern
# (e.g. "Looking at Series B data science openings this morning").
# Reasoning-shaped "looking at the strategy / the user / etc." variants
# live in _BRIEFING_MIDCONTENT_REASONING below where they get caught as
# mid-content fragments instead of false-tripping every topic-leading
# opener. See Artemis S-0511-07 § Architecture for the rationale.
_BRIEFING_LEADING_REASONING = (
    "now let me",
    "let me ",
    "let's ",
    "i'm looking at",
    "i should ",
    "i'll ",
    "i will ",
    "the strategy was last updated",
    "the emotional context shows",
    "the emotional context is",
    "wait —",
    "wait,",
    "wait -",
    "given:",
    "here is the situation",
    "here's the situation",
    "actually,",
    "actually let me",
    "first, let me",
    "scanning for",
    "per strategy direction",
)

# Phrases that are reasoning markers anywhere in the body (mid-content
# template-bypass signatures from the James 5/12 A'' fixture and the
# Garwin 5/2 A' multi-line reasoning prefix).
_BRIEFING_MIDCONTENT_REASONING = (
    "let me check",
    "let me write",
    "let me reconsider",
    "actually, let me",
    "wait — actually",
    "wait, actually",
    "per strategy direction",
    "i'll skip",
    "i should send",
    "i'll acknowledge",
    "scanning for executive-level signals",
    # "looking at <internal-state>" — narrowed from the previous leading
    # marker so legitimate "Looking at <topic>" openers (Artemis briefing
    # skill's documented form) pass layer 1, while these reasoning-shape
    # variants still trip layer 2.
    "looking at the strategy",
    "looking at the user",
    "looking at the user's",
    "looking at the profile",
    "looking at the emotional context",
    "looking at session",
    "looking at mem0",
    "looking at what the user",
    # B-class voice violations: third-person-about-user narration inside
    # briefing output. Added 2026-05-17 after Artemis B-0510-01 Phase 4
    # reopen (Crystal 5/16 13:52 Executor brief + Amy 5/16 16:02 quiet-day
    # briefing). These phrases mark the narrator addressing someone OTHER
    # than the user (a Strategist / Coach-self reader) about the user —
    # specifically incompatible with Coach's second-person voice contract.
    "she requested",
    "he requested",
    "she reaches out",
    "he reaches out",
    "she reaches",
    "he reaches",
    "if she reacts",
    "if he reacts",
    "if they react",
    "are already in your inbox",
    "ready to deploy",
    "loaded and ready",
    "the user is",
    "the user has",
    "the user's profile",
    "the user's strategy",
    # Bare "Wait —" / "Actually," opening a sentence — caught by regex
    # patterns below in addition to substring search.
)

# Sentence-start mid-content patterns (after ".", "!", "?", or newline).
_BRIEFING_SENTENCE_START_PATTERNS = (
    _re.compile(r"(?:^|[.!?\n])\s*Wait\s*[—\-,]", _re.IGNORECASE),
    _re.compile(r"(?:^|[.!?\n])\s*Actually,\s+(?:let me|the|user|user's)", _re.IGNORECASE),
)


def _scan_briefing_anti_patterns(text: str) -> tuple[bool, str]:
    """Scan Coach briefing output for reasoning-leak / template-bypass markers.

    Returns ``(True, "")`` when the output looks like a clean deliverable, or
    ``(False, reason)`` when it carries an anti-pattern marker that SKILL.md
    § Rules forbids.

    Detection layers:
      1. Leading-clause reasoning markers (first non-empty line).
      2. Mid-content reasoning fragments anywhere in the body.
      3. Sentence-start "Wait —" / "Actually, let me" patterns.

    The check is conservative — false negatives degrade to current behavior
    (delivery), false positives degrade to the deterministic fallback.  When
    in doubt, prefer to pass — the upstream ``_validate_output_quality`` gate
    already catches garbled output.
    """
    if not text or not text.strip():
        return True, ""

    # Layer 1: leading clause.
    first_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        # Drop common list / quote markers so "- Let me…" still trips the guard.
        stripped = _re.sub(r"^[-*>\d.)\s]+", "", stripped)
        if stripped:
            first_line = stripped
            break

    first_lc = first_line.lower()
    for marker in _BRIEFING_LEADING_REASONING:
        if first_lc.startswith(marker):
            return False, f"leading reasoning marker: {marker!r}"

    body_lc = text.lower()

    # Layer 2: mid-content reasoning fragments.
    for marker in _BRIEFING_MIDCONTENT_REASONING:
        if marker in body_lc:
            return False, f"mid-content reasoning fragment: {marker!r}"

    # Layer 3: sentence-start regex patterns.
    for pat in _BRIEFING_SENTENCE_START_PATTERNS:
        if pat.search(text):
            return False, f"sentence-start reasoning pattern: {pat.pattern!r}"

    return True, ""


def _quiet_day_fallback() -> str:
    """Deterministic quiet-day deliverable used when the anti-pattern guard
    suppresses Coach's output.

    Kept intentionally short, second-person, and emotionally neutral — the
    fallback runs precisely when the LLM's own briefing was unusable, so
    the message must not depend on any state the LLM was reasoning about.
    """
    return (
        "Nothing urgent on the board today. I'll keep scanning in the "
        "background and check back tomorrow. Reply any time if something "
        "shifts."
    )


# ---------------------------------------------------------------------------
# Artemis B-0510-01 Phase 4b — post-LLM semantic voice-scan.
# ---------------------------------------------------------------------------
#
# Phase 3's _scan_briefing_anti_patterns relies on literal substring markers,
# which miss third-person voice violations using recipient first names or
# paraphrased pronoun structures (Amy "if Amy responds", Crystal "her CS + SWE
# positioning"). Phase 4b adds a semantic check: a small LLM call to
# google/gemini-3-flash-preview (configurable) judges whether the briefing
# addresses its reader in second person or narrates about them in third
# person. Fail-open on any error — the scan is an enforcement layer, not the
# primary defense.
#
# Bench evidence: logs/bench/voice-scan-2026-05-18.md in the Artemis repo —
# gemini-3-flash-preview hit 7/7 on the calibration fixture set at ~1.7s
# median latency and ~$0.0003 per call.

_VOICE_SCAN_PROMPT = """You are auditing a Coach's daily briefing for a single user before delivery.

The briefing must be in SECOND PERSON ("you / your") when referring to the recipient.

A voice violation is when the briefing uses THIRD PERSON to refer to the recipient — either by name (e.g. "if Amy responds" when Amy IS the recipient) or by pronoun ("she / he / her / his / they") when the pronoun refers to the recipient.

Third-party names (events, companies, other people) are NOT violations even if they are proper nouns. Only the recipient being named/pronouned in third person counts.

Examples of violations (recipient is the named person):
- "if Amy responds" (recipient=Amy)
- "Crystal's positioning" (recipient=Crystal)
- "she reaches out" (referring to the recipient)
- "Maggie's bandwidth is the blocker" (recipient=Maggie)

Examples of OK content:
- "let me know if you're going" (second person — recipient addressed directly)
- "Women in Tech SF on 5/21" (Women in Tech SF is an event, not the recipient)
- "AIET 2026 in Zagreb" (event name)
- "Andiamo role" (company name)

Briefing content:
<<<
{text}
>>>

Respond with strict JSON only. No prose, no markdown fences, just JSON:
{{"verdict": "PASS" or "FAIL", "offending_phrases": ["..."]}}"""


def _voice_scan_log_path() -> Path:
    return get_hermes_home() / "logs" / "voice_scan.log"


def _voice_scan_log(level: str, job_id: str, msg: str) -> None:
    """Append a one-line log entry to voice_scan.log. Self-swallowing — voice
    scan must never break the cron tick."""
    try:
        path = _voice_scan_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts} {level} job={job_id} {msg}\n")
    except Exception:
        pass


def _voice_scan_check(text: str, job_id: str = "?") -> tuple[bool, str]:
    """Semantic voice-scan via OpenRouter LLM call.

    Returns ``(True, "")`` for PASS / inconclusive / any error (fail-open), or
    ``(False, reason)`` only when the model returns a confident FAIL verdict.

    Env knobs:
      - VOICE_SCAN_MODEL — OpenRouter model id (default
        ``google/gemini-3-flash-preview``).
      - VOICE_SCAN_ENABLED — set to "0" to disable the scan entirely.
      - OPENROUTER_API_KEY — required; missing key → fail-open + WARN log.

    No new dependencies — uses stdlib urllib so the scan works on every cron
    environment without venv changes.
    """
    if os.getenv("VOICE_SCAN_ENABLED", "1") != "1":
        return True, ""
    if not text or not text.strip():
        return True, ""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        _voice_scan_log("WARN", job_id, "no OPENROUTER_API_KEY — fail-open")
        return True, ""

    model = os.getenv("VOICE_SCAN_MODEL", "google/gemini-3-flash-preview")
    prompt = _VOICE_SCAN_PROMPT.format(text=text)

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 400,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elmtree-askmo/artemis",
            "X-Title": "Artemis voice-scan",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        _voice_scan_log("WARN", job_id, f"HTTP/parse error — fail-open: {exc!r}")
        return True, ""

    try:
        content = payload["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("content is None")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _voice_scan_log("WARN", job_id, f"no content in response — fail-open: {exc!r}")
        return True, ""

    raw = content.strip()
    # Tolerate fenced JSON.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        verdict_obj = json.loads(raw)
    except json.JSONDecodeError:
        _voice_scan_log("WARN", job_id, f"non-JSON model output — fail-open. raw={raw[:200]!r}")
        return True, ""

    verdict = str(verdict_obj.get("verdict", "")).upper()
    if verdict == "FAIL":
        offending = verdict_obj.get("offending_phrases") or []
        reason = f"voice-scan FAIL ({model}): {offending}"
        _voice_scan_log("HIT", job_id, reason)
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# Artemis S-0511-07 — briefing-output persistence (scheduler-side write).
# ---------------------------------------------------------------------------
#
# After _deliver_result succeeds for an artemis-briefing job, persist the
# exact delivered text (LLM draft OR guard-substituted fallback) so Artemis
# Coach can replay sections of the briefing on S2 ask without re-running the
# cron. The file shape matches the artemis MCP tool `save_briefing_output`'s
# write format — both writers share the storage layout, both readers (Coach
# S2 via MCP `get_recent_briefings`, retro analysis scripts) consume the
# same shape.
#
# Why scheduler-side and not LLM-side: the cron LLM has one final response
# that IS the deliverable, no second turn for a post-delivery tool call.
# The scheduler is also the only party that knows what was actually
# delivered (LLM draft vs. guard fallback). See Artemis S-0511-07 spec
# § Architecture for the full rationale.
#
# Failure handling: try/except swallows any persistence error, logs at WARN,
# and falls through to mark_job_run unchanged. A missing persisted entry
# means the next day's run produces a fresh one; the user has already
# received the briefing.

def _is_briefing_job(job: dict) -> bool:
    """True for artemis-briefing cron jobs (the only writer of the
    persisted-briefing store today). Skills list is the canonical marker;
    job names can drift."""
    skills = job.get("skills") or []
    if not isinstance(skills, (list, tuple)):
        return False
    return "artemis-briefing" in skills


def _persist_briefing_output(job: dict, delivered_text: str) -> None:
    """Write the delivered briefing text to the artemis per-user briefings
    directory. No-op + log on any failure — never raises.

    Schema is the three-field minimal entry shared with the artemis MCP
    tool `save_briefing_output`:
      - user_id: from origin.user_id (set by _resolve_origin)
      - briefing_timestamp: scheduler-side UTC at the moment of delivery
      - formatted_output: exact text sent to Slack (LLM draft OR fallback)
    """
    try:
        origin = _resolve_origin(job)
        if not origin:
            return
        user_id = origin.get("user_id")
        if not user_id:
            return
        if not isinstance(delivered_text, str) or not delivered_text.strip():
            return

        hermes_home_dyn = get_hermes_home()
        out_dir = hermes_home_dyn / "artemis" / user_id / "briefings"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_now = datetime.now(timezone.utc)
        ts_iso = ts_now.strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "user_id": user_id,
            "briefing_timestamp": ts_iso,
            "formatted_output": delivered_text,
        }
        out_path = out_dir / (ts_iso.replace(":", "-") + ".json")
        out_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(
            "briefing-persist failed for job %s (user=%s): %s",
            job.get("id", "?"),
            (job.get("origin") or {}).get("user_id", "?"),
            exc,
        )


# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = get_hermes_home()

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = _hermes_home / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Post-S-0429-01 G1, origin should carry ``user_id`` for cron-spawned
    sessions to bind ``HERMES_SESSION_USER_ID`` correctly. Legacy jobs
    persisted before the schema change won't have it; reverse-resolve via
    each artemis user's ``slack_channel.txt`` sidecar so they keep working
    without a one-shot migration.
    """
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if not (platform and chat_id):
        return None

    if not origin.get("user_id"):
        # Re-read HERMES_HOME at call time rather than relying on the
        # module-level ``_hermes_home`` snapshot — tests override it via
        # monkeypatch, and dev rigs may legitimately move the data dir
        # between process starts.
        hermes_home_dyn = get_hermes_home()
        artemis_root = hermes_home_dyn / "artemis"
        if artemis_root.exists():
            chat_id_str = str(chat_id)
            for user_dir in artemis_root.iterdir():
                if not user_dir.is_dir():
                    continue
                ch_file = user_dir / "slack_channel.txt"
                try:
                    if ch_file.exists() and ch_file.read_text().strip() == chat_id_str:
                        # Build a copy so we don't mutate the on-disk job dict.
                        origin = {**origin, "user_id": user_dir.name}
                        break
                except OSError:
                    continue

        # B-0504-01 followup #1: if reverse-resolve still didn't find a user_id,
        # log loudly. Without user_id, scheduler skips env+ContextVar injection
        # (~line 640) and the cron's MCP calls fail-closed with the generic
        # "no user_id available" error. Silent failure here was the symptom
        # mode of the 2026-05-05 james/crystal incident (sidecar files missing).
        # Post-A1+A2 (mcp-server `91d8f69`+`be62c81` + 2026-05-05 cron-data
        # backfill), all known cron entries carry user_id natively and this
        # path is essentially defensive — but if a regression ever lands a
        # cron without user_id and without a matching sidecar, this log makes
        # the failure mode visible instead of silent.
        if not origin.get("user_id"):
            logger.error(
                "cron job %r: origin lacks user_id and no slack_channel.txt "
                "matches chat_id=%s — cron will fail-closed at MCP layer "
                "(B-0504-01 family). Inspect ~/.hermes/cron/jobs.json + "
                "~/.hermes/artemis/<user>/slack_channel.txt.",
                job.get("id", "?"),
                chat_id,
            )
    return origin


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return None

    if deliver == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": origin.get("thread_id"),
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in ("matrix", "telegram", "discord", "slack"):
            chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": None,
                }
        return None

    if ":" in deliver:
        platform_name, rest = deliver.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import _parse_target_ref

        parsed_chat_id, parsed_thread_id, is_explicit = _parse_target_ref(platform_key, rest)
        if is_explicit:
            chat_id, thread_id = parsed_chat_id, parsed_thread_id
        else:
            chat_id, thread_id = rest, None

        # Resolve human-friendly labels like "Alice (dm)" to real IDs.
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_key, chat_id)
            if resolved:
                parsed_chat_id, parsed_thread_id, resolved_is_explicit = _parse_target_ref(platform_key, resolved)
                if resolved_is_explicit:
                    chat_id, thread_id = parsed_chat_id, parsed_thread_id
                else:
                    chat_id = resolved
        except Exception:
            pass

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

    platform_name = deliver
    if origin and origin.get("platform") == platform_name:
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if platform_name.lower() not in _KNOWN_DELIVERY_PLATFORMS:
        return None
    chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": None,
    }


# Media extension sets — keep in sync with gateway/platforms/base.py:_process_message_background
_AUDIO_EXTS = frozenset({'.ogg', '.opus', '.mp3', '.wav', '.m4a'})
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(adapter, chat_id: str, media_files: list, metadata: dict | None, loop, job: dict) -> None:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.
    """
    from pathlib import Path

    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if ext in _AUDIO_EXTS:
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            result = future.result(timeout=30)
            if result and not getattr(result, "success", True):
                logger.warning(
                    "Job '%s': media send failed for %s: %s",
                    job.get("id", "?"), media_path, getattr(result, "error", "unknown"),
                )
        except Exception as e:
            logger.warning("Job '%s': failed to send media %s: %s", job.get("id", "?"), media_path, e)


def _deliver_result(job: dict, content: str, adapters=None, loop=None) -> Optional[str]:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    Returns None on success, or an error string on failure.
    """
    target = _resolve_delivery_target(job)
    if not target:
        if job.get("deliver", "local") != "local":
            msg = f"no delivery target resolved for deliver={job.get('deliver', 'local')}"
            logger.warning("Job '%s': %s", job["id"], msg)
            return msg
        return None  # local-only jobs don't deliver — not a failure

    platform_name = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
        "signal": Platform.SIGNAL,
        "matrix": Platform.MATRIX,
        "mattermost": Platform.MATTERMOST,
        "homeassistant": Platform.HOMEASSISTANT,
        "dingtalk": Platform.DINGTALK,
        "feishu": Platform.FEISHU,
        "wecom": Platform.WECOM,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        msg = f"unknown platform '{platform_name}'"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        msg = f"platform '{platform_name}' not configured/enabled"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"Note: The agent cannot see this message, and therefore cannot respond to it."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import BasePlatformAdapter
    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)

    # Prefer the live adapter when the gateway is running — this supports E2EE
    # rooms (e.g. Matrix) where the standalone HTTP path cannot encrypt.
    runtime_adapter = (adapters or {}).get(platform)
    if runtime_adapter is not None and loop is not None and getattr(loop, "is_running", lambda: False)():
        send_metadata = {"thread_id": thread_id} if thread_id else None
        try:
            # Send cleaned text (MEDIA tags stripped) — not the raw content
            text_to_send = cleaned_delivery_content.strip()
            adapter_ok = True
            if text_to_send:
                future = asyncio.run_coroutine_threadsafe(
                    runtime_adapter.send(chat_id, text_to_send, metadata=send_metadata),
                    loop,
                )
                send_result = future.result(timeout=60)
                if send_result and not getattr(send_result, "success", True):
                    err = getattr(send_result, "error", "unknown")
                    logger.warning(
                        "Job '%s': live adapter send to %s:%s failed (%s), falling back to standalone",
                        job["id"], platform_name, chat_id, err,
                    )
                    adapter_ok = False  # fall through to standalone path

            # Send extracted media files as native attachments via the live adapter
            if adapter_ok and media_files:
                _send_media_via_adapter(runtime_adapter, chat_id, media_files, send_metadata, loop, job)

            if adapter_ok:
                logger.info("Job '%s': delivered to %s:%s via live adapter", job["id"], platform_name, chat_id)
                return None
        except Exception as e:
            logger.warning(
                "Job '%s': live adapter delivery to %s:%s failed (%s), falling back to standalone",
                job["id"], platform_name, chat_id, e,
            )

    # Standalone path: run the async send in a fresh event loop (safe from any thread)
    coro = _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files)
    try:
        result = asyncio.run(coro)
    except RuntimeError:
        # asyncio.run() checks for a running loop before awaiting the coroutine;
        # when it raises, the original coro was never started — close it to
        # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
        # fresh thread that has no running loop.
        coro.close()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files))
            result = future.result(timeout=30)
    except Exception as e:
        msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    if result and result.get("error"):
        msg = f"delivery error: {result['error']}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
    return None


_SCRIPT_TIMEOUT = 120  # seconds


def _run_job_script(script_path: str) -> tuple[bool, str]:
    """Execute a cron job's data-collection script and capture its output.

    Scripts must reside within HERMES_HOME/scripts/.  Both relative and
    absolute paths are resolved and validated against this directory to
    prevent arbitrary script execution via path traversal or absolute
    path injection.

    Args:
        script_path: Path to a Python script.  Relative paths are resolved
            against HERMES_HOME/scripts/.  Absolute and ~-prefixed paths
            are also validated to ensure they stay within the scripts dir.

    Returns:
        (success, output) — on failure *output* contains the error message so the
        LLM can report the problem to the user.
    """
    from hermes_constants import get_hermes_home

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    # Guard against path traversal, absolute path injection, and symlink
    # escape — scripts MUST reside within HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=str(path.parent),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        # Redact any secrets that may appear in script output before
        # they are injected into the LLM prompt context.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
        except Exception:
            pass
        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {_SCRIPT_TIMEOUT}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _build_job_prompt(job: dict) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first."""
    prompt = job.get("prompt", "")
    skills = job.get("skills")

    # Run data-collection script if configured, inject output as context.
    script_path = job.get("script")
    if script_path:
        success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
            else:
                prompt = (
                    "[Script ran successfully but produced no output.]\n\n"
                    f"{prompt}"
                )
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )

    # Always prepend cron execution guidance so the agent knows how
    # delivery works and can suppress delivery when appropriate.
    cron_hint = (
        "[SYSTEM: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []

    skill_names = [str(name).strip() for name in skills if str(name).strip()]

    # Onboarding footer for daily briefing: append a one-line opt-out reminder
    # to the first 3 briefing runs so new users discover the pause path.
    # Gated on skill=artemis-briefing to avoid leaking into other cron jobs.
    footer_line = ""
    if "artemis-briefing" in skill_names:
        completed = (job.get("repeat") or {}).get("completed", 0) or 0
        if completed < 3:
            footer_line = (
                "FOOTER_REQUIRED: append "
                "_(daily briefing — say \"pause\" anytime to stop)_ "
                "as the absolute last line."
            )

    if not skill_names:
        if footer_line:
            prompt = f"{prompt}\n\n{footer_line}"
        return prompt

    from tools.skills_tool import skill_view

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        loaded = json.loads(skill_view(skill_name))
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[SYSTEM: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])
    if footer_line:
        parts.extend(["", footer_line])
    return "\n".join(parts)


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    # Initialize SQLite session store so cron job messages are persisted
    # and discoverable via session_search (same pattern as gateway/run.py).
    _session_db = None
    try:
        from hermes_state import SessionDB
        _session_db = SessionDB()
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = _build_job_prompt(job)
    origin = _resolve_origin(job)
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    try:
        # Inject origin context so the agent's send_message tool knows the chat.
        # Must be INSIDE the try block so the finally cleanup always runs.
        if origin:
            os.environ["HERMES_SESSION_PLATFORM"] = origin["platform"]
            os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
            if origin.get("chat_name"):
                os.environ["HERMES_SESSION_CHAT_NAME"] = origin["chat_name"]
            if origin.get("user_id"):
                os.environ["HERMES_SESSION_USER_ID"] = origin["user_id"]
            # G1 (S-0429-01): also seed the asyncio-task ContextVars so
            # any MCP subprocess spawn downstream of this job (via
            # ``_run_stdio``) materializes ``HERMES_SESSION_USER_ID``
            # from the ContextVar — keeping the env-injection contract
            # consistent across gateway-driven and cron-driven sessions.
            from tools.session_context import set_session as _ctx_set_session
            _ctx_set_session(
                platform=origin["platform"],
                chat_id=str(origin["chat_id"]),
                chat_name=origin.get("chat_name"),
                user_id=origin.get("user_id"),
            )
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        from dotenv import load_dotenv
        try:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="latin-1")

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            os.environ["HERMES_CRON_AUTO_DELIVER_PLATFORM"] = delivery_target["platform"]
            os.environ["HERMES_CRON_AUTO_DELIVER_CHAT_ID"] = str(delivery_target["chat_id"])
            if delivery_target.get("thread_id") is not None:
                os.environ["HERMES_CRON_AUTO_DELIVER_THREAD_ID"] = str(delivery_target["thread_id"])

        model = job.get("model") or os.getenv("HERMES_MODEL") or ""

        # Load config.yaml for model, reasoning, prefill, toolsets, provider routing
        _cfg = {}
        try:
            import yaml
            _cfg_path = str(_hermes_home / "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                _model_cfg = _cfg.get("model", {})
                if not job.get("model"):
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        model = _model_cfg.get("default", model)
        except Exception as e:
            logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

        # Reasoning config from config.yaml
        from hermes_constants import parse_reasoning_effort
        effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)

        # Prefill messages from env or config.yaml
        prefill_messages = None
        prefill_file = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "") or _cfg.get("prefill_messages_file", "")
        if prefill_file:
            import json as _json
            pfpath = Path(prefill_file).expanduser()
            if not pfpath.is_absolute():
                pfpath = _hermes_home / pfpath
            if pfpath.exists():
                try:
                    with open(pfpath, "r", encoding="utf-8") as _pf:
                        prefill_messages = _json.load(_pf)
                    if not isinstance(prefill_messages, list):
                        prefill_messages = None
                except Exception as e:
                    logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
                    prefill_messages = None

        # Max iterations
        max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90

        # Provider routing
        pr = _cfg.get("provider_routing", {})
        smart_routing = _cfg.get("smart_model_routing", {}) or {}

        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )
        try:
            runtime_kwargs = {
                "requested": job.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
            }
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            raise RuntimeError(message) from exc

        from agent.smart_model_routing import resolve_turn_route
        turn_route = resolve_turn_route(
            prompt,
            smart_routing,
            {
                "model": model,
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            },
        )

        agent = AIAgent(
            model=turn_route["model"],
            api_key=turn_route["runtime"].get("api_key"),
            base_url=turn_route["runtime"].get("base_url"),
            provider=turn_route["runtime"].get("provider"),
            api_mode=turn_route["runtime"].get("api_mode"),
            acp_command=turn_route["runtime"].get("command"),
            acp_args=turn_route["runtime"].get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            quiet_mode=True,
            skip_memory=True,  # Cron system prompts would corrupt user representations
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
        )
        
        # Run the agent with an *inactivity*-based timeout: the job can run
        # for hours if it's actively calling tools / receiving stream tokens,
        # but a hung API call or stuck tool with no activity for the configured
        # duration is caught and killed.  Default 600s (10 min inactivity);
        # override via HERMES_CRON_TIMEOUT env var.  0 = unlimited.
        #
        # Uses the agent's built-in activity tracker (updated by
        # _touch_activity() on every tool call, API call, and stream delta).
        _cron_timeout = float(os.getenv("HERMES_CRON_TIMEOUT", 600))
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        _POLL_INTERVAL = 5.0
        _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _cron_future = _cron_pool.submit(agent.run_conversation, prompt)
        _inactivity_timeout = False
        try:
            if _cron_inactivity_limit is None:
                # Unlimited — just wait for the result.
                result = _cron_future.result()
            else:
                result = None
                while True:
                    done, _ = concurrent.futures.wait(
                        {_cron_future}, timeout=_POLL_INTERVAL,
                    )
                    if done:
                        result = _cron_future.result()
                        break
                    # Agent still running — check inactivity.
                    _idle_secs = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            _act = agent.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    if _idle_secs >= _cron_inactivity_limit:
                        _inactivity_timeout = True
                        break
        except Exception:
            _cron_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            _cron_pool.shutdown(wait=False)

        if _inactivity_timeout:
            # Build diagnostic summary from the agent's activity tracker.
            _activity = {}
            if hasattr(agent, "get_activity_summary"):
                try:
                    _activity = agent.get_activity_summary()
                except Exception:
                    pass
            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                job_name, _secs_ago, _cron_inactivity_limit,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )
            if hasattr(agent, "interrupt"):
                agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(
                f"Cron job '{job_name}' idle for "
                f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
                f"— last activity: {_last_desc}"
            )

        final_response = result.get("final_response", "") or ""
        # Use a separate variable for log display; keep final_response clean
        # for delivery logic (empty response = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{logged_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}
```
"""
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_USER_ID",
            "HERMES_CRON_AUTO_DELIVER_PLATFORM",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
        ):
            os.environ.pop(key, None)
        # Clear session ContextVars (G1) — same reason as env cleanup.
        try:
            from tools.session_context import clear_session as _ctx_clear_session
            _ctx_clear_session()
        except Exception:
            pass
        if _session_db:
            try:
                _session_db.end_session(_cron_session_id, "cron_complete")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to end session: %s", job_id, e)
            try:
                _session_db.close()
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def tick(verbose: bool = True, adapters=None, loop=None) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
        adapters: Optional dict mapping Platform → live adapter (from gateway)
        loop: Optional asyncio event loop (from gateway) for live adapter sends
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    lock_fd = None
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        if lock_fd is not None:
            lock_fd.close()
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                # For recurring jobs (cron/interval), advance next_run_at to the
                # next future occurrence BEFORE execution.  This way, if the
                # process crashes mid-run, the job won't re-fire on restart.
                # One-shot jobs are left alone so they can retry on restart.
                advance_next_run(job["id"])

                success, output, final_response, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # Deliver the final response to the origin/target chat.
                # If the agent responded with [SILENT], skip delivery (but
                # output is already saved above).  Failed jobs always deliver.
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                should_deliver = bool(deliver_content)
                if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
                    logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
                    should_deliver = False

                # Pre-delivery quality gate — suppress garbled/degenerate output
                if should_deliver and success:
                    valid, reason = _validate_output_quality(deliver_content)
                    if not valid:
                        logger.warning(
                            "Job '%s': output failed quality check (%s) — skipping delivery. "
                            "First 200 chars: %s",
                            job["id"], reason, deliver_content[:200],
                        )
                        should_deliver = False

                # Artemis B-0510-01 Phase 3 — briefing anti-pattern guard.
                # Catches reasoning-leak / template-bypass output that
                # SKILL.md prefix constraints fail to prevent under input
                # contradiction.  On hit, substitute a deterministic
                # quiet-day fallback so the user receives a coherent
                # deliverable instead of chain-of-thought.
                if should_deliver and success:
                    clean, ap_reason = _scan_briefing_anti_patterns(deliver_content)
                    if not clean:
                        logger.warning(
                            "Job '%s': output tripped briefing anti-pattern guard (%s) — "
                            "substituting deterministic quiet-day fallback. "
                            "Original first 200 chars: %s",
                            job["id"], ap_reason, deliver_content[:200],
                        )
                        deliver_content = _quiet_day_fallback()

                # Artemis B-0510-01 Phase 4b — semantic voice-scan.
                # Catches B-class voice violations (third-person narration
                # about the recipient by name or pronoun) that Phase 3's
                # literal-marker guard misses. Briefing-only, fail-open on
                # any LLM/HTTP/parse error.
                if should_deliver and success and _is_briefing_job(job):
                    vs_clean, vs_reason = _voice_scan_check(deliver_content, job["id"])
                    if not vs_clean:
                        logger.warning(
                            "Job '%s': output tripped voice-scan (%s) — "
                            "substituting deterministic quiet-day fallback. "
                            "Original first 200 chars: %s",
                            job["id"], vs_reason, deliver_content[:200],
                        )
                        deliver_content = _quiet_day_fallback()

                delivery_error = None
                if should_deliver:
                    try:
                        delivery_error = _deliver_result(job, deliver_content, adapters=adapters, loop=loop)
                    except Exception as de:
                        delivery_error = str(de)
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                # Artemis S-0511-07 — persist the exact delivered text for
                # briefing jobs so Coach S2 can replay sections later. Only
                # fires on successful delivery; helper is self-swallowing on
                # any failure (logs WARN, never raises). Runs BEFORE
                # mark_job_run so a persistence raise still wouldn't get here,
                # but the try/except inside the helper guarantees it.
                if should_deliver and delivery_error is None and _is_briefing_job(job):
                    _persist_briefing_output(job, deliver_content)

                mark_job_run(job["id"], success, error, delivery_error=delivery_error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
