"""
skill-sleep — Stage 1: MINE

Scan recent Hermes sessions for friction signals (user corrections, tool errors,
retry patterns) and output structured task cards for the optimizer.

Uses `hermes sessions export --after <time> --format jsonl --redact` — stable
CLI API, not raw SQLite. Always runs with --redact for transcript safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Allow `python3 pipeline/mine.py` direct execution and `python3 -m pipeline.mine`
try:
    from lib.task_card import TaskCard  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.task_card import TaskCard  # type: ignore

# ── Friction signal keywords ────────────────────────────────────────────────

CORRECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"不对",
        r"不是[这的]",
        r"错了",
        r"错[了误]",
        r"重[来做新]",
        r"重新",
        r"改[一下正]",
        r"修复",
        r"wrong",
        r"incorrect",
        r"not what",
        r"try again",
        r"redo",
        r"fix[\s:]",
        r"that['\u2019]s not",
        r"that['\u2019]s wrong",
        r"don['\u2019]t do that",
        r"stop[\s,\.!]",
        r"never mind",
        r"算了",
        r"当我没说",
        r"忽略",
        r"\bno\b",
    ]
]

TOOL_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"exit_code.*[1-9]",
        r"\b error \b",
        r"traceback",
        r"exception",
        r"\bfail",
        r"timeout",
        r"not found",
        r"permission denied",
        r"connection refused",
        r"exit status",
        r"non-zero",
    ]
]

# ── Session export ──────────────────────────────────────────────────────────


def export_sessions(
    after: str,
    *,
    redact: bool = True,
    timeout: int = 60,
) -> list[dict]:
    """Export sessions via `hermes sessions export` and parse JSONL."""
    cmd = [
        "hermes",
        "sessions",
        "export",
        "--after",
        after,
        "--format",
        "jsonl",
    ]
    if redact:
        cmd.append("--redact")
    cmd.append("-")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HERMES_NO_COLOR": "1"},
        )
    except FileNotFoundError:
        print("ERROR: 'hermes' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"WARN: hermes export timed out after {timeout}s", file=sys.stderr)
        return []

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"WARN: hermes export exited {proc.returncode}: {stderr}", file=sys.stderr)
        return []

    sessions: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sessions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: skipping malformed JSONL line: {e}", file=sys.stderr)
            continue
    return sessions


# ── Friction detection ──────────────────────────────────────────────────────


def _is_correction(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    # search full text; for long messages limit to first 300 chars to reduce FP
    haystack = lowered if len(text) < 500 else lowered[:300]
    for pat in CORRECTION_PATTERNS:
        if pat.search(haystack):
            return True
    return False


def _has_tool_error(msg: dict) -> bool:
    content = (msg.get("content") or "")
    effect = (msg.get("effect_disposition") or "")
    combined = f"{content} {effect}".lower()
    # explicit exit_code check
    raw = (msg.get("content") or "")
    if isinstance(raw, str):
        # JSON tool result may contain exit_code field
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and int(parsed.get("exit_code", 0)) != 0:
                return True
        except Exception:
            pass
    for pat in TOOL_ERROR_PATTERNS:
        if pat.search(combined):
            return True
    return False


def _extract_skill_names_from_system_prompt(system_prompt: str) -> list[str]:
    """Extract skill names from system_prompt without logging raw content."""
    if not system_prompt:
        return []
    names: list[str] = []
    # pattern: skill_view(name='...') or skill_view(name="...")
    for m in re.finditer(r"skill_view\s*\(\s*name\s*=\s*['\"]([^'\"]+)['\"]", system_prompt):
        n = m.group(1).strip()
        if n and n not in names:
            names.append(n)
    # also match bare skill identifiers like "hermes-agent" in skills listing
    # only keep names that look like skill slugs (alphanum + hyphen/underscore)
    return names


def _extract_skill_names_from_messages(messages: list[dict]) -> list[str]:
    """Extract skill names from messages/tool_calls without logging raw content."""
    names: list[str] = []
    for msg in messages or []:
        # tool_calls: skill_view etc
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            fname = fn.get("name", "") or tc.get("name", "")
            if fname in ("skill_view", "skill_manage"):
                try:
                    args_raw = fn.get("arguments", "") or ""
                    args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw.strip().startswith("{") else {}
                    n = str(args.get("name") or "").strip()
                    if n and n not in names:
                        names.append(n)
                except Exception:
                    pass
        # tool result content may be JSON with skill name
        content = msg.get("content") or ""
        if isinstance(content, str) and '"name"' in content and "skill" in content.lower():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    n = str(parsed.get("name") or "").strip()
                    if n and re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", n) and n not in names:
                        # only accept if it looks like a skill name and came from skill tool
                        tool_name = msg.get("tool_name") or ""
                        if tool_name in ("skill_view", "skill_manage"):
                            names.append(n)
            except Exception:
                pass
        # slash command in user message: /skill or /skill_view
        if msg.get("role") == "user" and isinstance(content, str):
            m = re.search(r"/skill[\s_]+(\S+)", content, re.IGNORECASE)
            if m:
                n = m.group(1).strip().strip(",.;:")
                if n and n not in names:
                    names.append(n)
    return names


def _get_skill_name(session: dict) -> str:
    # 1) explicit skills_used field if present (future hermes export)
    skills_used = session.get("skills_used")
    if isinstance(skills_used, list) and skills_used:
        first = str(skills_used[0]).strip()
        if first:
            return first
    if isinstance(skills_used, str) and skills_used.strip():
        return skills_used.strip()

    # 2) system_prompt: extract skill_view(name='...') mentions (privacy: never log raw)
    sp = session.get("system_prompt") or ""
    if isinstance(sp, str) and sp:
        sp_skills = _extract_skill_names_from_system_prompt(sp)
        if sp_skills:
            return sp_skills[0]

    # 3) messages: skill_view tool calls / tool results / /skill commands
    msgs = session.get("messages") or []
    if msgs:
        msg_skills = _extract_skill_names_from_messages(msgs)
        if msg_skills:
            return msg_skills[0]

    # 4) fallback: existing heuristics (title / cwd)
    title = session.get("title") or ""
    cwd = session.get("cwd") or ""
    m = re.search(r"skill[:\s]+(\S+)", title, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(",.;:")
    if cwd:
        return Path(cwd).name
    return "default"


def _extract_tool_calls(messages: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for msg in messages:
        tcs = msg.get("tool_calls")
        if not tcs:
            continue
        for tc in tcs:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append(
                {
                    "name": fn.get("name", "?"),
                    "arguments": (fn.get("arguments", "") or "")[:200],
                }
            )
    return calls


def detect_friction(session: dict) -> list[TaskCard]:
    """Scan a single session for friction episodes (0 or 1 card)."""
    messages: list[dict] = session.get("messages") or []
    if not messages:
        return []

    skill_name = _get_skill_name(session)
    session_id = str(session.get("id") or "?")
    started_at = session.get("started_at") or session.get("timestamp") or 0
    try:
        timestamp = float(started_at)
    except Exception:
        timestamp = 0.0

    evidence: list[str] = []
    tool_calls = _extract_tool_calls(messages)

    last_user_head = ""
    retry_count = 0
    last_tool_name = ""

    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()

        if role == "user":
            if _is_correction(content):
                evidence.append(f"user_correction: {content[:120]}")
            # retry: same head repeated
            head = content[:60]
            if last_user_head and head and head == last_user_head:
                retry_count += 1
                if retry_count >= 1:
                    evidence.append(f"retry_{retry_count + 1}: same request repeated")
            else:
                retry_count = 0
            last_user_head = head

        elif role == "tool":
            tool_name = (msg.get("tool_name") or "").strip()
            if _has_tool_error(msg):
                snippet = (msg.get("content") or "")[:120].replace("\n", " ")
                evidence.append(f"tool_error: {tool_name or '?'} — {snippet}")
            if tool_name and tool_name == last_tool_name and _has_tool_error(msg):
                evidence.append(f"tool_retry: {tool_name} errored multiple times")
            if tool_name:
                last_tool_name = tool_name

        elif role == "assistant":
            low = content.lower()
            if content and any(
                kw in low for kw in ["i couldn't", "i'm sorry", "something went wrong", "failed to"]
            ):
                # count as error signal
                if "error" in low or "couldn't" in low or "sorry" in low:
                    evidence.append(f"assistant_error: {content[:120]}")

    if not evidence:
        return []

    first_user_msg = ""
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            first_user_msg = (m.get("content") or "")[:500]
            break

    return [
        TaskCard(
            skill_name=skill_name,
            session_id=session_id,
            user_request=first_user_msg,
            friction_evidence=evidence,
            tool_calls=tool_calls,
            timestamp=timestamp,
        )
    ]


# ── Deduplication & Seen Tracking ──────────────────────────────────────────

DEFAULT_SEEN_FILE = str(Path.home() / ".hermes" / "skill-sleep-seen.json")


def compute_fingerprint(session_id: str, friction_evidence: list[str]) -> str:
    """Compute deterministic fingerprint for a task card: session_id + evidence sha256 first 12 chars."""
    raw = "\n".join(str(e) for e in friction_evidence)
    ev_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{session_id}:{ev_hash}"


def load_seen_fingerprints(seen_file: str | Path) -> set[str]:
    """Load set of seen fingerprints from JSON file."""
    p = Path(seen_file).expanduser()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(str(x) for x in data)
        if isinstance(data, dict):
            fps = data.get("fingerprints", [])
            if isinstance(fps, list):
                return set(str(x) for x in fps)
            if isinstance(fps, dict):
                return set(str(x) for x in fps.keys())
        return set()
    except Exception as e:
        print(f"[mine] WARN: could not read seen file {p}: {e}", file=sys.stderr)
        return set()


def save_seen_fingerprints(seen_file: str | Path, fingerprints: set[str]) -> None:
    """Save set of seen fingerprints to JSON file."""
    p = Path(seen_file).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_seen": len(fingerprints),
        "fingerprints": sorted(list(fingerprints)),
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def filter_seen_cards(
    cards: list[TaskCard],
    seen_fps: set[str],
) -> tuple[list[TaskCard], list[TaskCard]]:
    """Split cards into (fresh_cards, seen_cards) based on fingerprint."""
    fresh: list[TaskCard] = []
    seen: list[TaskCard] = []
    for card in cards:
        fp = compute_fingerprint(card.session_id, card.friction_evidence)
        if fp in seen_fps:
            seen.append(card)
        else:
            fresh.append(card)
    return fresh, seen


def deduplicate(cards: list[TaskCard]) -> list[TaskCard]:
    """Keep best card per skill + friction-type."""
    buckets: dict[str, list[TaskCard]] = {}
    for card in cards:
        ev_type = card.friction_evidence[0].split(":")[0] if card.friction_evidence else "unknown"
        key = f"{card.skill_name}::{ev_type}"
        buckets.setdefault(key, []).append(card)

    result: list[TaskCard] = []
    for group in buckets.values():
        group.sort(key=lambda c: len(c.friction_evidence), reverse=True)
        result.append(group[0])
    return result


# ── Output ──────────────────────────────────────────────────────────────────


def write_task_cards(
    cards: list[TaskCard],
    output_dir: str,
    *,
    total_sessions_scanned: int = 0,
    seen_cards_skipped: int = 0,
) -> str:
    path = Path(output_dir) / "tasks.json"
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_sessions_scanned": total_sessions_scanned,
        "total_cards": len(cards),
        "seen_cards_skipped": seen_cards_skipped,
        "tasks": [c.to_dict() for c in cards],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skill-sleep MINE: extract friction signals from Hermes sessions")
    p.add_argument("--after", default="7d", help='Time window: "7d", "24h", or ISO datetime (default: 7d)')
    p.add_argument("--output-dir", default=".", help="Output directory for tasks.json")
    p.add_argument("--seen-file", default=DEFAULT_SEEN_FILE, help=f"Path to seen fingerprints file (default: {DEFAULT_SEEN_FILE})")
    p.add_argument("--reset-seen", action="store_true", help="Reset/clear seen fingerprints record")
    p.add_argument("--no-redact", action="store_true", help="Disable transcript redaction (not recommended)")
    p.add_argument("--timeout", type=int, default=60, help="Timeout for hermes export (default: 60s)")
    return p


def resolve_after(arg: str) -> str:
    """Normalize --after to a value hermes understands."""
    arg = arg.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", arg):
        return arg
    m = re.match(r"^(\d+)([dhms])$", arg)
    if not m:
        return "7d"
    value = int(m.group(1))
    unit = m.group(2)
    now = datetime.now(timezone.utc)
    if unit == "d":
        # hermes accepts bare durations; keep as-is for d is simplest, but
        # return ISO date for consistency with prior behavior
        return (now - timedelta(days=value)).strftime("%Y-%m-%d")
    if unit == "h":
        return (now - timedelta(hours=value)).isoformat()
    if unit == "m":
        return (now - timedelta(minutes=value)).isoformat()
    if unit == "s":
        return (now - timedelta(seconds=value)).isoformat()
    return "7d"


def run_mine(
    after: str = "7d",
    output_dir: str = ".",
    seen_file: str | None = None,
    reset_seen: bool = False,
    no_redact: bool = False,
    timeout: int = 60,
) -> tuple[str, list[TaskCard], list[TaskCard]]:
    """Run the MINE pipeline and return (output_path, fresh_cards, seen_cards)."""
    after_val = resolve_after(after)
    target_seen_file = seen_file or DEFAULT_SEEN_FILE

    if reset_seen:
        print(f"[mine] Resetting seen fingerprints at {target_seen_file}")
        seen_fps: set[str] = set()
        save_seen_fingerprints(target_seen_file, seen_fps)
    else:
        seen_fps = load_seen_fingerprints(target_seen_file)
        if seen_fps:
            print(f"[mine] Loaded {len(seen_fps)} seen fingerprint(s) from {target_seen_file}")

    print(f"[mine] Scanning sessions since {after_val} ...")

    sessions = export_sessions(after_val, redact=not no_redact, timeout=timeout)

    user_sessions = [s for s in sessions if not str(s.get("id", "")).startswith("cron_")]
    skipped = len(sessions) - len(user_sessions)
    print(f"[mine] Got {len(sessions)} sessions ({skipped} cron/automation skipped)")

    all_cards: list[TaskCard] = []
    for sess in user_sessions:
        all_cards.extend(detect_friction(sess))

    print(f"[mine] Raw friction episodes: {len(all_cards)}")

    deduped = deduplicate(all_cards)
    print(f"[mine] After intra-run dedup: {len(deduped)} task cards")

    fresh_cards, seen_cards = filter_seen_cards(deduped, seen_fps)
    if seen_cards:
        print(f"[mine] Skipped {len(seen_cards)} seen task card(s) (already processed in previous run):")
        for sc in seen_cards:
            fp = compute_fingerprint(sc.session_id, sc.friction_evidence)
            print(f"  [seen] {sc.skill_name} ({sc.session_id}) — fingerprint: {fp}")

    print(f"[mine] Fresh candidate task cards: {len(fresh_cards)}")
    for card in fresh_cards:
        fp = compute_fingerprint(card.session_id, card.friction_evidence)
        print(f"  {card}")

    # Record newly seen fingerprints
    new_fps = {compute_fingerprint(c.session_id, c.friction_evidence) for c in fresh_cards}
    if new_fps:
        updated_fps = seen_fps | new_fps
        save_seen_fingerprints(target_seen_file, updated_fps)
        print(f"[mine] Updated seen file with {len(new_fps)} new fingerprint(s): {target_seen_file}")

    output_path = write_task_cards(
        fresh_cards,
        output_dir,
        total_sessions_scanned=len(user_sessions),
        seen_cards_skipped=len(seen_cards),
    )
    print(f"[mine] Wrote {output_path}")
    return output_path, fresh_cards, seen_cards


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_mine(
        after=args.after,
        output_dir=args.output_dir,
        seen_file=args.seen_file,
        reset_seen=args.reset_seen,
        no_redact=args.no_redact,
        timeout=args.timeout,
    )
