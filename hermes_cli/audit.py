"""General audit for Hermes Agent — used by the /audit slash command.

Lazy footprint: stdlib only, ~150 LOC. Reuses ``hermes_constants``,
``hermes_cli.config``, and ``cron.jobs.list_jobs`` instead of duplicating
their logic. Two entry points:

    format_audit_summary(deep: bool = False) -> str

``deep=True`` adds a secret-leak scan of ~/.hermes/.env and the gateway
log tail. Keep everything here synchronous — the gateway handler runs it
under ``asyncio.to_thread``.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home
from hermes_cli.config import load_config_readonly


# ponytail: regex set is a deliberate false-tolerance trade-off — we err on
# "warn" for anything that looks like a known provider key. Upgrade to a real
# entropy scan only when the false-positive rate becomes a real burden.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("github", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("google", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("stripe", re.compile(r"sk_live_[A-Za-z0-9]{20,}")),
    ("slack", re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}")),
    ("telegram bot", re.compile(r"[0-9]{8,10}:[A-Za-z0-9_-]{35}")),
)


def _section(title: str) -> str:
    return f"\n— {title} —"


def _ok_or_warn(ok: bool, label: str, ok_text: str, warn_text: str) -> str:
    mark = "OK" if ok else "WARN"
    body = ok_text if ok else warn_text
    return f"  [{mark}] {label}: {body}"


def _check_config() -> list[str]:
    """Light config sanity — file exists, parsable, no duplicates."""
    lines: list[str] = []
    home = get_hermes_home()
    cfg_path = home / "config.yaml"
    if not cfg_path.exists():
        lines.append(f"  [WARN] config.yaml missing at {cfg_path}")
        return lines
    try:
        cfg = load_config_readonly()
        lines.append(f"  [OK] config.yaml loaded ({cfg_path.stat().st_size} bytes)")
    except Exception as exc:  # pragma: no cover — surface to user
        lines.append(f"  [WARN] config.yaml parse failed: {exc}")
        return lines

    # Duplicate-key detection: yaml.safe_load already collapses duplicates, so
    # do a raw text scan instead. Only count lines that start at column 0 —
    # nested keys are legitimate reuses (e.g. `headers:` under multiple
    # providers). Cheap, catches the common copy-paste mistake at the root.
    try:
        raw = cfg_path.read_text(encoding="utf-8", errors="replace")
        keys: dict[str, int] = {}
        for line in raw.splitlines():
            if not line or line.startswith((" ", "\t", "#", "-")):
                continue
            if ":" not in line:
                continue
            k = line.split(":", 1)[0].strip()
            if k and re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", k):
                keys[k] = keys.get(k, 0) + 1
        dupes = sorted(k for k, n in keys.items() if n > 1)
        if dupes:
            lines.append(f"  [WARN] duplicate top-level keys: {', '.join(dupes)}")
        else:
            lines.append("  [OK] no duplicate top-level keys")
    except Exception:
        pass

    # Model + provider present?
    model = cfg.get("model") or {}
    if model.get("default") and model.get("provider"):
        lines.append(
            f"  [OK] model: {model.get('default')} via {model.get('provider')}"
        )
    else:
        lines.append("  [WARN] model.default or model.provider missing")

    return lines


def _check_state() -> list[str]:
    """SQLite session store health."""
    lines: list[str] = []
    home = get_hermes_home()
    for name in ("state.db", "sessions.db"):
        p = home / name
        if not p.exists():
            lines.append(f"  [WARN] {name} missing")
            continue
        try:
            size = p.stat().st_size
            age = int(time.time() - p.stat().st_mtime)
            lines.append(f"  [OK] {name}: {size} bytes, mtime {age}s ago")
        except Exception as exc:
            lines.append(f"  [WARN] {name} stat failed: {exc}")
    return lines


def _check_env() -> list[str]:
    """Env file present, mode, secret-leak peek."""
    lines: list[str] = []
    home = get_hermes_home()
    env_path = home / ".env"
    if not env_path.exists():
        lines.append("  [WARN] .env missing (no API keys configured)")
        return lines
    try:
        st = env_path.stat()
        lines.append(f"  [OK] .env present ({st.st_size} bytes)")
        # POSIX mode only — Windows ignores it.
        if os.name == "posix":
            mode = st.st_mode & 0o777
            if mode & 0o044:
                lines.append(
                    f"  [WARN] .env is group/world readable (mode {oct(mode)})"
                )
            else:
                lines.append(f"  [OK] .env mode {oct(mode)} (private)")
    except Exception as exc:
        lines.append(f"  [WARN] .env stat failed: {exc}")
    return lines


def _check_cron() -> list[str]:
    """Cron jobs — count, last errors, disabled."""
    lines: list[str] = []
    try:
        from cron.jobs import list_jobs  # lazy import — heavy
    except Exception as exc:
        lines.append(f"  [WARN] cron import failed: {exc}")
        return lines
    try:
        jobs = list_jobs(include_disabled=True)
        active = [j for j in jobs if j.get("enabled", True)]
        disabled = [j for j in jobs if not j.get("enabled", True)]
        lines.append(
            f"  [OK] {len(active)} active cron job(s), {len(disabled)} disabled"
        )
        # Surface any with last_status not in {ok, success, succeeded} if the field exists.
        _ok_vals = {"ok", "success", "succeeded", "completed", "delivered", None, ""}
        bad = [
            j for j in jobs
            if j.get("last_status") and str(j.get("last_status")).lower() not in _ok_vals
        ]
        if bad:
            for j in bad[:5]:
                lines.append(
                    f"  [WARN] cron '{j.get('name', '?')}' last_status={j.get('last_status')}"
                )
    except Exception as exc:
        lines.append(f"  [WARN] cron listing failed: {exc}")
    return lines


def _check_log_tail() -> list[str]:
    """Tail the gateway log for ERROR/Traceback lines."""
    lines: list[str] = []
    home = get_hermes_home()
    log = home / "logs" / "gateway.log"
    if not log.exists():
        lines.append("  [OK] no gateway.log yet (clean slate)")
        return lines
    try:
        # Last 200 lines, scan for errors. Cheap enough to run on every audit.
        raw = log.read_text(encoding="utf-8", errors="replace")
        tail = raw.splitlines()[-200:]
        errors = [
            ln for ln in tail
            if "ERROR" in ln or "Traceback" in ln or "CRITICAL" in ln
        ]
        if errors:
            lines.append(f"  [WARN] {len(errors)} error line(s) in last 200 log lines")
            for ln in errors[-3:]:
                lines.append(f"        {ln[:200]}")
        else:
            lines.append("  [OK] no errors in last 200 log lines")
    except Exception as exc:
        lines.append(f"  [WARN] log scan failed: {exc}")
    return lines


def _check_secret_leak() -> list[str]:
    """Deep-only: scan sessions/jsonl/log/state for raw secret patterns.

    Heuristic — matches look like real keys, but a hit only means a string
    *looks like* a token. We never print the match itself, only the count +
    file. The caller (the slash handler) returns this string to the user,
    so no follow-up action is implied.
    """
    lines: list[str] = []
    home = get_hermes_home()
    targets = [
        home / ".env",
        home / "logs" / "gateway.log",
        home / "state.db",
    ]
    # Also scan any jsonl session transcripts.
    sessions_dir = home / "sessions"
    if sessions_dir.exists():
        targets.extend(sessions_dir.glob("*.jsonl"))

    hits = 0
    for path in targets:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for _name, pat in _SECRET_PATTERNS:
            n = len(pat.findall(raw))
            if n:
                hits += n
                lines.append(f"  [WARN] {path.name}: {n} match(es) for {_name} pattern")
    if hits == 0:
        lines.append("  [OK] no obvious secret patterns in env/log/state/sessions")
    else:
        lines.append(f"  [WARN] total {hits} potential secret match(es) — review manually")
    return lines


def format_audit_summary(deep: bool = False) -> str:
    """Render the audit report. Plain text, terminal-friendly."""
    out: list[str] = []
    out.append(f"Hermes Agent audit — {display_hermes_home()}")
    out.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    out.append(f"Mode: {'deep' if deep else 'quick'}")

    out.append(_section("Config"))
    out.extend(_check_config())

    out.append(_section("State"))
    out.extend(_check_state())

    out.append(_section("Env"))
    out.extend(_check_env())

    out.append(_section("Cron"))
    out.extend(_check_cron())

    out.append(_section("Gateway log"))
    out.extend(_check_log_tail())

    if deep:
        out.append(_section("Secret leak scan (deep)"))
        out.extend(_check_secret_leak())

    out.append("")
    return "\n".join(out)
