"""Core helpers for the codex-ops plugin.

No external dependencies; all state is profile-aware under ``get_hermes_home()``.
The ledger intentionally stores operational metadata only — commands are
redacted, output bodies are not stored, and summaries are short redacted signal
lines.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shlex
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

_PLUGIN_KEY = "codex-ops"
_STATE_DIR = "codex-ops"
_LEDGER_FILE = "ledger.sqlite3"
_DEFAULT_COMPACT_THRESHOLD = 18000
_DEFAULT_HEAD_LINES = 80
_DEFAULT_TAIL_LINES = 120
_DEFAULT_SIGNAL_LIMIT = 80
_MAX_SUMMARY_CHARS = 4000
_LOCK = threading.Lock()

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_ACCESS_KEY_ID]"),
    (
        re.compile(r"(?i)\baws(.{0,20})?(secret|access).{0,20}\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{24,}['\"]?"),
        "AWS_SECRET=[REDACTED]",
    ),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[REDACTED_JWT]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|auth(?:orization)?|bearer|client[_-]?secret|password|secret|token)\b\s*[:=]\s*['\"]?[^\s'\"`]{16,}['\"]?"
        ),
        "secret=[REDACTED]",
    ),
)

_SIGNAL_RE = re.compile(
    r"(?i)(error|exception|traceback|failed|failure|fatal|panic|warning|warn|denied|refused|timeout|timed out|passed|success|summary|tests?\b|exit code|listening|started|ready|http://|https://)"
)
_CODEX_RE = re.compile(r"(?:^|[\s;&|()])(?:npx\s+)?codex(?:\s|$)|codex(?:\.cmd|\.exe)?(?:\s|$)", re.I)
_AGENT_COMMAND_RE = re.compile(r"(?:^|[\s;&|()])(?:opencode|claude|gemini|aider)(?:\s|$)", re.I)
_DEV_COMMAND_RE = re.compile(
    r"(?:^|[\s;&|()])(?:pytest|python\s+-m\s+pytest|uv\s+run|npm\s+(?:test|run|exec)|pnpm\s+(?:test|run|exec)|yarn\s+(?:test|run)|cargo\s+(?:test|build)|go\s+test|make\s+test)(?:\s|$)",
    re.I,
)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _cfg(key: str, default: Any) -> Any:
    """Read plugins.entries.codex-ops.<key> from config.yaml."""
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        return cfg_get(config, "plugins", "entries", _PLUGIN_KEY, key, default=default)
    except Exception:
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    value = _cfg(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_int(key: str, default: int, *, minimum: int = 1, maximum: int = 1_000_000) -> int:
    try:
        value = int(_cfg(key, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def redact_text(text: str) -> str:
    """Return *text* with high-confidence secrets replaced by placeholders."""
    if not text:
        return text
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def is_codex_command(command: str) -> bool:
    return bool(command and _CODEX_RE.search(command))


_SAFE_CODEX_VALUE_FLAGS = {
    "--cd",
    "-C",
    "--sandbox",
    "-s",
    "--model",
    "--provider",
    "--profile",
    "--approval-policy",
    "--config-profile",
}
_MAX_COMMAND_SUMMARY_CHARS = 1000


def _clip_text(text: str, *, limit: int = _MAX_COMMAND_SUMMARY_CHARS) -> str:
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{text[: limit - 40]}... <sha256={digest} chars={len(text)}>"


def _is_codex_executable(token: str) -> bool:
    name = Path(token).name.lower()
    return name in {"codex", "codex.cmd", "codex.exe"}


def summarize_command(command: str) -> str:
    """Return a privacy-preserving command summary for the ledger.

    `codex exec "<prompt>"` puts the task body on the shell command line. The
    ledger must not persist that body, even redacted, so Codex positional
    payloads are replaced by a hash/length marker while operational flags stay
    visible for debugging.
    """
    clean = redact_text(command or "")
    if not clean:
        return ""
    if not is_codex_command(clean):
        return _clip_text(clean)
    try:
        tokens = shlex.split(clean, posix=True)
    except ValueError:
        return "codex <unparseable-command>"

    summary: list[str] = []
    payload: list[str] = []
    seen_codex = False
    seen_subcommand = False
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if _is_codex_executable(token):
            if not seen_codex:
                summary.append("codex")
                seen_codex = True
            idx += 1
            continue
        if not seen_codex:
            # Preserve wrappers such as `npx` without recording arbitrary args.
            summary.append(Path(token).name)
            idx += 1
            continue
        if not seen_subcommand and not token.startswith("-"):
            summary.append(token)
            seen_subcommand = True
            idx += 1
            continue
        if token.startswith("--") and "=" in token:
            flag, _, value = token.partition("=")
            if flag in _SAFE_CODEX_VALUE_FLAGS:
                summary.append(f"{flag}={redact_text(value)}")
            else:
                summary.append(f"{flag}=<value-redacted>")
            idx += 1
            continue
        if token in _SAFE_CODEX_VALUE_FLAGS and idx + 1 < len(tokens):
            summary.extend([token, redact_text(tokens[idx + 1])])
            idx += 2
            continue
        if token.startswith("-"):
            summary.append(token)
            idx += 1
            continue
        payload.append(token)
        idx += 1

    if payload:
        payload_text = redact_text(" ".join(payload))
        digest = hashlib.sha256(payload_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        summary.append(f"<prompt sha256={digest} chars={len(payload_text)}>")
    return _clip_text(" ".join(summary) if summary else "codex <command>")


def _should_compact_command(command: str) -> bool:
    if not command:
        return False
    return bool(is_codex_command(command) or _AGENT_COMMAND_RE.search(command) or _DEV_COMMAND_RE.search(command))


def _line_indices(lines: list[str], selected: Iterable[int]) -> set[int]:
    max_idx = len(lines) - 1
    return {i for i in selected if 0 <= i <= max_idx}


def _signal_indices(lines: list[str], *, limit: int) -> list[int]:
    if limit <= 0:
        return []
    hits: list[int] = []
    for idx, line in enumerate(lines):
        if _SIGNAL_RE.search(line):
            hits.append(idx)
            if len(hits) >= limit:
                break
    return hits


def summarize_output(output: str, *, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    """Build a short redacted signal-line summary for ledger storage."""
    clean = redact_text(output or "")
    lines = clean.splitlines()
    if not lines:
        return ""
    selected = _signal_indices(lines, limit=24)
    if not selected:
        selected = list(range(min(3, len(lines))))
        if len(lines) > 6:
            selected += list(range(max(3, len(lines) - 3), len(lines)))
    chunks = [f"L{idx + 1}: {lines[idx]}" for idx in dict.fromkeys(selected)]
    summary = "\n".join(chunks)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 80] + "\n... [codex-ops summary truncated]"
    return summary


def compact_output(command: str, output: str, *, returncode: int = 0) -> str:
    """Redact and compact large Codex/dev command output."""
    clean = redact_text(output or "")
    if not clean:
        return clean
    if not _cfg_bool("compact_terminal_output", True):
        return clean
    threshold = _cfg_int("compact_threshold_chars", _DEFAULT_COMPACT_THRESHOLD, minimum=2000)
    if len(clean) <= threshold or not _should_compact_command(command):
        return clean

    lines = clean.splitlines()
    head_n = _cfg_int("compact_head_lines", _DEFAULT_HEAD_LINES, minimum=5, maximum=1000)
    tail_n = _cfg_int("compact_tail_lines", _DEFAULT_TAIL_LINES, minimum=5, maximum=1000)
    signal_limit = _cfg_int("compact_signal_limit", _DEFAULT_SIGNAL_LIMIT, minimum=0, maximum=1000)

    head = _line_indices(lines, range(min(head_n, len(lines))))
    tail_start = max(0, len(lines) - tail_n)
    tail = _line_indices(lines, range(tail_start, len(lines)))
    signal = _line_indices(lines, _signal_indices(lines, limit=signal_limit))
    middle_signal = [idx for idx in sorted(signal) if idx not in head and idx not in tail]

    omitted_count = max(0, len(lines) - len(head | tail | set(middle_signal)))
    output_hash = hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()[:16]
    parts: list[str] = [
        "[codex-ops] terminal output compacted",
        f"command_kind={'codex' if is_codex_command(command) else 'dev'} returncode={returncode} lines={len(lines)} chars={len(clean)} sha256={output_hash}",
        "Full output was not stored by codex-ops. Re-run the command or inspect its log file if the omitted stream is needed.",
        "",
        f"--- head ({min(head_n, len(lines))} lines) ---",
    ]
    parts.extend(lines[idx] for idx in sorted(head))
    if middle_signal:
        parts.append("")
        parts.append(f"--- signal lines outside head/tail ({len(middle_signal)} lines) ---")
        parts.extend(f"L{idx + 1}: {lines[idx]}" for idx in middle_signal)
    if omitted_count:
        parts.append("")
        parts.append(f"... [codex-ops omitted {omitted_count} non-signal middle lines] ...")
    parts.append("")
    parts.append(f"--- tail ({min(tail_n, len(lines))} lines) ---")
    parts.extend(lines[idx] for idx in sorted(tail - head))
    return "\n".join(parts)


def transform_terminal_output(
    *,
    command: str = "",
    output: str = "",
    returncode: int = 0,
    task_id: str = "",
    env_type: str = "",
) -> Optional[str]:
    """Plugin hook: return replacement output, or None to leave untouched."""
    if not isinstance(output, str) or not output:
        return None
    transformed = compact_output(command or "", output, returncode=returncode)
    if transformed != output:
        return transformed
    return None


_DANGEROUS_CODEX_FLAGS = {
    "--danger-full-access",
    "--dangerously-bypass-approvals-and-sandbox",
    "--yolo",
}
_SANDBOX_FLAGS = {"--sandbox", "-s"}
_DANGER_FULL_ACCESS = "danger-full-access"


def _token_value(token: str, flag: str) -> Optional[str]:
    prefix = f"{flag}="
    if token.startswith(prefix):
        return token[len(prefix) :]
    if flag == "-s" and token.startswith("-s") and len(token) > 2:
        return token[2:].lstrip("=")
    return None


def _uses_codex_full_access(tokens: list[str]) -> bool:
    """Return True when a Codex invocation disables the sandbox/approvals."""
    for idx, token in enumerate(tokens):
        lowered = token.lower()
        if lowered in _DANGEROUS_CODEX_FLAGS:
            return True
        for flag in _SANDBOX_FLAGS:
            if lowered == flag and idx + 1 < len(tokens) and tokens[idx + 1].lower() == _DANGER_FULL_ACCESS:
                return True
            if _token_value(lowered, flag) == _DANGER_FULL_ACCESS:
                return True
    return False


def guard_pre_tool_call(*, tool_name: str = "", args: Any = None) -> Optional[dict[str, str]]:
    """Plugin hook: block accidental dangerous Codex invocations."""
    if tool_name != "terminal" or not isinstance(args, dict):
        return None
    command = args.get("command") or ""
    if not isinstance(command, str) or not command:
        return None
    if not is_codex_command(command):
        return None
    if _cfg_bool("allow_danger_full_access", False):
        return None
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    if not _uses_codex_full_access(tokens):
        return None
    return {
        "action": "block",
        "message": (
            "codex-ops blocked a Codex no-sandbox invocation "
            "(`--danger-full-access`, `--dangerously-bypass-approvals-and-sandbox`, "
            "`--sandbox danger-full-access`, `-s danger-full-access`, or `--yolo`). Use "
            "`--sandbox workspace-write` with a bounded `--cd <repo>` for Hermes-run "
            "agents. If this is truly intentional, set "
            "`plugins.entries.codex-ops.allow_danger_full_access: true` in config.yaml "
            "and retry."
        ),
    }


def state_dir() -> Path:
    path = get_hermes_home() / _STATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_path() -> Path:
    return state_dir() / _LEDGER_FILE


def _connect() -> sqlite3.Connection:
    db = ledger_path()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS codex_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            profile_home TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            command TEXT NOT NULL,
            workdir TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            output_chars INTEGER NOT NULL DEFAULT 0,
            output_lines INTEGER NOT NULL DEFAULT 0,
            output_sha256 TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            task_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            tool_call_id TEXT NOT NULL DEFAULT '',
            turn_id TEXT NOT NULL DEFAULT '',
            error_type TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_codex_runs_created_at ON codex_runs(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_codex_runs_session ON codex_runs(session_id, created_at DESC)")
    conn.commit()
    return conn


def _parse_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _exit_code(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _should_record_terminal(command: str) -> bool:
    if _cfg_bool("record_all_terminal", False):
        return True
    return is_codex_command(command)


def record_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    duration_ms: int = 0,
    status: str = "",
    error_type: str = "",
    error_message: str = "",
) -> None:
    """Record a terminal Codex run. Fail-open; hooks must never break tools."""
    if tool_name != "terminal" or not isinstance(args, dict):
        return
    command = args.get("command") or ""
    if not isinstance(command, str) or not command or not _should_record_terminal(command):
        return
    parsed = _parse_result(result)
    output_raw = parsed.get("output")
    output: str = output_raw if isinstance(output_raw, str) else ""
    exit_code = _exit_code(parsed.get("exit_code"))
    command_safe = summarize_command(command)
    workdir = args.get("workdir") or ""
    if not isinstance(workdir, str):
        workdir = ""
    output_safe = redact_text(output)
    output_hash = hashlib.sha256(output_safe.encode("utf-8", errors="ignore")).hexdigest() if output_safe else ""
    try:
        with _LOCK:
            conn = _connect()
            try:
                conn.execute(
                    """
                    INSERT INTO codex_runs (
                        created_at, profile_home, tool_name, command, workdir,
                        status, exit_code, duration_ms, output_chars, output_lines,
                        output_sha256, summary, task_id, session_id, tool_call_id,
                        turn_id, error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now_iso(),
                        str(get_hermes_home()),
                        tool_name,
                        command_safe,
                        workdir,
                        status or ("error" if exit_code not in (None, 0) else "ok"),
                        exit_code,
                        int(duration_ms or 0),
                        len(output_safe),
                        len(output_safe.splitlines()),
                        output_hash,
                        summarize_output(output_safe),
                        task_id or "",
                        session_id or "",
                        tool_call_id or "",
                        turn_id or "",
                        redact_text(error_type or ""),
                        redact_text(error_message or ""),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        return


def status() -> dict[str, Any]:
    with _LOCK:
        conn = _connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM codex_runs").fetchone()[0]
            failures = conn.execute("SELECT COUNT(*) FROM codex_runs WHERE COALESCE(exit_code, 0) != 0").fetchone()[0]
            latest = conn.execute(
                "SELECT id, created_at, command, exit_code, duration_ms FROM codex_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return {
                "db": str(ledger_path()),
                "total": int(total),
                "failures": int(failures),
                "latest": dict(latest) if latest else None,
                "compact_terminal_output": _cfg_bool("compact_terminal_output", True),
                "compact_threshold_chars": _cfg_int("compact_threshold_chars", _DEFAULT_COMPACT_THRESHOLD, minimum=2000),
                "record_all_terminal": _cfg_bool("record_all_terminal", False),
                "allow_danger_full_access": _cfg_bool("allow_danger_full_access", False),
            }
        finally:
            conn.close()


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    limit = max(1, min(500, int(limit or 20)))
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, created_at, exit_code, duration_ms, output_lines, command, workdir
                FROM codex_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def get_run(run_id: int) -> Optional[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM codex_runs WHERE id = ?", (int(run_id),)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def prune(days: int) -> int:
    days = max(1, int(days))
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).isoformat(timespec="seconds")
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM codex_runs WHERE created_at < ?", (cutoff,))
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()
