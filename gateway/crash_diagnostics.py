"""Best-effort OS crash diagnostics for gateway restart notifications."""

from __future__ import annotations

import json
import platform
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_DEFAULT_TIMEOUT_SECONDS = 2.0
_MAX_BACKTRACE_FRAMES = 4


def recent_crashes(
    name_filter: str = "python",
    since_hours: int = 24,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return newest-first OS crash records.

    This is intentionally best-effort: unsupported platforms, missing tools,
    parse failures, permission errors, and timeouts all degrade to ``[]``.
    """
    try:
        system = platform.system().lower()
        if system == "darwin":
            return _macos_recent_crashes(name_filter, since_hours, limit)
        if system == "linux":
            return _linux_recent_crashes(name_filter, since_hours, limit)
        if system == "windows":
            return _windows_recent_crashes(name_filter, since_hours, limit)
    except Exception:
        return []
    return []


def restart_notice(
    name_filter: str = "python",
    since_hours: int = 24,
) -> str:
    """Format the newest crash as a restart-notification suffix."""
    try:
        crashes = recent_crashes(name_filter=name_filter, since_hours=since_hours, limit=1)
        if not crashes:
            return ""
        return _format_restart_notice(crashes[0])
    except Exception:
        return ""


def _format_restart_notice(record: dict[str, Any]) -> str:
    cause = _clean_text(record.get("cause")) or "native crash"
    signal = _clean_text(record.get("signal"))
    process = _clean_text(record.get("process"))
    backtrace = record.get("backtrace")
    frame = ""
    if isinstance(backtrace, list) and backtrace:
        frame = _clean_text(backtrace[0])

    line = f"Crash cause: {cause}"
    # When cause inference fell through to the bare signal name, don't
    # repeat it in the parenthetical ("SIGTRAP (Python, SIGTRAP)").
    details = [item for item in (process, signal) if item and item != cause]
    if details:
        line += f" ({', '.join(details)})"
    if frame:
        line += f"\nFaulting frame: {frame}"
    return f"\n\n{line}"


def _macos_recent_crashes(
    name_filter: str,
    since_hours: int,
    limit: int,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))
    reports: list[Path] = []
    for directory in (
        Path.home() / "Library" / "Logs" / "DiagnosticReports",
        Path("/Library/Logs/DiagnosticReports"),
    ):
        try:
            reports.extend(directory.glob("*.ips"))
        except Exception:
            continue

    records: list[dict[str, Any]] = []
    for report in sorted(reports, key=_safe_mtime, reverse=True):
        if len(records) >= limit:
            break
        try:
            mtime = datetime.fromtimestamp(report.stat().st_mtime, timezone.utc)
            if mtime < cutoff:
                continue
            parsed = _parse_macos_ips(report)
            if not parsed or not _matches_filter(parsed, name_filter):
                continue
            records.append(parsed)
        except Exception:
            continue
    return records


def _parse_macos_ips(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    header = None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        try:
            header_text, payload = raw.split("\n", 1)
            body = json.loads(payload)
        except Exception:
            return None
        try:
            header = json.loads(header_text)
        except json.JSONDecodeError:
            header = None

    if not isinstance(body, dict):
        return None

    # DiagnosticReports mixes report kinds in *.ips — hangs (288), jetsam
    # (298) — and only bug_type 309 is a crash report. Skip the rest; keep
    # reports whose header is missing or unreadable (best-effort).
    if isinstance(header, dict):
        bug_type = str(header.get("bug_type") or "")
        if bug_type and bug_type != "309":
            return None

    process = body.get("procName") or body.get("name") or path.stem.split("_")[0]
    exception = body.get("exception") if isinstance(body.get("exception"), dict) else {}
    signal = exception.get("signal") or exception.get("type")
    backtrace = _macos_triggered_backtrace(body)
    return {
        "when": body.get("captureTime") or datetime.fromtimestamp(
            _safe_mtime(path),
            timezone.utc,
        ).isoformat(),
        "process": process,
        "signal": signal,
        "cause": _infer_cause(signal, backtrace),
        "backtrace": backtrace,
    }


def _macos_triggered_backtrace(body: dict[str, Any]) -> list[str]:
    images = body.get("usedImages") if isinstance(body.get("usedImages"), list) else []
    threads = body.get("threads") if isinstance(body.get("threads"), list) else []
    selected = None
    for thread in threads:
        if isinstance(thread, dict) and thread.get("triggered"):
            selected = thread
            break
    if selected is None and threads:
        selected = threads[0] if isinstance(threads[0], dict) else None
    frames = selected.get("frames") if isinstance(selected, dict) else []
    if not isinstance(frames, list):
        return []

    out: list[str] = []
    for frame in frames[:_MAX_BACKTRACE_FRAMES]:
        if not isinstance(frame, dict):
            continue
        image_name = _macos_image_name(frame, images)
        symbol = frame.get("symbol") or frame.get("symbolLocation") or frame.get("imageOffset")
        line = ": ".join(str(part) for part in (image_name, symbol) if part)
        if line:
            out.append(line)
    return out


def _macos_image_name(frame: dict[str, Any], images: list[Any]) -> str:
    image_index = frame.get("imageIndex")
    try:
        image = images[int(image_index)]
    except Exception:
        image = None
    if isinstance(image, dict):
        value = image.get("name") or image.get("path")
        if value:
            return Path(str(value)).name
    value = frame.get("imageName") or frame.get("imagePath")
    return Path(str(value)).name if value else ""


def _linux_recent_crashes(
    name_filter: str,
    since_hours: int,
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if shutil.which("coredumpctl"):
        output = _run(
            [
                "coredumpctl",
                "--no-pager",
                "--json=short",
                "--reverse",
                "--since",
                f"{max(1, int(since_hours))} hours ago",
            ]
        )
        records.extend(_parse_coredumpctl_json(output, name_filter, limit))
        for record in records[:limit]:
            pid = record.pop("_pid", None)
            if not pid or record.get("backtrace"):
                continue
            info = _run(["coredumpctl", "--no-pager", "info", str(pid)])
            backtrace = _parse_coredumpctl_info_backtrace(info)
            if backtrace:
                record["backtrace"] = backtrace
                record["cause"] = _infer_cause(record.get("signal"), backtrace)

    if not records and shutil.which("journalctl"):
        output = _run(
            [
                "journalctl",
                "-k",  # kernel log only — segfault/oom lines drown in the full journal
                "--no-pager",
                "--since",
                f"{max(1, int(since_hours))} hours ago",
                "-n",
                "200",
                "-o",
                "short-iso",
            ]
        )
        records.extend(_parse_journal_crashes(output, name_filter, limit))

    return records[:limit]


def _parse_coredumpctl_json(
    text: str,
    name_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _coredumpctl_json_rows(text):
        if len(records) >= limit:
            break
        if not isinstance(row, dict):
            continue
        # ``coredumpctl --json=short`` emits one compact JSON array with
        # lowercase keys (time/pid/sig/exe) and the signal as a number; the
        # journal field spellings are kept as a fallback for other shapes.
        process = (
            row.get("exe")
            or row.get("COREDUMP_COMM")
            or row.get("COREDUMP_EXE")
            or row.get("EXE")
            or row.get("COMM")
        )
        signal_value = row.get("sig")
        if signal_value is None:
            signal_value = row.get("COREDUMP_SIGNAL_NAME") or row.get("SIGNAL")
        backtrace = _split_backtrace(row.get("STACKTRACE") or row.get("MESSAGE"))
        record = {
            "when": row.get("time") or row.get("__REALTIME_TIMESTAMP") or row.get("TIME"),
            "process": Path(str(process)).name if process else "",
            "signal": _signal_display_name(signal_value),
            "backtrace": backtrace,
            "_pid": row.get("pid") or row.get("COREDUMP_PID") or row.get("PID"),
        }
        if not _matches_filter(record, name_filter):
            continue
        record["cause"] = _infer_cause(record.get("signal"), backtrace)
        records.append(record)
    return records


def _coredumpctl_json_rows(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    rows: list[Any] = []
    for line in stripped.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, list):
            rows.extend(row)
        else:
            rows.append(row)
    return rows


def _signal_display_name(value: Any) -> str:
    """Map a numeric signal (how ``coredumpctl --json`` reports it) to its name."""
    try:
        return signal.Signals(int(value)).name
    except Exception:
        return _clean_text(value)


def _parse_coredumpctl_info_backtrace(text: str) -> list[str]:
    frames: list[str] = []
    in_stack = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_stack and frames:
                break
            continue
        if stripped.lower().startswith("stack trace"):
            in_stack = True
            continue
        if not in_stack:
            continue
        if stripped.startswith("#"):
            frames.append(_clean_text(stripped))
            if len(frames) >= _MAX_BACKTRACE_FRAMES:
                break
            continue
        if frames:
            break
    return frames


def _parse_journal_crashes(
    text: str,
    name_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    patterns = ("segfault", "core dumped", "oom-killer", "out of memory", "killed process")
    for line in reversed(text.splitlines()):
        if len(records) >= limit:
            break
        lower = line.lower()
        if not any(pattern in lower for pattern in patterns):
            continue
        process = _extract_process_name(line)
        record = {
            "when": line[:25].strip(),
            "process": process,
            "signal": _extract_signal(line),
            "cause": _infer_cause(_extract_signal(line), [line]),
            "backtrace": [line.strip()],
        }
        if _matches_filter(record, name_filter):
            records.append(record)
    return records


def _windows_recent_crashes(
    name_filter: str,
    since_hours: int,
    limit: int,
) -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    shell = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return []

    hours = max(1, int(since_hours))
    command = (
        "$events = Get-WinEvent -FilterHashtable "
        f"@{{LogName='Application'; Id=1000,1001; StartTime=(Get-Date).AddHours(-{hours})}} "
        f"-MaxEvents {max(1, int(limit))}; "
        "$events | Select-Object TimeCreated,ProviderName,Message | ConvertTo-Json -Depth 4 -Compress"
    )
    output = _run([shell, "-NoProfile", "-NonInteractive", "-Command", command])
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    records: list[dict[str, Any]] = []
    for row in rows:
        if len(records) >= limit:
            break
        if not isinstance(row, dict):
            continue
        message = str(row.get("Message") or "")
        process = _extract_windows_faulting_app(message)
        record = {
            "when": row.get("TimeCreated"),
            "process": process,
            "signal": "Windows Error Reporting",
            "cause": _infer_cause(None, [message]),
            "backtrace": _split_backtrace(message),
        }
        if _matches_filter(record, name_filter):
            records.append(record)
    return records


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DEFAULT_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0 and not result.stdout:
        return ""
    return result.stdout or ""


def _matches_filter(record: dict[str, Any], name_filter: str) -> bool:
    needle = str(name_filter or "").strip().lower()
    if not needle:
        return True
    haystack_parts: list[str] = []
    for key in ("process", "signal", "cause"):
        value = record.get(key)
        if value:
            haystack_parts.append(str(value))
    backtrace = record.get("backtrace")
    if isinstance(backtrace, list):
        haystack_parts.extend(str(item) for item in backtrace[:_MAX_BACKTRACE_FRAMES])
    return needle in "\n".join(haystack_parts).lower()


def _infer_cause(signal: Any, lines: list[str]) -> str:
    text = " ".join([str(signal or ""), *[str(line) for line in lines]]).lower()
    signal_text = str(signal or "").upper()

    if any(token in text for token in ("mlx", "metal", "mps")):
        return "GPU error (Metal/MLX)"
    if any(token in text for token in ("cuda", "nvidia", "cublas", "cudnn")):
        return "GPU error (CUDA/NVIDIA)"
    if any(token in text for token in ("oom-killer", "out of memory", "memory pressure", "killed process")):
        return "out of memory"
    if "SIGSEGV" in signal_text or "segfault" in text:
        return "segmentation fault"
    if "SIGABRT" in signal_text or "abort" in text:
        return "process aborted"
    if "SIGKILL" in signal_text:
        return "killed by SIGKILL"
    if signal_text:
        return signal_text
    return "native crash"


def _split_backtrace(value: Any) -> list[str]:
    if not value:
        return []
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    return lines[:_MAX_BACKTRACE_FRAMES]


def _extract_process_name(line: str) -> str:
    for pattern in (
        r"\bcomm=\"([^\"]+)\"",
        r"\bProcess\s+(\S+)",
        r"\bpython[0-9.]*\b",
    ):
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return ""


def _extract_signal(line: str) -> str:
    match = re.search(r"\bSIG[A-Z0-9]+\b", line)
    if match:
        return match.group(0)
    if "segfault" in line.lower():
        return "SIGSEGV"
    return ""


def _extract_windows_faulting_app(message: str) -> str:
    match = re.search(r"Faulting application name:\s*([^,\r\n]+)", message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0
