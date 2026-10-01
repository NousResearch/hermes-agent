"""``hermes logs`` — view and filter Hermes log files.

``hermes logs [name] [-n N] [-f] [--level L] [--session S] [--component C] [--since 1h]``;
``hermes logs list`` shows the available files.
"""

import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

from hermes_constants import get_hermes_home, display_hermes_home

# Known log files (name → filename)
LOG_FILES = {
    "agent": "agent.log",
    "errors": "errors.log",
    "gateway": "gateway.log",
    "gui": "gui.log",
    "desktop": "desktop.log",
    # Every stdio MCP subprocess's stderr (tools/mcp_tool.py redirects it
    # here, with per-server session markers) — the "MCP output channel".
    "mcp": "mcp-stderr.log",
}

# "2026-04-05 22:35:00[,123]" at the start of a line.
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
_LEVEL_RE = re.compile(r"\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s")
# Logger name: the token before ":" after the level and optional "[session]" tag,
# e.g. "INFO gateway.run:" or "INFO [sess_abc] tools.terminal_tool:".
_LOGGER_NAME_RE = re.compile(r"\s(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)(?:\s+\[.*?\])?\s+(\S+):")
_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


def _parse_since(since_str: str) -> Optional[datetime]:
    """Parse a relative time like '1h', '30m', '2d' into a cutoff; None if unparseable."""
    match = re.match(r"^(\d+)\s*([smhd])$", since_str.strip().lower())
    if not match:
        return None
    unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[match.group(2)]
    return datetime.now() - timedelta(**{unit: int(match.group(1))})


def _parse_line_timestamp(line: str) -> Optional[datetime]:
    m = _TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _extract_level(line: str) -> Optional[str]:
    m = _LEVEL_RE.search(line)
    return m.group(1) if m else None


def _extract_logger_name(line: str) -> Optional[str]:
    m = _LOGGER_NAME_RE.search(line)
    return m.group(1) if m else None


def _line_matches_component(line: str, prefixes: Sequence[str]) -> bool:
    name = _extract_logger_name(line)
    return name is not None and name.startswith(tuple(prefixes))


def _matches_filters(
    line: str,
    *,
    min_level: Optional[str] = None,
    session_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    component_prefixes: Optional[Sequence[str]] = None,
) -> bool:
    """Whether a line passes all active filters (lines without a timestamp/level pass those)."""
    if since is not None:
        ts = _parse_line_timestamp(line)
        if ts is not None and ts < since:
            return False
    if min_level is not None:
        level = _extract_level(line)
        if level is not None and _LEVEL_ORDER.get(level, 0) < _LEVEL_ORDER.get(min_level, 0):
            return False
    if session_filter is not None and session_filter not in line:
        return False
    return component_prefixes is None or _line_matches_component(line, component_prefixes)


def tail_log(
    log_name: str = "agent",
    *,
    num_lines: int = 50,
    follow: bool = False,
    level: Optional[str] = None,
    session: Optional[str] = None,
    since: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    """Print the filtered tail of a log, optionally following in real time."""
    filename = LOG_FILES.get(log_name)
    if filename is None:
        print(f"Unknown log: {log_name!r}. Available: {', '.join(sorted(LOG_FILES))}")
        sys.exit(1)

    log_path = get_hermes_home() / "logs" / filename
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("(Logs are created when Hermes runs — try 'hermes chat' first)")
        sys.exit(1)

    since_dt = None
    if since:
        since_dt = _parse_since(since)
        if since_dt is None:
            print(f"Invalid --since value: {since!r}. Use format like '1h', '30m', '2d'.")
            sys.exit(1)

    min_level = level.upper() if level else None
    if min_level and min_level not in _LEVEL_ORDER:
        print(f"Invalid --level: {level!r}. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        sys.exit(1)

    component_prefixes = None
    if component:
        from hermes_logging import COMPONENT_PREFIXES
        component_lower = component.lower()
        if component_lower not in COMPONENT_PREFIXES:
            available = ", ".join(sorted(COMPONENT_PREFIXES))
            print(f"Unknown component: {component!r}. Available: {available}")
            sys.exit(1)
        component_prefixes = COMPONENT_PREFIXES[component_lower]

    filters = dict(min_level=min_level, session_filter=session,
                   since=since_dt, component_prefixes=component_prefixes)
    has_filters = any(v is not None for v in filters.values())

    try:
        lines = _read_tail(log_path, num_lines, has_filters=has_filters, **filters)
    except PermissionError:
        print(f"Permission denied: {log_path}")
        sys.exit(1)

    filter_parts = [
        f"{label}={value}" for label, value in
        (("level>", min_level), ("session", session), ("component", component), ("since", since))
        if value
    ]
    filter_desc = f" [{', '.join(filter_parts)}]" if filter_parts else ""
    mode = "Ctrl+C to stop" if follow else f"last {num_lines}"
    print(f"--- {display_hermes_home()}/logs/{filename}{filter_desc} ({mode}) ---")

    for line in lines:
        print(line, end="")

    if not follow:
        return
    try:
        _follow_log(log_path, **filters)
    except KeyboardInterrupt:
        print("\n--- stopped ---")


def _read_tail(path: Path, num_lines: int, *, has_filters: bool = False, **filters) -> list:
    """Read the last *num_lines* matching lines; ``filters`` are ``_matches_filters`` kwargs."""
    if not has_filters:
        return _read_last_n_lines(path, num_lines)
    # Over-read so enough lines survive filtering.
    raw_lines = _read_last_n_lines(path, max(num_lines * 20, 2000))
    return [l for l in raw_lines if _matches_filters(l, **filters)][-num_lines:]


def _read_all_lines(path: Path) -> list:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def _read_last_n_lines(path: Path, n: int) -> list:
    """Read the last N lines; files over 1MB are read in growing chunks from the end."""
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        if size <= 1_048_576:
            return _read_all_lines(path)[-n:]

        with open(path, "rb") as f:
            chunk_size = 8192
            lines = []
            pos = size
            while pos > 0 and len(lines) <= n + 1:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk_lines = f.read(read_size).split(b"\n")
                if lines:
                    # Join the chunk's trailing partial line with our leading partial line.
                    lines[0] = chunk_lines[-1] + lines[0]
                    lines = chunk_lines[:-1] + lines
                else:
                    lines = chunk_lines
                chunk_size = min(chunk_size * 2, 65536)
            decoded = [raw.decode("utf-8", errors="replace") + "\n" for raw in lines if raw.strip()]
            return decoded[-n:]
    except Exception:
        return _read_all_lines(path)[-n:]


def _follow_log(path: Path, **filters) -> None:
    """Poll a log file for new content and print matching lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
            elif _matches_filters(line, **filters):
                print(line, end="")
                sys.stdout.flush()


def _size_label(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _age_label(mtime: datetime) -> str:
    age_s = (datetime.now() - mtime).total_seconds()
    if age_s < 60:
        return "just now"
    if age_s < 3600:
        return f"{int(age_s / 60)}m ago"
    if age_s < 86400:
        return f"{int(age_s / 3600)}h ago"
    return mtime.strftime("%Y-%m-%d")


def list_logs() -> None:
    """Print available log files with sizes."""
    log_dir = get_hermes_home() / "logs"
    if not log_dir.exists():
        print(f"No logs directory at {display_hermes_home()}/logs/")
        return

    print(f"Log files in {display_hermes_home()}/logs/:\n")
    found = False
    for entry in sorted(log_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".log":
            st = entry.stat()
            age_str = _age_label(datetime.fromtimestamp(st.st_mtime))
            print(f"  {entry.name:<25} {_size_label(st.st_size):>8}   {age_str}")
            found = True

    if not found:
        print("  (no log files yet — run 'hermes chat' to generate logs)")
