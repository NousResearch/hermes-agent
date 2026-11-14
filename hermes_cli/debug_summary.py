"""Allowlisted public summary for ``hermes debug share`` (#46006).

The public paste carries only values Hermes itself produces: versions, install kind, platform and
toolset names, the KEY shape of config.yaml, and counts of warnings/errors by logger, exception
class and HTTP status. Log lines and message text never take this path: subtractive redaction
cannot recognise IPs, room IDs, display names or message bodies, so the public report is built
from an allowlist instead. The full redacted logs go only to private storage (``--nous``).
"""

from __future__ import annotations

import builtins
import functools
import importlib.util
import logging
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ``hermes_logging._LOG_FORMAT``: ``<date> <time> LEVEL[ [session]] logger: message``.
_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T][\d:,.]+\s+(?P<level>WARNING|ERROR|CRITICAL)"
    r"(?: \[[^\]]*\])?\s+(?P<logger>[A-Za-z_][\w.]{0,80}):")
# Identifier-shaped exception class names only (``RateLimitError``, ``httpx.ReadTimeout``).
_EXC_NAME_RE = re.compile(
    r"\b((?:[a-z_][a-z0-9_]*\.){0,4}[A-Z][A-Za-z0-9]{0,60}(?:Error|Exception|Timeout))\b")
_HTTP_STATUS_RE = re.compile(r"\b(?:HTTP|status(?:_code)?|error code)[\s:=]+([1-5]\d\d)\b",
                             re.IGNORECASE)
# The last line of a traceback: ``pkg.ExcClass: ...`` at column 0.
_TRACEBACK_TAIL_RE = re.compile(r"^(?:[a-z_][a-z0-9_]*\.){0,4}[A-Z]\w*(?:Error|Exception|Timeout)\b")
_TOP_N = 20
_MAX_SHAPE_DEPTH = 2
# Logger names are module paths, but some embed runtime names (an MCP server, a plugin); only
# loggers rooted in a known package are named, the rest are counted as "(other)". Below the root,
# only segments that exist as modules on disk are kept (see _module_path).
_KNOWN_LOGGER_ROOTS = frozenset({
    "acp_adapter", "agent", "cli", "cron", "gateway", "hermes", "hermes_cli", "hermes_constants",
    "hermes_logging", "hermes_platform", "hermes_state", "model_tools", "pm", "plugins",
    "providers", "run_agent", "toolsets", "tools", "tui_gateway", "asyncio", "httpx", "httpcore",
    "openai", "anthropic", "urllib3", "websockets", "aiohttp", "uvicorn", "fastapi", "mcp"})

PUBLIC_SUMMARY_BANNER = (
    "[hermes debug share: PUBLIC SUMMARY. Allowlisted fields only (versions, config key "
    "shape, error counts); no log lines or message text. Full redacted logs go only to "
    "private storage: hermes debug share --nous]\n")


def _guarded(fn, default: Any = "unknown") -> Any:
    """One unreadable field must not sink the whole summary."""
    try:
        return fn()
    except Exception:  # health: allow BLE001 -- one probe (gateway config, git) failing must not drop the summary
        logger.debug("debug summary field failed", exc_info=True)
        return default


def _hermes_version() -> str:
    from hermes_cli.version_info import get_version_info
    return get_version_info().derived_version


def _install_kind() -> str:
    from hermes_cli.config import detect_install_method
    return detect_install_method()


def _platform_names() -> str:
    """Connected platforms, by name only when core ships the adapter (a built-in ``Platform``
    member or a bundled ``plugins/platforms/`` plugin); user plugins register arbitrary names at
    runtime, so those are counted, not named."""
    from gateway.config import load_gateway_config
    from gateway.platform_registry import core_ships_platform
    connected = [str(p.value) for p in load_gateway_config().get_connected_platforms()]
    names = sorted(name for name in connected if core_ships_platform(name))
    custom = len(connected) - len(names)
    return ", ".join(names + ([f"+{custom} custom"] if custom else [])) or "none"


def _toolset_names(raw_config: dict[str, Any]) -> str:
    """Configured toolsets, by name only when the name is a built-in toolset (user-named custom
    toolsets are counted, not named)."""
    from toolsets import TOOLSETS
    configured = raw_config.get("toolsets")
    if not isinstance(configured, list):
        return "(default)"
    known = sorted({str(n) for n in configured if isinstance(n, str) and n in TOOLSETS})
    custom = len(configured) - sum(1 for n in configured if isinstance(n, str) and n in TOOLSETS)
    parts = known + ([f"+{custom} custom"] if custom else [])
    return ", ".join(parts) or "none"


def _config_shape(raw: dict[str, Any], schema: dict[str, Any], depth: int = 0) -> list[str]:
    """Key names present in config.yaml, never values. Only keys defined by DEFAULT_CONFIG are
    named; user-named entries (provider names, MCP servers, unknown keys) are only counted."""
    lines: list[str] = []
    indent = "  " * (depth + 1)
    unknown = 0
    for key in sorted(k for k in raw if isinstance(k, str)):
        if key not in schema:
            unknown += 1
            continue
        value, default = raw[key], schema[key]
        if isinstance(value, dict) and isinstance(default, dict) and default and \
                depth < _MAX_SHAPE_DEPTH:
            lines.append(f"{indent}{key}:")
            lines.extend(_config_shape(value, default, depth + 1))
        elif isinstance(value, dict):
            lines.append(f"{indent}{key}: {{{len(value)} entries}}")
        elif isinstance(value, list):
            lines.append(f"{indent}{key}: [{len(value)} items]")
        else:
            lines.append(f"{indent}{key}")
    unknown += sum(1 for k in raw if not isinstance(k, str))
    if unknown:
        lines.append(f"{indent}(+{unknown} other keys)")
    return lines


@functools.lru_cache(maxsize=512)
def _resolve_module(name: str) -> tuple[str, Path | None] | None:
    """(longest real module prefix of ``name``, its source file when the whole name resolved).
    Located on disk only: a log-derived name is never imported, so no module code runs."""
    segments = name.split(".")
    if segments[0] not in _KNOWN_LOGGER_ROOTS:
        return None
    spec = _guarded(lambda: importlib.util.find_spec(segments[0]), None)
    dirs = [Path(d) for d in (getattr(spec, "submodule_search_locations", None) or [])]
    origin = getattr(spec, "origin", None)
    source = Path(origin) if origin and origin.endswith(".py") else None
    # A symlink below the package can point at any file on the host; its source must not vouch.
    roots = tuple(d.resolve() for d in dirs) or ((source.resolve().parent,) if source else ())
    kept = [segments[0]]
    for seg in segments[1:]:
        hit = seg and next(
            (d / seg for d in dirs if (d / seg).is_dir() or (d / f"{seg}.py").is_file()), None)
        if not hit:
            return ".".join(kept) + ".*", None
        kept.append(seg)
        dirs = [hit] if hit.is_dir() else []
        source = hit / "__init__.py" if hit.is_dir() else hit.with_name(f"{seg}.py")
    return ".".join(kept), source if source and _contained(source, roots) else None


def _contained(path: Path, roots: tuple[Path, ...]) -> bool:
    """``path``'s canonical file lies under one of ``roots`` (symlinks resolved)."""
    real = path.resolve()
    return any(real.is_relative_to(root) for root in roots)


def _module_path(name: str) -> str | None:
    """``name`` when every segment is a real module under a known root, else the longest real
    prefix + ``.*``; ``None`` when the root is unknown. Message-derived segments (a user name, a
    password baked into a logger or exception path) never match a file, so they never print."""
    resolved = _resolve_module(name)
    return resolved[0] if resolved else None


@functools.lru_cache(maxsize=512)
def _module_names(source: Path) -> frozenset[str]:
    """Identifiers in a module's source: a class it defines or re-exports appears there. Sibling
    ``from .x import *`` re-exports (``asyncio.CancelledError``) are followed one level."""
    text = _source_text(source)
    names = set(re.findall(r"[A-Za-z_]\w*", text))
    package = (source.resolve().parent,)
    for star in re.findall(r"^from \.(\w+) import \*", text, re.MULTILINE):
        sibling = source.with_name(f"{star}.py")
        if _contained(sibling, package):
            names.update(re.findall(r"[A-Za-z_]\w*", _source_text(sibling)))
    return frozenset(names)


def _source_text(path: Path) -> str:
    return _guarded(lambda: path.read_text(encoding="utf-8-sig", errors="replace"), "")


def _logger_label(name: str) -> str:
    return _module_path(name) or "(other)"


def _exception_label(token: str) -> str:
    """A dotted class keeps its prefix only when that prefix is a verified module path AND the
    class name appears in that module's source (a real prefix does not vouch for a class name
    quoted in a log); otherwise only a builtin exception name is shown, since a bare or foreign
    class name can come from user code or tool output in the message text."""
    prefix, _, cls = token.rpartition(".")
    resolved = _resolve_module(prefix) if prefix else None
    if resolved and resolved[1] is not None and cls in _module_names(resolved[1]):
        return token
    builtin = getattr(builtins, cls, None)
    return cls if isinstance(builtin, type) and issubclass(builtin, BaseException) else "(other)"


def _log_counts(text: str) -> tuple[Counter, Counter, Counter, int]:
    """(level, logger) counts, exception-class counts and HTTP-status counts for one log. Only
    WARNING+ log lines and traceback frames are mined; INFO lines (where adapters log message
    text) contribute nothing but the line count."""
    by_logger: Counter = Counter()
    exceptions: Counter = Counter()
    statuses: Counter = Counter()
    lines = text.splitlines()
    for line in lines:
        m = _LOG_LINE_RE.match(line)
        if m:
            by_logger[(m.group("level"), _logger_label(m.group("logger")))] += 1
        elif not _TRACEBACK_TAIL_RE.match(line):
            continue
        exceptions.update({_exception_label(t) for t in _EXC_NAME_RE.findall(line)})
        statuses.update(set(_HTTP_STATUS_RE.findall(line)))
    return by_logger, exceptions, statuses, len(lines)


def _log_section(log_lines: int) -> list[str]:
    from hermes_cli.debug import _REPORT_LOGS, _capture_log_snapshot
    from hermes_cli.logs import LOG_FILES
    out = ["", "logs (counts from the recent tail; no line text):"]
    by_logger: Counter = Counter()
    exceptions: Counter = Counter()
    statuses: Counter = Counter()
    for name in _REPORT_LOGS:
        snap = _capture_log_snapshot(name, tail_lines=log_lines)
        filename = LOG_FILES.get(name, f"{name}.log")
        if snap.full_text is None:
            out.append(f"  {filename:<28} absent or empty")
            continue
        loggers, excs, codes, scanned = _log_counts(snap.tail_text)
        levels = Counter()
        for (level, _logger), n in loggers.items():
            levels[level] += n
        summary = ", ".join(f"{lvl} {levels[lvl]}" for lvl in ("WARNING", "ERROR", "CRITICAL")
                            if levels[lvl]) or "no warnings/errors"
        out.append(f"  {filename:<28} {scanned} lines scanned; {summary}")
        by_logger.update(loggers)
        exceptions.update(excs)
        statuses.update(codes)
    out += ["", "warnings/errors by logger:"]
    out += [f"  {level:<8} {logger:<48} x{n}"
            for (level, logger), n in by_logger.most_common(_TOP_N)] or ["  none"]
    out += ["", "exception classes (lines mentioning):"]
    out += [f"  {name:<56} x{n}" for name, n in exceptions.most_common(_TOP_N)] or ["  none"]
    out += ["", "http status codes (lines mentioning):"]
    out += [f"  {code}  x{n}" for code, n in sorted(statuses.items())] or ["  none"]
    return out


def collect_public_summary(*, log_lines: int = 200) -> str:
    """The text uploaded to the PUBLIC paste: allowlisted fields only, never log or message text."""
    from hermes_cli.config import read_raw_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    raw_config = _guarded(read_raw_config, {})
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    lines = [
        "--- hermes debug summary ---",
        f"hermes:        {_guarded(_hermes_version)}",
        f"python:        {sys.version.split()[0]}",
        f"os:            {platform.system()} {platform.release()} {platform.machine()}",
        f"install_kind:  {_guarded(_install_kind)}",
        f"platforms:     {_guarded(_platform_names)}",
        f"toolsets:      {_guarded(lambda: _toolset_names(raw_config))}",
        "", "config.yaml keys (no values):",
        *(_config_shape(raw_config, DEFAULT_CONFIG) or ["  (empty or missing)"]),
        *_log_section(log_lines),
        "--- end summary ---",
    ]
    return PUBLIC_SUMMARY_BANNER + "\n".join(lines) + "\n"
