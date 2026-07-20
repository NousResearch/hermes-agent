"""Compact, messaging-friendly gateway ``/status`` card helpers."""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from agent.i18n import t


@dataclass(frozen=True)
class StatusCardSnapshot:
    """Local runtime values rendered by :func:`format_hermes_status_card`."""

    version: str
    commit: str | None
    gateway_uptime_seconds: int | None
    system_uptime_seconds: int | None
    model: str
    provider: str
    fallbacks: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None
    cost_status: str | None
    context_tokens: int | None
    context_limit: int | None
    compactions: int | None
    session_id: str
    task_state: str
    active_tasks: int
    queue_mode: str
    queue_depth: int


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_count(value: object) -> str:
    count = _as_int(value)
    sign = "-" if count < 0 else ""
    count = abs(count)
    if count >= 1_000_000:
        return f"{sign}{count / 1_000_000:.1f}m"
    if count >= 1_000:
        scaled = count / 1_000
        return f"{sign}{scaled:.1f}k" if count % 1_000 else f"{sign}{int(scaled)}k"
    return f"{sign}{count}"


def _format_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    total = _as_int(seconds, -1)
    if total < 0:
        return None
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts[:2])


def _format_context(current: int | None, limit: int | None) -> str | None:
    if current is None or limit is None or limit <= 0:
        return None
    percentage = min(100, round((max(0, current) / limit) * 100))
    return f"{_format_count(current)}/{_format_count(limit)} ({percentage}%)"


def _format_cost(value: float | None, status: str | None) -> str | None:
    if value is None:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount):
        return None
    amount = max(0.0, amount)
    prefix = "~" if status == "estimated" else ""
    return f"{prefix}${amount:.2f}"


def collect_uptime_seconds(
    *,
    pid: int | None = None,
    now: float | None = None,
    psutil_module: ModuleType | object | None = None,
) -> tuple[int | None, int | None]:
    """Return gateway and system uptime using psutil on every OS.

    ``psutil_module`` and ``now`` are injectable so Windows behavior can be
    tested on non-Windows CI without invoking platform-specific commands.
    Each value degrades independently when process metadata is unavailable.
    """

    if psutil_module is None:
        try:
            import psutil as psutil_module  # type: ignore[no-redef]
        except ImportError:
            return None, None

    current_time = time.time() if now is None else now
    process_id = os.getpid() if pid is None else pid

    gateway_uptime: int | None = None
    try:
        started_at = float(psutil_module.Process(process_id).create_time())
        gateway_uptime = max(0, int(current_time - started_at))
    except Exception:
        pass

    system_uptime: int | None = None
    try:
        booted_at = float(psutil_module.boot_time())
        system_uptime = max(0, int(current_time - booted_at))
    except Exception:
        pass

    return gateway_uptime, system_uptime


def get_hermes_revision(project_root: Path | None = None) -> str | None:
    """Return the local checkout or baked build revision without network I/O."""

    root = project_root or Path(__file__).resolve().parents[1]
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=windows_hide_flags(),
        )
        revision = (result.stdout or "").strip()
        if result.returncode == 0 and revision:
            return revision
    except Exception:
        pass

    try:
        from hermes_cli.build_info import get_build_sha

        return get_build_sha(short=8)
    except Exception:
        return None


def format_hermes_status_card(
    snapshot: StatusCardSnapshot,
    *,
    translate: Callable[..., str] = t,
) -> str:
    """Format a concise status card from already-collected local data."""

    unknown = translate("gateway.status.unknown")
    lines = [
        translate(
            "gateway.status.card_header",
            version=snapshot.version or unknown,
            commit=f" ({snapshot.commit[:8]})" if snapshot.commit else "",
        )
    ]

    gateway_uptime = _format_duration(snapshot.gateway_uptime_seconds)
    system_uptime = _format_duration(snapshot.system_uptime_seconds)
    lines.append(
        translate(
            "gateway.status.card_uptime",
            gateway=gateway_uptime or unknown,
            system=system_uptime or unknown,
        )
    )

    lines.append(
        translate(
            "gateway.status.card_model",
            model=snapshot.model or unknown,
            provider=snapshot.provider or unknown,
        )
    )
    lines.append(
        translate(
            "gateway.status.card_fallbacks",
            fallbacks=", ".join(snapshot.fallbacks) or translate("gateway.status.none"),
        )
    )

    cost = _format_cost(snapshot.cost_usd, snapshot.cost_status)
    token_key = "gateway.status.card_tokens_cost" if cost else "gateway.status.card_tokens"
    token_values = {
        "input": _format_count(snapshot.input_tokens),
        "output": _format_count(snapshot.output_tokens),
    }
    if cost:
        token_values["cost"] = cost
    lines.append(translate(token_key, **token_values))

    cache_total = (
        max(0, snapshot.input_tokens)
        + max(0, snapshot.cache_read_tokens)
        + max(0, snapshot.cache_write_tokens)
    )
    cache_hit = round((max(0, snapshot.cache_read_tokens) / cache_total) * 100) if cache_total else 0
    lines.append(
        translate(
            "gateway.status.card_cache",
            hit=cache_hit,
            read=_format_count(snapshot.cache_read_tokens),
            write=_format_count(snapshot.cache_write_tokens),
        )
    )

    context = _format_context(snapshot.context_tokens, snapshot.context_limit)
    context_parts: list[str] = []
    if context:
        context_parts.append(translate("gateway.status.card_context", context=context))
    if snapshot.compactions is not None:
        context_parts.append(
            translate(
                "gateway.status.card_compactions",
                count=_format_count(snapshot.compactions),
            )
        )
    if context_parts:
        lines.append(" · ".join(context_parts))

    lines.extend(
        [
            translate(
                "gateway.status.card_session",
                session_id=snapshot.session_id or unknown,
            ),
            translate(
                "gateway.status.card_tasks",
                state=snapshot.task_state,
                count=_format_count(snapshot.active_tasks),
            ),
            translate(
                "gateway.status.card_queue",
                mode=snapshot.queue_mode,
                depth=_format_count(snapshot.queue_depth),
            ),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "StatusCardSnapshot",
    "collect_uptime_seconds",
    "format_hermes_status_card",
    "get_hermes_revision",
]
