"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, quota, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, quota, cwd]   # order shown; drop any to hide

Available fields:
    model        — bare model id, vendor prefix dropped (``gpt-5.4``)
    context_pct  — last-call context occupancy as a percent (``5%``)
    quota        — available model/Supermemory usage windows (best-effort)
    latency      — wall-clock duration of the turn (``22s``, ``1m05s``)
    cwd          — home-relative working dir (``~``)

``latency`` is opt-in: it is NOT in the default field set. A footer whose
``fields`` are unset uses ``model``, ``context_pct``, ``quota``, and ``cwd``.

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "quota", "cwd")
_SEP = " · "
logger = logging.getLogger(__name__)


async def fetch_runtime_footer_quota_snapshot(
    provider: str | None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 5.0,
):
    """Fetch model and Supermemory quota snapshots without blocking the loop."""
    from agent.account_usage import AccountUsageSnapshot, fetch_account_usage

    normalized = str(provider or "").strip().lower()
    requests: list[tuple[str, str | None, str | None]] = []
    if normalized in {"anthropic", "openai-codex", "openrouter"}:
        requests.append((normalized, base_url, api_key))
    async def _fetch(
        one_provider: str,
        one_base_url: str | None,
        one_api_key: str | None,
    ):
        return await asyncio.to_thread(
            fetch_account_usage,
            one_provider,
            base_url=one_base_url,
            api_key=one_api_key,
        )

    def _fetch_supermemory_if_configured():
        """Resolve connection + fetch in one worker so footer rendering never blocks."""
        from plugins.memory.supermemory import resolve_supermemory_connection_settings

        if not str(resolve_supermemory_connection_settings().get("api_key") or "").strip():
            return None
        return fetch_account_usage("supermemory")

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    _fetch(name, request_base, request_key)
                    for name, request_base, request_key in requests
                ),
                asyncio.to_thread(_fetch_supermemory_if_configured),
                return_exceptions=True,
            ),
            timeout=max(0.001, float(timeout_seconds)),
        )
    except (asyncio.TimeoutError, ValueError, TypeError):
        logger.debug("runtime footer quota fetch timed out", exc_info=True)
        return None

    snapshots = [
        result
        for result in results
        if isinstance(result, AccountUsageSnapshot)
        and result.available
        and not result.unavailable_reason
    ]
    if not snapshots:
        return None
    if len(snapshots) == 1:
        return snapshots[0]

    return AccountUsageSnapshot(
        provider="combined",
        source="runtime_footer",
        fetched_at=snapshots[0].fetched_at,
        title="Account limits",
        windows=tuple(window for snapshot in snapshots for window in snapshot.windows),
        details=tuple(detail for snapshot in snapshots for detail in snapshot.details),
    )


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]

    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def _quota_block(snapshot: Any) -> str:
    """Render up to three available quota windows using provider labels."""
    if not snapshot or getattr(snapshot, "unavailable_reason", None):
        return ""
    lines: list[str] = []
    for window in list(getattr(snapshot, "windows", ()) or ()):
        try:
            label = str(getattr(window, "label", "") or "").strip()
            raw_used = getattr(window, "used_percent", None)
            if raw_used is None:
                continue
            used = float(raw_used)
            if not label or not math.isfinite(used):
                continue
            line = f"{label[:40]} - {max(0, min(100, round(used)))}%"
            detail = str(getattr(window, "detail", "") or "").strip()
            if detail:
                line += f" - {detail}"
            reset_at = getattr(window, "reset_at", None)
            if reset_at is not None:
                reset_text = reset_at.astimezone().strftime("%b %d %H:%M %Z").replace(" 0", " ")
                line += f" - Reset {reset_text}"
            lines.append(line)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if len(lines) >= 3:
            break
    return "Quota Used:\n" + "\n".join(lines) if lines else ""


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    quota_snapshot: Any = None,
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    Multiline fields such as quota are rendered beneath the compact inline
    metadata so later fields cannot be joined onto a quota detail line.
    """
    parts: list[str] = []
    blocks: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "quota":
            quota = _quota_block(quota_snapshot)
            if quota:
                blocks.append(quota)
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts and not blocks:
        return ""
    return "\n".join(part for part in (_SEP.join(parts), *blocks) if part)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    quota_snapshot: Any = None,
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        quota_snapshot=quota_snapshot,
        cwd=cwd,
        turn_seconds=turn_seconds,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )


async def build_footer_line_async(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    provider: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
) -> str:
    """Build the footer and fetch quota only when the effective config shows it."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = cfg.get("fields") or _DEFAULT_FIELDS
    quota_snapshot = None
    if "quota" in fields:
        quota_snapshot = await fetch_runtime_footer_quota_snapshot(
            provider,
            base_url=base_url,
            api_key=api_key,
        )
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        quota_snapshot=quota_snapshot,
        cwd=cwd,
        turn_seconds=turn_seconds,
        fields=fields,
    )
