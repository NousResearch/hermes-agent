"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide

Available fields (the default set is unchanged — ``model``, ``context_pct``,
``cwd`` — so an existing footer renders byte-identically; the new fields are
opt-in via ``fields``):
    model           — bare model id, vendor prefix dropped (``claude-opus-4-8``)
    provider_model  — ``provider/model`` (``claude-bridge-f3/claude-opus-4-8``)
    context_pct     — last-call occupancy as a percent (``5%``)
    context_full    — ``used/window (pct)``, both humanized (``50.2k/1M (5%)``)
    reasoning       — model reasoning-effort level, ``r:<level>`` (``r:xhigh``)
    latency         — wall-clock duration of the turn (``22s``, ``1m05s``)
    cwd             — home-relative working dir (``~``)

``latency`` is opt-in: it is NOT in the default field set, so a footer whose
``fields`` are unset renders exactly as before.

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

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


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


def _split_provider_model(
    provider: Optional[str], model: Optional[str]
) -> tuple[str, str]:
    """Resolve a clean ``(provider, model)`` pair.

    When ``provider`` is unset but ``model`` carries a ``provider/model``
    prefix, split it so the footer reads cleanly (``provider/model``, not
    ``unset/a/b``).

    When the ``model`` ALREADY carries a ``provider/`` prefix, that embedded
    prefix wins and any separately-supplied ``provider`` is ignored — this
    avoids an ugly triple like ``openai-codex/claude-app/claude-opus-4-8`` when
    a caller passes both a provider and a prefixed model. The model's own
    prefix is the more specific source.
    """
    prov = (provider or "").strip()
    mdl = (model or "").strip()
    if "/" in mdl:
        # The model carries its own provider prefix — it's authoritative.
        prov, _, mdl = mdl.partition("/")
    return prov, mdl


def _humanize_tok(n: Any) -> str:
    """Token count -> compact string (``50k``, ``1.5k``, ``1M``, ``1.0M``)."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    if abs(n) >= 1_000_000:
        return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1000:
        return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:.1f}k"
    return str(n)


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


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    provider: Optional[str] = None,
    reasoning: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "provider_model":
            prov, mdl = _split_provider_model(provider, model)
            if prov and mdl:
                parts.append(f"{prov}/{mdl}")
            elif mdl:
                parts.append(mdl)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "context_full":
            # Both used and window humanized (50.2k/1M); pct from raw values.
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(
                    f"{_humanize_tok(context_tokens)}/{_humanize_tok(context_length)} ({pct}%)"
                )
            elif context_tokens and context_tokens > 0:
                parts.append(_humanize_tok(context_tokens))
        elif field == "reasoning":
            # Model reasoning-effort level (none/minimal/low/medium/high/xhigh).
            r = (reasoning or "").strip()
            if r:
                parts.append(f"r:{r}")
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def _reasoning_label(reasoning_config: Any) -> str:
    """Render a parsed reasoning-config dict as a footer label.

    Accepts the dict shape produced by
    :func:`hermes_constants.parse_reasoning_effort` — ``{"enabled": True,
    "effort": "<level>"}`` or ``{"enabled": False}``.  Returns the bare level
    (``xhigh``), ``none`` when thinking is explicitly disabled, or ``""`` when
    unset (caller drops the field).
    """
    if not isinstance(reasoning_config, dict):
        return ""
    if not reasoning_config.get("enabled", True):
        return "none"
    return str(reasoning_config.get("effort", "") or "").strip()


def _reasoning_from_config(
    user_config: dict[str, Any] | None, model: Optional[str] = None
) -> str:
    """Resolve the effective reasoning level for *model* from *user_config*.

    Routes through the shared chokepoint
    :func:`hermes_constants.resolve_reasoning_config` so the footer honors
    per-model overrides (``agent.reasoning_overrides``) and the YAML-boolean
    "disabled" spelling exactly as the agent does, rather than re-reading
    ``agent.reasoning_effort`` raw.

    Session-scoped ``/reasoning`` overrides are resolved by the CALLER (they
    always win) and passed to :func:`build_footer_line` as ``reasoning_config``.
    """
    try:
        from hermes_constants import resolve_reasoning_config

        return _reasoning_label(
            resolve_reasoning_config(user_config or {}, model or "")
        )
    except Exception:
        return ""


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    provider: Optional[str] = None,
    reasoning: Optional[str] = None,
    reasoning_config: Any = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.

    ``reasoning_config`` is the caller's ALREADY-RESOLVED reasoning config for
    this session (gateway/run.py's ``_resolve_session_reasoning_config``, which
    honors a session-scoped ``/reasoning <level>``).  Passing it keeps the
    footer in step with what the session actually runs; without it the footer
    falls back to the config-level resolution, which can be stale for a session
    that set a session-scoped override.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    # Reasoning: prefer an explicit label, then the caller's session-resolved
    # config, then config-level resolution for this model.
    if reasoning is None:
        if reasoning_config is not None:
            reasoning = _reasoning_label(reasoning_config)
        else:
            reasoning = _reasoning_from_config(user_config, model)
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_seconds=turn_seconds,
        provider=provider,
        reasoning=reasoning,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )
