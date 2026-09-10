"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``cwd`` (home-relative), and three more opt-in fields resolved from the
live agent by ``resolve_agent_runtime_fields()``: ``account`` (credential-pool entry label of the
serving credential, falling back to the provider name), ``billing`` (``sub`` for OAuth subscription
seats, ``extra`` when the provider stamped the turn as paid overage via its
``anthropic-ratelimit-unified-overage-in-use`` response header, ``API`` for per-token keys) and
``effort`` (the agent's reasoning effort). Like ``latency``, none of the three is in the default
field set, so an unset ``fields`` renders byte-identically to before. ``gateway/run.py`` appends the
footer to the final response only (never to tool-progress or streaming partials); when streaming
already delivered the text, it goes out as a trailing message via ``send_trailing_footer()``."""

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
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
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


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          account_label: Optional[str] = None, billing: Optional[str] = None,
                          effort: Optional[str] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
        # Opt-in agent-identity fields (resolve_agent_runtime_fields); skipped when unresolved.
        "account": lambda: account_label or "",
        "billing": lambda: billing or "",
        "effort": lambda: f"effort {effort}" if effort else "",
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def resolve_agent_runtime_fields(agent: Any) -> dict[str, Optional[str]]:
    """Derive the ``account`` / ``billing`` / ``effort`` footer fields from a live agent.

    Reads the same attributes the dispatch path uses, so the footer can never disagree with the
    wire. Never raises; unresolvable fields come back ``None`` and render as skipped.

    - ``account``: credential-pool entry label via ``agent._credential_pool_entry_id`` looked up
      in the persisted pool (``hermes_cli.auth.read_credential_pool``); falls back to the raw
      entry id, then to the provider name for non-pool credentials.
    - ``billing``: ``sub`` when the serving credential is an OAuth subscription seat, ``extra``
      when the provider's own per-response header
      (``anthropic-ratelimit-unified-overage-in-use: true``) marked the turn as served on paid
      overage rather than the included allowance (captured per call in
      ``_capture_anthropic_response_headers`` as ``agent._last_anthropic_overage_in_use``), and
      ``API`` for per-token keys. ``openai-codex`` is a subscription-only backend → ``sub``.
    - ``effort``: the agent's ``reasoning_config`` effort value, if any.
    """
    out: dict[str, Optional[str]] = {"account": None, "billing": None, "effort": None}
    try:
        provider = (getattr(agent, "provider", "") or "").strip().lower()
        pool_id = getattr(agent, "_credential_pool_entry_id", None)
        label = None
        if pool_id:
            try:
                from hermes_cli.auth import read_credential_pool
                for prov_entries in (read_credential_pool() or {}).values():
                    for entry in prov_entries or []:
                        if isinstance(entry, dict) and entry.get("id") == pool_id:
                            label = entry.get("label") or str(pool_id)
                            break
                    if label:
                        break
            except Exception:
                label = None
            if not label:
                label = str(pool_id)
        if label:
            out["account"] = label
        elif provider:
            out["account"] = provider
        if provider == "anthropic":
            if getattr(agent, "_is_anthropic_oauth", False):
                # The provider's own per-response stamp: True = this turn was paid from the
                # overage top-up, not the subscription's included allowance.
                overage = getattr(agent, "_last_anthropic_overage_in_use", None)
                out["billing"] = "extra" if overage is True else "sub"
            else:
                out["billing"] = "API"
        elif provider == "openai-codex":
            out["billing"] = "sub"
        elif provider:
            out["billing"] = "API"
        rc = getattr(agent, "reasoning_config", None)
        if isinstance(rc, dict):
            eff = rc.get("effort") or rc.get("reasoning_effort")
            if eff:
                out["effort"] = str(eff)
        elif isinstance(rc, str) and rc:
            out["effort"] = rc
    except Exception:
        pass
    return out


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      account_label: Optional[str] = None, billing: Optional[str] = None,
                      effort: Optional[str] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field. ``account_label`` / ``billing`` / ``effort`` come from
    ``resolve_agent_runtime_fields()`` on the live agent; ``None`` skips each."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 account_label=account_label, billing=billing, effort=effort,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)
