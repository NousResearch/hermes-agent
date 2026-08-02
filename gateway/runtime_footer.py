"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, reasoning effort,
context %, cwd) and appends it to the FINAL message of an agent turn when
enabled.  Off by default to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, reasoning_effort, fast, context_pct, cwd]
        separator: " • "

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

from utils import base_url_host_matches

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_DEFAULT_SEPARATOR = " · "
_CREDIT_WARNING = "💸 consuming credits"

# Providers whose normal Hermes route is usage-metered. OAuth/subscription
# routes deliberately use distinct slugs (for example ``openai-codex`` and
# ``xai-oauth``) and are therefore excluded. Host matching catches generic
# OpenAI-compatible routes that actually terminate at a metered provider.
_METERED_PROVIDER_SLUGS = frozenset({
    "ai-gateway",
    "anthropic",
    "azure-openai",
    "bedrock",
    "cohere",
    "deepinfra",
    "deepseek",
    "fireworks",
    "gemini",
    "google",
    "groq",
    "mistral",
    "moonshot",
    "novita",
    "nvidia",
    "openai",
    "openrouter",
    "perplexity",
    "together",
    "vertex",
    "xai",
})
_METERED_PROVIDER_HOSTS = (
    "openrouter.ai",
    "api.openai.com",
    "api.anthropic.com",
    "api.x.ai",
    "api.deepseek.com",
    "api.groq.com",
    "api.mistral.ai",
    "api.together.ai",
    "api.together.xyz",
    "fireworks.ai",
    "perplexity.ai",
    "generativelanguage.googleapis.com",
    "api.cohere.com",
    "api.deepinfra.com",
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


def _cwd_short(cwd: str) -> str:
    """Return only the current directory name for compact footers."""
    rel = _home_relative_cwd(cwd)
    if rel in {"", "~", os.sep}:
        return rel
    return os.path.basename(rel.rstrip(os.sep)) or rel


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def provider_consumes_credits(
    provider: Optional[str],
    base_url: Optional[str] = None,
) -> bool:
    """Return whether the effective backend is a known usage-metered route."""
    slug = str(provider or "").strip().lower().replace("_", "-")
    if slug in _METERED_PROVIDER_SLUGS:
        return True
    url = str(base_url or "")
    return any(base_url_host_matches(url, host) for host in _METERED_PROVIDER_HOSTS)


def reasoning_effort_label(reasoning_config: dict[str, Any] | None) -> str:
    """Return the session-configured effort label used by the footer."""
    if reasoning_config is None:
        return "medium"
    if not reasoning_config.get("enabled", True):
        return "none"
    return str(reasoning_config.get("effort") or "medium").strip().lower()


def _reasoning_short(effort: Optional[str]) -> str:
    """Return a compact, human-readable reasoning effort label."""
    normalized = str(effort or "").strip().lower()
    aliases = {
        "minimal": "min",
        "medium": "med",
        "none": "off",
    }
    return aliases.get(normalized, normalized)


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
    resolved = {
        "enabled": False,
        "fields": list(_DEFAULT_FIELDS),
        "separator": _DEFAULT_SEPARATOR,
    }
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]
        if isinstance(global_cfg.get("separator"), str):
            resolved["separator"] = global_cfg["separator"]

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
                if isinstance(plat_footer.get("separator"), str):
                    resolved["separator"] = plat_footer["separator"]

    return resolved


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    cwd: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    fast_mode: bool = False,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    separator: str = _DEFAULT_SEPARATOR,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    field_names = tuple(str(field) for field in fields)
    fast_requested = any(field in {"fast", "service_tier"} for field in field_names)
    model_requested = "model" in field_names

    for field in field_names:
        if field == "model":
            m = _model_short(model)
            if m:
                if fast_mode and fast_requested:
                    m = f"{m} ⚡️"
                parts.append(m)
        elif field in {"reasoning", "reasoning_effort"}:
            effort = _reasoning_short(reasoning_effort)
            if effort:
                parts.append(f"🧠 {effort}")
        elif field in {"fast", "service_tier"}:
            # With a model field, Fast decorates the model (``gpt-5 ⚡️``) to
            # avoid spending a full footer segment on one icon.  Without a
            # model field, keep the icon useful as a standalone segment.
            if fast_mode and not model_requested:
                parts.append("⚡️")
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        elif field == "dir":
            short = _cwd_short(cwd or os.environ.get("TERMINAL_CWD", ""))
            if short:
                parts.append(short)
        # Unknown field names are silently ignored.

    footer = separator.join(parts)
    if provider_consumes_credits(provider, base_url):
        return f"{footer}\n{_CREDIT_WARNING}" if footer else _CREDIT_WARNING
    return footer


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    cwd: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    fast_mode: bool = False,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        provider=provider,
        base_url=base_url,
        cwd=cwd,
        reasoning_effort=reasoning_effort,
        fast_mode=fast_mode,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        separator=cfg.get("separator", _DEFAULT_SEPARATOR),
    )
