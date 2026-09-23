"""Gateway response prefix.

Opt-in counterpart of :mod:`gateway.runtime_footer`: where the footer appends
runtime metadata to the FINAL message of a turn, the prefix prepends a short
model/provider tag to the FIRST message of a turn.  Off by default.

Config (``~/.hermes/config.yaml``)::

    display:
      response_prefix:
        enabled: true                    # off by default
        template: "[{provider}/{model}]" # rendered once per turn

A bare string is accepted as shorthand (``response_prefix: "[{model}]"``
enables the prefix with that template).  Per-platform overrides live under
``display.platforms.<platform>.response_prefix`` with the same shape, and
users can toggle the global switch with ``/prefix on|off`` from the CLI and
any gateway platform.

Template variables (case-insensitive; unknown placeholders are left as-is):
    {model}       — bare model id, vendor prefix dropped (``gpt-5.4``)
    {modelFull}   — full model id (``openai/gpt-5.4``)
    {provider}    — provider slug (``openai``, ``anthropic``, …)

Delivery: for a non-streamed reply ``gateway/run.py`` prepends the rendered
prefix to the response text right before the adapter send.  For a streamed
reply the prefix is handed to :class:`gateway.stream_consumer.GatewayStreamConsumer`,
which folds it into the first visible delta exactly once so every preview
edit, the final seal, and the delivered-payload reconciliation all carry it.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_TEMPLATE_VAR_PATTERN = re.compile(r"\{([a-zA-Z][a-zA-Z0-9_]*)\}")

# Separator inserted between the rendered prefix and the response body.
PREFIX_SEPARATOR = " "


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def _apply_prefix_cfg(resolved: dict[str, Any], raw: Any) -> None:
    """Merge one config layer (string shorthand or dict) into *resolved*."""
    if isinstance(raw, str):
        resolved["enabled"] = bool(raw.strip())
        resolved["template"] = raw
    elif isinstance(raw, dict):
        if "enabled" in raw:
            resolved["enabled"] = bool(raw.get("enabled"))
        if isinstance(raw.get("template"), str):
            resolved["template"] = raw["template"]


def resolve_prefix_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve the effective response-prefix config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False, empty template)
        2. ``display.response_prefix``
        3. ``display.platforms.<platform_key>.response_prefix``
    """
    resolved: dict[str, Any] = {"enabled": False, "template": ""}
    display = (user_config or {}).get("display") or {}
    _apply_prefix_cfg(resolved, display.get("response_prefix"))
    if platform_key:
        plat_cfg = (display.get("platforms") or {}).get(platform_key)
        if isinstance(plat_cfg, dict):
            _apply_prefix_cfg(resolved, plat_cfg.get("response_prefix"))
    return resolved


def platform_has_prefix_override(
    user_config: dict[str, Any] | None,
    platform_key: str | None,
) -> bool:
    """True when ``display.platforms.<platform_key>.response_prefix`` is set."""
    if not platform_key:
        return False
    display = (user_config or {}).get("display") or {}
    plat_cfg = (display.get("platforms") or {}).get(platform_key)
    return isinstance(plat_cfg, dict) and "response_prefix" in plat_cfg


def interpolate_prefix_template(
    template: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Substitute ``{model}`` / ``{modelFull}`` / ``{provider}`` in *template*.

    Placeholder names are case-insensitive.  A placeholder whose value is
    unknown is left verbatim so a misconfigured template stays visible.
    """

    def _replace(match: re.Match) -> str:
        name = match.group(1).lower()
        if name == "model":
            return _model_short(model) or match.group(0)
        if name == "modelfull":
            return model or match.group(0)
        if name == "provider":
            return provider or match.group(0)
        return match.group(0)

    return _TEMPLATE_VAR_PATTERN.sub(_replace, template)


def build_prefix_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Top-level entry point used by the gateway.

    Returns the rendered prefix (no trailing separator) or ``""`` when the
    prefix is disabled, the template is empty, or rendering yields only
    whitespace.  *provider* falls back to the vendor part of *model*.
    """
    cfg = resolve_prefix_config(user_config, platform_key)
    if not cfg.get("enabled") or not (cfg.get("template") or "").strip():
        return ""
    if not provider and model and "/" in model:
        provider = model.split("/", 1)[0]
    rendered = interpolate_prefix_template(
        cfg["template"], model=model, provider=provider,
    ).strip()
    # A single-line tag: newlines in a template would break the "first
    # message" placement on platforms that collapse leading whitespace.
    return " ".join(rendered.split())


def apply_prefix(prefix: str, text: str) -> str:
    """Prepend *prefix* to *text* unless it is already there or empty."""
    if not prefix or not text:
        return text
    if text.startswith(prefix):
        return text
    return f"{prefix}{PREFIX_SEPARATOR}{text}"


def strip_prefix(prefix: str, text: str) -> str:
    """Inverse of :func:`apply_prefix` for delivered-payload reconciliation."""
    if not prefix or not text or not text.startswith(prefix):
        return text
    return text[len(prefix):].lstrip()
