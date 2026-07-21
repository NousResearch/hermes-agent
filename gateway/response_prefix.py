"""Gateway response prefix.

Renders a compact prefix showing runtime state (model, provider) and
prepends it to the FIRST message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    messages:
      response_prefix: "[{provider}/{model}]"     # template string
      # Available variables: {model}, {modelFull}, {provider}, {thinking}
      # Default: disabled (empty string)

Per-platform overrides live under ``messages.platforms.<platform>.response_prefix``.
Users can toggle with ``/prefix on|off`` from both the CLI and any gateway platform.

The prefix is prepended to the first response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the first message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the initial text has already been delivered
piecemeal, the prefix is NOT retroactively added (streaming partials cannot be
edited).

Template variables (case-insensitive):
    {model}       — Short model name (e.g. "claude-opus-4.6")
    {modelFull}   — Full model ID (e.g. "github-copilot/claude-opus-4.6")
    {provider}    — Provider name (e.g. "github-copilot", "openai-codex")
    {thinking}    — Current thinking level (e.g. "high", "low", "off")
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Regex for template variables: {variableName} or {variable_name} or {variable.Name}
_TEMPLATE_VAR_PATTERN = re.compile(r"\{([a-zA-Z][a-zA-Z0-9._]*)\}")


def _model_short(model: Optional[str]) -> str:
    """Drop ``provider/`` prefix for readability (``github-copilot/claude-opus-4.6`` → ``claude-opus-4.6``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def resolve_prefix_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective response-prefix config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False, empty template)
        2. ``messages.response_prefix``
        3. ``messages.platforms.<platform_key>.response_prefix``
    """
    resolved: dict[str, Any] = {"enabled": False, "template": ""}
    cfg = (user_config or {}).get("messages") or {}

    global_template = cfg.get("response_prefix")
    if isinstance(global_template, str):
        resolved["enabled"] = bool(global_template)
        resolved["template"] = global_template
    elif isinstance(global_template, dict):
        if "enabled" in global_template:
            resolved["enabled"] = bool(global_template.get("enabled"))
        if isinstance(global_template.get("template"), str):
            resolved["template"] = global_template["template"]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_prefix = plat_cfg.get("response_prefix")
            if isinstance(plat_prefix, str):
                resolved["enabled"] = bool(plat_prefix)
                resolved["template"] = plat_prefix
            elif isinstance(plat_prefix, dict):
                if "enabled" in plat_prefix:
                    resolved["enabled"] = bool(plat_prefix.get("enabled"))
                if isinstance(plat_prefix.get("template"), str):
                    resolved["template"] = plat_prefix["template"]

    return resolved


def interpolate_prefix_template(
    template: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    thinking: Optional[str] = None,
) -> str:
    """Interpolate template variables in a response prefix string.

    Variables are case-insensitive. Unresolved variables remain as literal text.

    Args:
        template: Template string with {variable} placeholders.
        model: Full model ID (e.g. "github-copilot/claude-opus-4.6").
        provider: Provider name (e.g. "github-copilot").
        thinking: Thinking level (e.g. "high", "low", "off").

    Returns:
        Interpolated string.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1).lower()
        if var_name == "model":
            return _model_short(model) or match.group(0)
        elif var_name == "modelfull":
            return model or match.group(0)
        elif var_name == "provider":
            return provider or match.group(0)
        elif var_name in ("thinking", "thinkinglevel", "thinking_level"):
            return thinking or match.group(0)
        else:
            # Leave unrecognized variables as-is
            return match.group(0)

    return _TEMPLATE_VAR_PATTERN.sub(_replace, template)


def build_prefix_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    thinking: Optional[str] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the prefix text (empty string when disabled or no template).
    Callers prepend this to the first response themselves, followed by a space
    or newline for separation.

    Provider resolution (priority order):
    1. ``provider`` argument (from agent_result.get("provider"))
    2. Model prefix (e.g. "github-copilot/claude-opus-4.6" → "github-copilot")
    3. Config ``model.provider`` (from user_config)
    """
    cfg = resolve_prefix_config(user_config, platform_key)
    if not cfg.get("enabled") or not cfg.get("template"):
        return ""

    if not provider:
        # Try deriving from model prefix
        if model:
            parts = model.split("/", 1)
            if len(parts) == 2:
                provider = parts[0]
        # Fall back to config model.provider
        if not provider:
            _mc = (user_config or {}).get("model") or {}
            provider = _mc.get("provider") or ""

    return interpolate_prefix_template(
        cfg["template"],
        model=model,
        provider=provider,
        thinking=thinking,
    )
