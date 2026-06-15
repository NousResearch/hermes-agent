"""Gateway runtime-metadata footer and compact runtime prefix.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide
      runtime_prefix:
        enabled: true                       # prepend compact model tag
        labels:
          gpt-5.5: "[gpt5.5]"

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``
and ``display.platforms.<platform>.runtime_prefix``.  Users can toggle the
footer with ``/footer on|off`` from both the CLI and any gateway platform.

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
_DEFAULT_PREFIX_LABELS: dict[str, str] = {
    "grok-composer": "[grok]",
    "grok": "[grok]",
    "gpt-5.5": "[gpt5.5]",
    "gpt": "[gpt]",
    "glm-5.1": "[glm]",
    "glm-5.2": "[glm2]",
    "glm": "[glm]",
}
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home) :]
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


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
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
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
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
        cwd=cwd,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )


def resolve_prefix_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-prefix config for *platform_key*.

    The prefix is intentionally separate from ``runtime_footer``: it provides a
    small model tag at the start of visible gateway replies without adding cwd
    or context details. Merge order mirrors ``resolve_footer_config``.
    """
    resolved: dict[str, Any] = {
        "enabled": False,
        "labels": dict(_DEFAULT_PREFIX_LABELS),
    }
    cfg = (user_config or {}).get("display") or {}

    def _merge_labels(prefix_cfg: dict[str, Any]) -> None:
        labels_cfg = (
            prefix_cfg.get("labels")
            or prefix_cfg.get("map")
            or prefix_cfg.get("markers")
        )
        if not isinstance(labels_cfg, dict):
            return
        labels = dict(resolved["labels"])
        for key, value in labels_cfg.items():
            label_key = str(key).strip()
            label_value = str(value).strip()
            if label_key and label_value:
                labels[label_key] = label_value
        resolved["labels"] = labels

    global_cfg = cfg.get("runtime_prefix")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        _merge_labels(global_cfg)

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_prefix = plat_cfg.get("runtime_prefix")
            if isinstance(plat_prefix, dict):
                if "enabled" in plat_prefix:
                    resolved["enabled"] = bool(plat_prefix.get("enabled"))
                _merge_labels(plat_prefix)

    return resolved


def _label_for_model(model: Optional[str], labels: dict[str, Any] | None = None) -> str:
    """Return the configured compact label for *model*, or a safe default."""
    short = _model_short(model)
    if not short:
        return ""

    labels = labels or {}
    lowered = {str(k).lower(): str(v) for k, v in labels.items() if str(v)}
    for candidate in (str(model or ""), short):
        val = lowered.get(candidate.lower())
        if val:
            return val

    # Fallback to longest prefix match so a broad label like ``gpt`` can cover
    # ``gpt-5.5`` while exact labels (checked above) still win.
    short_l = short.lower()
    for key in sorted(lowered, key=len, reverse=True):
        if short_l.startswith(key):
            return lowered[key]
    return f"[{short}]"


def build_runtime_prefix(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
) -> str:
    """Return compact model prefix, e.g. ``[gpt5.5]``, or empty if disabled."""
    cfg = resolve_prefix_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return _label_for_model(model, cfg.get("labels") or {})


def apply_runtime_prefix(response: str, prefix: str) -> str:
    """Prepend *prefix* once to a final gateway response."""
    if not response or not prefix:
        return response
    stripped = response.lstrip()
    if stripped.startswith(prefix):
        return response
    leading = response[: len(response) - len(stripped)]
    return f"{leading}{prefix} {stripped}"


def format_runtime_prefix(
    *,
    model: Optional[str],
    labels: dict[str, str] | None = None,
) -> str:
    """Compatibility helper returning the compact marker for *model*."""
    return _label_for_model(model, labels or _DEFAULT_PREFIX_LABELS)


def build_prefix_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
) -> str:
    """Compatibility alias for the first-line runtime prefix entry point."""
    return build_runtime_prefix(
        user_config=user_config,
        platform_key=platform_key,
        model=model,
    )
