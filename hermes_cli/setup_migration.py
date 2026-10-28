"""Post-import section-skip logic: after ``hermes setup`` imported another agent's settings, each
section that is already configured shows its summary and offers to skip (setup.py re-exports the
names it uses; setup helpers are imported lazily so test patches apply)."""

from functools import partial
from typing import Optional


# ── Post-Migration Section Skip Logic ──

_OPENROUTER_ENV_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


def _model_section_has_credentials(config: dict) -> bool:
    """True when any known inference provider has usable credentials: ``active_provider`` in the
    auth store (OAuth providers), ``PROVIDER_REGISTRY`` ``api_key_env_vars``, or the legacy
    OpenRouter aggregator env vars (``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY``)."""
    from hermes_cli.setup import get_env_value
    try:
        from hermes_cli.auth import get_active_provider
        if get_active_provider():
            return True
    except Exception:
        pass
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        PROVIDER_REGISTRY = {}  # type: ignore[assignment]

    def _has_key(pconfig) -> bool:
        # CLAUDE_CODE_OAUTH_TOKEN is set by Claude Code itself, not by the user —
        # mirrors is_provider_explicitly_configured in auth.py.
        return any(get_env_value(v) for v in pconfig.api_key_env_vars if v != "CLAUDE_CODE_OAUTH_TOKEN")

    any_openrouter_key = any(get_env_value(v) for v in _OPENROUTER_ENV_VARS)

    # Prefer the provider declared in config.yaml, avoids false positives from stray
    # env vars (GH_TOKEN, etc.) when the user has already picked a different provider.
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        provider_id = (model_cfg.get("provider") or "").strip().lower()
        if provider_id in PROVIDER_REGISTRY and _has_key(PROVIDER_REGISTRY[provider_id]):
            return True
        if provider_id == "openrouter" and any_openrouter_key:
            return True

    # OpenRouter aggregator fallback (no provider declared in config).
    if any_openrouter_key:
        return True
    # Skip copilot in auto-detect: GH_TOKEN / GITHUB_TOKEN are commonly set for git tooling.
    # Mirrors resolve_provider in auth.py.
    return any(_has_key(pconfig) for pid, pconfig in PROVIDER_REGISTRY.items() if pid != "copilot")


def _model_summary(config: dict) -> Optional[str]:
    if not _model_section_has_credentials(config):
        return None
    model = config.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    if isinstance(model, dict):
        return str(model.get("default") or model.get("model") or "configured")
    return "configured"


def _cfg_summary(config: dict, section: str, key: str, default, prefix: str) -> str:
    from hermes_cli.setup import cfg_get
    return f"{prefix}{cfg_get(config, section, key, default=default)}"


def _gateway_summary(config: dict) -> Optional[str]:
    from hermes_cli.gateway import _all_platforms, _platform_status
    # Any non-empty status other than "not configured" counts — WhatsApp ("enabled, not paired"),
    # Matrix ("configured + E2EE") mean setup already started.
    configured = [
        # Trailing parenthetical qualifiers are stripped from the label.
        plat["label"].split("(", 1)[0].strip() or plat["label"]
        for plat in _all_platforms()
        if _platform_status(plat) and _platform_status(plat) != "not configured"
    ]
    return ", ".join(configured) if configured else None


_TOOL_ENV_LABELS = (
    ("ELEVENLABS_API_KEY", "TTS/ElevenLabs"),
    ("BROWSERBASE_API_KEY", "Browser"),
    ("FIRECRAWL_API_KEY", "Firecrawl"),
)


def _tools_summary(config: dict) -> Optional[str]:
    from hermes_cli.setup import get_env_value
    tools = [label for env_var, label in _TOOL_ENV_LABELS if get_env_value(env_var)]
    return ", ".join(tools) if tools else None


_SECTION_SUMMARIES = {
    "model": _model_summary,
    "terminal": partial(_cfg_summary, section="terminal", key="backend", default="local", prefix="backend: "),
    "agent": partial(_cfg_summary, section="agent", key="max_turns", default=90, prefix="max turns: "),
    "gateway": _gateway_summary,
    "tools": _tools_summary,
}


def _get_section_config_summary(config: dict, section_key: str) -> Optional[str]:
    """Short summary if a setup section is already configured (post-import skip detection), else
    None. ``get_env_value`` is reached through hermes_cli.setup so test patches
    on ``setup_mod.get_env_value`` apply."""
    summarize = _SECTION_SUMMARIES.get(section_key)
    return summarize(config) if summarize else None


def _skip_configured_section(config: dict, section_key: str, label: str) -> bool:
    """Show an already-configured section summary and offer to skip; True when the user skips."""
    from hermes_cli.setup import print_success, prompt_yes_no
    summary = _get_section_config_summary(config, section_key)
    if not summary:
        return False
    print()
    print_success(f"  {label}: {summary}")
    return not prompt_yes_no(f"  Reconfigure {label.lower()}?", default=False)
