"""Dashboard config schema and model-assignment logic: CONFIG_SCHEMA construction, dynamic provider options, web<->config normalisation, main/aux model assignment.
"""

import logging
import os
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.config import (
    DEFAULT_CONFIG,
    build_cron_model_impact,
    cfg_get,
    clear_model_endpoint_credentials,
    find_provider_entry,
    read_raw_config,
    resolve_cron_model_drift_defaults,
)
from hermes_cli.web_server_memory import _normalize_memory_provider_name

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

def _memory_provider_options() -> List[str]:
    """Discovered memory providers for the ``memory.provider`` select.

    Directory-scan only (no provider imports), so safe at module import time. ``""``
    (built-in only) is always first; discovery failures degrade to the bundled defaults.
    The literal ``builtin`` alias is deliberately NOT offered — built-in memory is not a
    provider plugin; ``_normalize_memory_provider_name`` maps legacy aliases back to ``""``.

    See #49513.
    """
    options = [""]
    try:
        from plugins.memory import list_memory_provider_names

        options.extend(list_memory_provider_names())
    except Exception:
        options.extend(["honcho"])
    return list(dict.fromkeys(options))


def _timezone_options() -> List[str]:
    """Return sorted IANA timezone identifiers, cached at import time."""
    try:
        import zoneinfo
        return sorted(zoneinfo.available_timezones()) or ["UTC"]
    except Exception:  # pragma: no cover
        return ["UTC"]


def _select(description: str, *options: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "select", "description": description, "options": list(options), **extra}


# Manual overrides for fields that need select options or custom types.
_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "timezone": _select(
        "IANA timezone (e.g. America/New_York). Blank uses the system timezone.",
        *_timezone_options(), searchable=True, clearable=True,
    ),
    "memory.provider": _select("Memory provider plugin", *_memory_provider_options()),
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
        "category": "general",
    },
    "terminal.backend": _select(
        "Terminal execution backend",
        "local", "docker", "ssh", "modal", "daytona", "vercel_sandbox", "singularity",
    ),
    # sync with _SUPPORTED_VERCEL_RUNTIMES in terminal_tool.py
    "terminal.vercel_runtime": _select("Vercel Sandbox runtime", "node24", "node22", "python3.13"),
    "terminal.modal_mode": _select("Modal sandbox mode", "sandbox", "function"),
    "proxy.enabled": {
        "type": "boolean",
        "description": (
            "Docker-only egress credential firewall. Requires `hermes egress setup` "
            "and `hermes egress start`; Modal/SSH/Daytona are not wired yet."
        ),
        "category": "security",
    },
    "proxy.credential_source": _select(
        "Where iron-proxy loads real upstream secrets at start time", "env", "bitwarden", category="security"
    ),
    "proxy.enforce_on_docker": {
        "type": "boolean",
        "description": "Refuse Docker sandboxes when egress is enabled but not configured/running",
        "category": "security",
    },
    "tts.provider": _select(
        "Text-to-speech provider",
        "edge", "elevenlabs", "openai", "xai", "minimax", "mistral", "gemini", "neutts", "kittentts", "piper",
    ),
    # "mistral" temporarily removed — mistralai PyPI package quarantined
    # (malicious 2.4.6 release on 2026-05-12). Restore once available.
    "stt.provider": _select("Speech-to-text provider", "local", "groq", "openai", "xai", "elevenlabs"),
    "stt.local.model": _select("Local faster-whisper model size", "tiny", "base", "small", "medium", "large-v3"),
    "stt.groq.model": _select(
        "Groq Whisper model", "whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"
    ),
    "stt.openai.model": _select(
        "OpenAI transcription model", "whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"
    ),
    "stt.elevenlabs.model_id": _select("ElevenLabs Scribe model", "scribe_v2", "scribe_v1"),
    "display.skin": _select("CLI visual theme", "default", "ares", "mono", "slate"),
    "dashboard.theme": _select(
        "Web dashboard visual theme", "default", "midnight", "ember", "mono", "cyberpunk", "rose"
    ),
    "display.resume_display": _select("How resumed sessions display history", "minimal", "full", "off"),
    "display.busy_input_mode": _select("Input behavior while agent is running", "interrupt", "queue", "steer"),
    "approvals.mode": _select("Dangerous command approval mode", "manual", "smart", "off"),
    "context.engine": _select("Context management engine", "default", "custom"),
    "human_delay.mode": _select("Simulated typing delay mode", "off", "typing", "fixed"),
    "logging.level": _select("Log level for agent.log", "DEBUG", "INFO", "WARNING", "ERROR"),
    "agent.service_tier": _select(
        "Fast mode: fast = always, auto = first N seconds of each turn, cold = first turn only",
        "", "normal", "fast", "auto", "cold",
    ),
    "delegation.reasoning_effort": _select(
        "Reasoning effort for delegated subagents",
        "", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
    ),
    "updates.non_interactive_local_changes": _select(
        "When the chat app / gateway updates Hermes (no terminal prompt), "
        "what to do with uncommitted local source edits. 'stash' keeps them "
        "and re-applies them after the update; 'discard' throws them away. "
        "Terminal updates always ask, regardless of this setting.",
        "stash", "discard",
    ),
    "updates.refresh_cua_driver": {
        "type": "boolean",
        "description": (
            "Refresh an already-installed cua-driver during hermes update. "
            "Disable this on non-admin macOS accounts where /Applications is "
            "not writable."
        ),
    },
    "browser.headed": {
        "type": "boolean",
        "description": "Run the local browser in headed mode (visible window). Also keeps the window open between turns; idle sessions are still reaped after browser.inactivity_timeout.",
    },
    "plugins.hook_callback_timeout": {
        "type": "number",
        "description": (
            "Wall-clock cap (seconds) for timeout-bounded in-process Python "
            "plugin hook callbacks (hot-path observers + pre_tool_call). "
            "Timed-out pre_tool_call fails closed. 0 disables the cap; "
            "values above 600 are clamped. Caller-thread hooks such as "
            "subagent_stop are never moved onto a timeout worker."
        ),
    },
}

# Small categories fold into a bigger tab to avoid one-field orphan tabs. Several sources
# (models_dev, onboarding, mcp, computer_use, telemetry, plugins, doctor, runtime, session,
# nous, telegram) currently surface a single schema field each.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    "models_dev": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "dashboard": "display",
    "code_execution": "agent",
    "prompt_caching": "agent",
    "bot_mode": "agent",
    "goals": "agent",
    "updates": "general",
    "onboarding": "agent",
    "telegram": "discord",
    "mcp": "agent",
    "computer_use": "agent",
    "telemetry": "security",
    "plugins": "agent",
    "doctor": "general",
    # `runtime.nofile_soft_limit` (#78873) is the only schema-surfaced runtime field — fold it into the
    # agent tab rather than spawning a one-field orphan category.
    "runtime": "agent",
    "session": "general",
    "nous": "agent",
}


_UI_TYPES = ((bool, "boolean"), (int, "number"), (float, "number"), (list, "list"), (dict, "object"))


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    return next((ui for py, ui in _UI_TYPES if isinstance(value, py)), "string")


def _build_schema_from_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if full_key == "_config_version":
            continue
        if isinstance(value, dict):
            schema.update(_build_schema_from_config(value, full_key))
            continue
        # Category: first path component for nested keys, "general" for top-level scalars.
        entry: Dict[str, Any] = {
            "type": _infer_type(value),
            "description": full_key.replace(".", " → ").replace("_", " ").title(),
            "category": prefix.split(".")[0] if prefix else "general",
        }
        entry.update(_SCHEMA_OVERRIDES.get(full_key, {}))
        entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
        schema[full_key] = entry
    return schema


def _config_schema_with_virtual_fields() -> Dict[str, Dict[str, Any]]:
    """DEFAULT_CONFIG schema plus the virtual ``model_context_length`` field, inserted right
    after ``model`` so it renders adjacent in the frontend."""
    ordered: Dict[str, Dict[str, Any]] = {}
    for key, entry in _build_schema_from_config(DEFAULT_CONFIG).items():
        ordered[key] = entry
        if key == "model":
            ordered["model_context_length"] = _SCHEMA_OVERRIDES["model_context_length"]
    return ordered


CONFIG_SCHEMA = _config_schema_with_virtual_fields()


def _is_command_provider_block(value: Any) -> bool:
    """True when *value* declares a command-type voice provider.

    Mirrors the runtime discriminators (``tools.tts_command_provider._is_command_provider_config`` /
    ``tools.transcription_command._is_command_stt_provider_config``) and the desktop's
    ``isCommandProvider``: ``type`` is OPTIONAL and case/space-insensitive (absent or
    normalizing to ``"command"``); ``command`` MUST be a non-empty string.
    """
    if not isinstance(value, dict):
        return False
    ptype = str(value.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = value.get("command")
    return isinstance(command, str) and bool(command.strip())


def _custom_provider_options(kind: str, builtin_names: List[str], cfg: Dict[str, Any]) -> List[str]:
    """Merged ``tts``/``stt`` provider options without hard-coding vendor names.

    Built-in display names first (original order), then, deduped case-insensitively:
    1. Command-type providers from canonical ``<kind>.providers.<name>`` and the legacy
       top-level ``<kind>.<name>`` — the runtime's dual resolution order. Names colliding
       with a RUNTIME built-in are excluded (the runtime rejects them before config lookup);
       the runtime sets are used rather than the display shortlist, which drifts.
    2. Plugin-registered names from the tts/transcription registries — opportunistic: this
       process may never call ``discover_plugins()``, so the registry may be empty.
    3. The current ``<kind>.provider`` value, so a custom active name stays selectable.
    Guard semantics mirror the desktop's ``commandProviderNames`` so both surfaces agree.
    """
    names = [str(n) for n in builtin_names]
    seen = {n.strip().lower() for n in names}
    if kind == "tts":
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS as _runtime_builtins
    else:
        from tools.transcription_common import BUILTIN_STT_PROVIDERS as _runtime_builtins

    def _add(name: Any) -> None:
        stripped = name.strip() if isinstance(name, str) else ""
        if stripped and stripped.lower() not in seen:
            names.append(stripped)
            seen.add(stripped.lower())

    section = cfg.get(kind)
    if not isinstance(section, dict):
        section = {}
    providers_map = section.get("providers")
    candidate_blocks: List[Any] = [providers_map] if isinstance(providers_map, dict) else []
    candidate_blocks.append({k: v for k, v in section.items() if k != "providers"})
    for block in candidate_blocks:
        for name, value in block.items():
            if (
                isinstance(name, str)
                and name.strip().lower() not in _runtime_builtins
                and _is_command_provider_block(value)
            ):
                _add(name)

    try:
        if kind == "tts":
            from agent.tts_registry import list_providers as _list_voice_providers
        else:
            from agent.transcription_registry import list_providers as _list_voice_providers
        for _p in _list_voice_providers():
            _add(getattr(_p, "name", None))
    except Exception:  # pragma: no cover - registry import should not break schema
        pass

    # ``cfg_get`` takes *keys*, not dotted paths.
    _add(cfg_get(cfg, kind, "provider"))
    return names


def _memory_provider_schema_options(cfg: Dict[str, Any]) -> List[str]:
    """Discovered memory providers plus the currently-configured one, so a value that is no
    longer discoverable (e.g. plugin removed from disk) never vanishes from the dropdown."""
    options = _memory_provider_options()
    memory = cfg.get("memory")
    current = _normalize_memory_provider_name(memory.get("provider") if isinstance(memory, dict) else None)
    if current and current not in options:
        options = [*options, current]
    return options


def _schema_select_options(key: str) -> Optional[List[str]]:
    entry = CONFIG_SCHEMA.get(key)
    options = entry.get("options") if isinstance(entry, dict) else None
    return options if isinstance(options, list) else None


def _schema_with_dynamic_provider_options() -> Dict[str, Dict[str, Any]]:
    """CONFIG_SCHEMA with per-request discovery-driven ``*.provider`` options merged.

    ``_SCHEMA_OVERRIDES`` freezes option lists at import time, so a provider installed after
    the server started never appears. Recomputing at request time reflects the CURRENT
    (possibly profile-scoped) config.yaml and mid-session plugin installs for every surface
    that reads the schema. ``CONFIG_SCHEMA`` is never mutated; changed entries are
    shallow-copied onto a copied mapping.
    """
    from hermes_cli.web_server_profiles import _plugin_terminal_backend_rows
    from hermes_cli.config import load_config
    try:
        cfg = load_config()
    except Exception:  # pragma: no cover - schema must survive config errors
        return CONFIG_SCHEMA

    overlay: Dict[str, Dict[str, Any]] = {}

    def merge(key: str, options: List[str]) -> None:
        if _schema_select_options(key) is not None and options != CONFIG_SCHEMA[key]["options"]:
            overlay[key] = {**CONFIG_SCHEMA[key], "options": options}

    for kind in ("tts", "stt"):
        existing = _schema_select_options(f"{kind}.provider")
        if existing is not None:
            merge(f"{kind}.provider", _custom_provider_options(kind, list(existing), cfg))

    merge("memory.provider", _memory_provider_schema_options(cfg))

    tb_options = _schema_select_options("terminal.backend")
    if tb_options is not None:
        try:
            plugin_names = sorted({row["name"] for row in _plugin_terminal_backend_rows()} - set(tb_options))
        except Exception:
            plugin_names = []
        if plugin_names:
            merge("terminal.backend", [*tb_options, *plugin_names])

    return {**CONFIG_SCHEMA, **overlay} if overlay else CONFIG_SCHEMA


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The per-card "Use as → Main model" menu can send the model's VENDOR prefix as the
    provider (analytics rows with no ``billing_provider``), producing e.g.
    ``provider: anthropic`` + ``default: anthropic/claude-opus-4.6`` — an aggregator slug on
    the native provider, which 400s. Two repairs at this single chokepoint:

    1. Vendor-name → Hermes-provider: when the provider is not a known provider/alias but the
       model is a vendor-prefixed slug, keep the user's CURRENT aggregator if on one, else
       openrouter. User-declared ``providers:``/``custom_providers:`` entries resolve first,
       and durable named-custom slugs (``custom`` / ``custom:<name>``) are excluded —
       ``_KNOWN_PROVIDER_NAMES`` lists only the bare ``custom`` bucket, so without this a
       LiteLLM proxy serving ``ollama/glm-5.2`` would be silently reassigned to openrouter.
       Matching only that syntax (not ``startswith("custom")``) avoids swallowing
       unconfigured vendors like ``customproxy``.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (custom/user providers keep the model verbatim).
    """
    from hermes_cli.config import load_config
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.models import _AGGREGATOR_PROVIDERS, _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.providers import resolve_custom_provider, resolve_user_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    user_providers = cfg.get("providers")
    declared = resolve_user_provider(
        prov_in, user_providers if isinstance(user_providers, dict) else {}
    ) or resolve_custom_provider(prov_in, get_compatible_custom_providers(cfg))
    if declared is not None:
        return declared.id, model_in

    is_custom_provider_slug = canonical == "custom" or canonical.startswith("custom:")
    if canonical not in _KNOWN_PROVIDER_NAMES and not is_custom_provider_slug and "/" in model_in:
        try:
            cur_cfg = cfg.get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower() if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = prov_in = "openrouter"

    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            model_in = normalize_model_for_provider(model_in, canonical) or model_in
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles endpoint fields. ``base_url`` and the
    endpoint key share one lifecycle: an explicit value is always persisted; an existing
    value is cleared ONLY when switching to a *different* provider (it belonged to the old
    endpoint); a same-provider re-pick preserves it — re-picking a model used to wipe a
    user's custom host (e.g. a Xiaomi MiMo Token Plan URL) and break their keys. The
    runtime resolver reads ``model.base_url`` from config and only honors it when the
    configured provider matches, so preserving it here is what lets the override route.
    A stale secret may live under the legacy ``api`` alias with no ``api_key``, so the
    switch-clears-the-key path triggers on either field. ``context_length`` is always
    dropped (the new model may have a different window).

    Returns the same dict (a fresh dict if the input wasn't one).
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    new_provider = provider.strip().lower()
    switched = new_provider != prev_provider
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and switched:
        model_cfg["base_url"] = ""
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif (model_cfg.get("api_key") or model_cfg.get("api")) and switched:
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if switched:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a dict-form ``model`` to its string form (the schema is built from
    DEFAULT_CONFIG where ``model`` is a string) and surface ``model_context_length``
    as a top-level field (0 = auto-detect)."""
    config = dict(config)
    model_val = config.get("model")
    if isinstance(model_val, dict):
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
    return config


# ---------------------------------------------------------------------------
# Model assignment — main slot or auxiliary slots. Mirrors the model.options
# JSON-RPC from tui_gateway but over REST so the Models page can drive it.
# ---------------------------------------------------------------------------

# Canonical auxiliary task slots. Keep in sync with DEFAULT_CONFIG["auxiliary"]
# in hermes_cli/config.py — listed here for deterministic ordering in the UI.
_AUX_TASK_SLOTS: Tuple[str, ...] = (
    "vision", "compression", "skills_hub", "approval", "mcp", "title_generation", "review",
    "triage_specifier", "kanban_decomposer", "profile_describer", "curator",
)


def _dashboard_code_skew_guard() -> Optional[str]:
    """Return a "restart required" message when this process runs stale code, else None.

    Long-lived dashboard / Desktop-owned ``hermes serve`` processes freeze ``sys.modules``
    at boot; after ``hermes update`` replaces the checkout, a first-time lazy import can
    resolve a fresh consumer module against a stale cached dependency -> ImportError.
    Mirrors the gateway's ``_model_switch_skew_guard``: refuse the risky call with an
    actionable message. Never a false positive (non-git installs return None).

    ``/api/model/options`` 500 after the update added ``agent.model_metadata.is_grok_46_family`` while the
    running process kept serving the pre-update module (#86207).
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return (
        f"This process is running code from {boot_rev} but the checkout on "
        f"disk is now {disk_rev}. The model picker would risk a stale-module "
        f"crash — {_dashboard_skew_restart_hint()}"
    )


def _dashboard_skew_restart_hint() -> str:
    """Restart advice matching how this process is owned — the same app backs the browser
    dashboard and Desktop-owned ``hermes serve``; naming a systemd unit would mislead
    macOS/launchd hosts and Desktop SSH backends.

    See #97046.
    """
    if os.environ.get("HERMES_SERVE_HEADLESS") == "1":
        return (
            "restart the Desktop-owned backend to load the new code "
            "(use Restart backend in Hermes Desktop, or quit and reopen the app)"
        )
    return (
        "restart this Hermes process to load the new code "
        "(hermes dashboard --port <port>, or the equivalent service restart for this install)"
    )


def _resolve_assignment_credentials(model_cfg: dict, provider: str, provider_entry: Any) -> None:
    """Carry the provider's credential POINTER (``key_env`` / raw ``${VAR}``) onto ``model_cfg``.

    ``provider_entry`` comes from ``load_config()``, which expands ``${VAR}`` to plaintext;
    copying that into ``model.api_key`` would write the SECRET into config.yaml (and recreate
    it on every re-apply). Prefer the raw template; fall back to the expanded value only when
    the raw yaml itself stores the key as a literal (no new exposure).
    """
    try:
        _stored, raw_entry = find_provider_entry(read_raw_config().get("providers"), provider)
    except Exception:
        raw_entry = None
    if not isinstance(raw_entry, dict):
        raw_entry = {}
    key_env = str(raw_entry.get("key_env") or "").strip()
    if key_env:
        model_cfg["key_env"] = key_env
        # #88990: carry the credential POINTER, never a resolved secret.
        model_cfg.pop("api_key", None)
    elif isinstance(provider_entry, dict) and provider_entry.get("api_key"):
        raw_key = str(raw_entry.get("api_key") or "").strip()
        model_cfg["api_key"] = raw_key if raw_key.startswith("${") and raw_key.endswith("}") else provider_entry["api_key"]


def _apply_nous_gateway_defaults(cfg: dict) -> list:
    """Mirror the CLI's post-model-selection behaviour when switching main to Nous: route
    *unconfigured* tools through the Nous Tool Gateway. Purely additive — tools with a direct
    key or explicit backend are skipped. Failures never block saving the assignment."""
    try:
        from hermes_cli.nous_subscription import apply_nous_managed_defaults
        from hermes_cli.tools_config import _get_platform_tools

        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
        return sorted(apply_nous_managed_defaults(cfg, enabled_toolsets=enabled, force_fresh=True))
    except Exception:
        _log.debug("apply_nous_managed_defaults skipped", exc_info=True)
        return []


def _register_custom_endpoint(base_url: str, api_key: str, model: str) -> None:
    """Register a named ``custom_providers`` entry for a custom/local endpoint (mirrors the
    ``hermes model`` custom flow) so the picker gets a proper ready row instead of a "needs
    setup" dead-end. Dedups by base_url; never blocks the already-persisted assignment."""
    try:
        from hermes_cli.main_provider_setup import _auto_provider_name, _save_custom_provider

        _save_custom_provider(base_url, api_key, model, name=_auto_provider_name(base_url))
    except Exception:
        _log.debug("custom_providers registration skipped", exc_info=True)


def _stale_aux_pins(cfg: dict, new_provider: str) -> list:
    """Aux slots still pinned to a *different* provider than the new main one.

    Switching main never touches aux pins (independent, sticky per-task overrides) — a user
    leaving a now-unpaid provider keeps paying 402s on background calls until they reset
    them. We never auto-clear (pinning aux is legitimate) but report them so the UI can
    offer a "reset to main" nudge.
    """
    stale_aux: list[dict] = []
    aux_cfg = cfg.get("auxiliary", {})
    if not isinstance(aux_cfg, dict):
        return stale_aux
    for slot in _AUX_TASK_SLOTS:
        slot_cfg = aux_cfg.get(slot)
        if not isinstance(slot_cfg, dict):
            continue
        slot_provider = str(slot_cfg.get("provider", "") or "").strip()
        if slot_provider and slot_provider.lower() not in {"auto", ""} and slot_provider.lower() != new_provider:
            stale_aux.append({
                "task": slot, "provider": slot_provider, "model": str(slot_cfg.get("model", "") or ""),
            })
    return stale_aux


def _cron_model_impact(cfg: dict, provider: str, model: str) -> Any:
    from hermes_cli.config import load_config
    try:
        effective_config = load_config()
        effective_provider, effective_model = resolve_cron_model_drift_defaults(effective_config)
        return build_cron_model_impact(
            current_provider=effective_provider or provider,
            current_model=effective_model or model,
            config=effective_config,
        )
    except Exception:
        _log.debug("cron model impact inspection failed", exc_info=True)
        return build_cron_model_impact(config=cfg, jobs={})


def _apply_main_assignment_sync(cfg: dict, provider: str, model: str, base_url: str, api_key: str) -> dict:
    from hermes_cli.config import save_config
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model required for main")
    provider, model = _normalize_main_model_assignment(provider, model)
    providers_cfg = cfg.get("providers")
    provider_entry = providers_cfg.get(provider) if isinstance(providers_cfg, dict) else None
    if not base_url and isinstance(provider_entry, dict) and provider_entry.get("base_url"):
        base_url = str(provider_entry.get("base_url") or "").strip()
    model_cfg = _apply_main_model_assignment(cfg.get("model", {}), provider, model, base_url, api_key)
    _resolve_assignment_credentials(model_cfg, provider, provider_entry)
    cfg["model"] = model_cfg

    new_provider = provider.strip().lower()
    gateway_tools = _apply_nous_gateway_defaults(cfg) if new_provider == "nous" else []
    save_config(cfg)
    if new_provider in {"custom", "local"} and base_url:
        _register_custom_endpoint(base_url, api_key, model)

    return {
        "ok": True,
        "scope": "main",
        "provider": provider,
        "model": model,
        "base_url": model_cfg.get("base_url", ""),
        "gateway_tools": gateway_tools,
        "stale_aux": _stale_aux_pins(cfg, new_provider),
        "cron_model_impact": _cron_model_impact(cfg, provider, model),
    }


def _apply_aux_assignment_sync(cfg: dict, provider: str, model: str, task: str, base_url: str, api_key: str) -> dict:
    from hermes_cli.config import save_config
    aux = cfg.get("auxiliary")
    if not isinstance(aux, dict):
        aux = {}

    def _slot(slot: str) -> dict:
        slot_cfg = aux.get(slot)
        return slot_cfg if isinstance(slot_cfg, dict) else {}

    if task == "__reset__":
        # Reset every slot to provider="auto", model="" — keeps other fields intact.
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = _slot(slot)
            slot_cfg["provider"] = "auto"
            slot_cfg["model"] = ""
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
            aux[slot] = slot_cfg
        cfg["auxiliary"] = aux
        save_config(cfg)
        return {"ok": True, "scope": "auxiliary", "reset": True}

    if not provider:
        raise HTTPException(status_code=400, detail="provider required for auxiliary")

    targets = [task] if task else list(_AUX_TASK_SLOTS)
    new_provider = provider.strip().lower()
    for slot in targets:
        if slot not in _AUX_TASK_SLOTS:
            raise HTTPException(status_code=400, detail=f"unknown auxiliary task: {slot}")
        slot_cfg = _slot(slot)
        prev_provider = str(slot_cfg.get("provider") or "").strip().lower()
        slot_cfg["provider"] = provider
        slot_cfg["model"] = model
        if base_url:
            # Sibling of the main-slot endpoint handling: an aux assignment for a custom/local
            # endpoint must carry its own base_url/api_key (the auxiliary resolver reads
            # auxiliary.<task>.base_url/api_key), or it silently rebinds to model.base_url and
            # breaks once the main slot switches away.
            # The auxiliary resolver already reads auxiliary.<task>.base_url/api_key
            # (_resolve_task_provider_model), so persisting them here is what actually wires the endpoint
            # in. See #65254.
            slot_cfg["base_url"] = base_url
            if api_key:
                slot_cfg["api_key"] = api_key
        elif new_provider != prev_provider and new_provider != "custom":
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
        aux[slot] = slot_cfg

    cfg["auxiliary"] = aux
    save_config(cfg)
    return {"ok": True, "scope": "auxiliary", "tasks": targets, "provider": provider, "model": model}


def _apply_model_assignment_sync(
    scope: str, provider: str, model: str, task: str, base_url: str, api_key: str = ""
):
    """Synchronous body of POST /api/model/set.

    Runs inside ``_profile_scope`` (worker thread) so every load_config/save_config lands in
    the requested profile. Raises HTTPException for validation errors.
    """
    from hermes_cli.config import load_config
    cfg = load_config()
    if scope == "main":
        return _apply_main_assignment_sync(cfg, provider, model, base_url, api_key)
    return _apply_aux_assignment_sync(cfg, provider, model, task, base_url, api_key)


def _infer_provider_on_model_change(model_val: str, prev_provider: str) -> tuple[str, str]:
    """Infer which provider serves ``model_val`` when the flat Config-page Model field changes.

    Returns ``(provider, model)``; ``provider`` is empty when no switch is warranted. Signals,
    in order: curated-catalog detection (``detect_provider_for_model``), then the vendor-slug
    heuristic — a ``vendor/model`` slug cannot belong to a non-aggregator provider (e.g.
    ``ollama-local``), so return the sentinel ``"openrouter"``; the caller's
    ``_normalize_main_model_assignment`` resolves the real aggregator (keeps the current one).
    """
    name = (model_val or "").strip()
    if not name:
        return "", name
    try:
        from hermes_cli.models import _AGGREGATOR_PROVIDERS, detect_provider_for_model, normalize_provider
    except Exception:
        return "", name

    try:
        detected = detect_provider_for_model(name, prev_provider)
    except Exception:
        detected = None
    if detected:
        return detected[0], detected[1]

    if "/" in name:
        try:
            cur_is_aggregator = normalize_provider(prev_provider) in _AGGREGATOR_PROVIDERS
        except Exception:
            cur_is_aggregator = False
        if not cur_is_aggregator:
            return "openrouter", name
    return "", name


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse ``_normalize_config_for_web`` before saving.

    Reconstructs ``model`` as a dict from the on-disk config to recover subkeys (provider,
    base_url, api_mode, ...) the GET response stripped. When the model name actually changed,
    re-detects the serving provider and routes through the assignment chokepoints (a user
    picking an OpenRouter model while on ``ollama-local`` would otherwise keep the stale
    provider and 404); saving unrelated fields never overwrites an explicit provider.

    ``model_context_length`` is written back as ``context_length`` (0 = auto-detect, key
    removed). A partial update (Settings autosave diff) that OMITS the key means "unchanged"
    and must leave the on-disk override alone — not be treated as an explicit 0.
    """
    from hermes_cli.config import load_config
    config = dict(config)
    config.pop("_model_meta", None)

    ctx_sent = "model_context_length" in config
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    has_model = isinstance(model_val, str) and bool(model_val)
    if not (has_model or ctx_sent):
        return config
    try:
        disk_model = load_config().get("model")
        if isinstance(disk_model, dict):
            if has_model:
                prev_default = str(disk_model.get("default") or "").strip()
                prev_provider = str(disk_model.get("provider") or "").strip()
                if model_val != prev_default and prev_provider:
                    new_provider, resolved_model = _infer_provider_on_model_change(model_val, prev_provider)
                    if new_provider and new_provider.strip().lower() != prev_provider.lower():
                        norm_provider, norm_model = _normalize_main_model_assignment(new_provider, resolved_model)
                        disk_model = _apply_main_model_assignment(disk_model, norm_provider, norm_model)
                        model_val = norm_model
                disk_model["default"] = model_val
            if ctx_sent:
                if ctx_override > 0:
                    disk_model["context_length"] = ctx_override
                else:
                    disk_model.pop("context_length", None)
            config["model"] = disk_model
        elif ctx_sent and ctx_override > 0:
            # Model was a bare string (or absent) — upgrade to a dict for the override.
            if has_model:
                default = model_val
            elif isinstance(disk_model, str) and disk_model:
                default = disk_model
            else:
                default = ""
            config["model"] = {"default": default, "context_length": ctx_override}
    except Exception:
        pass  # can't read disk config — just use the string form
    return config
