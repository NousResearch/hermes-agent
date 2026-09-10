"""Configuration-file checks for hermes doctor: .env, config.yaml validation, drift, deprecations.
Split out of ``hermes_cli/doctor.py``."""

from __future__ import annotations

import os
import shutil
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, _section, check_bool, check_fail, check_info, check_ok, check_warn, doctor_check,
    warn_on_error,
)


def _has_provider_env_config(content: str) -> bool:
    """Return True when ~/.hermes/.env contains provider auth/base URL settings."""
    from hermes_cli.doctor import _PROVIDER_ENV_HINTS
    return any(key in content for key in _PROVIDER_ENV_HINTS)


# Legacy config keys still read for back-compat: warn-only with the modern replacement, never auto-migrated
# (migrations live in config.py). (section, key, replacement)
_DEPRECATED_CONFIG_KEYS: tuple[tuple[str, str, str], ...] = (
    ("display", "tool_progress_overrides", "display.platforms"),
    ("delegation", "max_async_children", "delegation.max_concurrent_children"),
    ("compression", "summary_model", "auxiliary.compression"),
    ("compression", "summary_provider", "auxiliary.compression"), ("compression", "summary_base_url", "auxiliary.compression"),
)


# Deprecated env vars (checked in the .env FILE, not process env, so config→env bridges like terminal.cwd →
# TERMINAL_CWD do not false-positive). HERMES_TOOL_PROGRESS is silently ignored since the v12 config floor
# removed its only consumer; HERMES_TOOL_PROGRESS_MODE is still read by the gateway as a back-compat fallback.
_DEPRECATED_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("HERMES_TOOL_PROGRESS", "display.tool_progress in config.yaml — ignored/unsupported since config floor v12"),
    ("HERMES_TOOL_PROGRESS_MODE", "display.tool_progress in config.yaml"),
    ("TERMINAL_CWD", "terminal.cwd in config.yaml"), ("MESSAGING_CWD", "terminal.cwd in config.yaml"),
    ("QQ_HOME_CHANNEL", "QQBOT_HOME_CHANNEL"), ("QQ_HOME_CHANNEL_NAME", "QQBOT_HOME_CHANNEL_NAME"),
)


def collect_deprecated_config_keys(raw_config: dict | None) -> list[tuple[str, str]]:
    """``(legacy_path, replacement)`` for deprecated keys in the on-disk YAML (empty containers still count)."""
    if not isinstance(raw_config, dict):
        return []
    return [(f"{section}.{key}", replacement) for section, key, replacement in _DEPRECATED_CONFIG_KEYS
            if isinstance(raw_config.get(section), dict) and key in raw_config[section]]


def collect_deprecated_env_vars(env_map: dict | None) -> list[tuple[str, str]]:
    """``(legacy_env, replacement)`` for deprecated vars in *env_map* (the on-disk ``.env``, not ``os.environ``,
    so bridged runtime vars do not false-positive)."""
    if not isinstance(env_map, dict):
        return []
    return [(name, replacement) for name, replacement in _DEPRECATED_ENV_VARS
            if env_map.get(name) is not None and str(env_map[name]).strip() != ""]


def collect_relay_plugin_cutover_findings(raw_config: dict | None, env_map: dict | None) -> list[tuple[str, str]]:
    """Return actionable findings for the removed Hermes Relay plugin."""
    from hermes_cli.relay_plugin_cutover import (LEGACY_RELAY_EXPORT_ENV_VARS, RELAY_PLUGINS_CONFIG_ENV,
                                                 configured_legacy_relay_env_vars, legacy_relay_plugin_keys)
    findings: list[tuple[str, str]] = []
    plugins = raw_config.get("plugins") if isinstance(raw_config, dict) else None
    if isinstance(plugins, dict):
        findings += [(f"plugins.enabled: {key}", f"remove it and configure {RELAY_PLUGINS_CONFIG_ENV}")
                     for key in legacy_relay_plugin_keys(plugins.get("enabled"))]
    effective_env = dict(env_map or {})
    # Fall through to process env ONLY when no explicit env_map was given: run_doctor passes None and wants
    # live-process vars, but an explicit map describes a complete environment (merging os.environ breaks hermeticity).
    if env_map is None:
        for name in (*LEGACY_RELAY_EXPORT_ENV_VARS, RELAY_PLUGINS_CONFIG_ENV):
            if name not in effective_env and os.environ.get(name) is not None:
                effective_env[name] = os.environ[name]
    if not str(effective_env.get(RELAY_PLUGINS_CONFIG_ENV, "")).strip():
        findings += [(name, f"move exporter settings to {RELAY_PLUGINS_CONFIG_ENV}; this variable is now ignored")
                     for name in configured_legacy_relay_env_vars(effective_env)]
    return findings


def report_deprecated_config_and_env(raw_config: dict | None = None, env_map: dict | None = None) -> list[tuple[str, str]]:
    """Emit non-failing doctor warnings for deprecated config keys and env vars; returns the findings reported.
    Does not mutate config/env and does not append to the blocking ``issues`` list."""
    deprecated = collect_deprecated_config_keys(raw_config) + collect_deprecated_env_vars(env_map)
    relay_cutover = collect_relay_plugin_cutover_findings(raw_config, env_map)
    findings = deprecated + relay_cutover
    if not findings:
        check_ok("No deprecated config keys or env vars")
        return findings
    for legacy, replacement in deprecated:
        check_warn(f"Deprecated: {legacy}", f"(use {replacement} instead)")
        check_info(f"Replace {legacy} → {replacement} (warn-only; not auto-migrated here)")
    for legacy, replacement in relay_cutover:
        check_warn(f"Breaking Relay migration: {legacy}", f"({replacement})")
        check_info(f"Migrate {legacy}: {replacement}")
    return findings


def managed_scope_check() -> None:
    """Report the active managed scope (resolved dir + pinned key counts); silent when none. A HERMES_MANAGED_DIR
    override is surfaced too — a redirected scope is the documented foot-gun (docs/design/managed-scope.md §7)."""
    managed_dir = None
    with warn_on_error(""):  # diagnostics must never crash
        from hermes_cli import managed_scope
        managed_dir = managed_scope.get_managed_dir()
    if managed_dir is None:
        return
    n_cfg, n_env = len(managed_scope.managed_config_keys()), len(managed_scope.load_managed_env())
    check_ok(f"Managed scope active: {n_cfg} config key(s), {n_env} env key(s) pinned by {managed_dir}")
    if os.environ.get("HERMES_MANAGED_DIR", "").strip():
        check_info(f"managed dir set via HERMES_MANAGED_DIR={managed_dir}")


@doctor_check("MCP security check failed: {e}")
def _check_mcp_security(should_fix: bool, f: Finding) -> None:
    """Flag mcp_servers entries with suspicious stdio commands."""
    from hermes_cli.config import load_config
    from hermes_cli.mcp_security import validate_mcp_server_entry
    servers = load_config().get("mcp_servers") or {}
    suspicious = 0
    for name, entry in sorted(servers.items()) if isinstance(servers, dict) else ():
        issues_found = validate_mcp_server_entry(name, entry) if isinstance(entry, dict) else None
        if not issues_found:
            continue
        suspicious += 1
        check_warn(f"MCP server '{name}' has suspicious stdio command", "; ".join(issues_found))
        f.manual_issues.append(f"Review/remove mcp_servers.{name} in config.yaml; rotate any credentials that may have been exposed.")
    if suspicious == 0:
        check_ok("No suspicious MCP stdio commands")


@doctor_check()
def _check_env_file(should_fix: bool, f: Finding) -> None:
    """Managed scope plus ~/.hermes/.env presence and provider credentials."""
    from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT, _DHH
    managed_scope_check()
    env_path = HERMES_HOME / '.env'
    if env_path.exists():
        check_ok(f"{_DHH}/.env file exists")
        # UTF-8 first; latin-1 fallback for Windows Notepad/cp1252 files (matches env_loader._load_dotenv_with_fallback).
        try:
            content = env_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = env_path.read_text(encoding="latin-1")
        if not check_bool(_has_provider_env_config(content), "API key or custom endpoint configured", f"No API key found in {_DHH}/.env"):
            f.issues.append("Run 'hermes setup' to configure API keys")
    elif (PROJECT_ROOT / '.env').exists():  # project root as fallback
        check_ok(".env file exists (in project directory)")
    else:
        check_fail(f"{_DHH}/.env file missing")
        if should_fix:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.touch()
            # .env holds API keys — touch() obeys umask (commonly 0o022, world-readable); tighten explicitly.
            with warn_on_error(""):
                os.chmod(str(env_path), 0o600)
            check_ok(f"Created empty {_DHH}/.env")
            check_info("Run 'hermes setup' to configure API keys")
            f.fixed += 1
        else:
            check_info("Run 'hermes setup' to create one")
            f.issues.append("Run 'hermes setup' to create .env")


def _known_provider_ids(cfg: dict) -> tuple[set, list, object, object, object]:
    """Return (known ids, custom providers, resolve_auth, normalize, resolve_full); any import failure leaves
    the matching resolver as None so validation degrades to "unavailable" rather than crashing doctor."""
    known: set = set()
    resolve_auth = normalize = resolve_full = aliases = None
    custom_providers: list = []
    with warn_on_error(""):
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider as resolve_auth
        known = set(PROVIDER_REGISTRY.keys()) | {"openrouter", "custom", "auto", "moa"}
    with warn_on_error(""):
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.providers import custom_provider_aliases as aliases, normalize_provider as normalize, resolve_provider_full as resolve_full
        with warn_on_error(""):
            custom_providers = get_compatible_custom_providers(cfg)
    user_providers = cfg.get("providers")
    if isinstance(user_providers, dict):
        from hermes_cli.config import is_provider_enabled
        known.update(str(name).strip().lower() for name, prov_cfg in user_providers.items()
                     if str(name).strip() and is_provider_enabled(prov_cfg))
    for entry in custom_providers if aliases is not None else ():
        name = str(entry.get("name") or "").strip() if isinstance(entry, dict) else ""
        if name:
            known.update(aliases(name, str(entry.get("provider_key") or "").strip()))
    return known, custom_providers, resolve_auth, normalize, resolve_full


# Vendor/model slugs are valid on aggregators and any custom provider; Fireworks' native IDs are slash-form
# (accounts/fireworks/models/...) and DeepInfra's catalog is exclusively vendor/model.
_VENDOR_SLUG_PROVIDERS = {
    "openrouter", "auto", "ai-gateway", "kilocode", "opencode-zen", "huggingface", "lmstudio", "nous", "nvidia",
    "fireworks", "deepinfra",
}


def _provider_has_credentials(runtime_provider: str) -> bool:
    """Only API-key providers in PROVIDER_REGISTRY are checked — OAuth/SDK/custom providers have their own
    checks elsewhere, and get_auth_status() returns a bare {logged_in: False} for anything it doesn't dispatch."""
    if runtime_provider == "openrouter":
        from hermes_cli.config import get_env_value
        return any(str(get_env_value(k) or "").strip() for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"))
    from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status
    pconfig = PROVIDER_REGISTRY.get(runtime_provider)
    if pconfig and getattr(pconfig, "auth_type", "") == "api_key":
        status = get_auth_status(runtime_provider) or {}
        return bool(status.get("configured") or status.get("logged_in") or status.get("api_key"))
    return True


def _validate_model_config(config_path, issues: list) -> None:
    """Validate model.provider / model.default against the provider registry (raw file)."""
    # Detect stale root-level model keys (known bug source — PR #4329)
    from hermes_cli.config import read_user_config_raw
    cfg = read_user_config_raw(config_path)
    model_section = cfg.get("model") or {}
    provider_raw = (model_section.get("provider") or "").strip()
    provider = provider_raw.lower()
    default_model = (model_section.get("default") or model_section.get("model") or "").strip()
    known_providers, custom_providers, resolve_auth, normalize, resolve_full = _known_provider_ids(cfg)
    valid_provider_ids = set(known_providers)
    accept = {provider} if provider else set()
    for known_provider in known_providers if normalize is not None else ():
        try:
            valid_provider_ids.add(normalize(known_provider))
        except Exception:
            continue
    runtime_provider = catalog_provider = provider
    if provider and provider not in {"auto", "custom"}:
        if resolve_auth is not None:
            try:
                runtime_provider = resolve_auth(provider)
                accept.add(runtime_provider)
            except Exception:
                runtime_provider = provider
        if resolve_full is not None:
            provider_def = resolve_full(provider, cfg.get("providers"), custom_providers)
            catalog_provider = provider_def.id if provider_def is not None else None
            accept.update({catalog_provider} - {None})
    if provider and provider != "auto" and (catalog_provider is None or (known_providers and not (accept & valid_provider_ids))):
        known_list = ", ".join(sorted(known_providers)) if known_providers else "(unavailable)"
        _fail_and_issue(f"model.provider '{provider_raw}' is not a recognised provider", f"(known: {known_list})",
                        f"model.provider '{provider_raw}' is unknown. Valid providers: {known_list}. "
                        f"Fix: run 'hermes config set model.provider <valid_provider>'", issues)
    policy_id = str(runtime_provider or catalog_provider or "").strip().lower()
    accepts_vendor_slug = policy_id in _VENDOR_SLUG_PROVIDERS or policy_id == "custom" or policy_id.startswith("custom:")
    if default_model and "/" in default_model and policy_id and not accepts_vendor_slug:
        check_warn(f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'",
                   "(vendor-prefixed slugs belong to aggregators like openrouter)")
        issues.append(f"model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. "
                      "Either set model.provider to 'openrouter', or drop the vendor prefix.")
    if runtime_provider and runtime_provider not in ("auto", "custom"):
        from hermes_cli.doctor import _DHH
        with warn_on_error(""):
            if not _provider_has_credentials(runtime_provider):
                _fail_and_issue(f"model.provider '{runtime_provider}' is set but no API key is configured",
                                "(check ~/.hermes/.env or run 'hermes setup')",
                                f"No credentials found for provider '{runtime_provider}'. Run 'hermes setup' or set the provider's "
                                f"API key in {_DHH}/.env, or switch providers with 'hermes config set model.provider <name>'", issues)


@doctor_check()
def _check_config_file(should_fix: bool, f: Finding) -> None:
    """config.yaml presence (project cli-config.yaml as fallback); model/provider validation."""
    from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT, _DHH
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        check_ok(f"{_DHH}/config.yaml exists")
        with warn_on_error("Could not validate model/provider config"):
            _validate_model_config(config_path, f.issues)
    elif (PROJECT_ROOT / 'cli-config.yaml').exists():
        check_ok("cli-config.yaml exists (in project directory)")
    elif should_fix:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        example_config = PROJECT_ROOT / 'cli-config.yaml.example'
        if example_config.exists():
            shutil.copy2(str(example_config), str(config_path))
        else:
            from hermes_cli.config import DEFAULT_CONFIG, save_config
            save_config(DEFAULT_CONFIG)
        check_ok(f"Created {_DHH}/config.yaml from {'cli-config.yaml.example' if example_config.exists() else 'defaults'}")
        f.fixed += 1
    else:
        check_warn("config.yaml not found", "(using defaults)")


def _drift_config_version(f: Finding, should_fix: bool, config_path) -> None:
    from hermes_cli.config import check_config_version, migrate_config
    current_ver, latest_ver = check_config_version()
    outdated = (f"Config version outdated (v{current_ver} → v{latest_ver})", "(new settings available)")
    if check_bool(current_ver >= latest_ver, f"Config version up to date (v{current_ver})", outdated):
        return
    if not should_fix:
        f.issues.append("Run 'hermes doctor --fix' or 'hermes setup' to migrate config")
        return
    try:
        migrate_config(interactive=False, quiet=False)
        check_ok("Config migrated to latest version")
        f.fixed += 1
    except Exception as mig_err:
        check_warn(f"Auto-migration failed: {mig_err}")
        f.issues.append("Run 'hermes setup' to migrate config")


def _drift_stale_root_keys(f: Finding, should_fix: bool, config_path) -> None:
    """Root-level ``provider``/``base_url`` belong under ``model:`` (raw-file diagnostic)."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    raw_config = read_user_config_raw(config_path)
    stale_root_keys = [k for k in ("provider", "base_url") if k in raw_config and isinstance(raw_config[k], str)]
    if not stale_root_keys:
        return
    check_warn(f"Stale root-level config keys: {', '.join(stale_root_keys)}", "(should be under 'model:' section)")
    if not should_fix:
        f.issues.append("Stale root-level provider/base_url in config.yaml — run 'hermes doctor --fix'")
        return
    # Coerce scalar/None ``model:`` into a dict before mutation (setdefault would hand back a scalar).
    raw_model = raw_config.get("model")
    if not isinstance(raw_model, dict):
        raw_model = raw_config["model"] = {"default": raw_model.strip()} if isinstance(raw_model, str) and raw_model.strip() else {}
    for k in stale_root_keys:
        value = raw_config.pop(k)
        if not raw_model.get(k):
            raw_model[k] = value
    atomic_config_write(config_path, raw_config)
    check_ok("Migrated stale root-level keys into model section")
    f.fixed += 1


def _drift_max_iterations_ghost(f: Finding, should_fix: bool, config_path) -> None:
    """A stale HERMES_MAX_ITERATIONS in .env shadows agent.max_turns in config.yaml.

    The setup wizard used to dual-write the budget. The gateway bridge derives HERMES_MAX_ITERATIONS from
    agent.max_turns, but if it bails on an earlier config-parse error the .env value silently wins. Read the
    .env FILE (load_env), not get_env_value/os.environ, which the bridge may have overridden already.
    """
    from hermes_cli.doctor import _DHH
    # Detect stale HERMES_MAX_ITERATIONS ghost in .env shadowing agent.max_turns in config.yaml (issue
    # #17534). The setup wizard used to dual-write the iteration budget to both stores; users who later edit
    # only config.yaml are left with a .env ghost. The gateway bridge normally derives HERMES_MAX_ITERATIONS
    # from agent.max_turns at startup, but if that bridge bails (any earlier config-parse error), the stale
    # .env value silently wins and the agent runs at the wrong budget — e.g. config says 400 but the
    # activity line reads N/90.
    from hermes_cli.config import load_env, read_user_config_raw, remove_env_value
    raw_config = read_user_config_raw(config_path)
    agent_cfg = raw_config.get("agent")
    cfg_max_turns = agent_cfg.get("max_turns") if isinstance(agent_cfg, dict) else None
    if cfg_max_turns is None:
        cfg_max_turns = raw_config.get("max_turns")  # legacy root-level key counts too
    env_ghost = load_env().get("HERMES_MAX_ITERATIONS")
    if cfg_max_turns is None or env_ghost is None or str(cfg_max_turns).strip() == str(env_ghost).strip():
        return
    check_warn(f"HERMES_MAX_ITERATIONS={env_ghost} in .env shadows agent.max_turns={cfg_max_turns} in config.yaml",
               "(stale ghost from an earlier `hermes setup` run)")
    if not should_fix:
        f.issues.append("Stale HERMES_MAX_ITERATIONS in .env shadows config.yaml — run 'hermes doctor --fix'")
    elif remove_env_value("HERMES_MAX_ITERATIONS"):
        check_ok(f"Removed stale HERMES_MAX_ITERATIONS from .env (config.yaml agent.max_turns={cfg_max_turns} is now authoritative)")
        f.fixed += 1
    else:
        check_warn("Could not remove HERMES_MAX_ITERATIONS from .env")
        f.manual_issues.append(f"Manually delete the HERMES_MAX_ITERATIONS line from {_DHH}/.env — config.yaml agent.max_turns is authoritative.")


def _drift_deprecations(f: Finding, should_fix: bool, config_path) -> None:
    """Warn-only deprecation sweep over the raw file + on-disk .env (process env would false-positive)."""
    from hermes_cli.config import load_env, read_user_config_raw
    raw = read_user_config_raw(config_path) if config_path is not None else {}
    env = {}
    with warn_on_error(""):
        env = load_env()
    report_deprecated_config_and_env(raw, env)


def _drift_structure(f: Finding, should_fix: bool, config_path) -> None:
    """Structural validation (malformed custom_providers, etc.)."""
    from hermes_cli.config import validate_config_structure
    config_issues = validate_config_structure()
    if not config_issues:
        return
    _section("Config Structure")
    for ci in config_issues:
        (check_fail if ci.severity == "error" else check_warn)(ci.message)
        for hint_line in ci.hint.splitlines():
            check_info(hint_line)
        f.issues.append(ci.message)


_CONFIG_DRIFT_STEPS = (
    _drift_config_version, _drift_stale_root_keys, _drift_max_iterations_ghost, _drift_deprecations, _drift_structure,
)


@doctor_check()
def _check_config_drift(should_fix: bool, f: Finding) -> None:
    """Config version, stale root keys, HERMES_MAX_ITERATIONS ghost, deprecations, structure.

    Each step is independent and best-effort: a failure in one never hides the next.
    """
    from hermes_cli.doctor import HERMES_HOME
    config_path = HERMES_HOME / 'config.yaml'
    if not config_path.exists():
        config_path = None
    for step in _CONFIG_DRIFT_STEPS if config_path else (_drift_deprecations,):
        with warn_on_error(""):
            step(f, should_fix, config_path)


@doctor_check("xAI retirement check skipped", "({e})")
def _check_xai_retirement(should_fix: bool, f: Finding) -> None:
    from hermes_cli.config import load_config
    from hermes_cli.xai_retirement import MIGRATION_GUIDE_URL, find_retired_xai_refs, format_issue
    retired_refs = find_retired_xai_refs(load_config())
    if not retired_refs:
        check_ok("No retired xAI models in config")
        return
    for ref in retired_refs:
        check_warn(format_issue(ref))
    check_info(f"Migration guide: {MIGRATION_GUIDE_URL}")
    f.manual_issues.append(f"Update {len(retired_refs)} retired xAI model reference(s) in config.yaml — see {MIGRATION_GUIDE_URL}")


@doctor_check("Plugin compat check skipped", "({e})")
def _check_plugin_compat(should_fix: bool, f: Finding) -> None:
    from hermes_cli.plugin_compat import ALLOW_KEY, COMPAT_REMOVAL, compat_report, removal_in_effect
    report = compat_report()
    if not report:
        check_ok(f"No enabled plugin imports paths removed on {COMPAT_REMOVAL}")
        return
    for name, hits in sorted(report.items()):
        (check_fail if removal_in_effect() else check_warn)(
            f"{name}: {len(hits)} import(s) of paths removed on {COMPAT_REMOVAL}", f"{hits[0].old} -> {hits[0].new}")
    check_info("Details: hermes plugins compat")
    f.manual_issues.append(
        f"Update {len(report)} plugin(s) still importing pre-decomposition paths (hermes plugins compat) — "
        + ("they are NOT being loaded" if removal_in_effect() else f"they stop loading on {COMPAT_REMOVAL}")
        + f"; escape hatch: plugins.{ALLOW_KEY}: true")
