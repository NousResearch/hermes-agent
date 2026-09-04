"""Table-driven config migration registry.

Each step is ``_migrate_to_N(results, quiet)``; the version gate and strict ascending order live
in :func:`run_migrations`. Every write goes through ``hermes_cli.config._persist_migration`` so a
step may only persist values that differ from the schema default (plus removals/renames).
"""

from __future__ import annotations

import copy
import functools
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

#: Auto-migration support floor. Configs whose on-disk ``_config_version`` is below this are NOT
#: auto-migrated (v12 predates ~two years of releases; carrying the sub-v12 steps and the env
#: bridges they consumed forever is not worth it). Below-floor configs are left byte-for-byte
#: untouched — the process continues with defaults deep-merged at read time, matching the
#: non-fatal posture for unparseable configs — and a message tells the user how to proceed.
SUPPORT_FLOOR_VERSION = 12


def support_floor_message() -> str:
    """Human-facing explanation shown when a config is below the floor."""
    from hermes_constants import display_hermes_home

    return (
        f"This config predates version {SUPPORT_FLOOR_VERSION} (~2 years old) "
        "and can no longer be auto-migrated. Back up "
        f"{display_hermes_home()}/config.yaml and run `hermes setup` to "
        f"regenerate, or manually set _config_version: {SUPPORT_FLOOR_VERSION} "
        "after reviewing the changelog.")


def _cfg():
    """Return the live ``hermes_cli.config`` module (lazy, cycle-free, monkeypatch-friendly)."""
    from hermes_cli import config

    return config


def read_raw_config():
    return _cfg().read_raw_config()


def _persist_migration(config):
    _cfg()._persist_migration(config)


def _dict_at(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """``config[key]`` when it is a mapping (same object, so writes alias), else a fresh ``{}``."""
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _commit(
    config: Dict[str, Any],
    results: Dict[str, Any],
    quiet: bool,
    added: Optional[str],
    message: Optional[str]) -> None:
    """Persist *config*, record *added* under ``config_added`` and print *message* unless quiet."""
    _persist_migration(config)
    if added:
        results["config_added"].append(added)
    if message and not quiet:
        print(message)


def _rewrite_key(
    results: Dict[str, Any],
    quiet: bool,
    *,
    section: str,
    key: str,
    match: Callable[[Any], bool],
    new: Any,
    added: str,
    message: str,
    extra_guard: Callable[[Dict[str, Any]], bool] = lambda _m: True,
    create_section: bool = False) -> None:
    """Rewrite ``<section>.<key>`` to *new* (None = delete) when ``match(current)`` holds; a
    missing section is skipped unless *create_section*."""
    config = read_raw_config()
    raw = config.get(section)
    if not isinstance(raw, dict):
        if not create_section:
            return
        raw = {}
    if match(raw.get(key)) and extra_guard(raw):
        if new is None:
            del raw[key]
        else:
            raw[key] = new
        config[section] = raw
        _commit(config, results, quiet, added, message)


def _rewrite_stale_default(*, old: Any, **kw: Any) -> Callable[[Dict[str, Any], bool], None]:
    """Step rewriting a key only while it still equals the OLD default — never clobbers a value
    the user customized; unset keys inherit the new default at read time."""
    return functools.partial(_rewrite_key, match=lambda cur: cur == old, **kw)


def _lower_is(word: str) -> Callable[[Any], bool]:
    return lambda cur: isinstance(cur, str) and cur.strip().lower() == word


def _migrate_to_12(results: Dict[str, Any], quiet: bool) -> None:
    # 11 → 12: custom_providers list → providers dict.
    _custom_provider_entry_to_provider_config = _cfg()._custom_provider_entry_to_provider_config

    config = read_raw_config()
    custom_list = config.get("custom_providers")
    if not (isinstance(custom_list, list) and custom_list):
        return
    providers_dict = _dict_at(config, "providers")
    migrated_count = 0
    for entry in custom_list:
        if not isinstance(entry, dict):
            continue
        old_name = entry.get("name", "")
        old_url = entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or ""
        if not old_url:
            continue

        # kebab-case key from the display name; fall back to the URL hostname.
        key = old_name.strip().lower().replace(" ", "-").replace("(", "").replace(")", "")
        key = re.sub(r"-{2,}", "-", key).strip("-")
        if not key:
            try:
                key = (urlparse(old_url).hostname or "endpoint").replace(".", "-")
            except Exception:
                key = f"endpoint-{migrated_count}"

        # Don't overwrite existing entries
        base_key = key
        suffix = migrated_count
        while key in providers_dict:
            key = f"{base_key}-{suffix}"
            suffix += 1

        new_entry = _custom_provider_entry_to_provider_config(entry, provider_key=key)
        if new_entry is None:
            continue
        if not old_name:
            new_entry.pop("name", None)
        if new_entry.get("api_key") in {"no-key", "no-key-required", ""}:
            new_entry.pop("api_key", None)

        providers_dict[key] = new_entry
        migrated_count += 1

    if migrated_count > 0:
        config["providers"] = providers_dict
        # Runtime reads the list view via get_compatible_custom_providers().
        config.pop("custom_providers", None)
        _persist_migration(config)
        if not quiet:
            print(f"  ✓ Migrated {migrated_count} custom provider(s) to providers: section")
            for key in list(providers_dict.keys())[-migrated_count:]:
                print(f"    → {key}: {providers_dict[key].get('api', '')}")


def _migrate_to_13(results: Dict[str, Any], quiet: bool) -> None:
    # 12 → 13: clear dead LLM_MODEL / OPENAI_MODEL from .env (written by the old setup wizard;
    # nothing reads them — config.yaml is the sole source of truth).
    _c = _cfg()
    for dead_var in ("LLM_MODEL", "OPENAI_MODEL"):
        try:
            if _c.get_env_value(dead_var):
                _c.save_env_value(dead_var, "")
                if not quiet:
                    print(f"  ✓ Cleared {dead_var} from .env (no longer used — config.yaml is source of truth)")
        except Exception:
            pass


_LOCAL_WHISPER_MODELS = frozenset({
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large-v1", "large-v2", "large-v3",
    "large", "distil-large-v2", "distil-medium.en",
    "distil-small.en", "distil-large-v3", "distil-large-v3.5",
    "large-v3-turbo", "turbo"})


def _migrate_to_14(results: Dict[str, Any], quiet: bool) -> None:
    # 13 → 14: legacy flat stt.model → provider section. A provider-agnostic `stt.model` fed
    # OpenAI names to faster-whisper ("Invalid model size"). Only the raw (user-written) config
    # decides; a nested model the user already set is never overwritten.
    raw_stt = read_raw_config().get("stt", {})
    if not (isinstance(raw_stt, dict) and "model" in raw_stt):
        return
    legacy_model = raw_stt["model"]
    provider = raw_stt.get("provider", "local")
    config = read_raw_config()
    stt = config.get("stt", {})
    stt.pop("model", None)

    def _place(section: str) -> None:
        existing = raw_stt.get(section, {})
        if not isinstance(existing, dict) or "model" not in existing:
            stt.setdefault(section, {})["model"] = legacy_model

    if provider in {"local", "local_command"}:
        # An OpenAI model name is dropped; the local section already defaults to "base".
        if legacy_model in _LOCAL_WHISPER_MODELS:
            _place("local")
    else:
        _place(provider)
    config["stt"] = stt
    _commit(
        config, results, quiet, None, "  ✓ Migrated legacy stt.model to provider-specific config")


def _migrate_to_16(results: Dict[str, Any], quiet: bool) -> None:
    # 15 → 16: display.tool_progress_overrides → display.platforms.<plat>.tool_progress.
    config = read_raw_config()
    display = _dict_at(config, "display")
    old_overrides = display.get("tool_progress_overrides")
    if not (isinstance(old_overrides, dict) and old_overrides):
        return
    platforms = _dict_at(display, "platforms")
    for plat, mode in old_overrides.items():
        if plat not in platforms:
            platforms[plat] = {}
        if "tool_progress" not in platforms[plat]:
            platforms[plat]["tool_progress"] = mode
    display["platforms"] = platforms
    config["display"] = display
    migrated = ", ".join(f"{p}={m}" for p, m in old_overrides.items())
    _commit(
        config, results, quiet,
        "display.platforms (migrated from tool_progress_overrides)",
        f"  ✓ Migrated tool_progress_overrides → display.platforms: {migrated}")


def _migrate_to_17(results: Dict[str, Any], quiet: bool) -> None:
    # 16 → 17: remove legacy compression.summary_* keys; non-empty, non-default values move to
    # auxiliary.compression without overriding an explicit (non-"auto") aux value.
    config = read_raw_config()
    comp = config.get("compression", {})
    if not isinstance(comp, dict):
        return
    legacy = {k: comp.pop(f"summary_{k}", None) for k in ("model", "provider", "base_url")}
    migrated_keys = []
    for k, raw in legacy.items():
        val = str(raw).strip() if raw else ""
        if not val or (k == "provider" and val == "auto"):
            continue
        aux_comp = config.setdefault("auxiliary", {}).setdefault("compression", {})
        cur = aux_comp.get(k)
        if not cur or (k == "provider" and cur == "auto"):
            aux_comp[k] = val
            migrated_keys.append(f"{k}={raw}")
    if migrated_keys or any(v is not None for v in legacy.values()):
        config["compression"] = comp
        message = (
            "  ✓ Migrated compression.summary_* → auxiliary.compression: "
            f"{', '.join(migrated_keys)}"
            if migrated_keys else "  ✓ Removed unused compression.summary_* keys")
        _commit(config, results, quiet, None, message)


def _installed_user_plugins(disabled: set) -> List[str]:
    """Names of plugins under ``$HERMES_HOME/plugins/`` with a manifest, minus *disabled*."""
    _c = _cfg()
    found: List[str] = []
    try:
        user_plugins_dir = _c.get_hermes_home() / "plugins"
        if user_plugins_dir.is_dir():
            for child in sorted(user_plugins_dir.iterdir()):
                if not child.is_dir():
                    continue
                manifest_file = child / "plugin.yaml"
                if not manifest_file.exists():
                    manifest_file = child / "plugin.yml"
                if not manifest_file.exists():
                    continue
                try:
                    with open(manifest_file, encoding="utf-8") as _mf:
                        manifest = _c.fast_safe_load(_mf) or {}
                except Exception:
                    manifest = {}
                name = manifest.get("name") or child.name
                if name not in disabled:
                    found.append(name)
    except Exception:
        return []
    return found


def _migrate_to_21(results: Dict[str, Any], quiet: bool) -> None:
    # 20 → 21: plugins are now opt-in (loader requires ``plugins.enabled``). Grandfather installed
    # user plugins not already disabled; bundled plugins ship off and need explicit opt-in.
    config = read_raw_config()
    plugins_cfg = _dict_at(config, "plugins")
    if "enabled" in plugins_cfg:
        return
    disabled = plugins_cfg.get("disabled", []) or []
    grandfathered = _installed_user_plugins(set(disabled) if isinstance(disabled, list) else set())
    plugins_cfg["enabled"] = grandfathered
    config["plugins"] = plugins_cfg
    message = (
        f"  ✓ Plugins now opt-in: grandfathered "
        f"{len(grandfathered)} existing plugin(s) into plugins.enabled"
        if grandfathered else
        "  ✓ Plugins now opt-in: no existing plugins to grandfather. "
        "Use `hermes plugins enable <name>` to activate.")
    _commit(
        config, results, quiet,
        f"plugins.enabled (opt-in allow-list, {len(grandfathered)} grandfathered)", message)


def _migrate_to_23(results: Dict[str, Any], quiet: bool) -> None:
    # 22 → 23: seed curator defaults + create logs/curator/. Older configs never wrote the curator
    # section; deep-merge made it work but users could not see/edit it and `hermes curator status`
    # had no stable logs dir. Only keys the user hasn't set are written.
    _c = _cfg()
    DEFAULT_CONFIG = _c.DEFAULT_CONFIG

    try:
        curator_dir = _c.get_hermes_home() / "logs" / "curator"
        curator_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        results["warnings"].append(f"Could not create {curator_dir}: {e}")

    config = read_raw_config()

    def _seed_missing(section: Dict[str, Any], defaults: Dict[str, Any]) -> List[str]:
        added = [k for k in defaults if k not in section]
        for k in added:
            section[k] = copy.deepcopy(defaults[k])
        return added

    raw_curator = _dict_at(config, "curator")
    added_curator = _seed_missing(raw_curator, DEFAULT_CONFIG.get("curator", {}))
    if added_curator:
        config["curator"] = raw_curator

    raw_aux = _dict_at(config, "auxiliary")
    raw_aux_curator = _dict_at(raw_aux, "curator")
    added_aux = _seed_missing(
        raw_aux_curator, DEFAULT_CONFIG.get("auxiliary", {}).get("curator", {}))
    if added_aux:
        raw_aux["curator"] = raw_aux_curator
        config["auxiliary"] = raw_aux

    if added_curator or added_aux:
        _persist_migration(config)
        for label, added in (("curator", added_curator), ("auxiliary.curator", added_aux)):
            if not added:
                continue
            results["config_added"].append(f"{label} ({len(added)} default key(s))")
            if not quiet:
                print(
                    f"  ✓ {'Curator' if label == 'curator' else label} settings now available "
                    f"({', '.join(added)}) — edit via `hermes config set`")


def _migrate_to_29(results: Dict[str, Any], quiet: bool) -> None:
    # 28 → 29: memory/skills tri-state write_mode (on|off|approve) → boolean write_approval.
    # Only "approve" carried gating intent → true; the old "off = block writes" mode is dropped
    # (memory_enabled: false disables memory). Only a persisted key is rewritten.
    config = read_raw_config()
    touched = False
    for subsystem in ("memory", "skills"):
        sub = config.get(subsystem)
        if not isinstance(sub, dict) or "write_mode" not in sub:
            continue
        old = sub.pop("write_mode")
        old_norm = old.strip().lower() if isinstance(old, str) else old
        sub["write_approval"] = (old_norm == "approve")
        config[subsystem] = sub
        touched = True
        results["config_added"].append(
            f"{subsystem}.write_mode → write_approval={sub['write_approval']}")
    if touched:
        _commit(config, results, quiet, None,
                "  ✓ Renamed write_mode → write_approval (boolean gate)")


# 29 → 30 (curator.consolidate defaults to false) is schema-default-only: deep-merge supplies it
# at read time and persisting a default would only bloat a lean config. No registry entry.


def _migrate_to_33(results: Dict[str, Any], quiet: bool) -> None:
    # 32 → 33: max_async_children is deprecated; fold a raised value into max_concurrent_children
    # (take the max so nobody loses headroom), then drop it.
    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if not (isinstance(raw_deleg, dict) and "max_async_children" in raw_deleg):
        return
    old_async = raw_deleg.pop("max_async_children")
    try:
        old_async_i = int(old_async)
    except (TypeError, ValueError):
        old_async_i = None
    if old_async_i is not None and old_async_i > 3:
        try:
            cur_children = int(raw_deleg.get("max_concurrent_children", 3))
        except (TypeError, ValueError):
            cur_children = 3
        if old_async_i > cur_children:
            raw_deleg["max_concurrent_children"] = old_async_i
            results["config_added"].append(
                f"delegation.max_concurrent_children={old_async_i} "
                f"(folded from deprecated max_async_children)")
    config["delegation"] = raw_deleg
    _commit(
        config, results, quiet, None,
        "  ✓ Removed deprecated delegation.max_async_children — "
        "delegation.max_concurrent_children now caps background "
        "delegations too.")


def _migrate_to_34(results: Dict[str, Any], quiet: bool) -> None:
    # 33 → 34: one-time personality reset. Persistence used to be split (TUI/desktop wrote the
    # NAME to display.personality, CLI/gateway wrote rendered TEXT to agent.system_prompt), so
    # once display.personality became authoritative, stale names resurrected personalities users
    # had turned off. Reset display.personality → "" and scrub agent.system_prompt ONLY when it
    # verbatim-equals a known personality's rendered text; any other text is user-owned.
    from hermes_cli.personality import (
        available_personalities, normalize_personality_name, prompt_text, render_personality_prompt)

    config = read_raw_config()
    touched = False

    raw_display = config.get("display")
    old_name = ""
    if isinstance(raw_display, dict):
        old_name = normalize_personality_name(raw_display.get("personality", ""))
        if old_name:
            raw_display["personality"] = ""
            config["display"] = raw_display
            touched = True

    raw_agent = config.get("agent")
    scrubbed_text = False
    if isinstance(raw_agent, dict):
        manual = prompt_text(raw_agent.get("system_prompt", ""))
        if manual:
            rendered = {
                render_personality_prompt(defn) for defn in available_personalities(config).values()
            }
            if manual in rendered:
                raw_agent["system_prompt"] = ""
                config["agent"] = raw_agent
                touched = True
                scrubbed_text = True

    if not touched:
        return
    _commit(config, results, quiet, "display.personality=none (one-time reset)", None)
    if quiet:
        return
    if old_name:
        print(
            f"  ✓ Personality reset to none (was '{old_name}'). Personality "
            "state was previously saved inconsistently across surfaces and "
            "could re-enable a personality you had turned off. "
            f"Run /personality {old_name} to turn it back on.")
    if scrubbed_text:
        print(
            "  ✓ Removed personality text from agent.system_prompt (written "
            "by an older /personality). That field is now reserved for "
            "manual system prompts; personalities live in display.personality.")


def _migrate_to_38(results: Dict[str, Any], quiet: bool) -> None:
    # 37 → 38: the bundled observability/nemo_relay plugin was removed (Relay lifecycle moved
    # into the agent core); drop it from plugins.enabled.
    from hermes_cli.relay_plugin_cutover import legacy_relay_plugin_keys

    config = read_raw_config()
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return
    enabled = plugins.get("enabled")
    removed = legacy_relay_plugin_keys(enabled)
    if not removed or not isinstance(enabled, list):
        return

    plugins["enabled"] = [value for value in enabled if value not in removed]
    config["plugins"] = plugins
    _persist_migration(config)
    message = (
        "Removed legacy Relay plugin from plugins.enabled: "
        f"{', '.join(removed)}. Configure native Relay plugins with "
        "HERMES_NEMO_RELAY_PLUGINS_TOML.")
    results["warnings"].append(message)
    if not quiet:
        print(f"  ⚠ {message}")


def _migrate_to_39(results: Dict[str, Any], quiet: bool) -> None:
    # 38 → 39: strip the retired `bfl` toolset wherever a backfill/picker save wrote it, so stale
    # config can't resurrect an unknown toolset.
    config = read_raw_config()
    changed = False
    for section in ("platform_toolsets", "known_builtin_toolsets"):
        mapping = config.get(section)
        if not isinstance(mapping, dict):
            continue
        for platform, toolsets in mapping.items():
            if isinstance(toolsets, list) and "bfl" in toolsets:
                mapping[platform] = [ts for ts in toolsets if ts != "bfl"]
                changed = True
        if changed:
            config[section] = mapping
    if changed:
        _commit(
            config, results, quiet,
            "removed retired 'bfl' toolset from saved toolset lists",
            "  ✓ Removed the retired BFL FLUX 3 toolset from saved toolset "
            "lists — video generation now lives under `hermes tools` → "
            "Video Generation (Nous Subscription or FAL).")


#: Registry of (target_version, step), strictly ascending; simple default-flip steps are
#: declared inline via _rewrite_stale_default / _rewrite_key partials. Later steps observe
#: earlier steps' writes via read_raw_config() (filesystem state). v12 is the support floor:
#: configs already AT v12 still get every step below; only configs BELOW 12 are refused by the
#: floor gate in run_migrations()'s caller. Versions absent here (15, 18-20, 22, 24, 26-28, 30)
#: only added a schema default that runtime merging supplies without a write.
MIGRATIONS: Tuple[Tuple[int, Callable[[Dict[str, Any], bool], None]], ...] = (
    (12, _migrate_to_12),
    (13, _migrate_to_13),
    (14, _migrate_to_14),
    (16, _migrate_to_16),
    (17, _migrate_to_17),
    (21, _migrate_to_21),
    (23, _migrate_to_23),
    # 24 → 25: model_catalog TTL 24h → 1h (only the OLD default 24).
    (25, _rewrite_stale_default(
        section="model_catalog", key="ttl_hours", old=24, new=1,
        added="model_catalog.ttl_hours 24→1",
        message="  ✓ Lowered model_catalog.ttl_hours to 1 (hourly picker refresh)")),
    (29, _migrate_to_29),
    # 30 → 31: verify_on_stop OFF (one-time). The "auto" sentinel was more noise than signal.
    # Rewrite only when missing or still "auto" — an explicit user true/false is preserved.
    (31, functools.partial(
        _rewrite_key, section="agent", key="verify_on_stop", new=False, create_section=True,
        match=lambda cur: cur is None or _lower_is("auto")(cur),
        added="agent.verify_on_stop=false",
        message=(
            "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false). "
            "Set it to true to re-enable, or \"auto\" for the legacy "
            "surface-aware behavior."))),
    # 31 → 32: flip the BAKED-IN literal true to OFF (one-time). v30 defaulted verify_on_stop to a
    # literal True and migrate_config persisted defaults, so installs that updated through v30 have
    # `verify_on_stop: true` written literally — never a user choice (no off-switch existed until
    # v31). A true set AFTER v32 is never touched.
    (32, _rewrite_stale_default(
        section="agent", key="verify_on_stop", old=True, new=False,
        added="agent.verify_on_stop=false",
        message=(
            "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false) — "
            "the old default was written into your config as a literal "
            "true. Set it to true again to re-enable, or \"auto\" for the "
            "legacy surface-aware behavior."),
        extra_guard=lambda raw: raw.get("verify_on_stop") is True)),
    (33, _migrate_to_33),
    (34, _migrate_to_34),
    # 34 → 35: background_process_notifications 'all' (old implicit default, rarely chosen on
    # purpose) → 'concise'. Explicit result/error/off choices are preserved.
    (35, functools.partial(
        _rewrite_key, section="display", key="background_process_notifications",
        match=_lower_is("all"), new="concise",
        added="display.background_process_notifications=concise (was: all)",
        message=(
            "  ✓ Background process notifications switched from 'all' to "
            "'concise' — completions now show a one-line status message "
            "instead of the raw output dump. Set "
            "display.background_process_notifications: all to restore "
            "the old behavior."))),
    # 35 → 36: subagent iteration cap 50 → 250 (50 truncated substantial delegated work).
    (36, _rewrite_stale_default(
        section="delegation", key="max_iterations", old=50, new=250,
        added="delegation.max_iterations=250 (was: 50)",
        message=(
            "  ✓ Raised delegation.max_iterations from 50 to 250 — subagents "
            "now get a larger per-child tool-call budget so delegated work "
            "finishes instead of truncating. Set delegation.max_iterations "
            "back to 50 to restore the old cap."))),
    # 36 → 37: delegation concurrency 3 → 10 (stays at/below the high-cost warning threshold).
    (37, _rewrite_stale_default(
        section="delegation", key="max_concurrent_children", old=3, new=10,
        added="delegation.max_concurrent_children=10 (was: 3)",
        message=(
            "  ✓ Raised delegation.max_concurrent_children from 3 to 10 — "
            "independent delegated children now fan out wider in parallel. "
            "Each child consumes API tokens independently; set "
            "delegation.max_concurrent_children back to 3 to restore the old cap."))),
    (38, _migrate_to_38),
    (39, _migrate_to_39),
    # 39 → 40: model_catalog.ttl_hours → ttl_minutes (default 20). Only the OLD default
    # (ttl_hours: 1, written by v25) is dropped; any other explicit ttl_hours is still honoured.
    (40, _rewrite_stale_default(
        section="model_catalog", key="ttl_hours", old=1, new=None,
        added="model_catalog.ttl_hours 1 → ttl_minutes 20 (default)",
        message="  ✓ Model catalog now refreshes every 20 minutes (model_catalog.ttl_minutes)",
        extra_guard=lambda raw: "ttl_minutes" not in raw)),
)


def run_migrations(current_ver: int, results: Dict[str, Any], quiet: bool) -> None:
    """Apply every registered migration whose target version exceeds *current_ver*.

    *current_ver* is the on-disk schema version captured ONCE before any step runs and does not
    advance between steps — each step is gated on the same initial value.
    """
    for target_ver, migration_fn in MIGRATIONS:
        if current_ver < target_ver:
            migration_fn(results, quiet)
