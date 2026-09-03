"""Table-driven config migration registry.

This module holds the per-version migration steps that used to live as a
768-line ladder of ``if current_ver < N:`` blocks inside
``hermes_cli.config.migrate_config``. Each step is a function
``_migrate_to_N(results, quiet)`` whose body is copied verbatim from the
original block; only the shared skeleton (the version gate and the strict
ascending ordering) lives in the :func:`run_migrations` driver.

Semantics preserved exactly from the original ladder:

* ``current_ver`` is computed ONCE by the caller (``check_config_version``)
  and never advances while the ladder runs — every step compares against the
  same initial value. The driver replicates that: it applies every registry
  entry whose target version is ``> current_ver``, in ascending order.
* Each step re-reads the raw on-disk config itself (``read_raw_config``) and
  persists via ``_persist_migration`` — steps therefore observe the writes of
  earlier steps through the filesystem, which is why strict ascending order
  is mandatory.
* All ``results['config_added']`` / ``results['warnings']`` appends and all
  conditional ``print`` output stay inside the step functions, byte-identical
  to the original blocks.

Import direction / cycle avoidance:

``hermes_cli.config`` imports :func:`run_migrations` lazily (inside
``migrate_config``), and every step function here resolves its helpers
(``read_raw_config``, ``_persist_migration``, ``get_env_value``, …) lazily
through the live ``hermes_cli.config`` module object at call time via
:func:`_cfg`. There is deliberately NO module-level import of
``hermes_cli.config`` here, so no circular import can form — and, just as
importantly, tests that monkeypatch helpers on ``hermes_cli.config`` (e.g.
``patch("hermes_cli.config.read_raw_config", ...)``) keep working, because
the steps always go through the module attribute rather than a bound-early
reference.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

#: Auto-migration support floor. Configs whose on-disk ``_config_version`` is
#: below this are NOT auto-migrated any more (policy decision, July 2026):
#: v12 predates roughly two years of releases, and carrying the sub-v12
#: migration steps (plus the env bridges they consumed, e.g.
#: HERMES_TOOL_PROGRESS*) forever is not worth it. Below-floor configs are
#: left byte-for-byte untouched — the process continues with the config as-is
#: (defaults deep-merged at read time, matching the non-fatal posture used
#: for unparseable configs) and a clear message tells the user how to
#: proceed. The removed steps were the <12 targets: v4 (tool-progress .env →
#: config.yaml), v5 (timezone seed), v9 (clear ANTHROPIC_TOKEN).
SUPPORT_FLOOR_VERSION = 12


def support_floor_message() -> str:
    """Human-facing explanation shown when a config is below the floor."""
    from hermes_constants import display_hermes_home

    return (
        f"This config predates version {SUPPORT_FLOOR_VERSION} (~2 years old) "
        "and can no longer be auto-migrated. Back up "
        f"{display_hermes_home()}/config.yaml and run `hermes setup` to "
        f"regenerate, or manually set _config_version: {SUPPORT_FLOOR_VERSION} "
        "after reviewing the changelog."
    )


def _cfg():
    """Return the live ``hermes_cli.config`` module (lazy, cycle-free)."""
    from hermes_cli import config

    return config


def _migrate_to_12(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 11 → 12: migrate custom_providers list → providers dict ──
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration
    _custom_provider_entry_to_provider_config = _c._custom_provider_entry_to_provider_config

    config = read_raw_config()
    custom_list = config.get("custom_providers")
    if isinstance(custom_list, list) and custom_list:
        providers_dict = config.get("providers", {})
        if not isinstance(providers_dict, dict):
            providers_dict = {}
        migrated_count = 0
        for entry in custom_list:
            if not isinstance(entry, dict):
                continue
            old_name = entry.get("name", "")
            old_url = entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or ""
            if not old_url:
                continue  # skip entries with no URL

            # Generate a kebab-case key from the display name
            key = old_name.strip().lower().replace(" ", "-").replace("(", "").replace(")", "")
            # Remove consecutive hyphens and trailing hyphens
            while "--" in key:
                key = key.replace("--", "-")
            key = key.strip("-")
            if not key:
                # Fallback: derive from URL hostname
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(old_url)
                    key = (parsed.hostname or "endpoint").replace(".", "-")
                except Exception:
                    key = f"endpoint-{migrated_count}"

            # Don't overwrite existing entries
            base_key = key
            suffix = migrated_count
            while key in providers_dict:
                key = f"{base_key}-{suffix}"
                suffix += 1

            new_entry = _custom_provider_entry_to_provider_config(
                entry,
                provider_key=key,
            )
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
            # Remove the old list — runtime reads via get_compatible_custom_providers()
            config.pop("custom_providers", None)
            _persist_migration(config)
            if not quiet:
                print(f"  ✓ Migrated {migrated_count} custom provider(s) to providers: section")
                for key in list(providers_dict.keys())[-migrated_count:]:
                    ep = providers_dict[key]
                    print(f"    → {key}: {ep.get('api', '')}")


def _migrate_to_13(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 12 → 13: clear dead LLM_MODEL / OPENAI_MODEL from .env ──
    # These env vars were written by the old setup wizard but nothing reads
    # them anymore (config.yaml is the sole source of truth since March 2026).
    # Stale entries cause user confusion — see issue report.
    _c = _cfg()
    get_env_value = _c.get_env_value
    save_env_value = _c.save_env_value

    for dead_var in ("LLM_MODEL", "OPENAI_MODEL"):
        try:
            old_val = get_env_value(dead_var)
            if old_val:
                save_env_value(dead_var, "")
                if not quiet:
                    print(f"  ✓ Cleared {dead_var} from .env (no longer used — config.yaml is source of truth)")
        except Exception:
            pass


def _migrate_to_14(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 13 → 14: migrate legacy flat stt.model to provider section ──
    # Old configs (and cli-config.yaml.example) had a flat `stt.model` key
    # that was provider-agnostic.  When the provider was "local" this caused
    # OpenAI model names (e.g. "whisper-1") to be fed to faster-whisper,
    # crashing with "Invalid model size".  Move the value into the correct
    # provider-specific section and remove the flat key.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    # Read raw config (no defaults merged) to check what the user actually
    # wrote, then apply changes to the merged config for saving.
    raw = read_raw_config()
    raw_stt = raw.get("stt", {})
    if isinstance(raw_stt, dict) and "model" in raw_stt:
        legacy_model = raw_stt["model"]
        provider = raw_stt.get("provider", "local")
        config = read_raw_config()
        stt = config.get("stt", {})
        # Remove the legacy flat key
        stt.pop("model", None)
        # Place it in the appropriate provider section only if the
        # user didn't already set a model there
        if provider in {"local", "local_command"}:
            # Don't migrate an OpenAI model name into the local section
            _local_models = {
                "tiny.en", "tiny", "base.en", "base", "small.en", "small",
                "medium.en", "medium", "large-v1", "large-v2", "large-v3",
                "large", "distil-large-v2", "distil-medium.en",
                "distil-small.en", "distil-large-v3", "distil-large-v3.5",
                "large-v3-turbo", "turbo",
            }
            if legacy_model in _local_models:
                # Check raw config — only set if user didn't already
                # have a nested local.model
                raw_local = raw_stt.get("local", {})
                if not isinstance(raw_local, dict) or "model" not in raw_local:
                    local_cfg = stt.setdefault("local", {})
                    local_cfg["model"] = legacy_model
            # else: drop it — it was an OpenAI model name, local section
            # already defaults to "base" via DEFAULT_CONFIG
        else:
            # Cloud provider — put it in that provider's section only
            # if user didn't already set a nested model
            raw_provider = raw_stt.get(provider, {})
            if not isinstance(raw_provider, dict) or "model" not in raw_provider:
                provider_cfg = stt.setdefault(provider, {})
                provider_cfg["model"] = legacy_model
        config["stt"] = stt
        _persist_migration(config)
        if not quiet:
            print("  ✓ Migrated legacy stt.model to provider-specific config")


def _migrate_to_16(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 15 → 16: migrate tool_progress_overrides into display.platforms ──
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    display = config.get("display", {})
    if not isinstance(display, dict):
        display = {}
    old_overrides = display.get("tool_progress_overrides")
    if isinstance(old_overrides, dict) and old_overrides:
        platforms = display.get("platforms", {})
        if not isinstance(platforms, dict):
            platforms = {}
        for plat, mode in old_overrides.items():
            if plat not in platforms:
                platforms[plat] = {}
            if "tool_progress" not in platforms[plat]:
                platforms[plat]["tool_progress"] = mode
        display["platforms"] = platforms
        config["display"] = display
        _persist_migration(config)
        if not quiet:
            migrated = ", ".join(f"{p}={m}" for p, m in old_overrides.items())
            print(f"  ✓ Migrated tool_progress_overrides → display.platforms: {migrated}")
        results["config_added"].append("display.platforms (migrated from tool_progress_overrides)")


def _migrate_to_17(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 16 → 17: remove legacy compression.summary_* keys ──
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    comp = config.get("compression", {})
    if isinstance(comp, dict):
        s_model = comp.pop("summary_model", None)
        s_provider = comp.pop("summary_provider", None)
        s_base_url = comp.pop("summary_base_url", None)
        migrated_keys = []
        # Migrate non-empty, non-default values to auxiliary.compression
        if s_model and str(s_model).strip():
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("model"):
                aux_comp["model"] = str(s_model).strip()
                migrated_keys.append(f"model={s_model}")
        if s_provider and str(s_provider).strip() not in {"", "auto"}:
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("provider") or aux_comp.get("provider") == "auto":
                aux_comp["provider"] = str(s_provider).strip()
                migrated_keys.append(f"provider={s_provider}")
        if s_base_url and str(s_base_url).strip():
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("base_url"):
                aux_comp["base_url"] = str(s_base_url).strip()
                migrated_keys.append(f"base_url={s_base_url}")
        if migrated_keys or s_model is not None or s_provider is not None or s_base_url is not None:
            config["compression"] = comp
            _persist_migration(config)
            if not quiet:
                if migrated_keys:
                    print(f"  ✓ Migrated compression.summary_* → auxiliary.compression: {', '.join(migrated_keys)}")
                else:
                    print("  ✓ Removed unused compression.summary_* keys")


def _migrate_to_21(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 20 → 21: plugins are now opt-in; grandfather existing user plugins ──
    # The loader now requires plugins to appear in ``plugins.enabled`` before
    # loading. Existing installs had all discovered plugins loading by default
    # (minus anything in ``plugins.disabled``). To avoid silently breaking
    # those setups on upgrade, populate ``plugins.enabled`` with the set of
    # currently-installed user plugins that aren't already disabled.
    #
    # Bundled plugins (shipped in the repo itself) are NOT grandfathered —
    # they ship off for everyone, including existing users, so any user who
    # wants one has to opt in explicitly.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration
    get_hermes_home = _c.get_hermes_home
    fast_safe_load = _c.fast_safe_load

    config = read_raw_config()
    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    # Only migrate if the enabled allow-list hasn't been set yet.
    if "enabled" not in plugins_cfg:
        disabled = plugins_cfg.get("disabled", []) or []
        if not isinstance(disabled, list):
            disabled = []
        disabled_set = set(disabled)

        # Scan ``$HERMES_HOME/plugins/`` for currently installed user plugins.
        grandfathered: List[str] = []
        try:
            user_plugins_dir = get_hermes_home() / "plugins"
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
                            manifest = fast_safe_load(_mf) or {}
                    except Exception:
                        manifest = {}
                    name = manifest.get("name") or child.name
                    if name in disabled_set:
                        continue
                    grandfathered.append(name)
        except Exception:
            grandfathered = []

        plugins_cfg["enabled"] = grandfathered
        config["plugins"] = plugins_cfg
        _persist_migration(config)
        results["config_added"].append(
            f"plugins.enabled (opt-in allow-list, {len(grandfathered)} grandfathered)"
        )
        if not quiet:
            if grandfathered:
                print(
                    f"  ✓ Plugins now opt-in: grandfathered "
                    f"{len(grandfathered)} existing plugin(s) into plugins.enabled"
                )
            else:
                print(
                    "  ✓ Plugins now opt-in: no existing plugins to grandfather. "
                    "Use `hermes plugins enable <name>` to activate."
                )


def _migrate_to_23(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 22 → 23: seed curator defaults + create logs/curator/ ──
    # The curator (background skill maintenance) was added in PR #16049, but
    # existing configs from before that PR (or before the April 2026
    # unification under `auxiliary.curator`) never wrote the curator section
    # to disk. The runtime deep-merge in `load_config()` fills defaults at
    # read time, so the curator *functions*; but users can't see/edit the
    # settings in their `config.yaml`, and `hermes curator status` has no
    # stable logs dir to point at until the first run mkdir's it.
    #
    # This migration:
    #   1. Writes the `curator` top-level section to config.yaml (enabled,
    #      interval_hours, min_idle_hours, stale_after_days, archive_after_days)
    #      — only keys the user hasn't already overridden.
    #   2. Writes the `auxiliary.curator` aux-task slot (provider, model,
    #      base_url, api_key, timeout, extra_body) — canonical slot for
    #      routing the curator fork to a cheaper aux model.
    #   3. Creates `~/.hermes/logs/curator/` if missing (belt-and-suspenders
    #      on top of ensure_hermes_home() — old profiles that predate this
    #      migration still benefit).
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration
    get_hermes_home = _c.get_hermes_home
    DEFAULT_CONFIG = _c.DEFAULT_CONFIG

    try:
        curator_dir = get_hermes_home() / "logs" / "curator"
        curator_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        results["warnings"].append(f"Could not create {curator_dir}: {e}")

    config = read_raw_config()
    touched = False

    # (1) Top-level curator section — only add missing keys
    _curator_defaults = DEFAULT_CONFIG.get("curator", {})
    raw_curator = config.get("curator")
    if not isinstance(raw_curator, dict):
        raw_curator = {}
    added_curator: List[str] = []
    for k, v in _curator_defaults.items():
        if k not in raw_curator:
            raw_curator[k] = copy.deepcopy(v)
            added_curator.append(k)
    if added_curator:
        config["curator"] = raw_curator
        touched = True

    # (2) auxiliary.curator task slot
    _aux_curator_defaults = (
        DEFAULT_CONFIG.get("auxiliary", {}).get("curator", {})
    )
    raw_aux = config.get("auxiliary")
    if not isinstance(raw_aux, dict):
        raw_aux = {}
    raw_aux_curator = raw_aux.get("curator")
    if not isinstance(raw_aux_curator, dict):
        raw_aux_curator = {}
    added_aux: List[str] = []
    for k, v in _aux_curator_defaults.items():
        if k not in raw_aux_curator:
            raw_aux_curator[k] = copy.deepcopy(v)
            added_aux.append(k)
    if added_aux:
        raw_aux["curator"] = raw_aux_curator
        config["auxiliary"] = raw_aux
        touched = True

    if touched:
        _persist_migration(config)
        if added_curator:
            results["config_added"].append(
                f"curator ({len(added_curator)} default key(s))"
            )
            if not quiet:
                print(
                    "  ✓ Curator settings now available "
                    f"({', '.join(added_curator)}) — edit via `hermes config set`"
                )
        if added_aux:
            results["config_added"].append(
                f"auxiliary.curator ({len(added_aux)} default key(s))"
            )
            if not quiet:
                print(
                    "  ✓ auxiliary.curator settings now available "
                    f"({', '.join(added_aux)}) — edit via `hermes config set`"
                )


def _migrate_to_25(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 24 → 25: lower model_catalog TTL 24h → 1h ──
    # The model picker now refreshes its curated list hourly so freshly
    # published model-catalog.json deploys reach users without a day-long
    # stale window. Only rewrite the OLD default (24) — never clobber a
    # value the user deliberately customized.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_mc = config.get("model_catalog")
    if isinstance(raw_mc, dict) and raw_mc.get("ttl_hours") == 24:
        raw_mc["ttl_hours"] = 1
        config["model_catalog"] = raw_mc
        _persist_migration(config)
        results["config_added"].append("model_catalog.ttl_hours 24→1")
        if not quiet:
            print("  ✓ Lowered model_catalog.ttl_hours to 1 (hourly picker refresh)")


def _migrate_to_29(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 28 → 29: rename memory/skills write_mode → write_approval ──
    # The tri-state write_mode (on|off|approve) was replaced by a clear boolean
    # write_approval (default false = gate off, writes flow freely; true =
    # require approval). Only an explicit "approve" carried gating intent, so
    # it maps to true; everything else (on/off/unset) → false. The old
    # "off = block all writes" mode is dropped — memory_enabled: false disables
    # memory entirely. Only rewrite a key the user actually persisted; never
    # invent one.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

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
            f"{subsystem}.write_mode → write_approval={sub['write_approval']}"
        )
    if touched:
        _persist_migration(config)
        if not quiet:
            print("  ✓ Renamed write_mode → write_approval (boolean gate)")


# ── Version 29 → 30: curator.consolidate defaults to false ──
# Consolidation (the LLM umbrella-building fork) is opt-in, OFF by default;
# the deterministic inactivity prune still runs whenever the curator is
# enabled. No write is needed: the schema default (curator.consolidate=false)
# is supplied by load_config()'s deep-merge at read time, and persisting a
# default-valued key would only bloat a lean config (it gets stripped on
# save anyway). Existing installs that WANT the old always-consolidate
# behavior set it to true explicitly via `hermes config set`.
# (No registry entry: this version bump has no migration step.)


def _migrate_to_31(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 30 → 31: switch verify_on_stop OFF (one-time) ──
    # verify_on_stop defaulted to the "auto" sentinel (surface-aware: on for
    # interactive coding surfaces). In practice the verification narrative was
    # more noise than signal — it even fired on doc/markdown/skill edits with
    # nothing to verify. The new default is OFF. This migration switches
    # existing installs off ONCE, but only when the user never expressed an
    # explicit preference: we rewrite the value only if it's missing or still
    # the "auto" sentinel. An explicit true/false the user set is preserved.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_agent = config.get("agent")
    if not isinstance(raw_agent, dict):
        raw_agent = {}
    cur = raw_agent.get("verify_on_stop")
    is_auto_sentinel = (
        isinstance(cur, str) and cur.strip().lower() == "auto"
    )
    # Only flip the non-committal states; leave explicit bool/on/off alone.
    if cur is None or is_auto_sentinel:
        raw_agent["verify_on_stop"] = False
        config["agent"] = raw_agent
        _persist_migration(config)
        results["config_added"].append("agent.verify_on_stop=false")
        if not quiet:
            print(
                "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false). "
                "Set it to true to re-enable, or \"auto\" for the legacy "
                "surface-aware behavior."
            )


def _migrate_to_32(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 31 → 32: flip the BAKED-IN literal true to OFF (one-time) ──
    # The v30→v31 flip above only caught missing/"auto" values. But the very
    # first ship of verify-on-stop (config v30, commit 2f1a47b90) defaulted
    # DEFAULT_CONFIG["agent"]["verify_on_stop"] to a literal True, and
    # migrate_config persists defaults with strip_defaults=False — so every
    # install that updated through v30 got `verify_on_stop: true` written into
    # config.yaml as a literal. v31's guard deliberately preserves an explicit
    # bool, so it skipped that whole population and left them ON. That literal
    # true was never a user choice: the feature had no off-switch worth setting
    # it against until v31 introduced one, so a true persisted before v32 is
    # always the old machine default. Flip it off once here. A true the user
    # sets AFTER v32 (config already at version 32) is never touched.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_agent = config.get("agent")
    if isinstance(raw_agent, dict) and raw_agent.get("verify_on_stop") is True:
        raw_agent["verify_on_stop"] = False
        config["agent"] = raw_agent
        _persist_migration(config)
        results["config_added"].append("agent.verify_on_stop=false")
        if not quiet:
            print(
                "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false) — "
                "the old default was written into your config as a literal "
                "true. Set it to true again to re-enable, or \"auto\" for the "
                "legacy surface-aware behavior."
            )


def _migrate_to_33(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 32 → 33: unify delegation concurrency caps ──
    # delegation.max_async_children is deprecated: max_concurrent_children now
    # caps both a single batch's parallelism and concurrent background
    # delegation units. Fold a raised max_async_children into
    # max_concurrent_children (take the max so nobody loses headroom), then
    # drop the stale key.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if isinstance(raw_deleg, dict) and "max_async_children" in raw_deleg:
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
                    f"(folded from deprecated max_async_children)"
                )
        config["delegation"] = raw_deleg
        _persist_migration(config)
        if not quiet:
            print(
                "  ✓ Removed deprecated delegation.max_async_children — "
                "delegation.max_concurrent_children now caps background "
                "delegations too."
            )


def _migrate_to_34(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 33 → 34: one-time personality reset (post-#81946 unification) ──
    # Personality persistence used to be split per surface: the TUI/desktop
    # wrote the NAME to display.personality while the CLI/gateway wrote the
    # rendered TEXT into agent.system_prompt (and their "/personality none"
    # only blanked the text, leaving the name behind). When #81946 made
    # display.personality authoritative everywhere, stale names written years
    # ago resurrected personalities users had already turned off ("kawaii
    # defaults on after updating"). There is no way to know which of the two
    # divergent fields reflects the user's intent, so reset the selection to
    # none once and tell the user how to re-enable it. Two scrubs:
    #
    # 1. display.personality → "" (announce the old name).
    # 2. agent.system_prompt → "" ONLY when it verbatim-equals the rendered
    #    text of a known personality — that shape was written by the old
    #    CLI/gateway /personality, never typed by hand. Any other text is a
    #    user-owned manual prompt and is never touched.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    from hermes_cli.personality import (
        available_personalities,
        normalize_personality_name,
        prompt_text,
        render_personality_prompt,
    )

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
                render_personality_prompt(defn)
                for defn in available_personalities(config).values()
            }
            if manual in rendered:
                raw_agent["system_prompt"] = ""
                config["agent"] = raw_agent
                touched = True
                scrubbed_text = True

    if touched:
        _persist_migration(config)
        results["config_added"].append("display.personality=none (one-time reset)")
        if not quiet:
            if old_name:
                print(
                    f"  ✓ Personality reset to none (was '{old_name}'). Personality "
                    "state was previously saved inconsistently across surfaces and "
                    "could re-enable a personality you had turned off. "
                    f"Run /personality {old_name} to turn it back on."
                )
            if scrubbed_text:
                print(
                    "  ✓ Removed personality text from agent.system_prompt (written "
                    "by an older /personality). That field is now reserved for "
                    "manual system prompts; personalities live in display.personality."
                )


def _migrate_to_35(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 34 → 35: background process notifications → concise ──
    # The old default mode 'all' pushed the raw output tail of every finished
    # background process into the chat ("[Background process proc_x finished
    # with exit code 0~ Here's the final output: ...]" walls). The new
    # 'concise' mode renders a one-line status message instead (with a short
    # output tail on failures) and is the new default. Move users still on
    # 'all' — the old implicit default, almost never chosen on purpose — to
    # 'concise'. Explicit non-default choices (result / error / off) are the
    # user's own and are preserved. Users with the key unset inherit the new
    # default automatically at read time (no write needed).
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_display = config.get("display")
    if isinstance(raw_display, dict):
        raw_val = raw_display.get("background_process_notifications")
        if isinstance(raw_val, str) and raw_val.strip().lower() == "all":
            raw_display["background_process_notifications"] = "concise"
            config["display"] = raw_display
            _persist_migration(config)
            results["config_added"].append(
                "display.background_process_notifications=concise (was: all)"
            )
            if not quiet:
                print(
                    "  ✓ Background process notifications switched from 'all' to "
                    "'concise' — completions now show a one-line status message "
                    "instead of the raw output dump. Set "
                    "display.background_process_notifications: all to restore "
                    "the old behavior."
                )


def _migrate_to_36(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 35 → 36: raise the subagent iteration cap default 50 → 250 ──
    # delegation.max_iterations is the per-subagent tool-call budget. The old
    # default of 50 truncated substantial delegated work (leaf agents spend
    # ~15-20 turns on recon before producing output, then ran out mid-task).
    # The shipped default is now 250. Configs still pinned at exactly the old
    # default 50 — almost always the inherited default rather than a deliberate
    # choice — are lifted to 250 so existing installs get the same headroom on
    # update. Any OTHER explicit value (a deliberate override, high or low) is
    # the user's own and is preserved; unset inherits 250 at read time.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if isinstance(raw_deleg, dict) and raw_deleg.get("max_iterations") == 50:
        raw_deleg["max_iterations"] = 250
        config["delegation"] = raw_deleg
        _persist_migration(config)
        results["config_added"].append("delegation.max_iterations=250 (was: 50)")
        if not quiet:
            print(
                "  ✓ Raised delegation.max_iterations from 50 to 250 — subagents "
                "now get a larger per-child tool-call budget so delegated work "
                "finishes instead of truncating. Set delegation.max_iterations "
                "back to 50 to restore the old cap."
            )


def _migrate_to_37(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 36 → 37: raise the delegation concurrency default 3 → 10 ──
    # delegation.max_concurrent_children caps how many children run in parallel
    # per batch (and concurrent background delegation units). The old default of
    # 3 needlessly serialized independent fan-outs (e.g. reviewing N PRs at
    # once). The shipped default is now 10, which stays at/below the high-cost
    # warning threshold. Configs still pinned at exactly the old default 3 —
    # almost always the inherited default rather than a deliberate choice — are
    # lifted to 10 so existing installs get the wider fan-out on update. Any
    # OTHER explicit value (a deliberate override) is preserved; unset inherits
    # 10 at read time.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if isinstance(raw_deleg, dict) and raw_deleg.get("max_concurrent_children") == 3:
        raw_deleg["max_concurrent_children"] = 10
        config["delegation"] = raw_deleg
        _persist_migration(config)
        results["config_added"].append("delegation.max_concurrent_children=10 (was: 3)")
        if not quiet:
            print(
                "  ✓ Raised delegation.max_concurrent_children from 3 to 10 — "
                "independent delegated children now fan out wider in parallel. "
                "Each child consumes API tokens independently; set "
                "delegation.max_concurrent_children back to 3 to restore the old cap."
            )


def _migrate_to_38(results: Dict[str, Any], quiet: bool) -> None:
    # Version 37 → 38: the bundled observability/nemo_relay plugin was
    # removed when Relay lifecycle ownership moved into the agent core.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

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
        "HERMES_NEMO_RELAY_PLUGINS_TOML."
    )
    results["warnings"].append(message)
    if not quiet:
        print(f"  ⚠ {message}")


def _migrate_to_39(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 38 → 39: remove the retired `bfl` toolset from saved lists ──
    # The six bfl_flux3_* core tools shipped for a free FLUX 3 promotional
    # period that has since ended server-side, leaving every Nous-signed-in
    # install paying ~2.7K tokens of schema per API call for tools that can
    # only refuse. They were removed in favor of the standard video_gen
    # provider surface (`video_generate`, `hermes tools` → Video Generation).
    # Strip the toolset key wherever the auto-backfill or a picker save wrote
    # it, so stale config can't resurrect an unknown toolset.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

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
        _persist_migration(config)
        results["config_added"].append("removed retired 'bfl' toolset from saved toolset lists")
        if not quiet:
            print(
                "  ✓ Removed the retired BFL FLUX 3 toolset from saved toolset "
                "lists — video generation now lives under `hermes tools` → "
                "Video Generation (Nous Subscription or FAL)."
            )


def _migrate_to_40(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 39 → 40: model_catalog.ttl_hours → ttl_minutes (default 20) ──
    # The picker catalogs now refresh every 20 minutes (and the gateway
    # refreshes them in the background on that cadence). Only the OLD default
    # (ttl_hours: 1, written by the v25 migration) is dropped so the new
    # default applies; any other explicit ttl_hours is a deliberate choice
    # and stays honoured by the loader.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_mc = config.get("model_catalog")
    if isinstance(raw_mc, dict) and raw_mc.get("ttl_hours") == 1 and "ttl_minutes" not in raw_mc:
        del raw_mc["ttl_hours"]
        config["model_catalog"] = raw_mc
        _persist_migration(config)
        results["config_added"].append("model_catalog.ttl_hours 1 → ttl_minutes 20 (default)")
        if not quiet:
            print("  ✓ Model catalog now refreshes every 20 minutes (model_catalog.ttl_minutes)")


#: Registry of (target_version, migration_fn), strictly ascending. The driver
#: applies every entry whose target version is greater than the on-disk
#: observe earlier steps' writes via read_raw_config() (filesystem state).
MIGRATIONS: Tuple[Tuple[int, Callable[[Dict[str, Any], bool], None]], ...] = (
    # v12 is the support floor: configs already AT v12 (or newer) still get
    # every remaining step below. Only configs BELOW 12 are refused by the
    # floor gate in run_migrations().
    (12, _migrate_to_12),
    (13, _migrate_to_13),
    (14, _migrate_to_14),
    # v15 only added a schema default; runtime merging supplies it without a
    # write. Registering a migration would falsely report or materialise it.
    (16, _migrate_to_16),
    (17, _migrate_to_17),
    (21, _migrate_to_21),
    (23, _migrate_to_23),
    (25, _migrate_to_25),
    (29, _migrate_to_29),
    (31, _migrate_to_31),
    (32, _migrate_to_32),
    (33, _migrate_to_33),
    (34, _migrate_to_34),
    (35, _migrate_to_35),
    (36, _migrate_to_36),
    (37, _migrate_to_37),
    (38, _migrate_to_38),
    (39, _migrate_to_39),
    (40, _migrate_to_40),
)


def run_migrations(current_ver: int, results: Dict[str, Any], quiet: bool) -> None:
    """Apply every registered migration whose target version exceeds *current_ver*.

    Replicates the original ladder's semantics exactly: *current_ver* is the
    on-disk schema version captured ONCE (via ``check_config_version()``)
    before any step runs, and it does not advance between steps — each step
    is gated on the same initial value, exactly like the original sequential
    ``if current_ver < N:`` blocks. Steps run in strict ascending registry
    order and mutate ``results`` in place. The final ``_config_version`` bump
    is NOT performed here; it stays in ``migrate_config`` (persisted once,
    after the informational missing-config scan), matching the original flow.
    """
    for target_ver, migration_fn in MIGRATIONS:
        if current_ver < target_ver:
            migration_fn(results, quiet)
