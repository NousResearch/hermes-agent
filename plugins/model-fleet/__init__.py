"""model-fleet — repoint a whole Hermes install at one provider/model in one command.

A Hermes install accumulates several independent model pins: the default model, the
sub-agent (delegation) model, every named profile's config, every cron job, and the
auxiliary task models. Keeping them aligned by hand means editing a dozen files and
repeating the edit on the next upgrade. This plugin does it in one step and shows
exactly what it will touch.

Slash flow (text args, staged through a short-lived cache so it works in Discord,
Telegram and the CLI alike):

    /model-fleet                            numbered list of authenticated providers
    /model-fleet <p>                        numbered model list for provider #p
    /model-fleet <p> <m>                    apply provider/model everywhere
    /model-fleet <provider> <model>         same, addressed by slug
    /model-fleet status                     current model map across the install
    /model-fleet auxiliary <provider> <m>   repoint the auxiliary.* task models too

Add ``--dry-run`` to any apply form to preview without writing.

Scope is controlled by the ``config_schema`` keys in plugin.yaml (see
``include_profiles``, ``include_cron``, ``include_auxiliary``, ``profile_allowlist``,
``profile_blocklist``, ``model_allowlist``, ``backup``), so a user with no profiles or
no cron jobs simply gets a smaller blast radius without editing this file.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes_model_fleet")

PLUGIN_ID = "model-fleet"
BACKUP_SUFFIX = "model-fleet"
_HELP = (
    "**/model-fleet** — repoint every agent, sub-agent and cron job at one provider/model\n"
    "```\n"
    "/model-fleet                            list providers\n"
    "/model-fleet <provider>                  list that provider's models\n"
    "/model-fleet <provider> <model>          apply across the install (backs up first)\n"
    "/model-fleet status                     current model map\n"
    "/model-fleet auxiliary <provider> <m>   also repoint auxiliary task models\n"
    "add --dry-run to any apply form to preview without writing\n"
    "```"
)

# Staged-pick cache. {"providers": [...], "at": ts} after a provider listing,
# {"models": [...], "provider": slug, "at": ts} after a model listing. Purely a
# convenience for numbered picks: every resolver falls back to live data, so a cold
# or stale cache can never change what a slug-addressed apply does.
_CACHE: Dict[str, Any] = {}
DEFAULT_MODEL_LIMIT = 25
# A numbered pick is a two-step interaction; the cache only has to outlive the gap
# between "/model-fleet" and "/model-fleet 3". Past this it is re-derived from live
# data so a config change (e.g. model_allowlist) cannot be silently bypassed.
CACHE_TTL_SECONDS = 900

_SETTING_DEFAULTS: Dict[str, Any] = {
    "include_profiles": True,
    "include_cron": True,
    "include_auxiliary": False,
    "profile_allowlist": [],
    "profile_blocklist": [],
    "model_allowlist": [],
    "backup": True,
}
_BOOL_SETTINGS = ("include_profiles", "include_cron", "include_auxiliary", "backup")
_LIST_SETTINGS = ("profile_allowlist", "profile_blocklist", "model_allowlist")


# --------------------------------------------------------------------------- config

def _settings() -> Dict[str, Any]:
    """Plugin settings from config.yaml (``plugins.entries.model-fleet``), defaults on miss."""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.plugins_state import _plugin_settings_entry

        entry = _plugin_settings_entry(load_config_readonly() or {}, PLUGIN_ID) or {}
    except Exception as exc:
        logger.debug("model-fleet: falling back to default settings (%s)", exc)
        entry = {}
    merged = dict(_SETTING_DEFAULTS)
    for key, default in _SETTING_DEFAULTS.items():
        if entry.get(key) is not None:
            merged[key] = entry[key]
    # Coerce defensively: config.yaml is user-editable text.
    for key in _BOOL_SETTINGS:
        merged[key] = bool(merged[key])
    for key in _LIST_SETTINGS:
        value = merged[key]
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        merged[key] = list(value) if isinstance(value, (list, tuple)) else []
    return merged


# --------------------------------------------------------------------------- paths

def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _profile_selected(label: str, settings: Dict[str, Any]) -> bool:
    """Both filters apply: the allowlist narrows, the blocklist always wins.

    An early return on a non-empty allowlist silently ignored the blocklist, so
    ``allowlist: [alpha, beta]`` + ``blocklist: [beta]`` still selected beta.
    """
    allow = {str(n) for n in settings["profile_allowlist"]}
    if allow and label not in allow:
        return False
    return label not in {str(n) for n in settings["profile_blocklist"]}


def _active_config_path() -> Path:
    """``config.yaml`` of the home this invocation is scoped to."""
    return _hermes_home() / "config.yaml"


def _install_root() -> Path:
    """The installation's real default home, independent of the caller's profile scope.

    ``_hermes_home()`` is context-local: under named-profile dispatch it returns that
    profile's home, so anchoring enumeration there made a fleet command from a
    secondary profile look for ``<named>/profiles/`` and miss the actual default home
    and its siblings. Walk up out of a profile home to the installation root.
    """
    from hermes_constants import named_profile_home

    home = _hermes_home()
    try:
        # Resolves <root>/profiles/<name> back to <root>; None when home IS the root.
        named = named_profile_home(home)
    except Exception:
        named = None
    if named is not None:
        return named.parent.parent
    return home


def _profile_homes(settings: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """[(label, home)] — the real default home always, named profiles when in scope."""
    root = _install_root()
    out: List[Tuple[str, Path]] = [("default", root)]
    if not settings["include_profiles"]:
        return out
    profiles_dir = root / "profiles"
    if not profiles_dir.is_dir():
        return out
    for child in sorted(profiles_dir.iterdir()):
        if not child.is_dir() or not (child / "config.yaml").is_file():
            continue
        if _profile_selected(child.name, settings):
            out.append((child.name, child))
    return out


# --------------------------------------------------------------------------- listing

def _list_providers_sync(limit: int = 200) -> List[dict]:
    from hermes_cli.config import load_config_readonly
    from hermes_cli.model_switch_providers import list_picker_providers

    cfg = load_config_readonly() or {}
    model_cfg = cfg.get("model") or {}
    rows = list_picker_providers(
        current_provider=model_cfg.get("provider", "") or "",
        current_base_url=model_cfg.get("base_url", "") or "",
        user_providers=cfg.get("providers"),
        custom_providers=cfg.get("custom_providers"),
        max_models=limit,
        current_model=model_cfg.get("default", "") or "",
        include_moa=False,
        # Cache-only catalog reads: a cold provider must never stall the slash
        # handler on a live /models round-trip. Rows warm in the background.
        non_blocking_catalogs=True,
        probe_custom_providers=False,
    )
    out = [r for r in rows if r.get("slug")]
    allow = {str(n).lower() for n in _settings()["model_allowlist"]}
    return [r for r in out if not allow or str(r.get("slug", "")).lower() in allow]


def _providers_view() -> str:
    rows = _list_providers_sync()
    _CACHE.clear()
    _CACHE.update({"providers": rows, "at": time.time()})
    lines = ["**Pick a provider** — reply `/model-fleet <n>` to see its models:", ""]
    for i, row in enumerate(rows, 1):
        mark = " ← current" if row.get("is_current") else ""
        lines.append(f"{i}. **{row.get('name')}** (`{row.get('slug')}`){mark} — {row.get('total_models', 0)} models")
    if not rows:
        lines.append("_No authenticated providers found. Run `hermes auth list`._")
    lines += ["", _HELP]
    return "\n".join(lines)


def _models_view(provider: str) -> str:
    rows = _list_providers_sync()
    row = _match_provider(rows, provider)
    if row is None:
        return f"Unknown provider `{provider}`. Run `/model-fleet` for the numbered list."
    models = [m for m in (row.get("models") or []) if m]
    _CACHE.clear()
    _CACHE.update({"models": models, "provider": row.get("slug"), "at": time.time()})
    lines = [
        f"**{row.get('name')}** (`{row.get('slug')}`) — {row.get('total_models', len(models))} models.",
        "",
        "Reply `/model-fleet <provider> <model>` to apply across the install:",
        "",
    ]
    lines += [f"{i}. `{m}`" for i, m in enumerate(models[:DEFAULT_MODEL_LIMIT], 1)]
    if len(models) > DEFAULT_MODEL_LIMIT:
        lines.append(f"… and {len(models) - DEFAULT_MODEL_LIMIT} more — name the model id directly to use one of those.")
    return "\n".join(lines)


def _cache_fresh(key: str, current_rows: Optional[List[dict]] = None) -> bool:
    """True when the cached listing for *key* is still safe to resolve a number against.

    Expired by age, and — when the caller supplies the freshly-derived rows — only
    fresh if the cache still agrees with them, so a config change (e.g. a tightened
    ``model_allowlist``) invalidates a pending numbered pick.
    """
    if not _CACHE.get(key):
        return False
    at = _CACHE.get("at")
    if not isinstance(at, (int, float)) or (time.time() - at) > CACHE_TTL_SECONDS:
        return False
    if current_rows is not None:
        cached_slugs = [str(r.get("slug", "")) for r in (_CACHE.get(key) or [])]
        live_slugs = [str(r.get("slug", "")) for r in current_rows]
        if cached_slugs != live_slugs:
            return False
    return True


def _match_provider(rows: List[dict], token: str) -> Optional[dict]:
    token = str(token or "").strip()
    if not token:
        return None
    if token.isdigit():
        # Resolve a number against the CACHED listing only while that cache is fresh
        # and was built from the same filter. Once stale, fall back to `rows`, which
        # the caller has just re-derived from live data through model_allowlist —
        # resolving against a stale snapshot could select a now-disallowed provider.
        pool = _CACHE.get("providers") or []
        if not _cache_fresh("providers", rows):
            pool = rows
        idx = int(token) - 1
        return pool[idx] if 0 <= idx < len(pool) else None
    low = token.lower()
    for row in rows:
        if str(row.get("slug", "")).lower() == low:
            return row
    for row in rows:  # display-name substring match
        if low in str(row.get("name", "")).lower():
            return row
    return None


def _match_model(provider_row: dict, token: str) -> Optional[str]:
    token = str(token or "").strip()
    models = [m for m in ((provider_row or {}).get("models") or []) if m]
    if not token:
        return None
    if token.isdigit():
        # Same staleness rule as providers: the cached list is only trusted while it is
        # fresh AND belongs to the provider being resolved against.
        cached: List[str] = []
        if (_CACHE.get("models")
                and _CACHE.get("provider") == str((provider_row or {}).get("slug"))
                and _cache_fresh("models")):
            cached = list(_CACHE.get("models") or [])
        if cached and cached == list(models):
            idx = int(token) - 1
            return cached[idx] if 0 <= idx < len(cached) else None
        idx = int(token) - 1
        return models[idx] if 0 <= idx < len(models) else None
    for model in models:
        if model.lower() == token.lower():
            return model
    low = token.lower()
    for model in models:  # substring before prefix: 'sonnet' should beat 'sonnet-4-8'
        if low in model.lower():
            return model
    for model in models:
        if model.lower().startswith(low):
            return model
    return None


# --------------------------------------------------------------------------- status

def _status_sync() -> str:
    settings = _settings()
    lines = ["**Model map**", ""]
    for label, home in _profile_homes(settings):
        try:
            import yaml

            cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8-sig")) or {}
        except Exception as exc:  # report, never swallow
            lines.append(f"- **{label}**: config unreadable ({exc})")
            continue
        model = cfg.get("model") or {}
        dele = cfg.get("delegation") or {}
        lines.append(
            f"- **{label}**: `{model.get('provider', '?')}/{model.get('default', '?')}`"
            f" · sub-agents `{dele.get('provider') or 'inherit'}/{dele.get('model') or 'inherit'}`"
        )

    if settings["include_auxiliary"]:
        try:
            import yaml

            cfg = yaml.safe_load((_hermes_home() / "config.yaml").read_text(encoding="utf-8-sig")) or {}
            for task, spec in sorted((cfg.get("auxiliary") or {}).items()):
                if isinstance(spec, dict) and spec.get("model"):
                    lines.append(f"- **auxiliary/{task}**: `{spec.get('provider', '?')}/{spec['model']}`")
        except Exception as exc:
            lines.append(f"- **auxiliary**: unreadable ({exc})")

    if settings["include_cron"]:
        from cron.jobs import load_jobs, use_cron_store

        for label, home in _profile_homes(settings):
            if not (home / "cron" / "jobs.json").is_file():
                continue
            try:
                with use_cron_store(home):
                    jobs = load_jobs()
            except Exception as exc:
                lines.append(f"- **cron/{label}**: unreadable ({exc})")
                continue
            counts: Dict[Tuple[str, str], int] = {}
            for job in jobs:
                key = (job.get("provider") or "inherit", job.get("model") or "inherit")
                counts[key] = counts.get(key, 0) + 1
            for (prov, model), n in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"- **cron/{label}**: {n} job(s) → `{prov}/{model}`")
    lines += ["", _HELP]
    return "\n".join(lines)


# --------------------------------------------------------------------------- switch

def _switch_result(provider: str, model: str):
    """Run the canonical model-switch pipeline against the active profile config.

    Reusing ``switch_model`` (rather than writing model.provider by hand) is what
    makes this safe: it re-resolves credentials, base_url, api_mode and the context
    pin for the target route instead of leaving the old provider's endpoint behind.
    """
    from hermes_cli.config import get_config_path, read_user_config_raw
    from hermes_cli.model_switch import switch_model

    cfg = read_user_config_raw(get_config_path())
    model_cfg = cfg.get("model") or {}
    return switch_model(
        raw_input=model,
        current_provider=model_cfg.get("provider", "") or "",
        current_model=model_cfg.get("default", "") or "",
        current_base_url=model_cfg.get("base_url", "") or "",
        current_api_key=str(model_cfg.get("api_key") or ""),
        is_global=True,
        explicit_provider=provider,
        user_providers=cfg.get("providers"),
        custom_providers=cfg.get("custom_providers"),
    )


def _backup(path: Path, stamp: str) -> Optional[Path]:
    """Copy *path* beside itself. Returns None on failure — callers decide the policy.

    With ``backup: true`` a caller must ABORT rather than continue: proceeding without
    the copy is how a config gets rewritten with no way back, which is strictly worse
    than refusing the switch.
    """
    try:
        bak = path.with_name(f"{path.name}.bak-{BACKUP_SUFFIX}-{stamp}")
        shutil.copy2(path, bak)
        return bak
    except OSError as exc:
        logger.warning("model-fleet: backup of %s failed: %s", path, exc)
        return None


class BackupFailed(RuntimeError):
    """A required backup could not be written; the apply must not proceed."""


def _backup_or_abort(path: Path, stamp: str) -> str:
    bak = _backup(path, stamp)
    if bak is None:
        raise BackupFailed(f"could not back up {path}")
    return str(bak)


def _resolve_in_profile(home: Path, provider: str, model: str):
    """Resolve provider/model inside *home*'s own scope.

    Returns ``(result, updates)`` for this profile, or ``None`` when the target does
    not resolve there. Profiles are independent islands, so a slug can point at a
    different endpoint in each one and each carries its own credentials; the caller's
    already-resolved result must not be reused for a sibling profile.

    ``set_hermes_home_override`` is context-local (it never touches ``os.environ``), so
    scoping here cannot leak into the caller's home or another thread.
    """
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.model_switch import model_selection_config_updates, switch_model
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    cfg_path = Path(home) / "config.yaml"
    # Read this profile's OWN config for the "current" side of the switch, under the
    # override so the config reader is scoped to it too. switch_model has no
    # ``provider=`` kwarg: the target is ``explicit_provider``, and current_provider /
    # current_model / base_url / api_key are required so it can re-resolve credentials
    # for this profile's own route rather than inheriting the caller's.
    token = set_hermes_home_override(Path(home))
    try:
        raw = read_user_config_raw(cfg_path) if cfg_path.is_file() else {}
        model_cfg = raw.get("model") or {}
        result = switch_model(
            raw_input=model,
            current_provider=model_cfg.get("provider", "") or "",
            current_model=model_cfg.get("default", "") or "",
            current_base_url=model_cfg.get("base_url", "") or "",
            current_api_key=str(model_cfg.get("api_key") or ""),
            is_global=True,
            explicit_provider=provider,
            user_providers=raw.get("providers"),
            custom_providers=raw.get("custom_providers"),
        )
        if not getattr(result, "success", False):
            return None
        return result, model_selection_config_updates(result, raw.get("model"))
    except Exception as exc:
        logger.debug("model-fleet: %s did not resolve in %s (%s)", model, home, exc)
        return None
    finally:
        reset_hermes_home_override(token)


def _apply_profiles(provider: str, model: str, result, settings: Dict[str, Any],
                   stamp: str, dry_run: bool) -> Tuple[List[str], List[str]]:
    """Write model.* + delegation.* into every in-scope profile config."""
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.model_switch import model_selection_config_updates
    from utils import atomic_roundtrip_yaml_update

    changed: List[str] = []
    backups: List[str] = []
    for label, home in _profile_homes(settings):
        cfg_path = home / "config.yaml"
        if not cfg_path.is_file():
            continue
        raw = read_user_config_raw(cfg_path)
        # Resolve the route IN THIS PROFILE'S SCOPE. Profiles are independent islands:
        # the same provider slug can mean a different endpoint here than in the profile
        # the switch was resolved from, and only this profile's credentials are checked.
        # Reusing the caller's result would write another profile's base_url/api_mode.
        scoped = _resolve_in_profile(home, provider, model)
        if scoped is None:
            changed.append(f"{label}: SKIPPED — {provider}/{model} does not resolve here")
            continue
        result, scoped_updates = scoped
        updates = scoped_updates
        delegation_updates: Dict[str, Any] = {"model": model, "provider": provider}
        current = raw.get("delegation") or {}
        if str(current.get("provider") or "") != provider:
            # endpoint / key pins belong to the old provider — clear them with the route
            for stale in ("base_url", "api_key", "api_mode"):
                if current.get(stale):
                    delegation_updates[stale] = None
        summary = f"{label}: model→{provider}/{model}, delegation→{provider}/{model}"
        if dry_run:
            changed.append(summary)
            continue
        if settings["backup"]:
            # The active home's own config.yaml is already backed up in _apply_sync,
            # before persist_model_selection rewrote it. Re-copying here would clobber
            # that pre-change copy with post-change content (same stamp, same filename).
            if cfg_path != _active_config_path():
                backups.append(_backup_or_abort(cfg_path, stamp))
        for key, value in updates.items():
            atomic_roundtrip_yaml_update(cfg_path, f"model.{key}", value)
        for key, value in delegation_updates.items():
            atomic_roundtrip_yaml_update(cfg_path, f"delegation.{key}", value)
        changed.append(summary)
    return changed, backups


def _apply_crons(provider: str, model: str, settings: Dict[str, Any], stamp: str,
                 dry_run: bool) -> Tuple[List[str], List[str], List[str]]:
    """Repoint agent-running cron jobs in every in-scope profile store.

    Load, mutate and save happen inside ONE ``_jobs_lock()`` hold. Splitting them
    (load under the lock, save under another) meant a scheduler tick that updated
    ``next_run_at`` / ``fire_claim`` / completion state in between got overwritten with
    the stale copy we read. ``save_jobs`` only re-inserts job IDs missing from the
    payload; it does not merge newer fields for an ID that is present, so a
    last-writer-wins save silently reverts live scheduler state. Same shape as core's
    ``_with_job()``.
    """
    from cron.jobs import _jobs_lock, load_jobs, save_jobs, use_cron_store

    changed: List[str] = []
    skipped: List[str] = []
    backups: List[str] = []
    for label, home in _profile_homes(settings):
        jobs_file = home / "cron" / "jobs.json"
        if not jobs_file.is_file():
            continue

        def _repoint(jobs: List[dict], provider: str = provider, model: str = model) -> Tuple[int, int]:
            touched = already = 0
            for job in jobs:
                if job.get("no_agent"):
                    continue  # pure script job — no model in that loop
                if job.get("provider") == provider and job.get("model") == model:
                    already += 1
                    continue
                for key, value in (("provider", provider), ("model", model),
                                   ("provider_snapshot", provider), ("model_snapshot", model)):
                    job[key] = value
                touched += 1
            return touched, already

        # Single locked transaction: the plan is computed from the same snapshot that
        # gets written, so nothing the scheduler changed in between can be clobbered.
        try:
            with use_cron_store(home), _jobs_lock():
                jobs = load_jobs()
                touched, already = _repoint(jobs)
                if touched and not dry_run and settings["backup"]:
                    backups.append(_backup_or_abort(jobs_file, stamp))
                if touched and not dry_run:
                    save_jobs(jobs)
        except BackupFailed:
            # A backup-policy abort is not an unreadable store. Re-raise so the caller
            # surfaces "Switch aborted" instead of silently skipping the profile.
            raise
        except Exception as exc:
            skipped.append(f"cron/{label}: unreadable ({exc})")
            continue
        if touched == 0:
            changed.append(f"cron/{label}: {already} agent job(s) already on {provider}/{model}")
            continue
        verb = "would repoint" if dry_run else "repointed"
        changed.append(f"cron/{label}: {verb} {touched} agent job(s) → {provider}/{model}")
    return changed, skipped, backups


def _apply_auxiliary(provider: str, model: str, settings: Dict[str, Any], stamp: str,
                     dry_run: bool) -> Tuple[List[str], List[str]]:
    """Repoint auxiliary.<task> for every task that declares a model."""
    from utils import atomic_roundtrip_yaml_update

    cfg_path = _hermes_home() / "config.yaml"
    if not cfg_path.is_file():
        return [], []
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:
        return [f"auxiliary: config unreadable ({exc})"], []
    aux = cfg.get("auxiliary") or {}
    if not isinstance(aux, dict):
        return [], []
    tasks = [t for t, spec in sorted(aux.items()) if isinstance(spec, dict) and spec.get("model")]
    if not tasks:
        return ["auxiliary: no task declares a model"], []
    if dry_run:
        return [f"auxiliary: would set {len(tasks)} task(s) → {provider}/{model}"], []
    backups: List[str] = []
    if settings["backup"]:
        # Skip the active config: it is already backed up in _apply_sync before the
        # first write. A second copy under the same stamp would overwrite that
        # pre-change copy with a post-change one, making a restore a silent no-op.
        if cfg_path != _active_config_path():
            backups.append(_backup_or_abort(cfg_path, stamp))
    for task in tasks:
        atomic_roundtrip_yaml_update(cfg_path, f"auxiliary.{task}.provider", provider)
        atomic_roundtrip_yaml_update(cfg_path, f"auxiliary.{task}.model", model)
        # endpoint / key pins belong to the old provider — clear them with the route
        atomic_roundtrip_yaml_update(cfg_path, f"auxiliary.{task}.base_url", None)
        atomic_roundtrip_yaml_update(cfg_path, f"auxiliary.{task}.api_key", None)
    return [f"auxiliary: set {len(tasks)} task(s) → {provider}/{model}"], backups


def _close_match_hint(token: str, models: List[str], limit: int = 5) -> str:
    """`` Close matches: `a`, `b`.`` for a mistyped model id, else ``""``.

    Substring alone is not enough: a typo like ``glm-5.3-flahs`` is not a substring of
    anything, which is precisely when a hint helps. difflib covers both the typo and
    the wrong-vendor (``gpt-5`` typed against a Nous-only provider) cases.
    """
    import difflib

    low = str(token).lower()
    near = [m for m in models if low in m.lower()]
    if not near:
        near = difflib.get_close_matches(low, [m.lower() for m in models], n=limit, cutoff=0.6)
        near = [m for m in models if m.lower() in near]
    if not near:
        return ""
    return " Close matches: " + ", ".join(f"`{m}`" for m in near[:limit]) + "."


def _resolve_target(provider_token: str, model_token: str) -> Tuple[Optional[str], Optional[str], str]:
    """→ (provider_slug, model_id, error). Model resolution needs the live provider row."""
    rows = _list_providers_sync()
    row = _match_provider(rows, provider_token)
    if row is None:
        return None, None, f"Unknown provider `{provider_token}`. Run `/model-fleet` for the numbered list."
    model = _match_model(row, model_token)
    if model is None:
        models = [m for m in (row.get("models") or []) if m]
        if model_token.isdigit() and int(model_token) - 1 >= len(models):
            return None, None, f"No model #{model_token} for `{row.get('slug')}` (it lists {len(models)})."
        return None, None, f"`{model_token}` is not a model of `{row.get('slug')}`.{_close_match_hint(model_token, models)}"
    return str(row.get("slug")), model, ""


def _apply_sync(provider: str, model: str, dry_run: bool, with_auxiliary: bool) -> str:
    settings = _settings()
    if with_auxiliary and not settings["include_auxiliary"]:
        # An explicit `auxiliary` subcommand is the user asking for it; honour that for
        # this run without silently rewriting their saved config.
        settings = dict(settings, include_auxiliary=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result = _switch_result(provider, model)
    if not getattr(result, "success", False):
        return (f"**Switch refused** for `{provider}/{model}`: "
                f"{getattr(result, 'error_message', 'unknown error')}")

    if dry_run:
        # Reuse the resolved route without persisting it, so the preview matches the write.
        profile_changes, _ = _apply_profiles(provider, model, result, settings, stamp, True)
        cron_changes, cron_skipped, _ = _apply_crons(provider, model, settings, stamp, True)
        aux_changes = (_apply_auxiliary(provider, model, settings, stamp, True)[0]
                       if settings["include_auxiliary"] else [])
        lines = [f"**Dry run** — would set the install to `{provider}/{model}`", ""]
        lines += [f"- {c}" for c in profile_changes]
        lines += [f"- {c}" for c in cron_changes]
        lines += [f"- {c}" for c in aux_changes]
        lines += [f"- skipped: {s}" for s in cron_skipped]
        lines += ["", "Re-run without `--dry-run` to apply."]
        return "\n".join(lines)

    # Back the active config up BEFORE anything writes to it. The active home is also
    # enumerated as the "default" profile by _apply_profiles(), which would otherwise
    # take a second copy under the same stamp AFTER persist_model_selection() and
    # overwrite this pre-change copy — making a restore a silent no-op.
    try:
        backups: List[str] = []
        if settings["backup"]:
            active_cfg = _active_config_path()
            if active_cfg.is_file():
                backups.append(_backup_or_abort(active_cfg, stamp))
    except BackupFailed as exc:
        # Refuse rather than continue: without the copy there is no way back.
        return f"**Switch aborted** for `{provider}/{model}`: {exc}. Nothing was written."

    try:
        return _commit(provider, model, result, settings, stamp, backups)
    except BackupFailed as exc:
        # A per-profile/cron/auxiliary target could not be backed up. The already-written
        # files are restorable from the backups taken above; say so instead of failing
        # silently, and name what to restore.
        return (f"**Switch aborted partway** for `{provider}/{model}`: {exc}.\n"
                "Restore from the backups listed below (or run with `backup: false` to "
                f"skip them deliberately).\n{chr(10).join(f'- `{b}`' for b in backups)}")


def _commit(provider: str, model: str, result, settings: Dict[str, Any], stamp: str,
            backups: List[str]) -> str:
    from hermes_cli.model_switch import persist_model_selection

    persist_model_selection(result)
    profile_changes, profile_backups = _apply_profiles(provider, model, result, settings, stamp, False)
    cron_changes, cron_skipped, cron_backups = _apply_crons(provider, model, settings, stamp, False)
    aux_changes, aux_backups = _apply_auxiliary(provider, model, settings, stamp, False) if \
        settings["include_auxiliary"] else ([], [])

    lines = [f"**Install switched to `{provider}/{model}`**", ""]
    lines.append(f"- default model: {provider}/{model} (config.yaml, credentials re-resolved)")
    lines += [f"- {c}" for c in profile_changes]
    lines += [f"- {c}" for c in cron_changes]
    lines += [f"- {c}" for c in aux_changes]
    if cron_skipped:
        lines += [f"- skipped: {s}" for s in cron_skipped]
    backups = backups + profile_backups + cron_backups + aux_backups
    if backups:
        lines += ["", f"Backups ({len(backups)}): " + ", ".join(f"`{b}`" for b in backups)]
    lines += [
        "",
        "New sessions and the next cron run pick this up. The chat that ran the command "
        "keeps its current model until a new session starts.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- routing

def _parse(raw: str) -> Tuple[List[str], bool]:
    tokens = re.split(r"\s+", str(raw or "").strip()) if raw else []
    dry = False
    kept: List[str] = []
    for tok in tokens:
        if tok in {"--dry-run", "-n", "--preview"}:
            dry = True
        else:
            kept.append(tok)
    return kept, dry


def _handle_sync(raw_args: str) -> Optional[str]:
    tokens, dry_run = _parse(raw_args)

    if tokens and tokens[0] in {"help", "-h", "--help"}:
        return _HELP
    if not tokens:
        return _providers_view()
    if tokens[0].lower() in {"status", "show", "current"}:
        return _status_sync()

    if tokens[0].lower() == "auxiliary":
        rest = tokens[1:]
        if not rest:
            return "Repoint the auxiliary task models too.\nUsage: `/model-fleet auxiliary <provider> <model>`"
        if len(rest) == 1:
            return _models_view(rest[0])
        provider, model, err = _resolve_target(rest[0], " ".join(rest[1:]).strip())
        if err:
            return err
        return _apply_sync(provider, model, dry_run, with_auxiliary=True)

    if len(tokens) == 1:
        return _models_view(tokens[0]) if _match_provider(_list_providers_sync(), tokens[0]) \
            else _providers_view()

    provider, model, err = _resolve_target(tokens[0], " ".join(tokens[1:]).strip())
    if err:
        return err
    return _apply_sync(provider, model, dry_run, with_auxiliary=False)


async def _handle(raw_args: str) -> Optional[str]:
    # Provider listing and the switch pipeline can each block on network I/O.
    return await asyncio.to_thread(_handle_sync, raw_args)


def register(ctx) -> None:  # noqa: ANN001
    ctx.register_command(
        "model-fleet",
        _handle,
        description="Switch the default model, sub-agents, profiles and cron jobs to one provider/model.",
        args_hint="[<provider> [<model>]] [--dry-run] | status | auxiliary <provider> <model>",
    )
