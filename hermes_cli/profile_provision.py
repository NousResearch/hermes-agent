"""The one way to create a profile: ``provision_profile(ProfileCreateRequest) -> ProvisionResult``.

Every surface (``hermes profile create``, ``POST /api/profiles``, the Desktop/TUI ``profiles.create``
RPC) builds a request and renders the result; none of them writes into a new profile itself. The
core directory work (validate, copy, strip channels, stage, publish, notify the multiplexer) stays
in :func:`hermes_cli.profiles.create_profile`; this module owns everything that runs after it, in
one fixed order, so a change to what a new profile gets is made once and reaches every surface.
``scripts/check_profile_create_entry.py`` keeps ``create_profile`` callable only from here.

Each surface still passes the options it used before this module existed (the CLI and the dashboard
do not mirror launch credentials, the RPC does), so routing a surface through here changes nothing
it does. Unifying those defaults is a later, separate change.

Every step after the core create is best-effort: the profile already exists, so a failing step never
undoes the create. A step that fails outright is listed in ``warnings``; credential mirroring and the
Honcho clone report per item (``mirrored``, ``honcho_*``) instead, as they did before.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileCreateRequest:
    """Everything any surface can ask of a create. Defaults are the bare core create."""

    name: str
    # Core create (hermes_cli.profiles.create_profile)
    clone_from: Optional[str] = None
    clone_all: bool = False
    clone_config: bool = False
    clone_channels: bool = False
    no_skills: bool = False
    sync_imports: bool = False
    description: Optional[str] = None
    soul: Optional[str] = None
    # After the core create
    alias: bool = True                 # ~/.local/bin/<name> wrapper when the name is free
    quiet: bool = True                 # bundled-skill seeding prints its own failures when False
    mirror_credentials: bool = False   # copy the launch profile's .env / auth.json / voice / model
    share_auth: bool = False           # with mirror_credentials: keep one auth pool, do not copy auth.json
    model: str = ""                    # explicit model pin (with provider)
    provider: str = ""
    model_validate_in: Optional[Path] = None  # home whose catalog vouches for the pick (default: the profile)
    mcp_servers: tuple = ()            # hermes_cli.web_models.MCPServerCreate entries
    keep_skills: tuple = ()            # non-empty: disable every installed skill not listed
    clone_honcho: bool = False         # clone the Honcho memory config (the CLI asks for it on clones)
    # Seed bundled skills after the create. None = when the profile has no source (fresh); surfaces pass
    # their historical rule explicitly until the defaults are unified.
    seed_skills: Optional[bool] = None

    @property
    def fresh(self) -> bool:
        """No source profile: the new profile gets bundled skills instead of a copy of the source's."""
        return not (self.clone_from or self.clone_all or self.clone_config)

    @property
    def seeds_skills(self) -> bool:
        return self.fresh if self.seed_skills is None else self.seed_skills


@dataclass
class ProvisionResult:
    """What the create did. ``warnings`` lists each step that failed outright."""

    path: Path
    skills_seeded: Optional[dict] = None   # seed_profile_skills() result; None when skipped or failed
    alias_path: Optional[Path] = None
    alias_collision: Optional[str] = None
    mirrored: dict[str, Any] = field(default_factory=lambda: {
        "env": False, "auth": False, "model_inherited": False, "voice": False})
    model_set: bool = False
    model_error: str = ""
    mcp_written: int = 0
    skills_disabled: int = 0
    honcho_cloned: bool = False
    honcho_refused: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

def provision_profile(request: ProfileCreateRequest) -> ProvisionResult:
    """Create the profile, then apply every requested step in order.

    Raises what ``create_profile`` raises (``ValueError`` / ``FileExistsError`` /
    ``FileNotFoundError`` for a bad request); nothing after the core create raises.
    """
    from hermes_cli import profiles

    path = profiles.create_profile(
        name=request.name, clone_from=request.clone_from, clone_all=request.clone_all,
        clone_config=request.clone_config, no_skills=request.no_skills,
        description=request.description, clone_channels=request.clone_channels,
        sync_imports=request.sync_imports, soul=request.soul,
    )
    result = ProvisionResult(path=path)
    result.mirrored["auth"] = "shared" if request.share_auth else False

    def step(label: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.warning("profile %s: %s failed", request.name, label, exc_info=True)
            result.warnings.append(f"{label}: {exc}")
            return default

    if request.seeds_skills:
        # Clones carry the source's skills; a --no-skills profile seeds only essentials.
        result.skills_seeded = step("bundled skills", lambda: profiles.seed_profile_skills(path, quiet=request.quiet))
    if request.alias:
        step("alias wrapper", lambda: _write_alias(request.name, result))
    if request.mirror_credentials:
        step("launch credentials", lambda: mirror_launch_credentials(
            path, result.mirrored, share_auth=request.share_auth, clone_channels=request.clone_channels))
    if request.model and request.provider:
        result.model_set = step("model", lambda: _pin_model(path, request, result), default=False)
    elif request.mirror_credentials:
        result.mirrored["model_inherited"] = step("model inherit", lambda: inherit_launch_model(path), default=False)
    if request.mcp_servers:
        result.mcp_written = step("MCP servers", lambda: write_profile_mcp_servers(path, list(request.mcp_servers)),
                                  default=0)
    if request.keep_skills:
        result.skills_disabled = step("skill selection", lambda: disable_unselected_skills(path, request.keep_skills),
                                      default=0)
    if request.clone_honcho:
        step("Honcho config", lambda: _clone_honcho(request.name, result))
    return result


# ── steps ─────────────────────────────────────────────────────────────────────────────────────────

def _write_alias(name: str, result: ProvisionResult) -> None:
    from hermes_cli.profiles import check_alias_collision, create_wrapper_script
    collision = check_alias_collision(name)
    if collision:
        result.alias_collision = collision
        return
    result.alias_path = create_wrapper_script(name)


def _pin_model(path: Path, request: ProfileCreateRequest, result: ProvisionResult) -> bool:
    """Validated main-model write (the same shape as ``POST /api/model/set``). A rejection is reported
    as ``model_error`` so a dialog can say why the model was not set."""
    from fastapi import HTTPException  # the shared main-slot validation refuses with it
    try:
        write_profile_model(path, request.provider, request.model, validate_in=request.model_validate_in)
    except HTTPException as exc:
        result.model_error = str(exc.detail)
        return False
    return True


def _clone_honcho(name: str, result: ProvisionResult) -> None:
    try:
        from plugins.memory import import_provider_module
        honcho_cli = import_provider_module("honcho", "cli")
    except Exception:
        logger.debug("Honcho plugin unavailable; skipping its profile clone", exc_info=True)
        return
    if honcho_cli is None:
        return
    try:
        result.honcho_cloned = bool(honcho_cli.clone_honcho_for_profile(name))
    except honcho_cli.ConfigWriteRefused as exc:
        result.honcho_refused = str(exc)
    except Exception:
        logger.debug("Honcho clone skipped for profile %s (not configured)", name, exc_info=True)


def disable_unselected_skills(profile_dir: Path, keep) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``; returns how many were newly
    disabled. Profiles manage activation via a *disabled* list, so the builder's skill step has
    "replace" semantics. Hub skills are installed separately and are active on install."""
    from hermes_cli.config import load_config
    from hermes_cli.config_mutation_lock import CONFIG_MUTATION_LOCK
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    from hermes_cli.web_server_profiles import _hermes_home_scope
    keep_set = {s.strip() for s in keep if s and s.strip()}
    with _hermes_home_scope(profile_dir), CONFIG_MUTATION_LOCK:  # RMW span
        skills_root = profile_dir / "skills"
        installed = [md.parent.name for md in skills_root.rglob("SKILL.md")] if skills_root.is_dir() else []
        cfg = load_config()
        disabled = get_disabled_skills(cfg)
        newly = 0
        for name in installed:
            if name not in keep_set and name not in disabled:
                disabled.add(name)
                newly += 1
        if newly:
            save_disabled_skills(cfg, disabled)
    return newly


# ── config writers the create runs (moved from web_routers/profiles.py and web_server_profiles.py; the
#    dashboard routes call these) ──────────────────────────────────────────────────────────────────

def scope_profile_name(path: Path) -> Optional[str]:
    """Map a profile directory onto the query name ``_config_profile_scope`` expects: None for
    the process home (current-profile semantics: launch secret scope, no home override),
    ``"default"`` for the default root (its basename -- ``.hermes`` or a custom root -- is not a
    profile name; launched from ``profiles/<name>`` the root is a *different* profile), the
    directory name for ``profiles/<name>``."""
    from hermes_cli.config import get_process_hermes_home
    from hermes_constants import get_default_hermes_root
    resolved = path.resolve()
    if resolved == get_process_hermes_home().resolve():
        return None
    if resolved == get_default_hermes_root().resolve():
        return "default"
    return path.name


def write_profile_model(profile_dir: Path, provider: str, model: str, validate_in: Optional[Path] = None,
                         base_url: str = "") -> None:
    """Write the main model assignment into ``profile_dir``'s config.yaml (HERMES_HOME-scoped)
    through the same validated /model shape as ``POST /api/model/set``.

    ``validate_in`` is the home whose ``providers:``/``.env``/catalog vouch for the pick (default:
    ``profile_dir`` itself). Profile-create passes the dashboard's own home: the picker that offered
    the model read THAT catalog, and a just-created profile has no credentials yet, so validating
    in the empty profile rejected every non-env provider (anthropic, ollama, custom).

    Both spans enter ``_config_profile_scope`` (home + secret scope), matching ``/api/model/set``:
    once the dashboard has served a secondary profile, ``switch_model``'s ``key_env`` probe goes
    through ``get_secret``, which fails closed without a scope and reports the provider as
    unconnected (UnscopedSecretError class, #114676).

    ``base_url`` is the endpoint a bare ``custom``/``local`` pick binds to (a fresh profile's config
    names none); other providers ignore it."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.config_mutation_lock import CONFIG_MUTATION_LOCK
    from hermes_cli.config_backend import refusing_locked_omissions
    # Model validation is the shared ``/model`` main-slot path (``POST /api/model/set`` uses it too);
    # the profile scope is the shared config scope (20+ callers). Both still live under web_*.
    from hermes_cli.web_server_config import (
        _apply_main_model_assignment, _normalize_main_model_assignment, _validated_main_model_selection,
    )
    from hermes_cli.web_server_profiles import _config_profile_scope
    with _config_profile_scope(scope_profile_name(validate_in or profile_dir)):
        provider, model = _normalize_main_model_assignment(provider, model)
        result = _validated_main_model_selection(load_config(), provider, model, base_url)
    with _config_profile_scope(scope_profile_name(profile_dir)), CONFIG_MUTATION_LOCK:  # RMW span
        cfg = load_config()
        cfg["model"] = _apply_main_model_assignment(cfg.get("model", {}), result)
        with refusing_locked_omissions("model"):
            save_config(cfg)


def write_profile_mcp_servers(profile_dir: Path, servers) -> int:
    """Write MCP server entries into ``profile_dir``'s config.yaml (HERMES_HOME-scoped).

    Mirrors the per-server shape ``POST /api/mcp/servers`` builds, batched so the whole
    profile-create write is one config save. Returns the number of servers written.
    """
    from hermes_cli.config import load_config, save_config
    from hermes_cli.config_mutation_lock import CONFIG_MUTATION_LOCK
    from hermes_cli.mcp_config import _save_bearer_auth_token
    from hermes_cli.web_server_mcp import _normalize_mcp_server_create  # shared with POST /api/mcp/servers
    from hermes_cli.web_server_profiles import _hermes_home_scope
    written = 0
    with _hermes_home_scope(profile_dir), CONFIG_MUTATION_LOCK:  # RMW span
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            try:
                name, entry, bearer_token = _normalize_mcp_server_create(server)
            except ValueError as exc:
                display_name = (server.name or "").strip() or "<unnamed>"
                # The dashboard's logger, as before the move (its log lines keep their source).
                logging.getLogger("hermes_cli.web_server").warning(
                    "Profile-create: skipping MCP server '%s': %s", display_name, exc)
                continue
            if bearer_token is not None:
                entry["headers"] = _save_bearer_auth_token(name, bearer_token)
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # Don't leave the stray empty key we just created in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    return written


# ── launch-profile mirroring (moved from tui_gateway/methods_profiles.py) ─────────────────────────

def _profile_scope(path: Path):
    """The RPC's full runtime scope for ``path`` (home + secrets + terminal; see
    ``tui_gateway.methods_profiles._hermes_home_scope``). Called through ``tui_gateway.server``, where
    ``bind_module`` publishes it rebound to the server's globals."""
    from tui_gateway import server
    return server._hermes_home_scope(path)


def _try(fn: Callable[[], Any], default: Any) -> Any:
    """``fn()`` or ``default`` on any exception (best-effort sections must never fail each other)."""
    try:
        return fn()
    except Exception:
        logger.debug("Best-effort profile-create section failed", exc_info=True)
        return default


def _mirror_secret(path: Path, launch_home: Path, name: str, wanted) -> bool:
    """Copy the launch ``name`` file into the profile (0600) when it exists and ``wanted(src, dst)``."""
    src, dst = launch_home / name, path / name
    if not (src.is_file() and wanted(src, dst)):
        return False
    shutil.copy2(src, dst)
    with contextlib.suppress(OSError):
        os.chmod(str(dst), 0o600)
    return True


def _env_has_content(env_path: Path) -> bool:
    lines = env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    return any(s and not s.startswith("#") for s in map(str.strip, lines))


def _mirror_voice_sections(path: Path) -> bool:
    """Copy stt/tts/voice sections from the launch profile (a fresh profile has only ``model``, so
    voice fell back to defaults); True if written."""
    try:
        from hermes_cli.config import read_user_config_raw, save_config
        # Launch file RAW: the loaded config has ${VAR} refs expanded, and the new profile must get the
        # ref (resolved against its own .env), never the launch profile's secret.
        src_cfg = read_user_config_raw()
        sections = {k: src_cfg[k] for k in ("stt", "tts", "voice") if src_cfg.get(k)}
        if not sections:
            return False
        from hermes_cli.config_mutation_lock import CONFIG_MUTATION_LOCK
        with _profile_scope(path), CONFIG_MUTATION_LOCK:  # RMW span
            # RAW file: load_config() merges DEFAULT_CONFIG (every section would look present).
            dst_cfg = read_user_config_raw() or {}
            missing = {k: v for k, v in sections.items() if k not in dst_cfg}
            if missing:
                save_config({**dst_cfg, **missing})
        return bool(missing)
    except Exception:
        return False


def inherit_launch_model(path: Path) -> bool:
    """Inherit the launch model.provider/default when the new profile has none.

    Gate on the MODEL SECTION being absent, not on config.yaml existing: voice mirroring creates the
    file first (#85755), and a file-existence gate silently skipped inheritance for every non-clone
    bot ("No inference provider configured" on first message). Clones bring their own model section.
    """
    from hermes_cli.config import load_config_readonly, read_user_config_raw, save_config
    from hermes_cli.config_mutation_lock import CONFIG_MUTATION_LOCK
    from hermes_cli.profile_model_seed import launch_model_seed, merge_endpoint_seed
    with _profile_scope(path):
        dst_model = (read_user_config_raw() or {}).get("model") or {}
    if dst_model.get("provider") and dst_model.get("default"):
        return False
    model_cfg = (load_config_readonly() or {}).get("model") or {}
    if not (model_cfg.get("provider") and model_cfg.get("default")):
        return False
    # A custom endpoint travels with the model it backs (same seed as the CLI path), written BEFORE the
    # pin: the pin validates inside the new profile, which rejects a provider it has not been told
    # about ("Unknown provider"). Seeded from the RAW launch file so a ${VAR} api_key travels as the ref.
    seed = launch_model_seed(read_user_config_raw())
    if seed.get("providers") or seed.get("custom_providers"):
        with _profile_scope(path), CONFIG_MUTATION_LOCK:  # RMW span
            save_config(merge_endpoint_seed(read_user_config_raw(), seed))
    # Bare ``custom``/``local`` carries its endpoint into validation (the new profile's config names
    # none). Validation needs the expanded URL; the profile keeps the launch file's spelling.
    write_profile_model(path, str(model_cfg["provider"]), str(model_cfg["default"]),
                         base_url=str(model_cfg.get("base_url") or ""))
    raw_url = ((read_user_config_raw() or {}).get("model") or {}).get("base_url")
    if isinstance(raw_url, str) and "${" in raw_url:
        with _profile_scope(path), CONFIG_MUTATION_LOCK:  # RMW span
            cfg = read_user_config_raw() or {}
            if isinstance(cfg.get("model"), dict) and cfg["model"].get("base_url") == model_cfg.get("base_url"):
                cfg["model"]["base_url"] = raw_url
                save_config(cfg)
    return True


def mirror_launch_credentials(path: Path, mirrored: dict[str, Any], *, share_auth: bool = False,
                              clone_channels: bool = False) -> None:
    """Copy the launch profile's .env / auth.json / voice sections into a new profile, recording each
    item in ``mirrored`` (best-effort per item). ``share_auth`` skips the auth copy."""
    from hermes_constants import get_hermes_home
    launch_home = get_hermes_home()
    # .env: only over the seeded comment-only stub (never a clone's secrets).
    mirrored["env"] = _try(lambda: _mirror_secret(path, launch_home, ".env", lambda src, dst: (
        _env_has_content(src) and not _try(lambda: _env_has_content(dst), False))), False)
    if mirrored["env"] and not clone_channels:
        # Provider/tool keys are what "mirror credentials" means; the launch profile's bot tokens and
        # allowlists would make the new bot collide with it over one Telegram/Discord bot.
        def _strip_channels():
            from hermes_cli.profile_channels import strip_channel_env_file
            strip_channel_env_file(path / ".env")
        _try(_strip_channels, None)
    if not share_auth:  # a copy forks token state: the first refresh in either store strands the other
        mirrored["auth"] = _try(lambda: _mirror_secret(path, launch_home, "auth.json",
                                                       lambda src, dst: not dst.exists()), False)
        if mirrored["auth"]:
            # Drop single-use OAuth grants (the first refresh strands every sibling); the profile reads
            # the root grant through the pool fallback. API keys stay.
            def _strip_grants():
                from hermes_cli.auth import strip_cloned_single_use_oauth_grants
                strip_cloned_single_use_oauth_grants(path)
            _try(_strip_grants, None)
    mirrored["voice"] = _mirror_voice_sections(path)
