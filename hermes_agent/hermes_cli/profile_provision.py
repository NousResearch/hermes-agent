"""The one way to create a profile: ``provision_profile(ProfileCreateRequest) -> ProvisionResult``.

``hermes_agent.hermes_cli.profiles.create_profile`` does the core directory work; this module runs every step
after it in a fixed order. Post-create steps are best-effort: a failing step is listed in
``warnings`` (credential mirroring reports per item in ``mirrored``) and never undoes the create.

Defaults are the Desktop bot page's "Share keys & accounts with the main profile" (ticked): the
launch profile's provider/tool keys (bot tokens and allowlists stripped) and its model unless one is
picked, so a profile made on any surface answers its first chat. Logins are shared by NOT copying
``auth.json``: a profile without one reads the default root's (``hermes_agent.hermes_cli.auth`` fallback). That only
shares the launch profile's logins when the launch profile IS the root, so a named launch profile's
``auth.json`` is copied instead. Opt-outs: ``share_auth=False`` (auth copied once, single-use OAuth
grants dropped) and ``mirror_credentials=False`` (no keys, logins or model copied; the root's logins
stay readable through the same fallback). Clones keep their own ``.env``.
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
    clone_from: Optional[str] = None
    clone_all: bool = False
    clone_config: bool = False
    clone_channels: bool = False
    no_skills: bool = False
    sync_imports: bool = False
    description: Optional[str] = None
    soul: Optional[str] = None
    alias: bool = True                 # ~/.local/bin/<name> wrapper when the name is free
    quiet: bool = True                 # bundled-skill seeding prints its own failures when False
    # Bot-page defaults (see module docstring); every surface exposes both opt-outs.
    mirror_credentials: bool = True    # launch .env keys / voice / model (and auth.json unless shared)
    share_auth: bool = True            # read the root's auth.json instead of copying one (root launch only)
    model: str = ""                    # explicit model pin (with provider)
    provider: str = ""
    model_validate_in: Optional[Path] = None  # home whose catalog vouches for the pick (default: the profile)
    mcp_servers: tuple = ()            # hermes_agent.hermes_cli.web_models.MCPServerCreate entries
    keep_skills: tuple[str, ...] = ()  # non-empty: disable every installed skill not listed

    @property
    def fresh(self) -> bool:
        """No source profile: the new profile gets bundled skills instead of a copy of the source's."""
        return not (self.clone_from or self.clone_all or self.clone_config)


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
    warnings: list[str] = field(default_factory=list)


def provision_profile(request: ProfileCreateRequest) -> ProvisionResult:
    """Create the profile, then apply every requested step in order.

    Raises what ``create_profile`` raises (``ValueError`` / ``FileExistsError`` /
    ``FileNotFoundError`` for a bad request); nothing after the core create raises.
    """
    from hermes_agent.hermes_cli import profiles

    path = profiles.create_profile(
        name=request.name, clone_from=request.clone_from, clone_all=request.clone_all,
        clone_config=request.clone_config, no_skills=request.no_skills,
        description=request.description, clone_channels=request.clone_channels,
        sync_imports=request.sync_imports, soul=request.soul, seed_model=request.mirror_credentials,
    )
    result = ProvisionResult(path=path)

    def step(label: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.warning("profile %s: %s failed", request.name, label, exc_info=True)
            result.warnings.append(f"{label}: {exc}")
            return default

    if request.fresh:
        result.skills_seeded = step("bundled skills", lambda: profiles.seed_profile_skills(path, quiet=request.quiet))
    if request.alias:
        step("alias wrapper", lambda: _write_alias(request.name, result))
    if request.mirror_credentials:
        from hermes_agent.hermes_cli.auth import _global_auth_file_path
        kept_root = step("launch credentials", lambda: mirror_launch_credentials(
            path, result.mirrored, share_auth=request.share_auth, clone_channels=request.clone_channels), default=())
        if kept_root and _global_auth_file_path() is not None:
            # A named launch profile's own subscription logins can't be copied (single-use refresh
            # tokens); the copy reads the default profile's for those providers instead.
            result.warnings.append(
                f"subscription logins not carried over ({', '.join(kept_root)}): the profile uses the default "
                f"profile's; run `hermes -p {request.name} auth add <provider>` to sign it in separately")
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
    if result.mirrored["auth"] is not True:
        # From the outcome, not the flags: a profile with no auth.json of its own reads the root's
        # (``--clone-all`` brings its own copy; ``mirror_credentials=False`` still reads the root's).
        result.mirrored["auth"] = "shared" if not (path / "auth.json").exists() else False
    return result


def _write_alias(name: str, result: ProvisionResult) -> None:
    from hermes_agent.hermes_cli.profiles import check_alias_collision, create_wrapper_script
    collision = check_alias_collision(name)
    if collision:
        result.alias_collision = collision
        return
    result.alias_path = create_wrapper_script(name)


def _pin_model(path: Path, request: ProfileCreateRequest, result: ProvisionResult) -> bool:
    from hermes_agent.hermes_cli.model_assignment import ModelAssignmentRejected
    try:
        write_profile_model(path, request.provider, request.model, validate_in=request.model_validate_in)
    except ModelAssignmentRejected as exc:
        result.model_error = str(exc)
        return False
    return True


def disable_unselected_skills(profile_dir: Path, keep: tuple[str, ...]) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``; returns how many were newly disabled."""
    from hermes_agent.hermes_cli.config import load_config
    from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_agent.hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope
    keep_set = {s.strip() for s in keep if s and s.strip()}
    with profile_config_scope(profile_dir), CONFIG_MUTATION_LOCK:
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


def write_profile_model(profile_dir: Path, provider: str, model: str, validate_in: Optional[Path] = None,
                         base_url: str = "") -> None:
    """Write a validated main-model assignment into ``profile_dir``'s config.yaml (same shape as
    ``POST /api/model/set``). ``validate_in`` is the home whose catalog validates the pick (default
    ``profile_dir``); profile create passes the dashboard's home, because the picker offered that catalog
    and a just-created profile has no credentials yet. Both spans bind a secret scope: without one
    ``switch_model``'s key probe fails closed and reports the provider unconnected (#114676).
    ``base_url`` binds a bare ``custom``/``local`` pick; other providers ignore it. A rejected pick
    raises :class:`hermes_agent.hermes_cli.model_assignment.ModelAssignmentRejected`."""
    from hermes_agent.hermes_cli.config import load_config, save_config
    from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK, refusing_locked_omissions
    from hermes_agent.hermes_cli.model_assignment import (
        apply_main_model_assignment, normalize_main_model_assignment, validated_main_model_selection,
    )
    from hermes_agent.agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_agent.hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import get_process_hermes_home
    from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope
    validate_home = validate_in or profile_dir
    foreign = validate_home.resolve() != get_process_hermes_home().resolve()
    if foreign:
        # The key probe may need the home's external secret sources (``op run`` / ``bws``).
        hydrate_profile_secret_sources(validate_home)
    with contextlib.ExitStack() as stack:
        stack.enter_context(profile_config_scope(validate_home))
        if foreign:
            # The provider adapters' ambient-credential refusals (Bedrock, Vertex, Entra) apply to a
            # foreign home under multiplex semantics: bound for this task, not process-wide, and
            # reset before the scope so they never outlive it.
            stack.callback(reset_multiplex_context, set_multiplex_context(True))
        provider, model = normalize_main_model_assignment(provider, model)
        result = validated_main_model_selection(load_config(), provider, model, base_url)
    with profile_config_scope(profile_dir), CONFIG_MUTATION_LOCK:
        cfg = load_config()
        cfg["model"] = apply_main_model_assignment(cfg.get("model", {}), result)
        with refusing_locked_omissions("model"):
            save_config(cfg)


def write_profile_mcp_servers(profile_dir: Path, servers: list) -> int:
    """Write ``hermes_agent.hermes_cli.web_models.MCPServerCreate`` entries into ``profile_dir``'s config.yaml.

    Mirrors the per-server shape ``POST /api/mcp/servers`` builds, batched so the whole
    profile-create write is one config save. Returns the number of servers written.
    """
    from hermes_agent.hermes_cli.config import load_config, save_config
    from hermes_agent.hermes_cli.mcp_config import _save_bearer_auth_token
    from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_agent.hermes_cli.web_server_mcp import normalize_mcp_server_create
    from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope
    written = 0
    with profile_config_scope(profile_dir), CONFIG_MUTATION_LOCK:
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            try:
                name, entry, bearer_token = normalize_mcp_server_create(server)
            except ValueError as exc:
                display_name = (server.name or "").strip() or "<unnamed>"
                # The dashboard's logger: hermes_logging routes it to gui.log / `hermes logs --component gui`.
                logging.getLogger("hermes_agent.hermes_cli.web_server").warning(
                    "Profile-create: skipping MCP server '%s': %s", display_name, exc)
                continue
            if bearer_token is not None:
                entry["headers"] = _save_bearer_auth_token(name, bearer_token)
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    return written


def _try(label: str, fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        logger.warning("profile-create %s failed", label, exc_info=True)
        return default


def _mirror_secret(path: Path, launch_home: Path, name: str, wanted) -> bool:
    src, dst = launch_home / name, path / name
    if not (src.is_file() and wanted(src, dst)):
        return False
    shutil.copy2(src, dst)
    with contextlib.suppress(OSError):
        os.chmod(str(dst), 0o600)
    return True


def _env_over_stub(src: Path, dst: Path) -> bool:
    """Mirror .env only over the seeded comment-only stub, never over a clone's own secrets."""
    from hermes_agent.agent.secret_scope import load_env_file
    return bool(load_env_file(src)) and not load_env_file(dst)


def _mirror_voice_sections(path: Path) -> bool:
    from hermes_agent.hermes_cli.config import read_user_config_raw, save_config
    from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope
    # RAW reads: the launch side keeps its ${VAR} refs (never its secrets), and load_config() on the
    # profile side would merge DEFAULT_CONFIG so every section would look present.
    launch_raw = read_user_config_raw()
    sections = {k: launch_raw[k] for k in ("stt", "tts", "voice") if launch_raw.get(k)}
    if not sections:
        return False
    with profile_config_scope(path), CONFIG_MUTATION_LOCK:
        profile_raw = read_user_config_raw() or {}
        missing = {k: v for k, v in sections.items() if k not in profile_raw}
        if missing:
            save_config({**profile_raw, **missing})
    return bool(missing)


def inherit_launch_model(path: Path) -> bool:
    """Pin the launch model.provider/default into ``path`` unless its model section already has both."""
    from hermes_agent.hermes_cli.config import load_config_readonly, read_user_config_raw, save_config
    from hermes_agent.hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_agent.hermes_cli.profile_model_seed import launch_model_seed, merge_endpoint_seed
    from hermes_agent.tui_gateway.launch_profile_policy import profile_config_scope
    with profile_config_scope(path):
        dst_model = (read_user_config_raw() or {}).get("model") or {}
    if dst_model.get("provider") and dst_model.get("default"):
        return False
    model_cfg = (load_config_readonly() or {}).get("model") or {}
    if not (model_cfg.get("provider") and model_cfg.get("default")):
        return False
    # The endpoint seed goes in BEFORE the pin: the pin validates inside the new profile, which rejects a
    # provider it has no definition for (#101885).
    seed = launch_model_seed(read_user_config_raw())
    if seed.get("providers") or seed.get("custom_providers"):
        with profile_config_scope(path), CONFIG_MUTATION_LOCK:
            save_config(merge_endpoint_seed(read_user_config_raw(), seed))
    # Validation needs the expanded base_url; the profile then gets back the launch file's ${VAR} spelling.
    write_profile_model(path, str(model_cfg["provider"]), str(model_cfg["default"]),
                         base_url=str(model_cfg.get("base_url") or ""))
    raw_url = ((read_user_config_raw() or {}).get("model") or {}).get("base_url")
    if isinstance(raw_url, str) and "${" in raw_url:
        with profile_config_scope(path), CONFIG_MUTATION_LOCK:
            cfg = read_user_config_raw() or {}
            if isinstance(cfg.get("model"), dict) and cfg["model"].get("base_url") == model_cfg.get("base_url"):
                cfg["model"]["base_url"] = raw_url
                save_config(cfg)
    return True


def mirror_launch_credentials(path: Path, mirrored: dict[str, Any], *, share_auth: bool = False,
                              clone_channels: bool = False) -> tuple[str, ...]:
    """Copy the launch profile's .env / auth.json / voice sections into a new profile, recording each
    item in ``mirrored`` (best-effort per item). ``share_auth`` skips the auth copy when the launch
    profile is the root: the new profile then reads the root's auth.json, i.e. the launch logins. From a
    named launch profile that fallback would pair its keys with the root's logins, so its auth is copied.
    Returns the providers whose single-use logins were left out of a copy (read from the root instead)."""
    from hermes_agent.hermes_cli.auth import _global_auth_file_path
    from hermes_constants import get_hermes_home
    launch_home = get_hermes_home()
    # Same test as the auth fallback that makes sharing work: share only when it reads the launch store.
    share_auth = share_auth and _global_auth_file_path() is None
    kept_root: tuple[str, ...] = ()
    mirrored["env"] = _try(".env copy", lambda: _mirror_secret(path, launch_home, ".env", _env_over_stub), False)
    if mirrored["env"] and not clone_channels:
        # Keep provider/tool keys; the launch bot's tokens would make the new bot collide with it.
        def _strip_channels():
            from hermes_agent.hermes_cli.profile_channels import strip_channel_env_file
            strip_channel_env_file(path / ".env")
        _try("channel strip", _strip_channels, None)
    if not share_auth:  # a copied auth.json forks token state: a refresh in either store strands the other
        mirrored["auth"] = _try("auth.json copy", lambda: _mirror_secret(path, launch_home, "auth.json",
                                                       lambda src, dst: not dst.exists()), False)
        if mirrored["auth"]:
            # Single-use OAuth grants stay in the root store; the profile reads them via the pool fallback.
            def _strip_grants():
                from hermes_agent.hermes_cli.auth import strip_cloned_single_use_oauth_grants
                stripped = strip_cloned_single_use_oauth_grants(path)
                return tuple(sorted({*stripped["pool"], *stripped["providers"]}))
            kept_root = _try("OAuth grant strip", _strip_grants, ()) or ()
    mirrored["voice"] = _try("voice sections", lambda: _mirror_voice_sections(path), False)
    return kept_root
