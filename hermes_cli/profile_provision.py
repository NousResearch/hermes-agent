"""The one way to create a profile: ``provision_profile(ProfileCreateRequest) -> ProvisionResult``.

``hermes_cli.profiles.create_profile`` does the core directory work; this module runs every step
after it in a fixed order. Post-create steps are best-effort: a failing step is listed in
``warnings`` (credential mirroring reports per item in ``mirrored``) and never undoes the create.

Defaults are the Desktop bot page's "Share keys & accounts with the main profile" (ticked): the
launch profile's provider/tool keys (bot tokens and allowlists stripped) and its model unless one is
picked, so a profile made on any surface answers its first chat. Logins are shared by NOT copying
``auth.json``: a profile without one reads the default root's (``hermes_cli.auth`` fallback). That only
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
import threading
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

# ``create_profile`` stages in ``profiles/.<name>.staging-<pid>``, so two in-process creates of one
# name (two terminals creating through one gateway) shared and removed each other's staging dir: the
# loser failed with a raw OSError instead of "already exists". One lock per canonical name (other
# processes have their own pid); weak values, so a name's lock lives only while a create holds it.
_CREATE_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_CREATE_LOCKS_GUARD = threading.Lock()


def _create_lock(name: str) -> threading.Lock:
    from hermes_cli.profiles import normalize_profile_name
    try:
        key = normalize_profile_name(name)
    except ValueError:
        key = ""  # create_profile refuses it before staging anything
    with _CREATE_LOCKS_GUARD:
        lock = _CREATE_LOCKS.get(key)
        if lock is None:
            lock = _CREATE_LOCKS[key] = threading.Lock()
        return lock


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
    mcp_servers: tuple = ()            # hermes_cli.web_models.MCPServerCreate entries
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


#: What a receipt's ``mirrored`` can hold: ``auth`` is ``"shared"`` when the profile reads the root's auth.json.
MirroredValue = bool | Literal["shared"]


@dataclass(frozen=True)
class ProfileCreateReceipt:
    """What ``hermes profile create`` prints about a new profile, read on the machine (and in the
    profile scope) that made it: built after a local create, or sent by the gateway that made it
    (``profiles.create`` with ``report``). The contract uses this type itself; a stdlib dataclass, so a
    local create never loads pydantic. A client treats a receipt missing a field, or with a wrong type,
    as an unknown outcome."""

    path: str
    display_path: str
    launch_label: str
    source_label: str
    skills_synced: Optional[int]  # bundled skills a fresh create copied; None: not seeded (clone, or it failed)
    skills_opted_out: bool        # ``--no-skills``: only the essentials were seeded
    alias_path: Optional[str]
    alias_collision: Optional[str]
    mirrored: dict[str, MirroredValue]  # a key a newer gateway adds is kept, never printed
    warnings: list[str]
    cloned_plugins: list[str]
    memory_provider: Optional[str]
    channels_left_behind: list[str]
    shared_channels: list[str]
    served_now: Optional[bool]  # None: no live multiplexer recorded what it serves

    @classmethod
    def of(cls, request: ProfileCreateRequest, result: ProvisionResult) -> ProfileCreateReceipt:
        """Every read is best-effort: the profile exists whatever a read says, so this never raises."""
        from hermes_cli import profiles
        from hermes_cli.gateway_multiplex_served import recorded_served_profiles
        from hermes_constants import display_hermes_home
        path = result.path
        launch = _try("launch label", profiles.get_active_profile_name, "default")
        source = request.clone_from or launch
        plugins, memory, left_behind, shared = [], None, [], []
        if not request.fresh:
            if not request.clone_all:
                from hermes_cli.profile_memory_config import cloned_memory_provider
                plugins = _try("cloned plugins", lambda: profiles.cloned_plugin_names(path), [])
                memory = _try("cloned memory provider", lambda: cloned_memory_provider(path), None)
            source_dir = _try("source profile", lambda: profiles.get_profile_dir(source), None)
            if source_dir is not None and source_dir.is_dir():
                from hermes_cli.profile_channels import channel_platforms_configured, shared_channel_credentials
                if request.clone_channels:
                    shared = _try("shared channels", lambda: shared_channel_credentials(path, source_dir), [])
                else:
                    left_behind = _try("channels left behind", lambda: channel_platforms_configured(source_dir), [])
        served = _try("served profiles", recorded_served_profiles, None)
        served_now = None if served is None else _try("served profiles", lambda: profiles.normalize_profile_name(
            request.name) in {profiles.normalize_profile_name(p) for p in served}, None)
        seeded = result.skills_seeded or {}
        copied = seeded.get("copied")
        return cls(
            path=str(path), display_path=display_hermes_home(path), launch_label=launch, source_label=source,
            skills_synced=len(copied) if isinstance(copied, list) else None,
            skills_opted_out=bool(seeded.get("skipped_opt_out")),
            alias_path=str(result.alias_path) if result.alias_path else None,
            alias_collision=result.alias_collision, mirrored=dict(result.mirrored), warnings=list(result.warnings),
            cloned_plugins=list(plugins), memory_provider=memory, channels_left_behind=list(left_behind),
            shared_channels=list(shared), served_now=served_now)

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire(cls, data: Any) -> Optional[ProfileCreateReceipt]:
        """The receipt a gateway sent, or None unless every field is there with its type, strictly (no
        ``"false"`` for a bool): a gateway that predates a field or sends a wrong type must not have the rest
        printed from guesses. Fields this version doesn't know are dropped."""
        import json
        from pydantic import TypeAdapter, ValidationError
        try:
            made = TypeAdapter(cls).validate_json(json.dumps(data), strict=True)
        except (TypeError, ValueError, ValidationError):  # not JSON-shaped, or not a receipt
            return None
        return made if made.path else None


#: What a create refuses with: a bad request, nothing made. Every surface maps these to its "refused"
#: answer (``profiles.create`` 4062, ``hermes profile create``'s ``Error:``, ``POST /api/profiles`` 400);
#: a subclass travels as the nearest of these (``refusal``), so a ``JSONDecodeError`` from a broken
#: source config stays a refusal.
CREATE_REFUSALS: tuple[type[Exception], ...] = (FileExistsError, FileNotFoundError, ValueError)


def redact_create_text(text: Any) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(str(text), force=True, redact_url_credentials=True)


def refusal(exc: BaseException) -> tuple[str, dict]:
    """``(error.message, error.data)`` for a failed create. The message is the whole text, redacted (a
    parse error can quote a line of the source's config/.env); ``data`` names the nearest
    :data:`CREATE_REFUSALS` type (none: not a refusal) and repeats the text in the short parts
    ``gateway_client`` keeps in ``error.data`` (it drops longer strings whole)."""
    from hermes_cli.gateway_client import ERROR_DATA_ITEMS, ERROR_DATA_TEXT
    text = redact_create_text(exc)
    kind = next((k for k in CREATE_REFUSALS if isinstance(exc, k)), None)
    kept = ERROR_DATA_TEXT * ERROR_DATA_ITEMS
    shown = text if len(text) <= kept else text[:kept - len(_TRUNCATED)] + _TRUNCATED
    parts = [shown[i:i + ERROR_DATA_TEXT] for i in range(0, len(shown), ERROR_DATA_TEXT)]
    return text, {"type": kind.__name__ if kind else type(exc).__name__, "message_parts": parts}


#: Ends a refusal longer than ``error.data`` keeps (``error.message`` still has all of it).
_TRUNCATED = " … [truncated]"


def message_from(data: dict) -> str:
    """The text a ``refusal`` ``error.data`` carries, redacted again here; "" when it carries none."""
    parts = data.get("message_parts")
    return redact_create_text("".join(p for p in parts if isinstance(p, str))) if isinstance(parts, list) else ""


def refusal_from(data: dict) -> Optional[Exception]:
    """The refusal a ``refusal`` ``error.data`` names, re-raised with its text; None when it names none."""
    kind = next((k for k in CREATE_REFUSALS if k.__name__ == data.get("type")), None)
    if kind is None:
        return None
    return kind(message_from(data) or "the gateway refused the create; nothing was created")

def provision_profile(request: ProfileCreateRequest) -> ProvisionResult:
    """Create the profile, then apply every requested step in order.

    Raises what ``create_profile`` raises (``ValueError`` / ``FileExistsError`` /
    ``FileNotFoundError`` for a bad request); nothing after the core create raises.
    """
    from hermes_cli import profiles

    with _create_lock(request.name):
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
        problems: list[str] = []
        result.skills_seeded = step("bundled skills", lambda: profiles.seed_profile_skills(
            path, quiet=request.quiet, problems=problems))
        if request.quiet:  # nothing printed them: the caller (a gateway's client) shows them
            result.warnings.extend(line.strip().removeprefix("⚠ ") for line in problems if line.strip())
        from hermes_cli.default_plugins import spawn_install
        step("default plugins", lambda: spawn_install(path))  # detached: a create never waits on git
    if request.alias:
        result.alias_path, result.alias_collision = step(
            "alias wrapper", lambda: write_alias(request.name), default=(None, None))
    if request.mirror_credentials:
        from hermes_cli.auth import _global_auth_file_path
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


def write_alias(name: str) -> tuple[Optional[Path], Optional[str]]:
    """``(wrapper path, collision)`` for the ``~/.local/bin/<name>`` wrapper: written on this machine,
    with this process's PATH and ``hermes``. Exactly one is set unless the write itself failed."""
    from hermes_cli.profiles import check_alias_collision, create_wrapper_script
    collision = check_alias_collision(name)
    if collision:
        return None, collision
    return create_wrapper_script(name), None


def _pin_model(path: Path, request: ProfileCreateRequest, result: ProvisionResult) -> bool:
    from hermes_cli.model_assignment import ModelAssignmentRejected
    try:
        write_profile_model(path, request.provider, request.model, validate_in=request.model_validate_in)
    except ModelAssignmentRejected as exc:
        result.model_error = str(exc)
        return False
    return True


def disable_unselected_skills(profile_dir: Path, keep: tuple[str, ...]) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``; returns how many were newly disabled."""
    from hermes_cli.config import load_config
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    from tui_gateway.launch_profile_policy import profile_config_scope
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
    raises :class:`hermes_cli.model_assignment.ModelAssignmentRejected`."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK, refusing_locked_omissions
    from hermes_cli.model_assignment import (
        apply_main_model_assignment, normalize_main_model_assignment, validated_main_model_selection,
    )
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import get_process_hermes_home
    from tui_gateway.launch_profile_policy import profile_config_scope
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
    """Write ``hermes_cli.web_models.MCPServerCreate`` entries into ``profile_dir``'s config.yaml.

    Mirrors the per-server shape ``POST /api/mcp/servers`` builds, batched so the whole
    profile-create write is one config save. Returns the number of servers written.
    """
    from hermes_cli.config import load_config, save_config
    from hermes_cli.mcp_config import _save_bearer_auth_token
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from hermes_cli.web_server_mcp import normalize_mcp_server_create
    from tui_gateway.launch_profile_policy import profile_config_scope
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
                logging.getLogger("hermes_cli.web_server").warning(
                    "Profile-create: skipping MCP server '%s': %s", display_name, exc)
                continue
            if bearer_token is not None:
                saved_headers = _save_bearer_auth_token(name, bearer_token)
                if entry.get("auth") != "query":
                    entry["headers"] = saved_headers
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
    from agent.secret_scope import load_env_file
    return bool(load_env_file(src)) and not load_env_file(dst)


def _mirror_voice_sections(path: Path) -> bool:
    from hermes_cli.config import read_user_config_raw, save_config
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK
    from tui_gateway.launch_profile_policy import profile_config_scope
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
    """Copy the launch model seed without revalidating it; keep an existing explicit pin."""
    from hermes_cli.config import load_config_readonly, read_user_config_raw, save_config
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK, refusing_locked_omissions
    from hermes_cli.config_model_section import _normalize_root_model_keys
    from hermes_cli.model_assignment import apply_inherited_model_assignment, normalize_main_model_assignment
    from hermes_cli.profile_model_seed import merge_endpoint_seed, prepare_inherited_endpoint_seed
    from tui_gateway.launch_profile_policy import profile_config_scope
    with CONFIG_MUTATION_LOCK:
        with profile_config_scope(path):
            cfg = _normalize_root_model_keys(read_user_config_raw() or {})
        dst_model = cfg.get("model") or {}
        if dst_model.get("provider") and dst_model.get("default"):
            return False
        model_cfg = (load_config_readonly() or {}).get("model") or {}
        if not (model_cfg.get("provider") and model_cfg.get("default")):
            return False
        # Canonicalize raw routing without resolving credentials. Selection identity is pinned
        # to the launch scope; URL/key refs intentionally bind the destination's secrets.
        provider, model = str(model_cfg["provider"]), str(model_cfg["default"])
        # Only the model-name repair: the provider stays the one the source runs (the provider
        # recovery can resolve a bare ``custom`` to some other declared endpoint).
        normalized_provider, normalized_model = normalize_main_model_assignment(provider, model)
        if normalized_provider == provider:
            model = normalized_model
        raw = _normalize_root_model_keys(read_user_config_raw() or {})
        raw["model"] = {**(raw.get("model") or {}), "provider": provider, "default": model}
        seed, claimed = prepare_inherited_endpoint_seed(raw)
        with profile_config_scope(path):
            cfg = merge_endpoint_seed(cfg, seed, claimed=claimed)
            cfg["model"] = apply_inherited_model_assignment(dst_model, seed["model"])
            with refusing_locked_omissions("model", "providers", "custom_providers"):
                save_config(cfg)
    return True



def mirror_launch_credentials(path: Path, mirrored: dict[str, Any], *, share_auth: bool = False,
                              clone_channels: bool = False) -> tuple[str, ...]:
    """Copy the launch profile's .env / auth.json / voice sections into a new profile, recording each
    item in ``mirrored`` (best-effort per item). ``share_auth`` skips the auth copy when the launch
    profile is the root: the new profile then reads the root's auth.json, i.e. the launch logins. From a
    named launch profile that fallback would pair its keys with the root's logins, so its auth is copied.
    Returns the providers whose single-use logins were left out of a copy (read from the root instead)."""
    from hermes_cli.auth import _global_auth_file_path
    from hermes_constants import get_hermes_home
    launch_home = get_hermes_home()
    # Same test as the auth fallback that makes sharing work: share only when it reads the launch store.
    share_auth = share_auth and _global_auth_file_path() is None
    kept_root: tuple[str, ...] = ()
    mirrored["env"] = _try(".env copy", lambda: _mirror_secret(path, launch_home, ".env", _env_over_stub), False)
    if mirrored["env"] and not clone_channels:
        # Keep provider/tool keys; the launch bot's tokens would make the new bot collide with it.
        def _strip_channels():
            from hermes_cli.profile_channels import strip_channel_env_file
            strip_channel_env_file(path / ".env")
        _try("channel strip", _strip_channels, None)
    if not share_auth:  # a copied auth.json forks token state: a refresh in either store strands the other
        mirrored["auth"] = _try("auth.json copy", lambda: _mirror_secret(path, launch_home, "auth.json",
                                                       lambda src, dst: not dst.exists()), False)
        if mirrored["auth"]:
            # Single-use OAuth grants stay in the root store; the profile reads them via the pool fallback.
            def _strip_grants():
                from hermes_cli.auth import strip_cloned_single_use_oauth_grants
                stripped = strip_cloned_single_use_oauth_grants(path)
                return tuple(sorted({*stripped["pool"], *stripped["providers"]}))
            kept_root = _try("OAuth grant strip", _strip_grants, ()) or ()
    mirrored["voice"] = _try("voice sections", lambda: _mirror_voice_sections(path), False)
    return kept_root
