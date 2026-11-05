"""Create a profile that can run its first turn: the shared body of the ``profiles.create`` RPC
and ``PluginContext.create_profile``.

A bare ``create_profile()`` seeds a comment-only ``.env``, no ``auth.json`` and no model, so a
profile born from a headless door has NO provider and its first message fails (#85111).
``mirror_credentials`` (default on) copies the source profile's ``.env`` / ``auth.json`` / voice
sections and inherits its model. The source is the profile bound when the call runs
(``get_hermes_home()``). The copy happens once, at creation: profiles stay islands, nothing is
inherited live. Callers get a receipt of booleans, never a credential value.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _best_effort(step: str, fn: Callable[[], Any], default: Any = False) -> Any:
    """``fn()`` or ``default``: the profile already exists, so a failed follow-up step is reported in
    the receipt instead of failing the create."""
    try:
        return fn()
    except Exception:  # health: allow BLE001 -- best-effort create step; the receipt reports it as False
        logger.warning("profile create: %s failed", step, exc_info=True)
        return default


@contextlib.contextmanager
def _hermes_home_scope(path):
    """Bind ``path``'s home and its OWN secret scope for config reads and writes, so a ``${VAR}``
    check inside the new profile never borrows the caller's keys. No external-source hydration:
    these steps write config, they never call a provider."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home_token = set_hermes_home_override(str(path))
    try:
        secret_token = set_secret_scope(build_profile_secret_scope(Path(path)), profile_home=str(path))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _pin_profile_model(profile_dir, provider, model, base_url="") -> None:
    from hermes_cli.web_routers.profiles import _write_profile_model
    _write_profile_model(profile_dir, provider, model, base_url=base_url)


def _mirror_secret(path, launch_home, name: str, wanted) -> bool:
    """Copy the launch ``name`` file into the profile (0600) when it exists and ``wanted(src, dst)``."""
    src, dst = launch_home / name, path / name
    if not (src.is_file() and wanted(src, dst)):
        return False
    shutil.copy2(src, dst)
    with contextlib.suppress(OSError):
        os.chmod(str(dst), 0o600)
    return True


def _env_has_content(env_path) -> bool:
    if not env_path.is_file():
        return False
    lines = env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    return any(s and not s.startswith("#") for s in map(str.strip, lines))


def _mirror_voice_sections(path) -> bool:
    """Copy stt/tts/voice sections from the launch profile (a fresh profile has only ``model``,
    so voice fell back to defaults); True if written."""
    try:
        from hermes_cli.config import read_user_config_raw, save_config
        # Launch file RAW too: the loaded config has ${VAR} refs expanded, and the new profile
        # must get the ref (resolved against its own .env), never the launch profile's secret.
        src_cfg = read_user_config_raw()
        sections = {k: src_cfg[k] for k in ("stt", "tts", "voice") if src_cfg.get(k)}
        if not sections:
            return False
        with _hermes_home_scope(path):
            # RAW file: load_config() merges DEFAULT_CONFIG (every section would look present).
            dst_cfg = read_user_config_raw() or {}
            missing = {k: v for k, v in sections.items() if k not in dst_cfg}
            if missing:
                save_config({**dst_cfg, **missing})
        return bool(missing)
    except Exception:
        return False


def _inherit_launch_model(path) -> bool:
    """Inherit launch model.provider/default when the new profile has none. Gate on the MODEL
    SECTION, not config.yaml existing: voice mirroring creates the file first."""
    # Gate on the MODEL SECTION being absent, not on config.yaml existing — earlier mirroring steps (voice
    # sections, #85755) legitimately create the file first, and a file-existence gate silently skipped
    # inheritance for every non-clone bot ("No inference provider configured" on first message, tester
    # report). Clones bring their own model section and stay untouched.
    from hermes_cli.config import load_config_readonly, read_user_config_raw
    from hermes_cli.profile_model_seed import launch_model_seed, merge_endpoint_seed
    with _hermes_home_scope(path):
        dst_model = (read_user_config_raw() or {}).get("model") or {}
    if dst_model.get("provider") and dst_model.get("default"):
        return False
    launch_cfg = load_config_readonly() or {}
    model_cfg = launch_cfg.get("model") or {}
    if not (model_cfg.get("provider") and model_cfg.get("default")):
        return False
    # A custom endpoint travels with the model it backs (same seed as the CLI path). It is
    # written BEFORE the pin: the pin validates the pick inside the new profile, and an empty profile
    # rejects a provider it has not been told about ("Unknown provider").
    # Seeded from the RAW launch file so a ${VAR} api_key travels as the ref, not its value.
    seed = launch_model_seed(read_user_config_raw())
    if seed.get("providers") or seed.get("custom_providers"):
        from hermes_cli.config import save_config
        with _hermes_home_scope(path):
            save_config(merge_endpoint_seed(read_user_config_raw(), seed))
    # Bare ``custom``/``local`` carries its endpoint into validation (the new profile's config names
    # none); other providers ignore it. Validation needs the expanded URL, the profile keeps the
    # launch file's spelling so a ${VAR} URL stays a ref.
    _pin_profile_model(path, str(model_cfg["provider"]), str(model_cfg["default"]),
                       str(model_cfg.get("base_url") or ""))
    raw_url = ((read_user_config_raw() or {}).get("model") or {}).get("base_url")
    if isinstance(raw_url, str) and "${" in raw_url:
        from hermes_cli.config import save_config
        with _hermes_home_scope(path):
            cfg = read_user_config_raw() or {}
            if isinstance(cfg.get("model"), dict) and cfg["model"].get("base_url") == model_cfg.get("base_url"):
                cfg["model"]["base_url"] = raw_url
                save_config(cfg)
    return True


def _mirror_launch_credentials(path, params: dict) -> dict:
    """Copy launch .env / auth.json / voice sections into a new profile (best-effort per item).
    ``share_auth`` reports ``auth: "shared"`` and skips the auth copy; ``mirror_credentials``
    false skips everything. ``model_inherited`` is filled in by the caller."""
    share_auth = is_truthy_value(params.get("share_auth", False))
    mirrored = {"env": False, "auth": "shared" if share_auth else False, "model_inherited": False,
                "voice": False}
    if not is_truthy_value(params.get("mirror_credentials", True)):
        return mirrored
    launch_home = get_hermes_home()
    # .env: only over the seeded comment-only stub (never a clone's secrets).
    mirrored["env"] = _best_effort("env mirror", lambda: _mirror_secret(path, launch_home, ".env", lambda src, dst: (
        _env_has_content(src) and not _env_has_content(dst))))
    if mirrored["env"] and not is_truthy_value(params.get("clone_channels", False)):
        # Provider/tool keys are what "mirror credentials" means; the launch profile's bot tokens
        # and allowlists would make the new bot collide with it over one Telegram/Discord bot.
        from hermes_cli.profile_channels import strip_channel_env_file
        _best_effort("channel strip", lambda: strip_channel_env_file(path / ".env"))
    if not share_auth:  # a copy forks token state: the first refresh in either store strands the other
        mirrored["auth"] = _best_effort("auth mirror", lambda: _mirror_secret(
            path, launch_home, "auth.json", lambda src, dst: not dst.exists()))
        if mirrored["auth"]:
            # Drop single-use OAuth grants (first refresh strands every sibling); they read from the
            # root grant via the pool fallback. API keys stay.
            from hermes_cli.auth import strip_cloned_single_use_oauth_grants
            _best_effort("single-use grant strip", lambda: strip_cloned_single_use_oauth_grants(path))
    mirrored["voice"] = _mirror_voice_sections(path)
    return mirrored


def create_profile_with_credentials(
    name: str, *, description: Optional[str] = None, clone_from: Optional[str] = None,
    clone_all: bool = False, clone_channels: bool = False, no_skills: bool = False,
    no_alias: bool = False, soul: Optional[str] = None, model: str = "", provider: str = "",
    share_auth: bool = False, mirror_credentials: bool = True,
) -> dict:
    """Create ``name`` the way the ``profiles.create`` RPC does and return its receipt:
    ``{name, path, soul_written, model_set, mirrored: {env, auth, model_inherited, voice}}``.

    ``create_profile()`` errors (``ValueError``, ``FileExistsError``, ``FileNotFoundError``)
    propagate unchanged; every later step is best-effort and reported in the receipt. With
    ``mirror_credentials=False`` the result is the bare ``create_profile()`` profile: comment-only
    ``.env``, no ``auth.json``, no inherited model."""
    from hermes_cli import profiles as profiles_mod
    clone_from = (clone_from or "").strip() or None
    path = profiles_mod.create_profile(
        name=name, clone_from=clone_from, clone_all=clone_all,
        clone_config=bool(clone_from) and not clone_all, no_skills=no_skills,
        description=(description or "").strip() or None, clone_channels=clone_channels)
    # CLI/REST create flow: bundled skills for fresh profiles, then the alias wrapper.
    if not clone_from:
        _best_effort("skill seed", lambda: profiles_mod.seed_profile_skills(path, quiet=True))
    if not no_alias:
        _best_effort("alias", lambda: profiles_mod.check_alias_collision(name)
                     or profiles_mod.create_wrapper_script(name))
    soul_written = isinstance(soul, str) and bool(soul.strip()) and _best_effort(
        "SOUL.md", lambda: ((path / "SOUL.md").write_text(soul, encoding="utf-8"), True)[1])
    mirrored = _mirror_launch_credentials(path, {
        "mirror_credentials": mirror_credentials, "share_auth": share_auth, "clone_channels": clone_channels})
    model, provider = (model or "").strip(), (provider or "").strip()
    model_set = False
    if model and provider:
        model_set = _best_effort("model pin", lambda: (_pin_profile_model(path, provider, model), True)[1])
    elif mirror_credentials:
        mirrored["model_inherited"] = _best_effort("model inherit", lambda: _inherit_launch_model(path))
    return {"name": name, "path": str(path), "soul_written": soul_written, "model_set": model_set,
            "mirrored": mirrored}
