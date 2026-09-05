"""Persistent Bot Mode profile management.

A Bot Mode bot is an ordinary Hermes profile. Model/provider configuration
stays in ``config.yaml``, its system prompt is ``SOUL.md``, and the execution
gate is the small ``bot.enabled`` field in ``profile.yaml``. Keeping those
existing authorities means CLI, gateway, Desktop, and future surfaces all run
the same profile instead of maintaining parallel bot records.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from hermes_cli import profiles as profiles_mod


@dataclass(frozen=True)
class BotProfile:
    """Resolved, runnable view of one Hermes profile."""

    name: str
    path: Path
    model: str
    provider: str
    system_prompt: str
    enabled: bool = True


def _read_system_prompt(profile_dir: Path) -> str:
    path = profile_dir / "SOUL.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _write_system_prompt(profile_dir: Path, prompt: str) -> None:
    from utils import atomic_write_text

    cleaned = str(prompt or "").strip()
    if not cleaned:
        raise ValueError("System prompt cannot be empty.")
    atomic_write_text(
        profile_dir / "SOUL.md",
        cleaned + "\n",
        preserve_mode=True,
        create_mode=0o600,
    )


def _normalize_model_assignment(provider: str, model: str) -> tuple[str, str]:
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if not provider:
        raise ValueError("Provider cannot be empty.")
    if not model:
        raise ValueError("Model cannot be empty.")

    return provider, model


def _write_profile_model(profile_dir: Path, provider: str, model: str) -> None:
    """Persist through Hermes' profile-scoped config loader and saver."""
    provider, model = _normalize_model_assignment(provider, model)
    from hermes_cli.config import (
        clear_model_endpoint_credentials,
        load_config,
        save_config,
    )
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.models import normalize_provider
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    provider = normalize_provider(provider) or provider
    model = normalize_model_for_provider(model, provider) or model
    token = set_hermes_home_override(str(profile_dir))
    try:
        config = load_config()
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            model_config = {}
        else:
            model_config = dict(model_config)
        previous_provider = str(model_config.get("provider") or "").strip().lower()
        model_config["provider"] = provider
        model_config["default"] = model
        if previous_provider != provider.lower():
            model_config["base_url"] = ""
            clear_model_endpoint_credentials(model_config)
        model_config.pop("context_length", None)
        config["model"] = model_config
        save_config(config)
    finally:
        reset_hermes_home_override(token)


def _has_real_env_content(path: Path) -> bool:
    try:
        return any(
            line.strip() and not line.lstrip().startswith("#")
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    except OSError:
        return False


def _mirror_credentials(source_dir: Path, target_dir: Path) -> None:
    """Mirror the same profile credentials Desktop bot creation mirrors."""
    source_env = source_dir / ".env"
    target_env = target_dir / ".env"
    if source_env.is_file() and _has_real_env_content(source_env):
        shutil.copy2(source_env, target_env)
        try:
            target_env.chmod(0o600)
        except OSError:
            pass

    source_auth = source_dir / "auth.json"
    target_auth = target_dir / "auth.json"
    if source_auth.is_file() and not target_auth.exists():
        shutil.copy2(source_auth, target_auth)
        try:
            target_auth.chmod(0o600)
        except OSError:
            pass


def get_bot_profile(name: str) -> BotProfile:
    """Resolve one bot by canonical profile name, case-insensitively."""
    canonical = profiles_mod.normalize_profile_name(name)
    profiles_mod.validate_profile_name(canonical)
    profile_dir = profiles_mod.get_profile_dir(canonical)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Bot '${name}' does not exist.")

    model, provider = profiles_mod._read_config_model(profile_dir)
    metadata = profiles_mod.read_profile_meta(profile_dir)
    return BotProfile(
        name=canonical,
        path=profile_dir,
        model=str(model or ""),
        provider=str(provider or ""),
        system_prompt=_read_system_prompt(profile_dir),
        enabled=bool(metadata.get("bot_enabled", True)),
    )


def list_bot_profiles() -> list[BotProfile]:
    """List every live local profile using the lightweight profile scan."""
    return [
        get_bot_profile(name)
        for name, _path in profiles_mod.profiles_to_serve(multiplex=True)
    ]


def check_bot_chain_profile_access(profile: BotProfile, home: Path | None = None) -> None:
    """Live execution authority shared by resolution and each dispatched step."""
    from hermes_constants import get_hermes_home
    from tools.bot_mode_probe import _configured_targets, _hermes_root, _profile_name, _is_roster_profile_dir, _is_bot_enabled

    home = home if home is not None else get_hermes_home()
    root = _hermes_root(home)
    allowed = _configured_targets(root, _profile_name(home))
    if not _is_roster_profile_dir(root, profile.path) or (allowed is not None and profile.name not in allowed):
        raise ValueError(f"Bot '${profile.name}' is not allowed by the active profile's Bot Mode roster.")
    if not _is_bot_enabled(profile.path):
        raise ValueError(f"Bot '${profile.name}' is disabled or its metadata is unreadable.")


def resolve_bot_chain(names: Sequence[str]) -> list[BotProfile]:
    """Resolve ordered nicknames and fail before any model turn starts."""
    available = list_bot_profiles()
    by_name = {profile.name.casefold(): profile for profile in available}
    display = (
        ", ".join(f"${profile.name}" for profile in available if profile.enabled)
        or "(none enabled)"
    )
    resolved: list[BotProfile] = []
    for nickname in names:
        profile = by_name.get(str(nickname).casefold())
        if profile is None:
            raise ValueError(
                f"Unknown bot '${nickname}'. Available bots: {display}."
            )
        check_bot_chain_profile_access(profile)
        if not profile.enabled:
            raise ValueError(
                f"Bot '${profile.name}' is disabled. Enable it with: "
                f"hermes bots enable {profile.name}"
            )
        if not profile.provider or not profile.model:
            raise ValueError(
                f"Bot '${profile.name}' has no model/provider configured. "
                f"Run: hermes bots configure {profile.name} "
                "--provider <provider> --model <model>"
            )
        resolved.append(profile)
    return resolved


def create_bot_profile(
    name: str,
    *,
    model: str,
    provider: str,
    system_prompt: str,
    enabled: bool = True,
    mirror_credentials: bool = True,
    seed_skills: bool = True,
) -> BotProfile:
    """Create a runnable bot as a fresh, isolated Hermes profile."""
    canonical = profiles_mod.normalize_profile_name(name)
    profiles_mod.validate_profile_name(canonical)
    provider, model = _normalize_model_assignment(provider, model)
    if not str(system_prompt or "").strip():
        raise ValueError("System prompt cannot be empty.")

    profile_dir = profiles_mod.create_profile(canonical, no_alias=True)
    try:
        if mirror_credentials:
            from hermes_constants import get_hermes_home

            _mirror_credentials(Path(get_hermes_home()), profile_dir)
        _write_profile_model(profile_dir, provider, model)
        _write_system_prompt(profile_dir, system_prompt)
        profiles_mod.write_profile_meta(profile_dir, bot_enabled=enabled)
    except Exception:
        try:
            profiles_mod.delete_profile(canonical, yes=True)
        except Exception:
            pass
        raise

    if seed_skills:
        profiles_mod.seed_profile_skills(profile_dir, quiet=True)
    return get_bot_profile(canonical)


def configure_bot_profile(
    name: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> BotProfile:
    """Update explicitly supplied bot fields, preserving every other field."""
    current = get_bot_profile(name)
    if model is not None or provider is not None:
        _write_profile_model(
            current.path,
            provider if provider is not None else current.provider,
            model if model is not None else current.model,
        )
    if system_prompt is not None:
        _write_system_prompt(current.path, system_prompt)
    if enabled is not None:
        profiles_mod.write_profile_meta(current.path, bot_enabled=enabled)
    return get_bot_profile(current.name)


def remove_bot_profile(name: str, *, confirmed: bool = False) -> Path:
    """Remove a named bot without ever opening an interactive prompt."""
    canonical = profiles_mod.normalize_profile_name(name)
    if canonical == "default":
        raise ValueError("The built-in $default bot cannot be removed.")
    if not confirmed:
        raise ValueError("Refusing to remove a bot without --yes.")
    get_bot_profile(canonical)
    return profiles_mod.delete_profile(canonical, yes=True)


def rename_bot_profile(old_name: str, new_name: str) -> BotProfile:
    """Rename a named bot profile and return its new resolved view."""
    old_canonical = profiles_mod.normalize_profile_name(old_name)
    if old_canonical == "default":
        raise ValueError("The built-in $default bot cannot be renamed.")
    new_canonical = profiles_mod.normalize_profile_name(new_name)
    profiles_mod.rename_profile(old_canonical, new_canonical, no_alias=True)
    return get_bot_profile(new_canonical)


def _print_bot(profile: BotProfile, *, include_prompt: bool = False) -> None:
    status = "enabled" if profile.enabled else "disabled"
    print(f"Name:          ${profile.name}")
    print(f"Status:        {status}")
    print(f"Provider:      {profile.provider or '-'}")
    print(f"Model:         {profile.model or '-'}")
    print(f"Path:          {profile.path}")
    if include_prompt:
        print("System prompt:")
        print(profile.system_prompt or "-")


def run_bots_command(args) -> int:
    """Implementation for the ``hermes bots`` argparse surface."""
    action = getattr(args, "bots_action", None) or "list"
    try:
        if action == "list":
            profiles = list_bot_profiles()
            if not profiles:
                print("No bot profiles found.")
                return 0
            for index, profile in enumerate(profiles):
                if index:
                    print()
                _print_bot(profile)
            return 0

        if action == "create":
            profile = create_bot_profile(
                args.name,
                model=args.model,
                provider=args.provider,
                system_prompt=args.system_prompt,
                enabled=not args.disabled,
            )
            print(f"Created bot ${profile.name}.")
            _print_bot(profile)
            return 0

        if action == "info":
            _print_bot(get_bot_profile(args.name), include_prompt=True)
            return 0

        if action == "configure":
            enabled = True if args.enable else False if args.disable else None
            if (
                args.model is None
                and args.provider is None
                and args.system_prompt is None
                and enabled is None
            ):
                raise ValueError("No changes supplied.")
            profile = configure_bot_profile(
                args.name,
                model=args.model,
                provider=args.provider,
                system_prompt=args.system_prompt,
                enabled=enabled,
            )
            print(f"Updated bot ${profile.name}.")
            _print_bot(profile)
            return 0

        if action in {"enable", "disable"}:
            profile = configure_bot_profile(args.name, enabled=action == "enable")
            print(f"Bot ${profile.name} is now {'enabled' if profile.enabled else 'disabled'}.")
            return 0

        if action == "rename":
            profile = rename_bot_profile(args.old_name, args.new_name)
            print(f"Bot nickname is now ${profile.name}.")
            return 0

        if action == "remove":
            removed = remove_bot_profile(args.name, confirmed=args.yes)
            print(f"Removed bot ${profiles_mod.normalize_profile_name(args.name)} ({removed}).")
            return 0

        raise ValueError(f"Unknown bots action: {action}")
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
