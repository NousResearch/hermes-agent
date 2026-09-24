"""Atomic materialization of reviewed bot blueprints as complete Hermes profiles."""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from hermes_cli.bot_catalog import (
    BotCatalogEntry,
    resolve_reviewed_skill_path,
    validate_bot_catalog_dependencies,
)

CredentialPolicy = Literal["none", "copy_api_keys"]
SetupState = Literal["ready", "needs_setup"]


@dataclass(frozen=True)
class BotProfileInstallSpec:
    catalog_name: str
    name: str
    source_profile: str
    credentials: CredentialPolicy = "none"


@dataclass(frozen=True)
class BotProfileInstallReceipt:
    committed: bool
    name: str
    path: Path
    catalog_name: str
    catalog_version: str
    source_profile: str
    copied_credentials: tuple[str, ...]
    oauth_setup_required: tuple[str, ...]
    setup_state: SetupState
    setup_requirements: tuple[str, ...]
    post_publish_warnings: tuple[str, ...]


@contextlib.contextmanager
def _home_scope(home: Path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _resolve_source_profile(name: str) -> Path:
    if not name:
        raise ValueError("source_profile is required")
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import named_profile_has_identity

    source = get_profile_dir(name)
    if not source.is_dir() or (name != "default" and not named_profile_has_identity(source)):
        raise FileNotFoundError(f"source profile {name!r} does not exist")
    return source


def _copy_reviewed_skills(entry: BotCatalogEntry, staging: Path) -> None:
    for identifier in entry.capabilities.skills:
        source = resolve_reviewed_skill_path(identifier)
        if source is None:
            raise FileNotFoundError(f"reviewed skill {identifier!r} is unavailable")
        relative = Path(*identifier.split("/")[1:])
        destination = staging / "skills" / relative
        shutil.copytree(source, destination, symlinks=False)


def _copy_api_credentials(source: Path, staging: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    copied: list[str] = []
    oauth_setup: list[str] = []
    source_env = source / ".env"
    if source_env.is_file():
        shutil.copy2(source_env, staging / ".env")
        os.chmod(staging / ".env", 0o600)
        from hermes_cli.profile_channels import strip_channel_settings

        strip_channel_settings(staging, include_state=False, source_dir=source)
        copied.append(".env")

    source_auth = source / "auth.json"
    if source_auth.is_file():
        shutil.copy2(source_auth, staging / "auth.json")
        os.chmod(staging / "auth.json", 0o600)
        from hermes_cli.auth import strip_cloned_single_use_oauth_grants

        # Single-use OAuth grants are never forked into the bot profile. Current auth resolution is
        # profile-first with a global-root fallback, so a usable root Codex/Nous sign-in remains
        # available and refreshes are persisted by the canonical resolver to their owning store.
        # Readiness is decided later by setup.runtime_check, not by the fact a clone strip occurred.
        strip_cloned_single_use_oauth_grants(staging)
        if (staging / "auth.json").is_file():
            copied.append("auth.json")
    return tuple(copied), tuple(sorted(set(oauth_setup)))


def _write_bot_config(staging: Path, entry: BotCatalogEntry) -> None:
    path = staging / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config = raw if isinstance(raw, dict) else {}
    required = list(dict.fromkeys([
        *entry.capabilities.toolsets,
        *(item.id for item in entry.setup.requirements if item.kind == "toolset"),
    ]))
    from hermes_cli.tools_config import _platform_default_toolset

    platform_toolsets_raw = config.get("platform_toolsets")
    platform_toolsets = dict(platform_toolsets_raw) if isinstance(platform_toolsets_raw, dict) else {}
    for platform in ("cli", "desktop", "cron"):
        platform_toolsets[platform] = list(dict.fromkeys([_platform_default_toolset(platform), *required]))
    config["platform_toolsets"] = platform_toolsets
    # Tool availability is platform-scoped. Remove the legacy profile-editor pin so it cannot
    # contradict the canonical CLI/Desktop selections above.
    tools_raw = config.get("tools")
    if isinstance(tools_raw, dict) and "enabled_toolsets" in tools_raw:
        tools = dict(tools_raw)
        tools.pop("enabled_toolsets", None)
        if tools:
            config["tools"] = tools
        else:
            config.pop("tools", None)
    from utils import atomic_yaml_write

    atomic_yaml_write(path, config, sort_keys=False)


def _write_bot_blueprint(staging: Path, entry: BotCatalogEntry) -> None:
    """Persist the reviewed install-time blueprint; later catalog edits must not mutate this bot."""
    from utils import atomic_yaml_write

    atomic_yaml_write(
        staging / "bot-blueprint.yaml",
        {"schema_version": 1, "catalog_entry": entry.model_dump(mode="json")},
        sort_keys=False,
    )


def _write_bot_metadata(
    staging: Path, entry: BotCatalogEntry, spec: BotProfileInstallSpec, *, oauth_setup: tuple[str, ...]
) -> None:
    from utils import atomic_yaml_write

    title = entry.title.strip()
    description = entry.profile.description.strip()
    requirement_ids = [f"{item.kind}:{item.id}" for item in entry.setup.requirements if item.required]
    requirement_ids.extend(f"oauth:{name}" for name in oauth_setup)
    metadata = {
        "description": description,
        "description_auto": False,
        "display_name": title,
        "ui_meta": {
            "hermes-bots": {
                "title": title,
                "emoji": entry.presentation.emoji,
                "color": entry.presentation.color,
                "starter_prompt": entry.profile.starter_prompt,
            }
        },
        "provenance": {
            "kind": "bot-catalog",
            "catalog_name": entry.name,
            "catalog_version": entry.version,
            "maintainer": entry.maintainer,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "source_profile": spec.source_profile,
        },
        "setup_state": "needs_setup",
        "setup_requirements": requirement_ids,
        "first_task": {"status": "pending", "session_id": None},
        "routines": [
            {
                "id": routine.id,
                "name": routine.name,
                "prompt": routine.prompt,
                "schedule": routine.schedule,
                "state": "paused",
                "job_id": None,
                "timezone": None,
                "destination": None,
            }
            for routine in entry.routines
        ],
    }
    atomic_yaml_write(staging / "profile.yaml", metadata, sort_keys=False)


def install_bot_profile(entry: BotCatalogEntry, spec: BotProfileInstallSpec) -> BotProfileInstallReceipt:
    """Prepare every profile artifact while hidden and atomically publish exactly once."""
    if spec.catalog_name != entry.name:
        raise ValueError("resolved catalog entry does not match catalog_name")
    if spec.credentials not in ("none", "copy_api_keys"):
        raise ValueError("credentials must be 'none' or 'copy_api_keys'")
    validate_bot_catalog_dependencies(entry, label=f"bot {entry.name!r} install")
    source = _resolve_source_profile(spec.source_profile)
    state: dict[str, tuple[str, ...]] = {"copied": (), "oauth": ()}

    def prepare(staging: Path) -> None:
        from hermes_cli import profiles

        with _home_scope(source):
            profiles._bootstrap_profile_dir(staging, None)
            profiles._finish_profile_layout(
                staging, no_skills=False, clone_all=False,
                description=entry.profile.description,
            )
        (staging / "SOUL.md").write_text(entry.profile.soul, encoding="utf-8")
        _write_bot_config(staging, entry)
        _copy_reviewed_skills(entry, staging)
        _write_bot_blueprint(staging, entry)
        if spec.credentials == "copy_api_keys":
            state["copied"], state["oauth"] = _copy_api_credentials(source, staging)
        _write_bot_metadata(staging, entry, spec, oauth_setup=state["oauth"])

    from hermes_cli.profiles import _publish_staged_profile

    path = _publish_staged_profile(spec.name, prepare)
    setup_requirements = tuple([
        *(f"{item.kind}:{item.id}" for item in entry.setup.requirements if item.required),
        *(f"oauth:{name}" for name in state["oauth"]),
    ])
    # A committed profile is not ready until the runtime probe and a successful first Bot Chat task prove it.
    setup_state: SetupState = "needs_setup"
    return BotProfileInstallReceipt(
        committed=True,
        name=path.name,
        path=path,
        catalog_name=entry.name,
        catalog_version=entry.version,
        source_profile=spec.source_profile,
        copied_credentials=state["copied"],
        oauth_setup_required=state["oauth"],
        setup_state=setup_state,
        setup_requirements=setup_requirements,
        post_publish_warnings=(),
    )
