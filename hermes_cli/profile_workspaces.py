"""Profile-owned layout and workspace configuration before atomic publication."""

import contextlib
import os
from pathlib import Path


def seed_profile_workspace(staging_home: Path, destination_home: Path, source_home: Path | None) -> None:
    """Persist the published home, never the temporary staging directory."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from tools.terminal_scope import build_profile_terminal_scope

    path = staging_home / "config.yaml"
    config = read_user_config_raw(path)
    terminal = config.get("terminal") or {}
    # Read the copied destination files, never the launch process's TERMINAL_ENV.
    backend = build_profile_terminal_scope(staging_home)["TERMINAL_ENV"]
    # Presence wins even for null: the builder skips null and inherits .env/defaults.
    # Main's runtime builder does not normalize env_type; keep this alias creation-only.
    if "backend" not in terminal and terminal.get("env_type") is not None:
        backend = terminal["env_type"]
    if str(backend or "local").strip().lower() != "local":
        return  # A host profile path is not an SSH/container working directory.
    if source_home is not None:
        raw = str(terminal.get("cwd") or "").strip()
        # Normalize lexical aliases without following symlinks to explicit external projects.
        source_paths = {os.path.abspath(source_home), os.path.abspath(source_home / "workspace")}
        if raw not in {"", ".", "auto", "cwd"} and os.path.abspath(Path(raw).expanduser()) not in source_paths:
            return
    terminal["cwd"] = os.path.abspath(destination_home / "workspace")
    config["terminal"] = terminal
    atomic_config_write(path, config)


def finish_profile_layout(profile_dir: Path, *,
                          no_skills: bool, clone_all: bool, description: str | None) -> None:
    """Seed files a fresh profile owns from day one; runs on the staging tree before publish."""
    from hermes_cli import profiles as p

    # The profile owns a credentials file from day one, rather than inheriting shell API keys.
    # Skipped when a clone copied one.
    p._seed_file_if_missing(profile_dir / ".env", p._PLACEHOLDER_ENV, 0o600)

    # Default SOUL.md to customize immediately (skipped when a clone already provided one).
    with contextlib.suppress(Exception):  # best-effort — don't fail profile creation over this
        from hermes_cli.default_soul import DEFAULT_SOUL_MD
        p._seed_file_if_missing(profile_dir / "SOUL.md", DEFAULT_SOUL_MD)

    # Opt-out marker read by seed_profile_skills() and `hermes update`'s all-profile sync.
    if no_skills:
        p._seed_file_if_missing(
            profile_dir / p.NO_BUNDLED_SKILLS_MARKER,
            "This profile opted out of bundled-skill seeding (`hermes profile create --no-skills`).\n"
            "Delete this file to re-enable sync on the next `hermes update`.\n",
        )

    # Full clones retain their config version; only the workspace policy below changes it.
    if not clone_all:
        p._migrate_profile_config_if_outdated(profile_dir)

    # Description last, so a partial-create failure doesn't strand a description file.
    if description and description.strip():
        with contextlib.suppress(Exception):  # non-fatal — `hermes profile describe` works later
            p.write_profile_meta(profile_dir, description=description.strip(), description_auto=False)
