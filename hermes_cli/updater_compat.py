"""Symbols frozen for legacy-updater compatibility.

See docs/updater-world.md §2.13.

Every entry here is imported/called by SOME historical ``hermes update``
after it has pulled current code. Changing a signature or deleting an
entry bricks that population's next update. Guarded by
tests/test_updater_compat_fence.py. Sunset: see docs/plans/updater-rework/
06-phase5-ledger-and-sunset.md.

The fence freezes signatures *as they exist on current main*. If a frozen
symbol already drifted between some historical release and today, the fence
enshrines the drifted shape and the fence alone won't notice. The fence
is necessary-not-sufficient: it stops FUTURE drift. The authority on
whether hop 1 actually works for a given vintage is the phase-2 E2E
(task 2.8), which runs a real old release against current main — when the
E2E and the fence disagree, the E2E wins and the fence gets corrected.
"""

from __future__ import annotations

# ─── Frozen callables ─────────────────────────────────────────────────
#
# "module:qualname" -> frozen signature string (inspect.signature format)
#
# These are the symbols that old updaters import/call AFTER pulling new
# code onto disk. The old in-memory code crosses the update boundary on
# every first-time lazy import in the post-pull phase.

FROZEN_CALLABLES: dict[str, str] = {
    # ── managed_uv.py ── uv resolution + venv management ──
    "hermes_cli.managed_uv:ensure_uv": "()",
    "hermes_cli.managed_uv:update_managed_uv": "() -> 'Optional[str]'",
    "hermes_cli.managed_uv:rebuild_venv": (
        "(uv_bin: 'str', venv_dir: 'Path', python_version: 'str' = '3.11') -> 'bool'"
    ),

    # ── main.py ── the update flow's post-pull steps ──
    "hermes_cli.main:_install_python_dependencies_with_optional_fallback": (
        "(install_cmd_prefix: list[str], *, env: dict[str, str] | None = None, "
        "group: str = 'all') -> None"
    ),
    "hermes_cli.main:_update_node_dependencies": "() -> None",
    "hermes_cli.main:_build_web_ui": "(web_dir: pathlib.Path, *, fatal: bool = False) -> bool",
    "hermes_cli.main:_refresh_active_lazy_features": "() -> None",

    # ── skills_sync.py ── bundled skills seeding ──
    "tools.skills_sync:sync_skills": "(quiet: bool = False) -> dict",

    # ── lazy_deps.py ── optional feature management ──
    "tools.lazy_deps:active_features": "() -> 'list[str]'",
    "tools.lazy_deps:refresh_active_features": "(*, prompt: 'bool' = False) -> 'dict[str, str]'",

    # ── profiles.py ── per-profile skill/env sync ──
    "hermes_cli.profiles:list_profiles": "() -> List[hermes_cli.profiles.ProfileInfo]",
    "hermes_cli.profiles:seed_profile_skills": (
        "(profile_dir: pathlib.Path, quiet: bool = False) -> Optional[dict]"
    ),
    "hermes_cli.profiles:backfill_profile_envs": "(quiet: bool = False) -> List[str]",

    # ── model_catalog.py ── cache seed from checkout ──
    "hermes_cli.model_catalog:seed_cache_from_checkout": (
        "(project_root: \"'Path | str'\") -> 'bool'"
    ),

    # ── config.py ── config migration + env checks ──
    "hermes_cli.config:get_missing_env_vars": (
        "(required_only: bool = False) -> List[Dict[str, Any]]"
    ),
    "hermes_cli.config:get_missing_config_fields": "() -> List[Dict[str, Any]]",
    "hermes_cli.config:check_config_version": "() -> Tuple[int, int]",
    "hermes_cli.config:migrate_config": (
        "(interactive: bool = True, quiet: bool = False) -> Dict[str, Any]"
    ),
    "hermes_cli.config:load_config": "() -> Dict[str, Any]",

    # ── hermes_constants.py ── attributes read post-reload ──
    "hermes_constants:find_node_executable": "(command: str) -> str | None",
    "hermes_constants:with_hermes_node_path": (
        "(env: dict[str, str] | None = None) -> dict[str, str]"
    ),
    "hermes_constants:display_hermes_home": "() -> str",

    # ── Additional post-pull symbols discovered by archaeology ──
    # These are called post-pull but were not in the initial §2.13 list.
    # Added after a full _cmd_update_impl audit (subagent archaeology).

    # backup.py — snapshot + cron safety net
    "hermes_cli.backup:create_quick_snapshot": (
        "(label: Optional[str] = None, hermes_home: Optional[pathlib.Path] = None, "
        "keep: Optional[int] = None) -> Optional[str]"
    ),
    "hermes_cli.backup:restore_cron_jobs_if_emptied": (
        "(snapshot_id: str, hermes_home: Optional[pathlib.Path] = None) -> Optional[Dict[str, Any]]"
    ),

    # config.py — install method detection (used in the no-.git branch)
    "hermes_cli.config:detect_install_method": "(project_root: Optional[pathlib.Path] = None) -> str",

    # tools_config.py — cua-driver refresh
    "hermes_cli.tools_config:install_cua_driver": "(upgrade: bool = False) -> bool",

    # gateway.status — survivor sweep
    "gateway.status:terminate_pid": "(pid: int, *, force: bool = False) -> None",
}

# ─── Frozen CLI surfaces ──────────────────────────────────────────────
#
# Argv shapes that old updaters shell out to via subprocess. The fence
# verifies these are accepted by the current argparse parser without
# SystemExit.

FROZEN_CLI_SURFACES: list[list[str]] = [
    # Gateway detached update
    ["update", "--gateway"],
    # Tauri/desktop forced update
    ["update", "--yes", "--gateway", "--force", "--branch"],
    # Desktop rebuild
    ["desktop", "--build-only"],
]

# ─── Frozen file/path contract ────────────────────────────────────────
#
# Files that must exist for old updaters to function. Removing any of
# these bricks the update flow for installs parked on old versions.

FROZEN_PATHS: list[str] = [
    "pyproject.toml",  # must remain editable-installable with [all] extra
    "constraints-termux.txt",  # Termux pip constraint file
]
