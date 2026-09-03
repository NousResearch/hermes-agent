"""Stable compatibility facade for the Hermes update pipeline.

The implementation is split by operational ownership.  This module deliberately
keeps the historical imports, constants, and callables because ``hermes_cli.main``
and the test suite patch this surface.  Domain modules resolve those names back
through this facade at call time.
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home
from hermes_cli.update_abort_recovery import (
    _abort_recovery_is_complete,
    _qualified_serve_skips,
    _recover_gateway_restart_after_abort,
    _serve_unit_recovery_available,
    _surviving_pre_update_serve_runtimes,
    _warn_stale_serve_runtimes,
)
from hermes_constants import get_default_hermes_root, venv_python_path

logger = logging.getLogger(__name__)


def _m():
    """Lazy ``hermes_cli.main`` reference preserving the historical patch surface."""
    from hermes_cli import main

    return main


_LAST_SIBLING_SNAPSHOTS: dict = {}
_holder_value_flags_cache: frozenset | None = None

from hermes_cli import update_runtime_refresh as _update_runtime_refresh
from hermes_cli import update_notices as _update_notices
from hermes_cli import update_zip as _update_zip
from hermes_cli import update_source as _update_source
from hermes_cli import update_reconciliation as _update_reconciliation
from hermes_cli import update_dependencies as _update_dependencies
from hermes_cli import update_backup as _update_backup
from hermes_cli import update_process_guard as _update_process_guard
from hermes_cli import update_gateway_windows as _update_gateway_windows
from hermes_cli import update_gateway_posix as _update_gateway_posix
from hermes_cli import update_desktop as _update_desktop
from hermes_cli import update_fleet_restart as _update_fleet_restart
from hermes_cli import update_orchestrator as _update_orchestrator

_UPDATE_RUNTIME_RELOAD_MODULES = _update_runtime_refresh._UPDATE_RUNTIME_RELOAD_MODULES
_STALE_PURGE_PREFIXES = _update_runtime_refresh._STALE_PURGE_PREFIXES
_STALE_PURGE_PROTECTED = _update_runtime_refresh._STALE_PURGE_PROTECTED
_purge_stale_hermes_modules = _update_runtime_refresh._purge_stale_hermes_modules
_reload_updated_runtime_modules = _update_runtime_refresh._reload_updated_runtime_modules
_reload_config_modules = _update_runtime_refresh._reload_config_modules
_run_config_check_fresh = _update_runtime_refresh._run_config_check_fresh
_run_migrate_config_fresh = _update_runtime_refresh._run_migrate_config_fresh
_migrate_sibling_profile_configs = _update_runtime_refresh._migrate_sibling_profile_configs
_check_and_apply_config_migration = _update_runtime_refresh._check_and_apply_config_migration
_UPDATE_CRITICAL_FILES = _update_runtime_refresh._UPDATE_CRITICAL_FILES
_capture_head_sha = _update_runtime_refresh._capture_head_sha
_INSTALL_DEFINING_FILES = _update_runtime_refresh._INSTALL_DEFINING_FILES
_editable_install_is_current = _update_runtime_refresh._editable_install_is_current
_validate_critical_files_syntax = _update_runtime_refresh._validate_critical_files_syntax
_UPDATE_CRITICAL_MODULES = _update_runtime_refresh._UPDATE_CRITICAL_MODULES
_validate_critical_modules_import = _update_runtime_refresh._validate_critical_modules_import
_gateway_prompt = _update_notices._gateway_prompt
_npm_bin_exists = _update_notices._npm_bin_exists
_web_build_toolchain_ready = _update_notices._web_build_toolchain_ready
_web_toolchain_roots = _update_notices._web_toolchain_roots
_print_curator_first_run_notice = _update_notices._print_curator_first_run_notice
_print_fts_optimize_available_notice = _update_notices._print_fts_optimize_available_notice
_print_curator_recent_run_notice = _update_notices._print_curator_recent_run_notice
_format_time_ago = _update_notices._format_time_ago
_reload_process_scan_modules = _update_notices._reload_process_scan_modules
_finish_dashboard_update_cleanup = _update_notices._finish_dashboard_update_cleanup
_atomic_replace_dir = _update_zip._atomic_replace_dir
_stage_replacement = _update_zip._stage_replacement
_discard_staged = _update_zip._discard_staged
_commit_staged_replacements = _update_zip._commit_staged_replacements
_branch_head_label = _update_zip._branch_head_label
_branch_head_suffix = _update_zip._branch_head_suffix
_assess_parked_branch_switch = _update_zip._assess_parked_branch_switch
_print_parked_branch_skip_warning = _update_zip._print_parked_branch_skip_warning
_print_parked_branch_kept_notice = _update_zip._print_parked_branch_kept_notice
_print_update_completion = _update_zip._print_update_completion
_called_process_error_cmd_parts = _update_zip._called_process_error_cmd_parts
_called_process_error_is_git = _update_zip._called_process_error_is_git
_called_process_error_is_python_dep_install = _update_zip._called_process_error_is_python_dep_install
_format_update_failure_stage = _update_zip._format_update_failure_stage
_shim_quarantine_error_type = _update_zip._shim_quarantine_error_type
_refuse_update_for_contended_shims = _update_zip._refuse_update_for_contended_shims
_should_zip_fallback_on_update_error = _update_zip._should_zip_fallback_on_update_error
_print_called_process_error_tail = _update_zip._print_called_process_error_tail
_zip_overlay_block_reason = _update_zip._zip_overlay_block_reason
_ZIP_STAGING_ARTIFACT_SUFFIXES = _update_zip._ZIP_STAGING_ARTIFACT_SUFFIXES
_ZIP_PRESERVED_TOP_LEVEL = _update_zip._ZIP_PRESERVED_TOP_LEVEL
_is_zip_preserved_entry_status_line = _update_zip._is_zip_preserved_entry_status_line
_is_zip_staging_artifact_status_line = _update_zip._is_zip_staging_artifact_status_line
_abort_zip_update_if_dirty_tree = _update_zip._abort_zip_update_if_dirty_tree
_read_project_version = _update_zip._read_project_version
_update_complete_message = _update_zip._update_complete_message
_clear_stale_sqlite_sidecars = _update_zip._clear_stale_sqlite_sidecars
_print_update_summary = _update_zip._print_update_summary
_write_gateway_update_exit_code = _update_zip._write_gateway_update_exit_code
_restore_state_db_from_snapshot = _update_zip._restore_state_db_from_snapshot
_update_via_zip = _update_zip._update_via_zip
_stash_local_changes_if_needed = _update_source._stash_local_changes_if_needed
_resolve_stash_selector = _update_source._resolve_stash_selector
_print_stash_cleanup_guidance = _update_source._print_stash_cleanup_guidance
_stash_apply_failed_only_on_existing_untracked = _update_source._stash_apply_failed_only_on_existing_untracked
_park_stashed_changes = _update_source._park_stashed_changes
_restore_stashed_changes = _update_source._restore_stashed_changes
_discard_stashed_changes = _update_source._discard_stashed_changes
OFFICIAL_REPO_URLS = _update_source.OFFICIAL_REPO_URLS
OFFICIAL_REPO_URL = _update_source.OFFICIAL_REPO_URL
SKIP_UPSTREAM_PROMPT_FILE = _update_source.SKIP_UPSTREAM_PROMPT_FILE
_get_origin_url = _update_source._get_origin_url
_no_prompt_git_kwargs = _update_source._no_prompt_git_kwargs
_is_fork = _update_source._is_fork
_has_upstream_remote = _update_source._has_upstream_remote
_add_upstream_remote = _update_source._add_upstream_remote
_count_commits_between = _update_source._count_commits_between
_should_skip_upstream_prompt = _update_source._should_skip_upstream_prompt
_mark_skip_upstream_prompt = _update_source._mark_skip_upstream_prompt
_sync_fork_with_upstream = _update_source._sync_fork_with_upstream
_sync_with_upstream_if_needed = _update_source._sync_with_upstream_if_needed
_invalidate_update_cache = _update_reconciliation._invalidate_update_cache
_write_marker_file = _update_reconciliation._write_marker_file
_write_update_incomplete_marker = _update_reconciliation._write_update_incomplete_marker
_write_lazy_refresh_incomplete_marker = _update_reconciliation._write_lazy_refresh_incomplete_marker
_FLEET_RESTART_PENDING_NAME = _update_reconciliation._FLEET_RESTART_PENDING_NAME
_fleet_restart_pending_marker_path = _update_reconciliation._fleet_restart_pending_marker_path
_write_fleet_restart_pending_marker = _update_reconciliation._write_fleet_restart_pending_marker
_clear_fleet_restart_pending_marker = _update_reconciliation._clear_fleet_restart_pending_marker
_current_checkout_sha = _update_reconciliation._current_checkout_sha
_receipt_looks_unfinished = _update_reconciliation._receipt_looks_unfinished
_receipt_reports_stale_runtime = _update_reconciliation._receipt_reports_stale_runtime
_pending_fleet_restart_needed = _update_reconciliation._pending_fleet_restart_needed
_warn_pending_fleet_restart = _update_reconciliation._warn_pending_fleet_restart
_warn_pending_fleet_restart_on_startup = _update_reconciliation._warn_pending_fleet_restart_on_startup
_restart_systemd_gateway_units_best_effort = _update_reconciliation._restart_systemd_gateway_units_best_effort
_run_pending_fleet_restart = _update_reconciliation._run_pending_fleet_restart
_apply_pending_fleet_restart_catchup = _update_reconciliation._apply_pending_fleet_restart_catchup
_format_concurrent_instances_message = _update_process_guard._format_concurrent_instances_message
_classify_concurrent_instance = _update_process_guard._classify_concurrent_instance
_filter_non_gateway_concurrent_instances = _update_process_guard._filter_non_gateway_concurrent_instances
_upgrade_pip_before_lazy_refresh = _update_dependencies._upgrade_pip_before_lazy_refresh
_capture_active_lazy_features = _update_dependencies._capture_active_lazy_features
_capture_active_tool_dependencies = _update_dependencies._capture_active_tool_dependencies
_restore_active_tool_dependencies = _update_dependencies._restore_active_tool_dependencies
_refresh_active_lazy_features = _update_dependencies._refresh_active_lazy_features
_refresh_active_memory_provider_dependencies = _update_dependencies._refresh_active_memory_provider_dependencies
_is_android_python = _update_dependencies._is_android_python
_install_psutil_android_compat = _update_dependencies._install_psutil_android_compat
_ensure_uv_for_termux = _update_dependencies._ensure_uv_for_termux
_npm_manifest_paths = _update_dependencies._npm_manifest_paths
_npm_manifests_digest = _update_dependencies._npm_manifests_digest
_npm_lockfile_changed = _update_dependencies._npm_lockfile_changed
_record_npm_lockfile_hash = _update_dependencies._record_npm_lockfile_hash
_repair_node_deps_on_current_checkout = _update_dependencies._repair_node_deps_on_current_checkout
_update_node_dependencies = _update_dependencies._update_node_dependencies
_log_only_write = _update_dependencies._log_only_write
_run_logged_subprocess = _update_dependencies._run_logged_subprocess
_classify_fetch_failure = _update_dependencies._classify_fetch_failure
_print_fetch_failure = _update_dependencies._print_fetch_failure
_cmd_update_check = _update_source._cmd_update_check
_ensure_fhs_path_guard = _update_dependencies._ensure_fhs_path_guard
_ensure_acp_launcher = _update_dependencies._ensure_acp_launcher
_PRE_UPDATE_SNAPSHOT_KEEP = _update_backup._PRE_UPDATE_SNAPSHOT_KEEP
_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = _update_backup._PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE
_resolve_pre_update_backup_mode = _update_backup._resolve_pre_update_backup_mode
_run_pre_update_backup = _update_backup._run_pre_update_backup
_write_update_planned_stop_marker = _update_process_guard._write_update_planned_stop_marker
_wait_for_windows_update_gateway_exit = _update_process_guard._wait_for_windows_update_gateway_exit
_venv_core_imports_healthy = _update_process_guard._venv_core_imports_healthy
_detect_venv_python_processes = _update_process_guard._detect_venv_python_processes
_SELF_LOCKING_NATIVE_MODULES = _update_process_guard._SELF_LOCKING_NATIVE_MODULES
_dependency_sync_would_rewrite = _update_process_guard._dependency_sync_would_rewrite
_detect_self_loaded_native_modules = _update_process_guard._detect_self_loaded_native_modules
_abort_dependency_sync_if_self_locked = _update_process_guard._abort_dependency_sync_if_self_locked
_defer_update_for_self_lock = _update_process_guard._defer_update_for_self_lock
_HOLDER_VALUE_FLAGS_FALLBACK = _update_process_guard._HOLDER_VALUE_FLAGS_FALLBACK
_holder_value_flags = _update_process_guard._holder_value_flags
_hermes_holder_subcommand = _update_process_guard._hermes_holder_subcommand
_format_venv_python_holders_message = _update_process_guard._format_venv_python_holders_message
_venv_launcher_ancestors = _update_process_guard._venv_launcher_ancestors
_leftover_pausable_gateway_pids = _update_process_guard._leftover_pausable_gateway_pids
_ledger_manual_serve_holders = _update_process_guard._ledger_manual_serve_holders
_serve_relaunch_commands = _update_process_guard._serve_relaunch_commands
_relaunch_stopped_serves = _update_process_guard._relaunch_stopped_serves
_orphaned_desktop_backend_pids = _update_process_guard._orphaned_desktop_backend_pids
_ledger_reapable_backend_pids = _update_process_guard._ledger_reapable_backend_pids
_handoff_reapable_backend_pids = _update_process_guard._handoff_reapable_backend_pids
_stop_process_trees = _update_process_guard._stop_process_trees
_looks_like_desktop_control_plane = _update_process_guard._looks_like_desktop_control_plane
_desktop_owns_gateway_lifecycle = _update_process_guard._desktop_owns_gateway_lifecycle
_stop_windows_gateway_service = _update_gateway_windows._stop_windows_gateway_service
_start_windows_gateway_service = _update_gateway_windows._start_windows_gateway_service
_restore_windows_gateway_service = _update_gateway_windows._restore_windows_gateway_service
_pause_windows_gateways_for_update = _update_gateway_windows._pause_windows_gateways_for_update
_cold_start_windows_gateway_after_update = _update_gateway_windows._cold_start_windows_gateway_after_update
_for_each_systemd_gateway_unit = _update_gateway_posix._for_each_systemd_gateway_unit
_service_unit_supports_graceful_sigusr1_restart = _update_gateway_posix._service_unit_supports_graceful_sigusr1_restart
_warn_incomplete_gateway_fleet_restart = _update_gateway_posix._warn_incomplete_gateway_fleet_restart
_restart_launchd_gateway_after_update = _update_gateway_posix._restart_launchd_gateway_after_update
_restart_macos_launchd_gateways = _update_gateway_posix._restart_macos_launchd_gateways
_surviving_gateway_pids_after_failed_restart = _update_reconciliation._surviving_gateway_pids_after_failed_restart
_FRESH_RESTART_SUPERVISORS = _update_reconciliation._FRESH_RESTART_SUPERVISORS
_gateway_service_matches_profile = _update_reconciliation._gateway_service_matches_profile
_gateway_recovery_partition = _update_reconciliation._gateway_recovery_partition
_gateway_restart_recovery_profiles = _update_reconciliation._gateway_restart_recovery_profiles
_warn_gateway_restart_phase_aborted = _update_reconciliation._warn_gateway_restart_phase_aborted
_refresh_windows_gateway_launchers = _update_gateway_windows._refresh_windows_gateway_launchers
_refresh_bootstrap_cache_scripts = _update_gateway_windows._refresh_bootstrap_cache_scripts
_resume_windows_gateways_after_update = _update_gateway_windows._resume_windows_gateways_after_update
_discard_lockfile_churn = _update_source._discard_lockfile_churn
_normalize_managed_eol = _update_source._normalize_managed_eol
_desktop_app_present = _update_desktop._desktop_app_present
_rebuild_desktop_after_update = _update_desktop._rebuild_desktop_after_update
_cmd_update_impl = _update_orchestrator._cmd_update_impl
_restart_phase_failure_is_incomplete = _update_fleet_restart._restart_phase_failure_is_incomplete
_fleet_probe_expected_runtimes = _update_fleet_restart._fleet_probe_expected_runtimes
_print_items = _update_fleet_restart._print_items
_wait_for_service_active = _update_fleet_restart._wait_for_service_active
_service_restart_sec = _update_fleet_restart._service_restart_sec

_validate_python_files_syntax = _update_runtime_refresh._validate_python_files_syntax
_critical_module_import_failures = _update_runtime_refresh._critical_module_import_failures
_post_update_sqlite_runtime_status = _update_runtime_refresh._post_update_sqlite_runtime_status
_print_verified_update_completion = _update_runtime_refresh._print_verified_update_completion
_verify_and_restore_one_state_db = _update_zip._verify_and_restore_one_state_db
_verify_and_restore_state_dbs_post_update = _update_zip._verify_and_restore_state_dbs_post_update
_ORPHAN_RESCUE_REFS_TO_KEEP = _update_source._ORPHAN_RESCUE_REFS_TO_KEEP
_ORPHAN_RESCUE_REF_MAX_AGE_DAYS = _update_source._ORPHAN_RESCUE_REF_MAX_AGE_DAYS
_prune_orphan_rescue_refs = _update_source._prune_orphan_rescue_refs
_AUTOSTASH_NAME_PREFIX = _update_source._AUTOSTASH_NAME_PREFIX
_AUTOSTASH_WARN_AGE_DAYS = _update_source._AUTOSTASH_WARN_AGE_DAYS
_warn_orphaned_update_autostashes = _update_source._warn_orphaned_update_autostashes
_git_untracked_paths = _update_source._git_untracked_paths
_restored_python_paths = _update_source._restored_python_paths
_reject_unsafe_stash_restore = _update_source._reject_unsafe_stash_restore
_git_is_trampoline = _update_source._git_is_trampoline
_portable_git_candidates = _update_source._portable_git_candidates
_locate_real_git = _update_source._locate_real_git
_ensure_non_trampoline_git = _update_source._ensure_non_trampoline_git
_refuse_gateway_ancestor_tree_kill = _update_process_guard._refuse_gateway_ancestor_tree_kill
_path_uid = _update_process_guard._path_uid
_venv_foreign_owned_paths = _update_process_guard._venv_foreign_owned_paths
_refuse_update_if_venv_foreign_owned = _update_process_guard._refuse_update_if_venv_foreign_owned
_drain_or_signal_gateway_for_update = _update_fleet_restart._drain_or_signal_gateway_for_update

_COMPAT_CONSUMERS = {
    'OFFICIAL_REPO_URL': (_update_source,),
    'OFFICIAL_REPO_URLS': (_update_source,),
    'Optional': (_update_backup, _update_process_guard, _update_source, _update_zip),
    'Path': (_update_dependencies, _update_desktop, _update_gateway_windows, _update_notices, _update_process_guard, _update_reconciliation, _update_runtime_refresh, _update_source, _update_zip),
    'SKIP_UPSTREAM_PROMPT_FILE': (_update_source,),
    '_AUTOSTASH_NAME_PREFIX': (_update_source,),
    '_AUTOSTASH_WARN_AGE_DAYS': (_update_source,),
    '_FLEET_RESTART_PENDING_NAME': (_update_reconciliation,),
    '_FRESH_RESTART_SUPERVISORS': (_update_reconciliation,),
    '_HOLDER_VALUE_FLAGS_FALLBACK': (_update_process_guard,),
    '_INSTALL_DEFINING_FILES': (_update_runtime_refresh,),
    '_LAST_SIBLING_SNAPSHOTS': (_update_runtime_refresh,),
    '_ORPHAN_RESCUE_REFS_TO_KEEP': (_update_source,),
    '_ORPHAN_RESCUE_REF_MAX_AGE_DAYS': (_update_orchestrator, _update_source),
    '_PRE_UPDATE_SNAPSHOT_KEEP': (_update_backup,),
    '_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE': (_update_backup,),
    '_SELF_LOCKING_NATIVE_MODULES': (_update_process_guard,),
    '_STALE_PURGE_PREFIXES': (_update_runtime_refresh,),
    '_STALE_PURGE_PROTECTED': (_update_runtime_refresh,),
    '_UPDATE_CRITICAL_FILES': (_update_runtime_refresh,),
    '_UPDATE_CRITICAL_MODULES': (_update_runtime_refresh,),
    '_UPDATE_RUNTIME_RELOAD_MODULES': (_update_runtime_refresh,),
    '_ZIP_PRESERVED_TOP_LEVEL': (_update_zip,),
    '_ZIP_STAGING_ARTIFACT_SUFFIXES': (_update_zip,),
    '_abort_dependency_sync_if_self_locked': (_update_process_guard,),
    '_abort_recovery_is_complete': (_update_fleet_restart,),
    '_abort_zip_update_if_dirty_tree': (_update_zip,),
    '_add_upstream_remote': (_update_source,),
    '_apply_pending_fleet_restart_catchup': (_update_orchestrator, _update_reconciliation),
    '_assess_parked_branch_switch': (_update_zip,),
    '_atomic_replace_dir': (_update_zip,),
    '_branch_head_label': (_update_zip,),
    '_branch_head_suffix': (_update_orchestrator, _update_zip),
    '_called_process_error_cmd_parts': (_update_zip,),
    '_called_process_error_is_git': (_update_zip,),
    '_called_process_error_is_python_dep_install': (_update_orchestrator, _update_zip),
    '_capture_active_lazy_features': (_update_dependencies,),
    '_capture_active_tool_dependencies': (_update_dependencies,),
    '_capture_head_sha': (_update_orchestrator, _update_reconciliation, _update_runtime_refresh),
    '_check_and_apply_config_migration': (_update_dependencies, _update_orchestrator, _update_runtime_refresh),
    '_classify_concurrent_instance': (_update_process_guard,),
    '_classify_fetch_failure': (_update_dependencies,),
    '_clear_fleet_restart_pending_marker': (_update_orchestrator, _update_reconciliation),
    '_clear_stale_sqlite_sidecars': (_update_zip,),
    '_cmd_update_check': (_update_source,),
    '_cmd_update_impl': (_update_orchestrator,),
    '_cold_start_windows_gateway_after_update': (_update_gateway_windows,),
    '_commit_staged_replacements': (_update_zip,),
    '_count_commits_between': (_update_orchestrator, _update_source),
    '_critical_module_import_failures': (_update_runtime_refresh, _update_source),
    '_current_checkout_sha': (_update_reconciliation,),
    '_defer_update_for_self_lock': (_update_process_guard,),
    '_dependency_sync_would_rewrite': (_update_process_guard,),
    '_desktop_app_present': (_update_desktop, _update_orchestrator),
    '_desktop_owns_gateway_lifecycle': (_update_gateway_windows, _update_process_guard),
    '_detect_self_loaded_native_modules': (_update_process_guard,),
    '_detect_venv_python_processes': (_update_process_guard,),
    '_discard_lockfile_churn': (_update_orchestrator, _update_source),
    '_discard_staged': (_update_zip,),
    '_discard_stashed_changes': (_update_source,),
    '_drain_or_signal_gateway_for_update': (_update_fleet_restart,),
    '_editable_install_is_current': (_update_orchestrator, _update_runtime_refresh),
    '_ensure_acp_launcher': (_update_dependencies, _update_orchestrator),
    '_ensure_fhs_path_guard': (_update_dependencies, _update_orchestrator),
    '_ensure_non_trampoline_git': (_update_orchestrator, _update_source),
    '_ensure_uv_for_termux': (_update_dependencies, _update_orchestrator, _update_zip),
    '_filter_non_gateway_concurrent_instances': (_update_process_guard,),
    '_finish_dashboard_update_cleanup': (_update_notices, _update_orchestrator, _update_zip),
    '_fleet_probe_expected_runtimes': (_update_fleet_restart,),
    '_fleet_restart_pending_marker_path': (_update_reconciliation,),
    '_for_each_systemd_gateway_unit': (_update_fleet_restart, _update_gateway_posix, _update_orchestrator, _update_reconciliation),
    '_format_concurrent_instances_message': (_update_orchestrator, _update_process_guard),
    '_format_time_ago': (_update_notices,),
    '_format_update_failure_stage': (_update_orchestrator, _update_zip),
    '_format_venv_python_holders_message': (_update_orchestrator, _update_process_guard),
    '_gateway_prompt': (_update_notices, _update_orchestrator, _update_runtime_refresh),
    '_gateway_recovery_partition': (_update_reconciliation,),
    '_gateway_restart_recovery_profiles': (_update_reconciliation,),
    '_gateway_service_matches_profile': (_update_fleet_restart, _update_orchestrator, _update_reconciliation),
    '_get_origin_url': (_update_source,),
    '_git_is_trampoline': (_update_source,),
    '_git_untracked_paths': (_update_source,),
    '_handoff_reapable_backend_pids': (_update_process_guard,),
    '_has_upstream_remote': (_update_source,),
    '_hermes_holder_subcommand': (_update_process_guard,),
    '_holder_value_flags': (_update_process_guard,),
    '_holder_value_flags_cache': (_update_process_guard,),
    '_install_psutil_android_compat': (_update_dependencies, _update_orchestrator),
    '_invalidate_update_cache': (_update_orchestrator, _update_reconciliation),
    '_is_android_python': (_update_dependencies, _update_orchestrator),
    '_is_fork': (_update_orchestrator, _update_source),
    '_is_zip_preserved_entry_status_line': (_update_zip,),
    '_is_zip_staging_artifact_status_line': (_update_zip,),
    '_ledger_manual_serve_holders': (_update_process_guard,),
    '_ledger_reapable_backend_pids': (_update_process_guard,),
    '_leftover_pausable_gateway_pids': (_update_process_guard,),
    '_locate_real_git': (_update_source,),
    '_log_only_write': (_update_dependencies,),
    '_looks_like_desktop_control_plane': (_update_process_guard,),
    '_m': (_update_dependencies, _update_desktop, _update_fleet_restart, _update_gateway_windows, _update_notices, _update_orchestrator, _update_process_guard, _update_reconciliation, _update_runtime_refresh, _update_source, _update_zip),
    '_mark_skip_upstream_prompt': (_update_source,),
    '_migrate_sibling_profile_configs': (_update_runtime_refresh,),
    '_normalize_managed_eol': (_update_orchestrator, _update_source),
    '_no_prompt_git_kwargs': (_update_orchestrator, _update_source),
    '_npm_bin_exists': (_update_notices,),
    '_npm_lockfile_changed': (_update_dependencies,),
    '_npm_manifest_paths': (_update_dependencies,),
    '_npm_manifests_digest': (_update_dependencies,),
    '_orphaned_desktop_backend_pids': (_update_process_guard,),
    '_park_stashed_changes': (_update_source,),
    '_path_uid': (_update_process_guard,),
    '_pause_windows_gateways_for_update': (_update_gateway_windows,),
    '_pending_fleet_restart_needed': (_update_reconciliation,),
    '_portable_git_candidates': (_update_source,),
    '_post_update_sqlite_runtime_status': (_update_runtime_refresh, _update_zip),
    '_print_called_process_error_tail': (_update_orchestrator, _update_zip),
    '_print_curator_first_run_notice': (_update_notices, _update_orchestrator, _update_zip),
    '_print_curator_recent_run_notice': (_update_notices, _update_orchestrator, _update_zip),
    '_print_fetch_failure': (_update_dependencies, _update_orchestrator, _update_source),
    '_print_fts_optimize_available_notice': (_update_notices, _update_orchestrator),
    '_print_items': (_update_fleet_restart, _update_runtime_refresh),
    '_print_parked_branch_kept_notice': (_update_zip,),
    '_print_parked_branch_skip_warning': (_update_zip,),
    '_print_stash_cleanup_guidance': (_update_source,),
    '_print_update_completion': (_update_orchestrator, _update_runtime_refresh, _update_zip),
    '_print_update_summary': (_update_orchestrator, _update_zip),
    '_print_verified_update_completion': (_update_orchestrator, _update_runtime_refresh),
    '_prune_orphan_rescue_refs': (_update_orchestrator, _update_source),
    '_purge_stale_hermes_modules': (_update_runtime_refresh,),
    '_read_project_version': (_update_orchestrator, _update_zip),
    '_rebuild_desktop_after_update': (_update_dependencies, _update_desktop, _update_orchestrator, _update_zip),
    '_receipt_looks_unfinished': (_update_reconciliation,),
    '_receipt_reports_stale_runtime': (_update_reconciliation,),
    '_record_npm_lockfile_hash': (_update_dependencies,),
    '_recover_gateway_restart_after_abort': (_update_fleet_restart,),
    '_refresh_active_lazy_features': (_update_dependencies,),
    '_refresh_active_memory_provider_dependencies': (_update_dependencies,),
    '_refresh_bootstrap_cache_scripts': (_update_gateway_windows,),
    '_refresh_windows_gateway_launchers': (_update_gateway_windows,),
    '_refuse_gateway_ancestor_tree_kill': (_update_orchestrator, _update_process_guard),
    '_refuse_update_for_contended_shims': (_update_orchestrator, _update_zip),
    '_refuse_update_if_venv_foreign_owned': (_update_orchestrator, _update_process_guard),
    '_reject_unsafe_stash_restore': (_update_source,),
    '_relaunch_stopped_serves': (_update_process_guard,),
    '_reload_config_modules': (_update_runtime_refresh,),
    '_reload_process_scan_modules': (_update_notices,),
    '_reload_updated_runtime_modules': (_update_runtime_refresh,),
    '_repair_node_deps_on_current_checkout': (_update_dependencies, _update_orchestrator),
    '_resolve_pre_update_backup_mode': (_update_backup,),
    '_resolve_stash_selector': (_update_source,),
    '_restart_launchd_gateway_after_update': (_update_gateway_posix,),
    '_restart_macos_launchd_gateways': (_update_fleet_restart, _update_gateway_posix, _update_orchestrator, _update_reconciliation),
    '_restart_phase_failure_is_incomplete': (_update_fleet_restart,),
    '_restart_systemd_gateway_units_best_effort': (_update_reconciliation,),
    '_restore_active_tool_dependencies': (_update_dependencies,),
    '_restore_stashed_changes': (_update_source,),
    '_restore_state_db_from_snapshot': (_update_orchestrator, _update_zip),
    '_restore_windows_gateway_service': (_update_gateway_windows,),
    '_restored_python_paths': (_update_source,),
    '_resume_windows_gateways_after_update': (_update_gateway_windows,),
    '_run_config_check_fresh': (_update_runtime_refresh,),
    '_run_logged_subprocess': (_update_dependencies,),
    '_run_migrate_config_fresh': (_update_runtime_refresh,),
    '_run_pending_fleet_restart': (_update_reconciliation,),
    '_run_pre_update_backup': (_update_backup,),
    '_serve_relaunch_commands': (_update_process_guard,),
    '_service_restart_sec': (_update_fleet_restart,),
    '_service_unit_supports_graceful_sigusr1_restart': (_update_fleet_restart, _update_gateway_posix, _update_orchestrator),
    '_shim_quarantine_error_type': (_update_orchestrator, _update_zip),
    '_should_skip_upstream_prompt': (_update_source,),
    '_should_zip_fallback_on_update_error': (_update_orchestrator, _update_zip),
    '_stage_replacement': (_update_zip,),
    '_start_windows_gateway_service': (_update_gateway_windows,),
    '_stash_apply_failed_only_on_existing_untracked': (_update_source,),
    '_stash_local_changes_if_needed': (_update_source,),
    '_stop_process_trees': (_update_process_guard,),
    '_stop_windows_gateway_service': (_update_gateway_windows,),
    '_surviving_gateway_pids_after_failed_restart': (_update_fleet_restart, _update_orchestrator, _update_reconciliation),
    '_surviving_pre_update_serve_runtimes': (_update_fleet_restart, _update_orchestrator),
    '_sync_fork_with_upstream': (_update_source,),
    '_sync_with_upstream_if_needed': (_update_source,),
    '_time': (_update_backup, _update_fleet_restart, _update_gateway_windows, _update_notices, _update_orchestrator, _update_process_guard, _update_reconciliation),
    '_update_complete_message': (_update_zip,),
    '_update_node_dependencies': (_update_dependencies, _update_orchestrator, _update_zip),
    '_update_via_zip': (_update_orchestrator, _update_zip),
    '_upgrade_pip_before_lazy_refresh': (_update_dependencies,),
    '_validate_critical_files_syntax': (_update_orchestrator, _update_runtime_refresh),
    '_validate_critical_modules_import': (_update_orchestrator, _update_runtime_refresh, _update_zip),
    '_validate_python_files_syntax': (_update_runtime_refresh, _update_source),
    '_venv_core_imports_healthy': (_update_orchestrator, _update_process_guard),
    '_venv_foreign_owned_paths': (_update_process_guard,),
    '_venv_launcher_ancestors': (_update_process_guard,),
    '_verify_and_restore_one_state_db': (_update_zip,),
    '_verify_and_restore_state_dbs_post_update': (_update_orchestrator, _update_zip),
    '_wait_for_service_active': (_update_fleet_restart,),
    '_wait_for_windows_update_gateway_exit': (_update_process_guard,),
    '_warn_gateway_restart_phase_aborted': (_update_fleet_restart, _update_orchestrator, _update_reconciliation),
    '_warn_incomplete_gateway_fleet_restart': (_update_fleet_restart, _update_gateway_posix, _update_orchestrator, _update_reconciliation),
    '_warn_orphaned_update_autostashes': (_update_source,),
    '_warn_pending_fleet_restart': (_update_reconciliation,),
    '_warn_pending_fleet_restart_on_startup': (_update_reconciliation,),
    '_warn_stale_serve_runtimes': (_update_fleet_restart, _update_orchestrator),
    '_web_build_toolchain_ready': (_update_dependencies, _update_notices),
    '_web_toolchain_roots': (_update_dependencies, _update_notices),
    '_write_fleet_restart_pending_marker': (_update_orchestrator, _update_reconciliation),
    '_write_gateway_update_exit_code': (_update_fleet_restart, _update_orchestrator, _update_zip),
    '_write_lazy_refresh_incomplete_marker': (_update_orchestrator, _update_reconciliation),
    '_write_marker_file': (_update_reconciliation,),
    '_write_update_incomplete_marker': (_update_orchestrator, _update_reconciliation, _update_zip),
    '_write_update_planned_stop_marker': (_update_gateway_windows, _update_process_guard),
    '_zip_overlay_block_reason': (_update_zip,),
    'datetime': (_update_notices, _update_process_guard, _update_source),
    'get_default_hermes_root': (_update_source,),
    'get_hermes_home': (_update_backup, _update_fleet_restart, _update_notices, _update_orchestrator, _update_reconciliation, _update_source, _update_zip),
    'hashlib': (_update_dependencies,),
    'json': (_update_dependencies, _update_reconciliation),
    'logger': (_update_dependencies, _update_fleet_restart, _update_gateway_windows, _update_notices, _update_orchestrator, _update_process_guard, _update_reconciliation, _update_runtime_refresh, _update_source, _update_zip),
    'os': (_update_dependencies, _update_fleet_restart, _update_gateway_windows, _update_orchestrator, _update_process_guard, _update_reconciliation, _update_source, _update_zip),
    'shlex': (_update_process_guard, _update_zip),
    'shutil': (_update_dependencies, _update_orchestrator, _update_reconciliation, _update_zip),
    'subprocess': (_update_dependencies, _update_fleet_restart, _update_gateway_posix, _update_gateway_windows, _update_orchestrator, _update_process_guard, _update_reconciliation, _update_runtime_refresh, _update_source, _update_zip),
    'sys': (_update_desktop, _update_notices, _update_orchestrator, _update_process_guard, _update_reconciliation, _update_runtime_refresh, _update_source, _update_zip),
    'venv_python_path': (_update_orchestrator, _update_process_guard, _update_runtime_refresh),
}

for _name, _modules in _COMPAT_CONSUMERS.items():
    _value = globals()[_name]
    for _module in _modules:
        setattr(_module, _name, _value)


class _UpdateFacadeModule(type(sys)):
    """Propagate patches on the historical module to extracted consumers."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _COMPAT_CONSUMERS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _UpdateFacadeModule
