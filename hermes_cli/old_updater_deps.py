"""Retired dependency hooks requested by already-running historical updaters."""
from typing import NoReturn

from hermes_cli._old_updater import stop_historical_updater


def _capture_active_lazy_features() -> NoReturn:
    # PM owns the feature ledger; an old updater must not start a second install path.
    stop_historical_updater("hermes_cli.old_updater_deps._capture_active_lazy_features")


def _refresh_active_lazy_features(*args, **kwargs) -> NoReturn:
    # Historical signatures varied; none may mutate the new environment.
    stop_historical_updater("hermes_cli.old_updater_deps._refresh_active_lazy_features", ImportError)


def _refresh_active_memory_provider_dependencies() -> NoReturn:
    # Plugin dependencies participate in PM's union, not a last-writer reinstall.
    stop_historical_updater("hermes_cli.old_updater_deps._refresh_active_memory_provider_dependencies", ImportError)


def _npm_lockfile_changed(hermes_root) -> NoReturn:
    # Do not authorize an old npm install through a fabricated currency result.
    stop_historical_updater("hermes_cli.old_updater_deps._npm_lockfile_changed")


def _update_node_dependencies() -> NoReturn:
    # Source builds use the shared Node dependency provider.
    stop_historical_updater("hermes_cli.old_updater_deps._update_node_dependencies")


def _rebuild_desktop_after_update(desktop_dir, *, had_desktop_app_before_update) -> NoReturn:
    # Old callers must not report completion after skipping the build.
    stop_historical_updater("hermes_cli.old_updater_deps._rebuild_desktop_after_update")


def _write_update_incomplete_marker() -> NoReturn:
    # PM's successful facts determine completion; hermes_cli.main exports this lazily.
    stop_historical_updater("hermes_cli.old_updater_deps._write_update_incomplete_marker")


def _path_uid(path) -> NoReturn:
    # PM never mutates the old venv; do not re-enable its ownership preflight.
    stop_historical_updater("hermes_cli.old_updater_deps._path_uid")


# Historical main's lazy exports can request these after swapping the checkout.
# No holder classification or tree kill is needed to prepare a new PM generation.
def _leftover_pausable_gateway_pids(matches: list[tuple[int, str, str]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._leftover_pausable_gateway_pids")


def _ledger_manual_serve_holders(matches: list[tuple[int, str, str]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._ledger_manual_serve_holders")


def _relaunch_stopped_serves(token: dict) -> None:
    from hermes_cli._old_updater import relaunch_stopped_serves
    relaunch_stopped_serves(token)


def _orphaned_desktop_backend_pids(matches: list[tuple[int, str, str]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._orphaned_desktop_backend_pids")


def _ledger_reapable_backend_pids(matches: list[tuple[int, str, str]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._ledger_reapable_backend_pids")


def _handoff_reapable_backend_pids(matches: list[tuple[int, str, str]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._handoff_reapable_backend_pids")


def _stop_process_trees(pids: list[int] | list[tuple[int, int]]) -> NoReturn:
    stop_historical_updater("hermes_cli.old_updater_deps._stop_process_trees")
