"""Unattended plugin updates shared by every surface: the opt-in ``plugins.auto_apply`` switch, the
one-click "update all" (Desktop, dashboard, ``/plugins update``), the one-updater-per-plugin lock,
and the notice ``hermes update`` prints when it moved catalog pins.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade is imported late, never at module level).
"""

from __future__ import annotations

import logging
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# A second updater of the same plugin is refused rather than queued: the first one's result already
# answers the second's question, and a queued pull would run against a tree the first just replaced.
_LOCK_WAIT_SECONDS = 1.0
_lock_holders: dict[str, threading.local] = {}
_lock_holders_guard = threading.Lock()


def _pc():
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _holder(key: str) -> threading.local:
    with _lock_holders_guard:
        return _lock_holders.setdefault(key, threading.local())


@contextmanager
def plugin_update_lock(target: Path) -> Iterator[None]:
    """Hold the one-updater lock for the plugin at *target*, across threads and processes (the
    gateway's auto-apply tick, a terminal ``hermes plugins update``, Desktop and the dashboard).
    A busy lock raises ``PluginOperationError`` after a short wait, so every caller reports it like
    any other refusal (nothing changed); reentrant per thread."""
    from hermes_cli.auth import _file_lock
    from hermes_constants import get_hermes_home

    lock = get_hermes_home() / "cache" / "plugin-update-locks" / f"{target.name}.lock"
    stack = ExitStack()
    try:
        stack.enter_context(_file_lock(lock, _holder(str(lock)), _LOCK_WAIT_SECONDS, "plugin update busy"))
    except TimeoutError as exc:
        raise _pc().PluginOperationError(
            f"Plugin '{target.name}' is already being updated by another Hermes process; "
            "nothing was changed. Try again when that update finishes.") from exc
    with stack:
        yield


# ── installs the user held ────────────────────────────────────────────────────────────────────

HELD_WITH_REF = "held with --ref (installed off the catalog pin); `hermes plugins update {name}` moves it"


def held_with_ref(record: Any) -> bool:
    """A catalog install the user put on a commit of their choosing (``install <name> --ref <sha>``:
    ``pinned`` with an empty ``catalog.pin``). Only an explicit ``hermes plugins update <name>`` moves
    it: unattended paths would otherwise silently undo a deliberate rollback."""
    block = record.get("catalog") if isinstance(record, dict) else None
    return isinstance(block, dict) and record.get("pinned") is True and not block.get("pin")


# ── the auto-update switch ────────────────────────────────────────────────────────────────────

def auto_update_state() -> dict[str, Any]:
    """``{"auto_apply", "check_hours"}`` for the active profile."""
    from hermes_cli.plugins_cadence import auto_apply_enabled, check_interval_hours
    return {"auto_apply": auto_apply_enabled(), "check_hours": check_interval_hours()}


def set_auto_update(enabled: bool) -> dict[str, Any]:
    """Write ``plugins.auto_apply`` for the active profile and return the new state. Turning it on also
    clears the check marker, so the gateway's next housekeeping tick checks and applies instead of
    waiting out the rest of the interval."""
    from hermes_cli.config import get_config_path, is_managed
    from hermes_cli.config_backend import write_config_key
    from hermes_constants import get_hermes_home

    if is_managed():
        raise _pc().PluginOperationError(
            "This Hermes install's configuration is managed by a package manager; set plugins.auto_apply there.")
    write_config_key(get_config_path(), "plugins.auto_apply", bool(enabled))
    if enabled:
        _clear_check_marker(get_hermes_home())
    return auto_update_state()


def _clear_check_marker(home: Path) -> None:
    from hermes_cli.plugins_cadence import _MARKERS_DIR
    (home / _MARKERS_DIR / "last-run").unlink(missing_ok=True)


# ── update all, unattended ────────────────────────────────────────────────────────────────────

def _apply_one(name: str) -> dict[str, str]:
    """One plan row through the dashboard's non-interactive update. A widened catalog pin or a
    caution scan is NOT accepted on the user's behalf: the row comes back ``needs_review``."""
    from hermes_cli.plugins_cmd_update import dashboard_update_user_plugin
    try:
        res = dashboard_update_user_plugin(name)
    except Exception as exc:  # per-plugin batch boundary: one crash must not stop the rest
        logger.warning("update of plugin %s failed", name, exc_info=True)
        return {"name": name, "status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
    if res.get("consent_required") or res.get("scan_blocked"):
        return {"name": name, "status": "needs_review", "detail": str(res.get("error") or "")}
    if not res.get("ok"):
        return {"name": name, "status": "failed", "detail": str(res.get("error") or "")}
    return {"name": str(res.get("name") or name), "status": "current" if res.get("unchanged") else "updated",
            "detail": ""}


def update_all_unattended() -> dict[str, Any]:
    """Check every installed plugin, then update each one ``hermes plugins update`` would move (same
    plan: catalog re-pins and tracked Git pulls; pinned, held and pip rows are skipped with why).

    Returns ``{"results": [{name, status, detail}], "skipped": [{name, reason}], "current": n}``;
    status is ``updated`` / ``current`` / ``failed`` / ``needs_review`` (nothing changed; the
    plugin's own Update action walks the consent or scan review)."""
    from hermes_cli.plugins_cadence import record_check_results, snapshot_installs
    from hermes_cli.plugins_cmd_update_all import plan_update_all
    from hermes_cli.plugins_updates import run_checks

    plugins_dir = _pc()._plugins_dir()
    checked = snapshot_installs(plugins_dir)
    results = run_checks(plugins_dir)
    record_check_results(results, checked)
    plan = plan_update_all(results)
    held = [row for row in plan.movable if held_with_ref(checked.get(row.name))]
    return {"results": [_apply_one(row.name) for row in plan.movable if row not in held],
            "skipped": [{"name": row.name, "reason": reason} for row, reason in plan.skipped]
                       + [{"name": row.name, "reason": HELD_WITH_REF.format(name=row.name)} for row in held],
            "current": plan.current}


_STATUS_MARKS = {"updated": "✓ updated", "current": "= already current", "failed": "✗ failed",
                 "needs_review": "! needs review"}


def summary_lines(outcome: dict[str, Any]) -> list[str]:
    """Plain-text lines for an :func:`update_all_unattended` result (``/plugins update``)."""
    lines = [f"  {_STATUS_MARKS[r['status']]}  {r['name']}" + (f"  ({r['detail']})" if r["detail"] else "")
             for r in outcome["results"]]
    lines += [f"  - skipped {s['name']}: {s['reason']}" for s in outcome["skipped"]]
    if not outcome["results"]:
        lines.append("  No plugin updates to apply.")
    if any(r["status"] == "needs_review" for r in outcome["results"]):
        lines.append("  Review those with `hermes plugins update <name>` in a terminal.")
    if any(r["status"] == "updated" for r in outcome["results"]):
        lines.append("  Updated plugin code loads in new sessions.")
    return lines


def plugins_slash_action(parts: list[str]) -> list[str]:
    """``/plugins update`` (update every plugin with an update, unattended) and ``/plugins auto-update
    [on|off]`` for the classic CLI and the TUI (which runs the same handler in its slash worker)."""
    if parts[0] == "update":
        return ["Plugin updates:", *summary_lines(update_all_unattended())]
    arg = parts[1].lower() if len(parts) > 1 else ""
    if arg not in ("", "on", "off"):
        return ["Usage: /plugins auto-update [on|off]"]
    try:
        state = auto_update_state() if not arg else set_auto_update(arg == "on")
    except _pc().PluginOperationError as exc:
        return [f"  {exc}"]
    word = "on" if state["auto_apply"] else "off"
    return [f"Plugin auto-update: {word} (gateway check every {state['check_hours']:g}h; "
            "updates load in new sessions)"]


# ── after `hermes update` ─────────────────────────────────────────────────────────────────────

def catalog_pins_behind(home: Path) -> list[str]:
    """Catalog installs under *home* whose installed pin is not the in-tree catalog's pin. Offline:
    reads the install records and the catalog shipped with this checkout. An install held with
    ``--ref`` is not behind: the user chose that commit."""
    from hermes_cli.plugin_catalog import get_catalog_entry
    from hermes_cli.plugins_cmd_catalog import at_catalog_pin
    from hermes_cli.plugins_provenance import read_sidecar_rows

    behind = []
    for name, row in read_sidecar_rows(home / "plugins").items():
        block = row.get("catalog") if isinstance(row, dict) else None
        if not isinstance(block, dict) or not block.get("name") or held_with_ref(row):
            continue
        entry = get_catalog_entry(str(block["name"]))
        if entry is not None and not at_catalog_pin(block, entry.sha):
            behind.append(str(name))
    return sorted(behind)


def report_catalog_updates_after_hermes_update(home: Path, sibling_homes: list[tuple[str, Path]]) -> None:
    """An update can move catalog pins; installed plugins stay on their old pin until updated. Name
    them per profile, and clear that profile's check marker so its gateway re-checks on the next tick
    (which applies them when that profile opted into ``plugins.auto_apply``)."""
    for label, profile_home in [("this profile", home), *sibling_homes]:
        behind = catalog_pins_behind(profile_home)
        if not behind:
            continue
        _clear_check_marker(profile_home)
        print(f"  → {len(behind)} plugin update(s) from the new catalog in {label}: {', '.join(behind)}")
        print("    apply: hermes plugins update   (or turn on: hermes plugins auto-update on)")
