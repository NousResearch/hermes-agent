"""The plugin update-check cadence: read-only, receipt-surfaced.

Runs at gateway start + the periodic tick when
``plugins.auto_update_check_hours`` (default 24, 0 disables) says it's
due, writes a plugin-check receipt (pm.receipt, kind 'plugin-check'),
and logs ONE actionable line when updates exist. Applying updates
stays explicit — ``plugins.auto_apply: true`` (default false) opts into
unattended apply for git-row plugins ONLY, scan-gated like cmd_update.

Pure, injectable clock/check/update seams for hermetic tests.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

_MARKERS_DIR = "plugin-update-checks"
_DEFAULT_INTERVAL_HOURS = 24


def _marker_path() -> Path:
    """The last-run marker, resolved per call (tests monkeypatch
    get_hermes_home). Derivation only — NO mkdir: a due-check or any
    read path must not create user state."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _MARKERS_DIR / "last-run"


def _markers_dir() -> Path:
    """The markers dir, created on demand — WRITE paths only."""
    d = _marker_path().parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_interval_hours(config_get: Callable = None) -> float:
    """plugins.auto_update_check_hours; 0 disables; default 24."""
    if config_get is None:
        config_get = _default_config_get
    try:
        value = config_get("plugins", "auto_update_check_hours")
    except Exception:
        return _DEFAULT_INTERVAL_HOURS
    if value is None:
        return _DEFAULT_INTERVAL_HOURS
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_INTERVAL_HOURS
    return max(0.0, hours)


def auto_apply_enabled(config_get: Callable = None) -> bool:
    if config_get is None:
        config_get = _default_config_get
    try:
        return bool(config_get("plugins", "auto_apply"))
    except Exception:
        return False


def _default_config_get(section: str, key: str):
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        return cfg_get(load_config_readonly(), section, key, default=None)
    except Exception:
        return None


def check_due(now: Optional[float] = None, interval_hours: Optional[float] = None,
              config_get: Callable = None) -> bool:
    """The clock gate: last-run marker vs the interval."""
    if interval_hours is None:
        interval_hours = check_interval_hours(config_get)
    if interval_hours <= 0:
        return False
    now = time.time() if now is None else now
    marker = _marker_path()
    try:
        last = marker.stat().st_mtime
    except OSError:
        return True
    return (now - last) >= interval_hours * 3600


#: Minimal singleflight for one home: two gateway ticks overlapping in
#: this process (boot pass + periodic tick) must not double-fetch or
#: double-write receipts. A plain non-blocking lock — not a manager.
_tick_lock = threading.Lock()


def run_scheduled_check(
    *,
    run_checks_fn: Callable[..., list],
    plugins_dir: Path,
    apply_updates_fn: Optional[Callable[[str], None]] = None,
    log=None,
    config_get: Callable = None,
    now: Optional[float] = None,
) -> Optional[list]:
    """One cadence tick: gate → check → receipt → (opt-in) apply.

    Returns the check results, or None when not due / disabled / another
    tick is already in flight. NEVER raises — a cadence failure is
    logged, never fatal.
    """
    import logging

    if log is None:
        log = logging.getLogger(__name__)
    if not check_due(now=now, config_get=config_get):
        return None
    if not _tick_lock.acquire(blocking=False):
        log.debug("plugin update check already in flight — skipping tick")
        return None
    try:
        return _run_check_locked(
            run_checks_fn=run_checks_fn,
            plugins_dir=plugins_dir,
            apply_updates_fn=apply_updates_fn,
            log=log,
            config_get=config_get,
        )
    finally:
        _tick_lock.release()


def _run_check_locked(
    *,
    run_checks_fn: Callable[..., list],
    plugins_dir: Path,
    apply_updates_fn: Optional[Callable[[str], None]],
    log,
    config_get: Callable = None,
) -> list:
    check_ok = True
    warning = ""
    try:
        results = run_checks_fn(plugins_dir)
    except Exception:
        warning = "plugin update check failed"
        log.warning(warning, exc_info=True)
        results = []
        check_ok = False

    try:
        from pm import receipt
        token = receipt.begin("plugin-check")
        receipt.record_plugin_checks(results)
        if warning:
            receipt.record_warning(warning)
        updates = [r for r in results if getattr(r, "update_available", None) is True]
        receipt.finalize("failed" if not check_ok else "updates-available" if updates else "ok",
                         exit_code=0 if check_ok else 1, token=token)
    except Exception:
        log.debug("plugin-check receipt write failed", exc_info=True)

    # ONE actionable line (a log, not a system-prompt mutation — cache safe)
    updates = [r for r in results if getattr(r, "update_available", None) is True]
    needs_fixing = [r for r in results if getattr(r, "needs_fixing", None)]
    if updates:
        names = ", ".join(r.name for r in updates)
        log.info(
            "plugin updates available: %s — run `hermes plugins check-updates` "
            "and `hermes plugins update <name>`",
            names,
        )
    if needs_fixing:
        names = ", ".join(r.name for r in needs_fixing)
        log.warning(
            "plugin update_url mismatches need attention: %s — run "
            "`hermes plugins trust-update-url <name>` after review",
            names,
        )

    # Opt-in unattended apply: git rows ONLY, scan-gated by the update
    # path itself. Pinned/manual/drift/pip are never auto-applied
    # (their update_available is False/None or their class excludes it).
    if apply_updates_fn and updates and auto_apply_enabled(config_get):
        for r in updates:
            if getattr(r, "klass", "") == "git":
                try:
                    apply_updates_fn(r.name)
                except (Exception, SystemExit):
                    log.warning("auto-apply of %s failed", r.name, exc_info=True)

    # stamp the marker AFTER a completed run (even a failed one — a
    # failing check retrying every tick would hammer the network)
    try:
        marker = _markers_dir() / "last-run"
        marker.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass
    return results


# ---------------------------------------------------------------------------
# the production caller: real seams for the gateway boot/housekeeping tick
# ---------------------------------------------------------------------------

def maybe_run_gateway_check(
    *,
    run_checks_fn: Optional[Callable[..., list]] = None,
    apply_updates_fn: Optional[Callable[[str], None]] = None,
    plugins_dir: Optional[Path] = None,
    log=None,
    now: Optional[float] = None,
) -> Optional[list]:
    """The tick every gateway boot / housekeeping pass calls.

    Fills in the REAL seams ``run_scheduled_check`` leaves injectable:
    the same read-only ``plugins_updates.run_checks`` the manual
    ``hermes plugins check-updates`` uses (urllib feed fetch, git
    ls-remote, PyPI latest), the real plugins dir, and the manual
    ``hermes plugins update <name>`` flow as the opt-in apply path —
    auto-apply rides the identical security/consent/scan pipeline.

    Returns the check results, or None when not due / disabled. Due-gated
    by ``plugins.auto_update_check_hours`` (0 disables); a network error
    costs one warning and a stamped marker — never an apply.
    """
    if plugins_dir is None:
        from hermes_cli import plugins_cmd

        plugins_dir = plugins_cmd._plugins_dir()
    if run_checks_fn is None:
        # ONE shared network-default implementation — plugins_updates
        # owns default_fetch / default_ls_remote / _default_pypi_latest;
        # the manual `hermes plugins check-updates` passes the same
        # defaults. No boilerplate re-derivation here.
        from hermes_cli.plugins_updates import run_checks

        run_checks_fn = run_checks
    if apply_updates_fn is None:
        from hermes_cli import plugins_cmd

        from functools import partial
        apply_updates_fn = partial(plugins_cmd.cmd_update, interactive=False)
    return run_scheduled_check(
        run_checks_fn=run_checks_fn,
        plugins_dir=plugins_dir,
        apply_updates_fn=apply_updates_fn,
        log=log,
        now=now,
    )
