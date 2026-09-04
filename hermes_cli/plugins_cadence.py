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

import time
from pathlib import Path
from typing import Callable, Optional

_MARKERS_DIR = "plugin-update-checks"
_DEFAULT_INTERVAL_HOURS = 24


def _markers_dir() -> Path:
    # resolved per call (tests monkeypatch get_hermes_home)
    from hermes_constants import get_hermes_home

    d = get_hermes_home() / _MARKERS_DIR
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
    marker = _markers_dir() / "last-run"
    try:
        last = marker.stat().st_mtime
    except OSError:
        return True
    return (now - last) >= interval_hours * 3600


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

    Returns the check results, or None when not due / disabled. NEVER
    raises — a cadence failure is logged, never fatal.
    """
    import logging

    if log is None:
        log = logging.getLogger(__name__)
    if not check_due(now=now, config_get=config_get):
        return None

    try:
        results = run_checks_fn(plugins_dir)
    except Exception:
        log.warning("plugin update check failed", exc_info=True)
        results = []

    # the receipt — the surface every medium reads
    try:
        from pm import receipt

        receipt.begin("plugin-check")
        receipt.record_plugin_checks(results)
        updates = [
            r for r in results
            if getattr(r, "update_available", None) is True
        ]
        receipt.finalize("ok" if not updates else "updates-available")
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
                except Exception:
                    log.warning("auto-apply of %s failed", r.name, exc_info=True)

    # stamp the marker AFTER a completed run (even a failed one — a
    # failing check retrying every tick would hammer the network)
    try:
        marker = _markers_dir() / "last-run"
        marker.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass
    return results