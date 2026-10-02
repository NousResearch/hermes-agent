"""Bounded delayed retry for failed plugin loads (#126356).

A plugin whose import + ``register()`` overruns ``plugins.load_timeout_seconds`` — or raises during a
transient boot window (an unclean reboot leaves the state-db integrity check holding the progress
lease, starving plugin imports of I/O until the deadline bites) — used to be lost for the rest of the
process lifetime: the failure was recorded, the deferred platform loader was consumed one-shot, and
nothing ever re-attempted the load. A gateway booting in that window ran with the platform silently
unserved (dead Telegram for 49h in the field report) until someone restarted it.

Every load failure funnels through ``PluginLoaderMixin._load_plugin_scoped``'s exception path, which
seeds this scheduler. Retries are bounded (``plugins.load_retry_attempts``; ``0`` disables),
exponential-backoff (``plugins.load_retry_base_seconds``, doubling, capped at
``_RETRY_MAX_DELAY_SECS``), and run on one daemon worker per manager that exits when the queue
drains. A retry re-runs the normal load path under the same per-plugin deadline; on success the
manager fires ``on_plugin_loaded`` so live consumers (the gateway's handler re-wire) pick the plugin
up. Permanent skips (version gates, compat gates, "no register()") never reach the failure path and
are never retried; a plugin that fails deterministically simply exhausts the budget and stays failed
with a loud give-up warning.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from hermes_cli.plugins import PluginManager, PluginManifest

logger = logging.getLogger("hermes_cli.plugins")

_DEFAULT_RETRY_ATTEMPTS = 5
_DEFAULT_RETRY_BASE_SECS = 30.0
_RETRY_MAX_DELAY_SECS = 300.0
_MAX_RETRY_BASE_SECS = 600.0


def resolve_load_retry_policy() -> Tuple[int, float]:
    """``(max_attempts, base_delay_seconds)`` from ``plugins.load_retry_attempts`` /
    ``plugins.load_retry_base_seconds`` (defaults 5 / 30s; 0 attempts disables retries)."""
    attempts: Any = _DEFAULT_RETRY_ATTEMPTS
    base: Any = _DEFAULT_RETRY_BASE_SECS
    try:
        from hermes_cli.config import load_config_readonly
        plugins_cfg = (load_config_readonly() or {}).get("plugins")
        if isinstance(plugins_cfg, dict):
            if plugins_cfg.get("load_retry_attempts") is not None:
                attempts = plugins_cfg["load_retry_attempts"]
            if plugins_cfg.get("load_retry_base_seconds") is not None:
                base = plugins_cfg["load_retry_base_seconds"]
    except Exception:
        return _DEFAULT_RETRY_ATTEMPTS, _DEFAULT_RETRY_BASE_SECS
    try:
        attempts = int(attempts)
    except (TypeError, ValueError):
        logger.warning("plugins.load_retry_attempts is not a number; using default %d",
                       _DEFAULT_RETRY_ATTEMPTS)
        attempts = _DEFAULT_RETRY_ATTEMPTS
    try:
        base = float(base)
    except (TypeError, ValueError):
        logger.warning("plugins.load_retry_base_seconds is not a number; using default %gs",
                       _DEFAULT_RETRY_BASE_SECS)
        base = _DEFAULT_RETRY_BASE_SECS
    if attempts < 0:
        logger.warning("plugins.load_retry_attempts=%d is negative; retries disabled", attempts)
        attempts = 0
    if base < 0:
        logger.warning("plugins.load_retry_base_seconds=%g is negative; using default %gs",
                       base, _DEFAULT_RETRY_BASE_SECS)
        base = _DEFAULT_RETRY_BASE_SECS
    if base > _MAX_RETRY_BASE_SECS:
        logger.warning("plugins.load_retry_base_seconds=%g exceeds max %gs; clamping",
                       base, _MAX_RETRY_BASE_SECS)
        base = _MAX_RETRY_BASE_SECS
    return attempts, base


def retry_delay_secs(attempt: int, base: float) -> float:
    """Backoff for retry *attempt* (1-based): base, 2x, 4x, ... capped at ``_RETRY_MAX_DELAY_SECS``."""
    return min(base * (2 ** (attempt - 1)), _RETRY_MAX_DELAY_SECS)


class PluginLoadRetryScheduler:
    """Per-manager bounded retry of failed plugin loads on one daemon worker.

    The attempt budget is anchored to the manifest object: each retry re-loads the SAME manifest, so
    a failed retry re-seeds with the budget intact, while a force re-discovery (which re-collects
    fresh manifest objects) resets it — and makes any still-pending entry a no-op via the identity
    check in :meth:`_retry_one`.
    """

    def __init__(self, manager: "PluginManager") -> None:
        self._manager = manager
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: Dict[str, Dict[str, Any]] = {}  # plugin_key -> {manifest, attempt, due}
        # Budget anchored to the manifest object, independent of the pending queue (which is popped
        # before each retry): each retry re-loads the SAME manifest, so a failed retry re-seeds with
        # the budget intact, while a force re-discovery (fresh manifest objects) resets it.
        self._attempts: Dict[str, tuple] = {}  # plugin_key -> (manifest, attempts_used)
        self._worker: Optional[threading.Thread] = None

    # -- seeding (called from _load_plugin_scoped's failure path) -----------------------------

    def seed(self, manifest: "PluginManifest", plugin_key: str) -> None:
        """Record one fresh load failure; schedule the next retry or give up loudly."""
        max_attempts, base = resolve_load_retry_policy()
        if max_attempts <= 0:
            return
        with self._lock:
            record = self._attempts.get(plugin_key)
            if record is None or record[0] is not manifest:
                record = (manifest, 0)  # new failure generation (force re-discovery): fresh budget
            attempt = record[1] + 1
            if attempt > max_attempts:
                self._attempts.pop(plugin_key, None)
                self._pending.pop(plugin_key, None)
                logger.warning(
                    "Plugin '%s' failed to load %d time(s); not retrying again. Fix the cause, then "
                    "run `hermes plugins reload` (or restart) to load it.", plugin_key, attempt,
                )
                return
            self._attempts[plugin_key] = (manifest, attempt)
            delay = retry_delay_secs(attempt, base)
            self._pending[plugin_key] = {
                "manifest": manifest, "attempt": attempt, "due": time.monotonic() + delay,
            }
            logger.info(
                "Plugin '%s' load will be retried in %gs (attempt %d/%d)",
                plugin_key, delay, attempt, max_attempts,
            )
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name=f"plugin-load-retry:{plugin_key}", daemon=True,
                )
                self._worker.start()
            self._wake.set()

    # -- worker -------------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._worker = None
                    return
                key, state = min(self._pending.items(), key=lambda kv: kv[1]["due"])
                wait = state["due"] - time.monotonic()
            if wait > 0:
                self._wake.wait(wait)
                self._wake.clear()
                continue
            with self._lock:
                if self._pending.pop(key, None) is None:
                    continue  # consumed/given-up concurrently
            self._retry_one(key, state["manifest"], state.get("deferrals", 0))

    def _retry_one(self, plugin_key: str, manifest: "PluginManifest", deferrals: int = 0) -> None:
        """Re-run the normal load path once; notify listeners when the plugin comes back."""
        manager = self._manager
        # The deadline returns while its worker is still alive: never start a second
        # import+register() alongside an in-flight one (the abandoned-context guard cannot undo
        # arbitrary import side effects). Defer WITHOUT spending budget; a worker that outlives
        # the whole budget's worth of deferrals means the plugin is terminally stuck — give up.
        from hermes_cli.plugins_loader import plugin_load_in_flight
        if plugin_load_in_flight(plugin_key):
            max_attempts, base = resolve_load_retry_policy()
            with self._lock:
                record = self._attempts.get(plugin_key)
                attempt = record[1] if record is not None and record[0] is manifest else 1
                deferrals += 1
                if deferrals > max_attempts:
                    self._attempts.pop(plugin_key, None)
                    logger.warning(
                        "Plugin '%s': earlier load attempt is STILL running after %d deferral(s); "
                        "not retrying. Fix the hang, then `hermes plugins reload`.",
                        plugin_key, deferrals - 1,
                    )
                    return
                self._pending[plugin_key] = {
                    "manifest": manifest, "attempt": attempt, "deferrals": deferrals,
                    "due": time.monotonic() + retry_delay_secs(attempt, base),
                }
            logger.info(
                "Plugin '%s' retry deferred: its earlier load attempt is still running "
                "(deferral %d/%d)", plugin_key, deferrals, max_attempts,
            )
            if self._worker is None or not self._worker.is_alive():
                with self._lock:
                    if self._worker is None or not self._worker.is_alive():
                        self._worker = threading.Thread(
                            target=self._run, name=f"plugin-load-retry:{plugin_key}", daemon=True,
                        )
                        self._worker.start()
            self._wake.set()
            return
        # The staleness check and the re-load must be one discovery-lock transaction: a force
        # re-discovery in between would otherwise let a stale manifest double-load its plugin.
        with manager._discovery_lock:
            current = manager._plugins.get(plugin_key)
            if (
                current is None
                or current.enabled
                or current.error is None
                or current.manifest is not manifest
            ):
                return  # healed, removed, or replaced by a force re-discovery while waiting
            logger.info("Retrying load of plugin '%s' after its earlier failure", plugin_key)
            loaded_before = frozenset(
                k for k, p in manager._plugins.items() if not p.error and not p.deferred
            )
            manager._load_plugin(manifest)
            reloaded = manager._plugins.get(plugin_key)
        if reloaded is not None and reloaded.enabled:
            with self._lock:
                self._attempts.pop(plugin_key, None)  # healed: a later fresh failure gets a fresh budget
            logger.info("Plugin '%s' loaded successfully on retry", plugin_key)
            manager._notify_plugin_loaded(loaded_before)
        # A fresh failure re-seeds from _load_plugin_scoped's failure path with the same manifest,
        # so the backoff continues until the budget is spent.
