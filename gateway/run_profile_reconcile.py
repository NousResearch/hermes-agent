"""Hot-serve for ``gateway.multiplex_profiles``: keep the served-profile set in step with ``profiles/``
while the multiplexer runs, instead of snapshotting it once at boot.

Three things were start-time snapshots: the secondary adapter set (``_start_secondary_profile_adapters``),
the ``served_profiles`` record in ``gateway_state.json`` (``_record_served_profiles``) and the cron
ticker's ``profile_homes`` list. Everything else (``/p/<profile>/`` prefixes, profile-route eligibility,
handoff/kanban watchers, shared ingress) already reads ``profiles_to_serve()`` / ``_profile_adapters``
live, so reconciling those three is enough for a profile created after boot to be served.

``reconcile_served_profiles`` runs on the loop under one lock, triggered by the ``rescan-profiles`` control
verb (``hermes_cli/profiles.py`` create/delete fire it through the control socket) and by the supervised
``_profile_reconcile_watcher`` every ``_PROFILE_RESCAN_INTERVAL_SECS`` as the safety net. A served profile whose ``config.yaml``/``.env``
changed since its adapters were last built is re-scanned too: creators make the profile first and add the
bot token afterwards, and without this an adapter-less profile would stay adapter-less forever.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.run_shutdown import _log_suppressed
from utils import file_signature

logger = logging.getLogger(__name__)

_PROFILE_RESCAN_INTERVAL_SECS = 30.0
_PROFILE_SIGNATURE_FILES = ("config.yaml", ".env")


def profile_serve_signature(home: "Path") -> tuple:
    """Cheap change detector for a served profile's credentials/config: file signature per file."""
    sig = []
    for name in _PROFILE_SIGNATURE_FILES:
        try:
            st = os.stat(Path(home) / name)
            sig.append(file_signature(st))
        except OSError:
            sig.append(None)
    return tuple(sig)


class GatewayProfileReconcileMixin:
    """Runtime reconciliation of the multiplexed served-profile set (hot add / unroute / credential-add)."""

    _served_profile_homes: Optional[Dict[str, "Path"]] = None
    _served_profile_signatures: Optional[Dict[str, tuple]] = None
    _profile_reconcile_lock: Optional[asyncio.Lock] = None

    # ── state helpers ─────────────────────────────────────────────────────────────────────────────

    def _reconcile_lock(self) -> asyncio.Lock:
        if self._profile_reconcile_lock is None:
            self._profile_reconcile_lock = asyncio.Lock()
        return self._profile_reconcile_lock

    def served_profile_names(self) -> list:
        """Profiles this multiplexer currently serves (active first), from the live bookkeeping."""
        homes = self._served_profile_homes or {}
        active = getattr(self, "_primary_profile_name", None) or "default"
        return ([active] if active in homes or not homes else []) + sorted(n for n in homes if n != active)

    def _note_served_profiles(self, profile_homes) -> None:
        """Called by ``_record_served_profiles``: remember the served set and each home's signature."""
        homes = {str(name): Path(home) for name, home in profile_homes}
        self._served_profile_homes = homes
        sigs = self._served_profile_signatures if isinstance(self._served_profile_signatures, dict) else {}
        self._served_profile_signatures = {name: sigs.get(name) or profile_serve_signature(home)
                                           for name, home in homes.items()}

    # ── watcher ───────────────────────────────────────────────────────────────────────────────────

    async def _profile_reconcile_watcher(self, interval: float = _PROFILE_RESCAN_INTERVAL_SECS) -> None:
        """Supervised safety net: rescan ``profiles/`` every ``interval`` seconds (creators signal the
        control socket for an immediate rescan). Returns at once, never respawned, when multiplexing is off."""
        if not self._multiplex_on():
            return
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                return
            try:
                await self.reconcile_served_profiles(reason="watcher")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Served-profile reconcile failed; retrying next cycle", exc_info=True)

    # ── reconcile ─────────────────────────────────────────────────────────────────────────────────

    async def reconcile_served_profiles(self, *, reason: str = "request") -> Dict[str, Any]:
        """Diff ``profiles/`` against the served set: start adapters for new profiles, tear down and
        unroute deleted ones, (re)build adapters for served profiles whose config/.env changed. Other
        profiles' adapters are never touched. Returns ``{"added", "removed", "rescanned", "served_profiles"}``."""
        from gateway.run import MultiplexConfigError, _multiplex_profile_homes
        result: Dict[str, Any] = {"added": [], "removed": [], "rescanned": [], "reason": reason}
        if not self._multiplex_on():
            return {**result, "multiplex": False, "served_profiles": self.served_profile_names()}
        if not self._running or self._served_profile_homes is None:
            # Startup enumerates profiles/ itself; a rescan before it finishes has nothing to diff against.
            return {**result, "pending": True, "served_profiles": self.served_profile_names()}
        async with self._reconcile_lock():
            active = getattr(self, "_primary_profile_name", None) or "default"
            current = {str(name): Path(home) for name, home in _multiplex_profile_homes(self.config)}
            known = dict(self._served_profile_homes or {})
            sigs = self._served_profile_signatures or {}
            added = [n for n in current if n not in known and n != active]
            removed = [n for n in known if n not in current and n != active]
            changed = [n for n in current if n in known and n != active and n not in added
                       and profile_serve_signature(current[n]) != sigs.get(n)]
            if not (added or removed or changed):
                return {**result, "served_profiles": self.served_profile_names()}
            for name in removed:
                await self._unserve_profile(name, known[name])
                result["removed"].append(name)
            claimed = self._live_resource_claims(active)
            for name in added + changed:
                # Only acknowledge the configuration observed before connecting;
                # a setup save during an awaited handshake needs another scan.
                scan_signature = profile_serve_signature(current[name])
                try:
                    connected = await self._start_one_profile_adapters(name, current[name], claimed)
                except MultiplexConfigError as exc:
                    # Boot refuses to run with such a profile; at runtime we park just this profile.
                    logger.error("[MULTIPLEX] Profile '%s' not served: %s", name, exc)
                    connected = 0
                except Exception:
                    logger.error("[MULTIPLEX] Failed to start adapters for profile '%s'", name, exc_info=True)
                    connected = 0
                sigs[name] = scan_signature
                if name in added:
                    logger.info("[MULTIPLEX] Now serving profile '%s' (%s adapter(s) connected; %s)", name, connected, reason)
                    result["added"].append(name)
                else:
                    logger.info("[MULTIPLEX] Re-scanned profile '%s' after config/.env change (%s adapter(s) connected)", name, connected)
                    result["rescanned"].append(name)
            self._served_profile_signatures = sigs
            # A profile deleted while an adapter above was still connecting must not be recorded back
            # (the deleter's signal timed out against this lock and rmtree already ran).
            live_now = {str(name) for name, _home in _multiplex_profile_homes(self.config)}
            for name in [n for n in current if n not in live_now and n != active]:
                await self._unserve_profile(name, current.pop(name))
                result["removed"].append(name)
                added = [n for n in added if n != name]
            self._record_served_profiles(active, list(current.items()))
            if added:
                await self._after_profiles_added([(n, current[n]) for n in added])
            result["served_profiles"] = self.served_profile_names()
            return result

    def _live_resource_claims(self, active: str) -> Dict[tuple, str]:
        """Startup's ``claimed`` map rebuilt from what is live now: primary claims plus every connected
        secondary's credential/listener, so a hot-added profile reusing a token is parked, never a
        second poller."""
        claimed = self._primary_resource_claims(active)
        for profile_name, adapters in (getattr(self, "_profile_adapters", None) or {}).items():
            for platform, adapter in list(adapters.items()):
                for claim in (self._adapter_credential_claim(platform, adapter),
                              self._adapter_listener_claim(platform, adapter)):
                    if claim is not None:
                        claimed[claim] = profile_name
        return claimed

    async def _after_profiles_added(self, profile_homes) -> None:
        """Per-profile startup side effects for hot-added profiles: log routing + scoped MCP discovery."""
        from gateway.run import _enable_multiplex_log_routing, _profile_runtime_scope
        from contextvars import copy_context
        with _log_suppressed(logging.DEBUG, "log routing refresh failed", exc_info=True):
            _enable_multiplex_log_routing(self.config)
        from tools.mcp_oauth import suppress_interactive_oauth
        loop = asyncio.get_running_loop()
        for profile_name, profile_home in profile_homes:
            try:
                from tools.mcp_tool_discovery import discover_mcp_tools
                with _profile_runtime_scope(Path(profile_home)), suppress_interactive_oauth():
                    await loop.run_in_executor(None, copy_context().run, discover_mcp_tools)
            except Exception:
                logger.warning("MCP tool discovery failed for profile '%s'", profile_name, exc_info=True)

    async def _unserve_profile(self, name: str, home: "Path") -> None:
        """Stop and unroute one deleted profile: cancel its reconnects, tear down its adapters, drop its
        bookkeeping and release this process's handles into its home so the deleter's rmtree succeeds."""
        from gateway.run import _write_runtime_status_quiet
        pending = (getattr(self, "_profile_failed_platforms", None) or {}).pop(name, None) or {}
        tasks = [t for t in pending.values() if isinstance(t, asyncio.Task) and not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self._adapter_disconnect_timeout_secs())
        adapters = (getattr(self, "_profile_adapters", None) or {}).pop(name, None) or {}
        for platform, adapter in list(adapters.items()):
            await self._bounded_adapter_teardown(adapter, platform, profile=name)
        # Its ``<name>:<platform>`` runtime entries describe a profile that no longer exists.
        _write_runtime_status_quiet(drop_profile_platforms=name)
        for attr in ("pairing_stores", "_busy_text_modes_by_profile", "_busy_input_modes_by_profile"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(name, None)
        if isinstance(self._served_profile_homes, dict):
            self._served_profile_homes.pop(name, None)
        if isinstance(self._served_profile_signatures, dict):
            self._served_profile_signatures.pop(name, None)
        from gateway.session import _session_key_namespace
        prefix = _session_key_namespace(name) + ":"
        cache = getattr(self, "_agent_cache", None)
        for key in [k for k in list(cache or {}) if str(k).startswith(prefix)]:
            with _log_suppressed(logging.DEBUG, "agent eviction failed for %s", key, exc_info=True):
                self._evict_cached_agent(key)
        with _log_suppressed(logging.DEBUG, "profile handle release failed", exc_info=True):
            from hermes_state_registry import close_all_under
            close_all_under(home)
        with _log_suppressed(logging.DEBUG, "memory-store release failed", exc_info=True):
            from plugins.memory.holographic.store import MemoryStore
            MemoryStore.release_all_under(home)
        logger.info("[MULTIPLEX] Profile '%s' deleted — %d adapter(s) stopped and unrouted", name, len(adapters))


def _mcp_config_reconciler(runner=None):
    """Housekeeping chore keeping live MCP servers in step with ``mcp_servers`` on disk, every tick
    after the first (startup discovery owns that one). Reconciling on DRIFT rather than only on a
    config EDIT is what brings back a server whose FIRST connect failed (#112445): it never reached
    ``_servers``, so the parked self-probe — a property of a task that connected once — cannot revive
    it, and its config never changes. The reconcile is a cached config read plus set compares when
    nothing moved; a server dropped from config is torn down (a parked one otherwise self-probes
    every ``_PARKED_RETRY_INTERVAL`` for the life of the process) and a missing one is reconnected
    only once its per-server connect cooldown (30s→600s backoff) has lapsed, so a chronically failing
    server is retried on that schedule, not every tick. Interactive OAuth is suppressed — this runs
    on a housekeeping thread nobody is watching."""
    primed: set = set()

    def _reconcile_current(label: str) -> None:
        from tools.mcp_oauth import suppress_interactive_oauth
        from tools.mcp_tool_discovery import reconcile_mcp_servers_with_config
        if label not in primed:
            primed.add(label)
            return  # first tick: startup discovery already reflects this config (or is still running)
        with suppress_interactive_oauth():
            result = reconcile_mcp_servers_with_config()
        if result["removed"] or result["added"]:
            logger.info("MCP servers reconciled with config (%s): removed=%s added=%s",
                        label, result["removed"], result["added"])

    return lambda: _for_each_served_profile(runner, _reconcile_current)


def _for_each_served_profile(runner, body) -> None:
    """Run ``body(profile_label)`` once per served profile, inside that profile's runtime scope.

    Housekeeping runs on a bare thread with no turn on the stack, so nothing binds a profile for it:
    ``get_hermes_home()`` and ``get_secret()`` see the LAUNCH profile's values, and under
    ``gateway.multiplex_profiles`` a fail-closed credential read logs ``no profile secret scope on a
    multiplexed call`` on every tick (the skills-sync pulls resolved Nous credentials this way, four
    WARNINGs per hourly tick per chore). A single-profile gateway runs ``body`` once, unscoped:
    there the process env IS the profile's own value — unless a hosted room already flipped the
    process-wide guard (#112878), in which case the launch profile's OWN scope is bound, as
    ``run_turn.py::_standalone_launch_scope`` does for turns."""
    from gateway.run import _multiplex_profile_homes, _profile_runtime_scope
    config = getattr(runner, "config", None)
    if not getattr(config, "multiplex_profiles", False):
        from gateway.run_turn import GatewayTurnMixin
        with GatewayTurnMixin._standalone_launch_scope():
            body("default")
        return
    for profile_name, profile_home in _multiplex_profile_homes(config):
        with _profile_runtime_scope(Path(profile_home)):
            body(str(profile_name))


def profile_scoped_chore(runner, chore):
    """Wrap a zero-arg housekeeping chore that reads the profile's home, config or credentials so it
    runs once per served profile under that profile's scope (see ``_for_each_served_profile``)."""
    return lambda: _for_each_served_profile(runner, lambda _label: chore())


def migrate_profile_identity_verb(runner):
    """Build the ``migrate-profile-identity`` control-verb handler for ``hermes profile rename``
    (#111926). The live multiplexer owns the routing index in memory and writes it back
    periodically, so a CLI-side rewrite of ``agent:<old>:*`` would be clobbered on the next save;
    the CLI therefore asks this process to rekey both durable stores AND ``SessionStore._entries``.
    Runs on the control-socket executor thread; ``rekey_profile_routing`` takes the store lock."""

    def _handler(params: dict) -> dict:
        old, new = str(params.get("old") or "").strip(), str(params.get("new") or "").strip()
        if not old or not new or old == new:
            return {"ok": False, "error": "old/new required and must differ"}
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"ok": False, "error": "live gateway has no session store"}
        acquired = []
        try:
            from hermes_state_registry import acquire, release_or_close
            db_counts: Dict[str, Dict[str, int]] = {}
            routing_db = getattr(store, "_routing_db", None)
            if routing_db is not None and hasattr(routing_db, "rekey_profile_state"):
                db_counts["routing"] = routing_db.rekey_profile_state(old, new)
            routing_home = getattr(store, "_routing_home", None)
            profile_path = Path(routing_home) / "profiles" / new / "state.db" if routing_home else None
            if profile_path is not None and profile_path.exists():
                profile_db = acquire(profile_path)
                acquired.append(profile_db)
                db_counts["profile"] = profile_db.rekey_profile_state(old, new)
            rekeyed = store.rekey_profile_routing(old, new)
            return {"ok": True, "rekeyed": rekeyed, "db": db_counts}
        except Exception as exc:
            logger.warning("Profile identity migration failed for %r->%r: %s", old, new, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            for db in acquired:
                try:
                    release_or_close(db)
                except Exception:
                    logger.debug("Failed to release renamed profile state DB", exc_info=True)

    return _handler


def purge_profile_identity_verb(runner):
    """Build the ``purge-profile-identity`` control-verb handler for ``hermes profile delete``
    (#111926, delete side). The live multiplexer owns the routing index in memory and writes it back
    periodically, so a CLI-side DELETE of ``agent:<name>:*`` rows would be undone by its next save;
    the CLI therefore asks this process to drop the durable rows AND ``SessionStore._entries``.

    Deliberately NOT part of ``_unserve_profile()``: that path also unserves names that are still
    alive elsewhere in the identity story — a rename's old name leaves the served set exactly like a
    delete does (its directory is gone either way) — and purging there would race the rekey it is
    supposed to leave intact. Only the delete path invokes this verb. Runs on the control-socket
    executor thread; ``purge_profile_routing`` takes the store lock."""

    def _handler(params: dict) -> dict:
        name = str(params.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name required"}
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"ok": False, "error": "live gateway has no session store"}
        try:
            db_counts: Dict[str, Dict[str, int]] = {}
            routing_db = getattr(store, "_routing_db", None)
            if routing_db is not None and hasattr(routing_db, "purge_profile_state"):
                db_counts["routing"] = routing_db.purge_profile_state(name)
            dropped = store.purge_profile_routing(name)
            return {"ok": True, "dropped": dropped, "db": db_counts}
        except Exception as exc:
            logger.warning("Profile identity purge failed for %r: %s", name, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return _handler
