"""Hot platform reconcile: a running gateway starts, restarts or stops ONE platform adapter when a
served profile's platform configuration changes, instead of waiting for a gateway restart.

Every served profile is covered, the launch (primary) profile of a single-profile gateway included:
its ``.env`` / ``config.yaml`` signature is watched by ``_profile_reconcile_watcher``; a plugin that
registers a platform mid-run fires the ``on_plugin_loaded`` subscription (``run_plugin_rewire``); the
``reconcile-platforms`` control verb (dashboard/Desktop save, ``hermes config set``) runs it at once and
answers each platform's live state, so a caller can say "applied" or "restart needed" truthfully.

The diff is generic: the platforms the reloaded config wants (enabled, and under multiplex carrying a
credential in that profile's scope) against the live or reconnect-queued adapters. A wanted platform
with no adapter is started through the same create/wire/connect path boot uses; a live one whose build
material (token, api_key, extra) changed is retired and rebuilt; a live one no longer wanted is retired.
Retiring waits (bounded by ``agent.restart_drain_timeout``) for that platform's in-flight turns, so a
disabled adapter delivers the reply it is producing. Other adapters are never touched.

Tools and system prompt of an already-running session never change here (prompt-cache invariant): a
plugin's tools reach the next session, as with ``/skills install``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger(__name__)

# ``_instantiate_adapter`` stamps these into ``extra`` at build time; they are runner settings, not
# platform build material, and would read as a change on every comparison.
_RUNNER_STAMPED_EXTRA = frozenset({"group_sessions_per_user", "thread_sessions_per_user"})
_TURN_POLL_SECS = 0.5
# The control verb answers within this; a longer connect keeps running and the answer says ``pending``.
_VERB_WAIT_SECS = 10.0


def platform_build_fingerprint(platform_config: Any) -> str:
    """Log-safe hash of what an adapter is built from (token, api_key, platform ``extra``): a change
    means the live adapter runs on stale credentials/settings and must be rebuilt."""
    extra = {k: v for k, v in (getattr(platform_config, "extra", None) or {}).items()
             if k not in _RUNNER_STAMPED_EXTRA}
    material = json.dumps([getattr(platform_config, "token", None), getattr(platform_config, "api_key", None),
                           extra], sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()[:16]


@dataclass(frozen=True)
class PlatformPlan:
    """One profile's adapter changes: ``start`` has no adapter yet, ``restart`` is live on stale build
    material, ``stop`` is live but no longer wanted."""
    start: tuple
    restart: tuple
    stop: tuple

    def __bool__(self) -> bool:
        return bool(self.start or self.restart or self.stop)


def plan_platform_changes(wanted: dict, live: dict) -> PlatformPlan:
    """``wanted``/``live``: platform -> the config it should run on / the config its adapter was built from."""
    return PlatformPlan(
        start=tuple(p for p in wanted if p not in live),
        restart=tuple(p for p in wanted if p in live and live[p] is not None
                      and platform_build_fingerprint(wanted[p]) != platform_build_fingerprint(live[p])),
        stop=tuple(p for p in live if p not in wanted),
    )


def _turn_platform(session_key: str) -> tuple:
    """``(namespace, platform)`` of a ``build_session_key`` key (``agent:<ns>:<platform>:...``)."""
    parts = str(session_key).split(":")
    return (":".join(parts[:2]), parts[2]) if len(parts) > 2 else ("", "")


class GatewayPlatformReconcileMixin:
    """Per-profile platform adapter reconcile (primary and multiplexed secondaries)."""

    _primary_platform_signature: Optional[tuple] = None

    # ── triggers ─────────────────────────────────────────────────────────────────────────────────

    async def _note_primary_platform_baseline(self) -> None:
        """Watcher start (startup done): the files the boot adapters were built from are the baseline."""
        from gateway.run_profile_reconcile import profile_serve_signature
        with _log_suppressed(logging.DEBUG, "primary platform baseline failed", exc_info=True):
            self._primary_platform_signature = await asyncio.to_thread(
                profile_serve_signature, self._primary_platform_home())

    async def _reconcile_primary_platforms_if_changed(self) -> None:
        """Watcher tick: rebuild the launch profile's changed platforms when its config/.env changed."""
        from gateway.run_profile_reconcile import profile_serve_signature
        signature = await asyncio.to_thread(profile_serve_signature, self._primary_platform_home())
        if signature != self._primary_platform_signature:
            await self.reconcile_profile_platforms(None, reason="watcher")

    def _schedule_platform_reconcile(self, profile_name: Optional[str], reason: str) -> None:
        """Loop-thread entry for event triggers (a plugin registering a platform): reconcile in the background."""
        if not getattr(self, "_running", False):
            return
        task = asyncio.ensure_future(self.reconcile_profile_platforms(profile_name, reason=reason))
        self._retain_background_task(task)

    # ── reconcile ────────────────────────────────────────────────────────────────────────────────

    async def reconcile_profile_platforms(self, profile_name: Optional[str], *, reason: str) -> dict:
        """Apply one profile's platform diff now; ``None`` (or the primary's name) = the launch profile.
        Returns ``{"applied", "profile", "platforms": {platform: state}}``."""
        primary = getattr(self, "_primary_profile_name", None) or "default"
        if not getattr(self, "_running", False):
            return {"applied": False, "pending": True, "error": "gateway is not running yet"}
        if profile_name in (None, primary):
            from gateway.run_profile_reconcile import profile_serve_signature
            async with self._reconcile_lock():
                # Taken before the read: an edit landing during this apply is seen by the next tick.
                signature = await asyncio.to_thread(profile_serve_signature, self._primary_platform_home())
                plan = await self._apply_primary_platform_plan(reason)
                self._primary_platform_signature = signature
            return {"applied": True, "profile": primary, "changed": bool(plan),
                    "started": [p.value for p in plan.start], "restarted": [p.value for p in plan.restart],
                    "stopped": [p.value for p in plan.stop], "platforms": self._primary_platform_states()}
        home = (self._served_profile_homes or {}).get(profile_name)
        if home is None:
            return {"applied": False, "error": f"profile '{profile_name}' is not served by this gateway"}
        async with self._reconcile_lock():
            await self._apply_profile_changes(dict(self._served_profile_homes or {}), [], [], [profile_name],
                                              reason=reason)
        return {"applied": True, "profile": profile_name, "platforms": self._secondary_platform_states(profile_name)}

    def _primary_platform_home(self) -> Path:
        from hermes_constants import get_default_hermes_root, get_process_hermes_home
        return Path(get_default_hermes_root() if self._multiplex_on() else get_process_hermes_home())

    def _load_primary_config_now(self):
        """The launch profile's gateway config as the files say now (blocking: run off-loop). Standalone
        re-reads ``.env`` into ``os.environ`` first (main's ``/reload``: its adapters read it there);
        under multiplex the default profile's scope is the credential source, never ``os.environ``."""
        from gateway.config import load_gateway_config
        from gateway.run import _profile_runtime_scope
        home = self._primary_platform_home()
        if not self._multiplex_on():
            from hermes_cli.config import reload_env
            reload_env()
            return load_gateway_config()
        from agent.secret_scope import invalidate_env_file_cache
        invalidate_env_file_cache(home / ".env")
        with _profile_runtime_scope(home):
            return load_gateway_config()

    def _primary_wants(self, platform, platform_config) -> bool:
        """Startup's primary filter (``_start_prefilter_platforms``): enabled, and under multiplex only
        with a credential in the default profile's own secrets."""
        from gateway.run import _platform_has_bot_credential
        if not platform_config.enabled:
            return False
        return not self._multiplex_on() or _platform_has_bot_credential(platform, platform_config)

    async def _apply_primary_platform_plan(self, reason: str) -> PlatformPlan:
        fresh = await self._run_housekeeping_in_executor(self._load_primary_config_now)
        wanted = {p: c for p, c in fresh.platforms.items() if self._primary_wants(p, c)}
        live = {p: self.config.platforms.get(p)
                for p in {*self.adapters, *(getattr(self, "_failed_platforms", None) or {})}}
        plan = plan_platform_changes(wanted, live)
        if not plan:
            return plan
        logger.info("Platform reconcile (%s): start=%s restart=%s stop=%s", reason,
                    [p.value for p in plan.start], [p.value for p in plan.restart], [p.value for p in plan.stop])
        for platform in (*plan.stop, *plan.restart):
            await self._retire_primary_platform(platform)
        for platform in plan.stop:
            if platform in fresh.platforms:
                self.config.platforms[platform] = fresh.platforms[platform]
            else:
                self.config.platforms.pop(platform, None)
        await self._hot_start_primary_platforms([(p, wanted[p]) for p in (*plan.restart, *plan.start)])
        return plan

    async def _hot_start_primary_platforms(self, items: list) -> None:
        """Boot's create -> wire -> connect -> publish-or-queue path for platforms started mid-run."""
        pending = []
        for platform, platform_config in items:
            self.config.platforms[platform] = platform_config
            adapter = await self._create_adapter_off_loop_deps(platform, platform_config)
            if not adapter:
                heals = self._adapter_may_heal(platform, platform_config)
                logger.warning("No adapter for %s; it stays down until its plugin or dependencies load",
                               platform.value)
                self._mark_adapter_unavailable(platform.value, platform, retrying=heals, needs_attention=True)
                if heals:
                    self._failed_platforms[platform] = self._startup_retry_entry(platform, None, platform_config)
                continue
            if self._primary_claim_refused(platform, adapter):
                from gateway.run import _dispose_unused_adapter
                await _dispose_unused_adapter(adapter)
                continue
            self._wire_adapter_handlers(adapter)
            pending.append((platform, platform_config, adapter))
        raw = await self._start_connect_pending(pending)
        await self._start_aggregate_connect_results(raw or [], [], [])
        self.delivery_router.adapters = self.adapters
        if getattr(self, "_failed_platforms", None):
            self._ensure_reconnect_watcher_running()

    def _primary_claim_refused(self, platform, adapter) -> bool:
        """A secondary profile already polls this bot / binds this listener: never a second consumer."""
        primary = getattr(self, "_primary_profile_name", None) or "default"
        owners = {c: n for c, n in self._current_claim_owners().items() if n != primary}
        for kind, claim in zip(("credential", "listener"), self._adapter_claims(platform, adapter)):
            if claim is not None and self._refuse_duplicate_claim(claim, owners, primary, platform, kind):
                return True
        return False

    async def _retire_primary_platform(self, platform) -> None:
        """Take one primary platform down: drop its reconnect slot, let its in-flight turns finish, then
        tear the adapter down. Nothing else is touched."""
        (getattr(self, "_failed_platforms", None) or {}).pop(platform, None)
        adapter = self.adapters.get(platform)
        if adapter is None:
            return
        await self._await_platform_turns(platform, None)
        if self.adapters.get(platform) is adapter:
            self.adapters.pop(platform, None)
        await self._bounded_adapter_teardown(adapter, platform)
        self._update_platform_runtime_status(platform.value, platform_state="disabled",
                                             error_code=None, error_message=None)

    async def _retire_stale_profile_platforms(self, profile_name: str, profile_home: Path,
                                              previous, current) -> None:
        """Secondary rescan: retire live adapters whose platform is no longer wanted or whose build material
        changed; ``_start_one_profile_adapters`` then builds whatever is missing (the restart's second half)."""
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        profile_map = self._profile_adapters.get(profile_name) or {}
        queued = (getattr(self, "_profile_failed_platforms", None) or {}).get(profile_name) or {}
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            wanted = {p: c for p, c in current.platforms.items()
                      if c.enabled and _platform_has_bot_credential(p, c)}
        live = {p: previous.platforms.get(p) for p in {*profile_map, *queued}}
        plan = plan_platform_changes(wanted, live)
        for platform in (*plan.stop, *plan.restart):
            task = queued.pop(platform, None)
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
            adapter = profile_map.get(platform)
            if adapter is None:
                continue
            await self._await_platform_turns(platform, profile_name)
            if profile_map.get(platform) is adapter:
                profile_map.pop(platform, None)
            self._hold_secondary_claims(profile_name, platform, adapter, self._adapter_claims(platform, adapter))
            try:
                with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                    await self._bounded_adapter_teardown(adapter, platform, profile=profile_name)
            finally:
                self._release_secondary_claims(profile_name, platform, adapter)
            self._update_platform_runtime_status(f"{profile_name}:{platform.value}", platform_state="disabled",
                                                 error_code=None, error_message=None)
            logger.info("Platform reconcile: %s %s (profile: %s)", platform.value,
                        "restarting" if platform in plan.restart else "stopped", profile_name)

    async def _await_platform_turns(self, platform, profile_name: Optional[str]) -> None:
        """Wait until no turn received on ``platform`` (for a secondary: in that profile's namespace) is in
        flight, bounded by ``agent.restart_drain_timeout``: a retired adapter delivers the reply it owes."""
        from gateway.session import _session_key_namespace
        from hermes_cli.gateway import _get_restart_drain_timeout
        namespace = _session_key_namespace(profile_name) if profile_name else None
        deadline = time.monotonic() + await asyncio.to_thread(_get_restart_drain_timeout)

        def _busy() -> list:
            return [key for key in self._snapshot_running_agents()
                    if _turn_platform(key)[1] == platform.value
                    and (namespace is None or _turn_platform(key)[0] == namespace)]

        busy = _busy()
        if busy:
            logger.info("Platform reconcile: %s waits for %d in-flight turn(s) before it goes down",
                        platform.value, len(busy))
        while busy and time.monotonic() < deadline and getattr(self, "_running", False):
            await asyncio.sleep(_TURN_POLL_SECS)
            busy = _busy()

    # ── live state ───────────────────────────────────────────────────────────────────────────────

    # ``unavailable``: wanted, yet no adapter runs or is queued (no plugin/dependency, refused claim,
    # non-retryable connect failure). Every other state needs nothing a restart would add.
    def _primary_platform_states(self) -> dict:
        failed = getattr(self, "_failed_platforms", None) or {}
        return {p.value: ("connected" if p in self.adapters else "retrying" if p in failed
                          else "disabled" if not c.enabled
                          else "unavailable" if self._primary_wants(p, c) else "not_configured")
                for p, c in self.config.platforms.items()}

    def _secondary_platform_states(self, profile_name: str) -> dict:
        from gateway.config import SHARED_LISTENER_MIRROR_PLATFORMS, Platform
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        cfg = (getattr(self, "_profile_configs", None) or {}).get(profile_name)
        home = (self._served_profile_homes or {}).get(profile_name)
        live = self._profile_adapters.get(profile_name) or {}
        queued = (getattr(self, "_profile_failed_platforms", None) or {}).get(profile_name) or {}
        states = {}
        if cfg is None or home is None:
            return states
        with _profile_runtime_scope(home, hydrate_secrets=False):
            for p, c in cfg.platforms.items():
                states[p.value] = (
                    "connected" if p in live else "retrying" if p in queued else "disabled" if not c.enabled
                    # Served through the default profile's listener / shared ingress (startup's skips).
                    else "shared" if p is Platform.RELAY or p.value in SHARED_LISTENER_MIRROR_PLATFORMS
                    else "unavailable" if _platform_has_bot_credential(p, c) else "not_configured")
        return states


def reconcile_platforms_verb(runner: Any, loop: asyncio.AbstractEventLoop):
    """Control-socket ``reconcile-platforms`` ``{"home": <profile home>}`` (default: the gateway's own):
    apply that served profile's platform diff now and answer its per-platform live state. Runs on the
    socket executor: marshal onto the loop, bounded; a longer connect answers ``pending``."""
    import concurrent.futures

    def _handler(params: Optional[dict] = None) -> dict:
        from hermes_constants import get_process_hermes_home, hermes_home_key
        requested = Path(str((params or {}).get("home") or get_process_hermes_home())).expanduser()
        key = hermes_home_key(requested)
        primary_home = runner._primary_platform_home()
        if key in (hermes_home_key(primary_home), hermes_home_key(get_process_hermes_home())):
            name = None
        else:
            served = (getattr(runner, "_served_profile_homes", None) or {}).items()
            name = next((str(n) for n, h in served if hermes_home_key(h) == key), None)
            if name is None:
                return {"applied": False, "error": "home is not served by this gateway", "home": str(requested)}
        future = asyncio.run_coroutine_threadsafe(
            runner.reconcile_profile_platforms(name, reason="control-socket"), loop)
        try:
            return future.result(timeout=_VERB_WAIT_SECS)
        except concurrent.futures.TimeoutError:
            return {"applied": True, "pending": True}

    return _handler
