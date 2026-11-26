"""Hot platform reconcile: a running gateway starts, restarts or stops ONE platform adapter when a
served profile's platform configuration changes, instead of waiting for a gateway restart.

Every served profile is covered, the launch (primary) profile of a single-profile gateway included:
its ``.env`` / ``config.yaml`` signature is watched by ``_profile_reconcile_watcher``; a plugin that
registers a platform mid-run fires the ``on_plugin_loaded`` subscription (``run_plugin_rewire``); the
``reconcile-platforms`` control verb (dashboard/Desktop save, ``hermes config set``) runs it at once and
answers each platform's live state, so a caller can say "applied" or "restart needed" truthfully.

A platform whose plugin is not installed yet arrives the same way: when the profile's credential layer
(its ``.env``) changed, the reconcile first runs the gateway-start plugin recovery for that profile
(``gateway.run_startup.recover_left_core_in``: the catalog plugin of a platform it now has credentials
for), then starts what the installed plugin registers. A gateway start runs that recovery beside the
start (:func:`recover_plugins_for_start`), never long ahead of the platforms that are ready: a plugin
it (or another installer) lands later is started by this reconcile once it loads.

The diff is generic: every build attempt (boot's and a mid-run one, through the one prepare path) records
a :class:`BuildPrint` of what it read (typed settings, ``extra``, plugin registration, the ``.env`` values
the platform owns); a reconcile compares the prints of what the reloaded config wants (enabled, and under
multiplex carrying a credential in that profile's scope). Wanted and never attempted: started; attempted
from other inputs: rebuilt; attempted but no longer wanted: retired. Adapters no build attempt owns (the
LOCAL session adapter) are never touched. Retiring waits (bounded by ``agent.restart_drain_timeout``) for
the turns the retiring adapter answers, so a disabled bot delivers the reply it is producing. A reconnect
already in flight for a retired slot discards what it built instead of publishing it.

Tools and system prompt of an already-running session never change here (prompt-cache invariant): a
plugin's tools reach the next session, as with ``/skills install``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger(__name__)

# ``_instantiate_adapter`` stamps these into ``extra`` at build time; they are runner settings, not
# platform build material, and would read as a change on every comparison.
_RUNNER_STAMPED_EXTRA = frozenset({"group_sessions_per_user", "thread_sessions_per_user"})
# The runner, not the adapter, reads these (delivery target): a /sethome is not a rebuild.
_RUNNER_OWNED_FIELDS = frozenset({"home_channel"})
_TURN_POLL_SECS = 0.5
# The control verb answers within this; a longer connect keeps running and the answer says ``pending``.
_VERB_WAIT_SECS = 10.0
# A gateway start gives its plugin recovery this head start: an install that lands within it is in the
# runner's first config (served by the start itself); a longer one (a clone, another installer's lock)
# goes on beside the start and its platform is started live when it lands.
_RECOVERY_HEAD_START_SECS = 5.0


# ── build prints ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildPrint:
    """Log-safe hashes of what one adapter build read: ``config`` = the platform's typed settings and
    ``extra`` plus whether a plugin registers it; ``env`` = the profile ``.env`` values the platform owns
    (its env prefixes), which adapters read directly at connect (Slack's app token never reaches the
    config). ``env`` None = not taken (a reconnect-queued entry rereads env at its next attempt)."""
    config: str
    env: Optional[str] = None

    def differs(self, other: BuildPrint) -> bool:
        return self.config != other.config or (
            self.env is not None and other.env is not None and self.env != other.env)


def _digest(material: Any) -> str:
    raw = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _runner_read_env(entry: Any) -> frozenset:
    """Env names the RUNNER reads per message/delivery (the entry's declared gates, its home channel):
    a change there applies without rebuilding the adapter, so it is not build material."""
    names = {getattr(entry, attr, "") for attr in (
        "allowed_users_env", "allow_all_env", "group_allowed_chats_env", "allow_bots_env", "cron_deliver_env_var")}
    return frozenset(n for n in names if n)


def platform_build_print(platform: Any, platform_config: Any, home: Optional[Path]) -> BuildPrint:
    """The :class:`BuildPrint` of building ``platform`` from ``platform_config`` with ``home``'s ``.env``
    (None: config only). Blocking (reads ``.env``, may resolve a deferred plugin entry): run off-loop."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import settled_plugin_registrations
    # Read between sweeps: mid force-reload (another plugin installed or enabled) every plugin platform is
    # briefly unregistered, which read as "changed" and restarted live adapters nothing touched.
    with settled_plugin_registrations():
        registered = platform.value in platform_registry.registered_names()  # no load, no probe side effects
        entry = platform_registry.get(platform.value) if registered and home is not None else None
    settings = {f.name: getattr(platform_config, f.name) for f in fields(platform_config)
                if f.compare and f.name not in _RUNNER_OWNED_FIELDS}
    settings["extra"] = {k: v for k, v in (settings.get("extra") or {}).items() if k not in _RUNNER_STAMPED_EXTRA}
    config = _digest([settings, registered])
    if home is None:
        return BuildPrint(config)
    from agent.secret_scope import load_env_file
    from hermes_cli.profile_channels import platform_env_prefixes
    skip = _runner_read_env(entry)
    prefixes = platform_env_prefixes(platform.value)
    owned = {k: v for k, v in load_env_file(Path(home) / ".env").items()
             if k.startswith(prefixes) and k not in skip and not k.startswith(tuple(f"{n}_" for n in skip))}
    return BuildPrint(config, _digest(owned))


@dataclass(frozen=True)
class PlatformPlan:
    """One profile's adapter changes: ``start`` has no adapter yet, ``restart`` is live on stale build
    material, ``stop`` is live but no longer wanted; ``forget`` was attempted, never came up, and is no
    longer wanted (nothing to tear down)."""
    start: tuple
    restart: tuple
    stop: tuple
    forget: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.start or self.restart or self.stop)


def plan_platform_changes(wanted: dict, built: dict, live) -> PlatformPlan:
    """``wanted``: platform -> :class:`BuildPrint` the config asks for; ``built``: platform -> the print
    its last build attempt read; ``live``: platforms with an adapter or a queued reconnect. Only platforms
    a build attempt owns are ever stopped: runtime-owned adapters (the LOCAL session adapter) have none."""
    changed = {p for p in wanted if p in built and built[p].differs(wanted[p])}
    gone = [p for p in built if p not in wanted]
    return PlatformPlan(
        start=tuple(p for p in wanted if p not in built or (p in changed and p not in live)),
        restart=tuple(p for p in wanted if p in changed and p in live),
        stop=tuple(p for p in gone if p in live),
        forget=tuple(p for p in gone if p not in live),
    )


def _turn_platform(session_key: str) -> str:
    """The platform of a ``build_session_key`` key (``agent:<ns>:<platform>:...``)."""
    parts = str(session_key).split(":")
    return parts[2] if len(parts) > 2 else ""


def _receipt(profile: str, started, restarted, stopped, states: dict, reasons: dict, installed=()) -> dict:
    """The one ``reconcile-platforms`` answer shape, primary and secondary alike. ``installed``: plugins
    this reconcile installed for the profile (its credentials named a platform that had none)."""
    return {"applied": True, "profile": profile, "changed": bool(started or restarted or stopped),
            "started": [p.value for p in started], "restarted": [p.value for p in restarted],
            "stopped": [p.value for p in stopped], "platforms": states, "reasons": reasons,
            "installed": list(installed)}


class PlatformConfigUnreadable(Exception):
    """A served profile's ``config.yaml`` cannot be loaded: the reconcile keeps the last good
    configuration (adapters, runner settings, ``os.environ``) and acknowledges nothing, so the next
    readable edit is applied as usual."""


def _unreadable_receipt(profile: str, error: Exception, states: dict) -> dict:
    """The ``reconcile-platforms`` answer when the profile's config cannot be read: nothing applied,
    every platform keeps running as it was (``platforms``), and why (``error``)."""
    return {"applied": False, "profile": profile, "changed": False, "started": [], "restarted": [],
            "stopped": [], "platforms": states, "reasons": {}, "installed": [],
            "reason": "config_unreadable",
            "error": f"config.yaml cannot be read; the running gateway kept its current platforms ({error})"}


def load_platform_config_strict():
    """The live reconcile's ONE config read: the startup loader's fallback to ``.env`` /
    ``gateway.json`` on a YAML failure would read as "every YAML-only platform is gone"."""
    from gateway.config import load_gateway_config
    try:
        return load_gateway_config(strict=True)
    except Exception as exc:  # any read failure keeps the last good config
        raise PlatformConfigUnreadable(f"{type(exc).__name__}: {' '.join(str(exc).split())}") from exc


def publish_runner_owned_settings(live_platforms: dict, fresh_platforms: dict) -> None:
    """Copy the settings the runner (not the adapter) reads, e.g. ``home_channel``, from a fresh load
    into the live platform configs in place: no rebuild is owed, and every holder of the live object
    (the delivery router, the adapter built from it) sees the new value. A served profile needs none:
    its rescan replaces the cached config the runner reads."""
    for platform, current in live_platforms.items():
        new = fresh_platforms.get(platform)
        if new is None or new is current:
            continue
        for name in _RUNNER_OWNED_FIELDS:
            if getattr(current, name) != getattr(new, name):
                setattr(current, name, getattr(new, name))


def _read_dotenv(home: Path) -> dict:
    from agent.secret_scope import load_env_file
    return load_env_file(home / ".env")


def _credential_print(home: Path) -> str:
    """Digest of ``home``'s ``.env``: the layer a platform's credentials arrive in (``hermes config set``,
    the dashboard / Desktop save, an edit). Only its change re-runs the plugin recovery, so other config
    writes never re-attempt an install that keeps failing."""
    try:
        return hashlib.sha256((home / ".env").read_bytes()).hexdigest()
    except OSError:
        return ""


def _dependency_blocked(config: Any) -> dict:
    """Blocking. Configured plugin platforms (activation evidence present) the config sweep left out
    because their dependency probe fails in this process: ``{name: why}``. A dependency installed after
    the gateway started is not importable here until it restarts, so the save is not live."""
    from gateway.platform_activation import configured_plugin_entries
    from gateway.platform_registry import platform_registry
    names = {p.value for p in config.platforms}
    blocked = {}
    for entry in configured_plugin_entries(platform_registry, yaml_cfg={}, platform_names=config.platforms):
        if entry.name in names:
            continue
        with _log_suppressed(logging.DEBUG, "check_fn for %s raised", entry.name, exc_info=True):
            if entry.check_fn():
                continue
        blocked[entry.name] = ("the running gateway cannot import this platform's Python dependencies "
                               "(installed after it started, or missing); restart the gateway")
    return blocked


class GatewayPlatformReconcileMixin:
    """Per-profile platform adapter reconcile (primary and multiplexed secondaries)."""

    _primary_platform_signature: Optional[tuple] = None
    _primary_platform_plan: Optional[PlatformPlan] = None
    _primary_credential_print: Optional[str] = None
    _primary_dotenv: Optional[dict] = None  # the .env values the last read put into os.environ (standalone)
    _primary_installing: bool = False  # the reconcile is installing a plugin the new credentials need
    _unreadable_signatures: Optional[dict] = None  # profile -> the file signature whose config failed to load

    @property
    def _unreadable_platform_configs(self) -> dict:
        if self._unreadable_signatures is None:
            self._unreadable_signatures = {}
        return self._unreadable_signatures

    def _note_unreadable_platform_config(self, profile: str, signature, error: Exception) -> None:
        """One WARNING per unreadable edit, not one per watcher tick."""
        if self._unreadable_platform_configs.get(profile) != signature:
            logger.warning("Platform reconcile: profile '%s' config.yaml cannot be read; keeping its current "
                           "platforms until it is fixed: %s", profile, error)
        self._unreadable_platform_configs[profile] = signature

    # ── triggers ─────────────────────────────────────────────────────────────────────────────────

    async def _note_boot_platform_signature(self) -> None:
        """Startup, before any adapter is built: the files the boot config was just read from are the
        baseline, so an edit landing during a slow startup reads as a change at the watcher's start."""
        from gateway.run_profile_reconcile import profile_serve_signature
        home = self._primary_platform_home()
        with _log_suppressed(logging.DEBUG, "primary platform baseline failed", exc_info=True):
            self._primary_credential_print = await asyncio.to_thread(_credential_print, home)
            self._primary_dotenv = await asyncio.to_thread(_read_dotenv, home)
            self._primary_platform_signature = await asyncio.to_thread(profile_serve_signature, home)

    async def _start_primary_platform_watch(self) -> None:
        """Watcher start: apply at once whatever changed since boot read its config."""
        if self._primary_platform_signature is None:
            await self._note_boot_platform_signature()
            return
        with _log_suppressed(logging.WARNING, "startup platform reconcile failed", exc_info=True):
            await self._reconcile_primary_platforms_if_changed(reason="startup")

    async def _reconcile_primary_platforms_if_changed(self, reason: str = "watcher") -> None:
        """Watcher tick: rebuild the launch profile's changed platforms when its config/.env changed."""
        from gateway.run_profile_reconcile import profile_serve_signature
        signature = await asyncio.to_thread(profile_serve_signature, self._primary_platform_home())
        primary = getattr(self, "_primary_profile_name", None) or "default"
        if signature not in (self._primary_platform_signature, self._unreadable_platform_configs.get(primary)):
            await self.reconcile_profile_platforms(None, reason=reason)

    def _schedule_platform_reconcile(self, profile_name: Optional[str], reason: str) -> None:
        """Loop-thread entry for event triggers (a plugin registering a platform, a cancelled reconnect
        that settled late): reconcile in the background."""
        if not getattr(self, "_running", False):
            return
        task = asyncio.ensure_future(self.reconcile_profile_platforms(profile_name, reason=reason))
        self._retain_background_task(task)

    # ── reconcile ────────────────────────────────────────────────────────────────────────────────

    async def reconcile_profile_platforms(self, profile_name: Optional[str], *, reason: str) -> dict:
        """Apply one profile's platform diff now; ``None`` (or the primary's name) = the launch profile.
        Returns ``{"applied", "profile", "changed", "started", "restarted", "stopped", "platforms": {platform:
        state}, "reasons": {platform: why it is unavailable}}`` for every served profile alike."""
        primary = getattr(self, "_primary_profile_name", None) or "default"
        if not getattr(self, "_running", False):
            return {"applied": False, "pending": True, "error": "gateway is not running yet"}
        if profile_name in (None, primary):
            from gateway.run_profile_reconcile import profile_serve_signature
            async with self._reconcile_lock():
                # Taken before the read: an edit landing during this apply is seen by the next tick.
                signature = await asyncio.to_thread(profile_serve_signature, self._primary_platform_home())
                try:
                    plan, installed = await self._apply_primary_platform_plan(reason)
                except PlatformConfigUnreadable as exc:
                    # Not acknowledged: the edit that makes it readable again is applied as usual.
                    self._note_unreadable_platform_config(primary, signature, exc)
                    return _unreadable_receipt(primary, exc, self._primary_platform_states())
                finally:
                    self._primary_platform_plan = None
                self._primary_platform_signature = signature
                self._unreadable_platform_configs.pop(primary, None)
            return _receipt(primary, plan.start, plan.restart, plan.stop, self._primary_platform_states(),
                            dict(getattr(self, "_primary_blocked_platforms", None) or {}), installed)
        home = (self._served_profile_homes or {}).get(profile_name)
        if home is None:
            return {"applied": False, "error": f"profile '{profile_name}' is not served by this gateway"}
        async with self._reconcile_lock():
            before = self._secondary_running_platforms(profile_name)
            (getattr(self, "_profile_platform_plans", None) or {}).pop(profile_name, None)
            applied = await self._apply_profile_changes(
                dict(self._served_profile_homes or {}), [], [], [profile_name], reason=reason)
            unreadable = (applied.get("unreadable") or {}).get(profile_name)
            if unreadable is not None:
                return _unreadable_receipt(profile_name, unreadable, self._secondary_platform_states(profile_name))
            plan = (getattr(self, "_profile_platform_plans", None) or {}).pop(
                profile_name, PlatformPlan((), (), ()))
            after = self._secondary_running_platforms(profile_name)
        started = tuple(p for p in after if p not in before and p not in plan.restart)
        return _receipt(profile_name, started, plan.restart, plan.stop,
                        self._secondary_platform_states(profile_name), {})

    def platform_plan_in_flight(self, profile_name: Optional[str]) -> Optional[PlatformPlan]:
        """The plan a still-running reconcile of ``profile_name`` (None = the launch profile) applies."""
        if profile_name is None:
            return self._primary_platform_plan
        return (getattr(self, "_profile_platform_plans", None) or {}).get(profile_name)

    def _secondary_running_platforms(self, profile_name: str) -> set:
        queued = (getattr(self, "_profile_failed_platforms", None) or {}).get(profile_name) or {}
        return {*(self._profile_adapters.get(profile_name) or {}), *queued}

    def _primary_platform_home(self) -> Path:
        from hermes_constants import get_default_hermes_root, get_process_hermes_home
        return Path(get_default_hermes_root() if self._multiplex_on() else get_process_hermes_home())

    def _primary_platform_wanted(self, platform, platform_config) -> bool:
        """Boot's and the reconcile's one filter: enabled, and under multiplex only with a credential in
        the default profile's own secrets (a secondary may own the token instead)."""
        from gateway.run import _platform_has_bot_credential
        if not platform_config.enabled:
            return False
        return not self._multiplex_on() or _platform_has_bot_credential(platform, platform_config)

    def _primary_build_print(self, platform, platform_config) -> BuildPrint:
        """Blocking. Always taken unscoped (boot and reconcile alike), so equal inputs hash equal."""
        return platform_build_print(platform, platform_config, self._primary_platform_home())

    async def _prepare_primary_adapter(self, platform, platform_config, build_print: BuildPrint):
        """Create + wire one primary adapter (no connect): boot's per-platform step, shared with a
        mid-run start. Records the build print; a platform with no adapter is flagged unavailable and,
        when it can heal on its own, queued for the reconnect watcher. Returns the adapter or None."""
        from gateway.config import Platform
        prints = getattr(self, "_primary_build_prints", None)
        if prints is None:
            prints = self._primary_build_prints = {}
        prints[platform] = build_print
        adapter = await self._create_adapter_off_loop_deps(platform, platform_config)
        if adapter:
            # Under multiplexing the default profile needs the same whole-handler runtime scope as a
            # secondary (authorization and prompt rendering run before the agent-turn scope).
            self._wire_adapter_handlers(adapter)
            return adapter
        if platform.value in {m.value for m in Platform.__members__.values()}:
            logger.warning("No adapter available for %s", platform.value)
        else:
            logger.warning("No adapter for '%s' -- is the plugin installed? (platform is enabled in "
                           "config.yaml but no plugin registered it)", platform.value)
        # Only a platform that can heal on its own is queued for the reconnect watcher; either way flag
        # it so the unserved enabled platform is visible.
        heals = self._adapter_may_heal(platform, platform_config)
        self._mark_adapter_unavailable(platform.value, platform, retrying=heals, needs_attention=True)
        if heals:
            self._failed_platforms[platform] = self._startup_retry_entry(platform, None, platform_config)
        return None

    def _recover_primary_plugins(self, home: Path) -> list:
        """Blocking. When ``home``'s credential layer changed since the last look: the gateway-start
        plugin recovery for it (a platform whose credentials just arrived gets its catalog plugin, loaded
        in this process), so the reconcile below starts it. Returns the installed plugin names."""
        from gateway.run_startup import recover_left_core_in
        from hermes_constants import get_process_hermes_home, hermes_home_key
        credentials = _credential_print(home)
        if credentials == self._primary_credential_print:
            return []
        self._primary_installing = True
        try:
            # The launch home's secret sources were hydrated at boot (as recover_left_core_at_gateway_start).
            launch = hermes_home_key(home) == hermes_home_key(get_process_hermes_home())
            installed = recover_left_core_in(home, hydrate_secrets=not launch)
        finally:
            self._primary_installing = False
        self._primary_credential_print = credentials
        if installed:
            logger.info("Platform reconcile: installed %s for the credentials this profile now has", installed)
        return installed

    def _load_primary_platforms_now(self) -> tuple:
        """Blocking (off-loop). ``(config, wanted prints, blocked, installed)``: the launch profile's gateway
        config as the files say now (after :meth:`_recover_primary_plugins`), the build print of every
        platform it wants, the configured plugin platforms the config sweep dropped because their
        dependencies do not import here, and the plugins the recovery installed. Standalone re-reads
        ``.env`` changes into ``os.environ`` first (:meth:`_apply_dotenv_changes`: its adapters read it
        there); under multiplex the default profile's scope is the credential source, never ``os.environ``."""
        home = self._primary_platform_home()
        credentials = self._primary_credential_print
        installed = self._recover_primary_plugins(home)
        try:
            return self._read_primary_platforms(home, installed)
        except PlatformConfigUnreadable:
            self._primary_credential_print = credentials  # the fixed config re-runs this recovery too
            raise

    def _read_primary_platforms(self, home: Path, installed: list) -> tuple:
        from gateway.run import _profile_runtime_scope
        if self._multiplex_on():
            from agent.secret_scope import invalidate_env_file_cache
            invalidate_env_file_cache(home / ".env")
            with _profile_runtime_scope(home):
                fresh = load_platform_config_strict()
                wanted = [(p, c) for p, c in fresh.platforms.items() if self._primary_platform_wanted(p, c)]
        else:
            undo = self._apply_dotenv_changes(home)
            try:
                fresh = load_platform_config_strict()
            except PlatformConfigUnreadable:
                undo()  # the running adapters keep reading the environment they were built from
                raise
            wanted = [(p, c) for p, c in fresh.platforms.items() if self._primary_platform_wanted(p, c)]
        prints = {p: (c, self._primary_build_print(p, c)) for p, c in wanted}
        return fresh, prints, _dependency_blocked(fresh), installed

    def _apply_dotenv_changes(self, home: Path):
        """Standalone: what changed in ``home``'s ``.env`` since the last read, into ``os.environ`` (its
        adapters read it there). A key leaves ``os.environ`` only when the ``.env`` held it and no longer
        does: a value the process was launched with (Docker ``-e``, a unit's ``Environment=``) and the
        ``.env`` never named is not the ``.env``'s to remove. Returns the undo for a reconcile that
        cannot go on (its config is unreadable): the environment and the last read, as they were."""
        import os
        from hermes_constants import is_launch_only_env_key
        current = _read_dotenv(home)
        previous = self._primary_dotenv
        before = {}
        for key, value in current.items():
            if not is_launch_only_env_key(key) and os.environ.get(key) != value:
                before.setdefault(key, os.environ.get(key))
                os.environ[key] = value
        for key, value in (previous or {}).items():
            if key not in current and os.environ.get(key) == value:
                before.setdefault(key, value)
                os.environ.pop(key, None)
        self._primary_dotenv = current

        def _undo() -> None:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self._primary_dotenv = previous
        return _undo

    async def _apply_primary_platform_plan(self, reason: str) -> tuple:
        """``(plan applied, plugins installed first)``."""
        fresh, wanted, self._primary_blocked_platforms, installed = await self._run_housekeeping_in_executor(
            self._load_primary_platforms_now)
        built = getattr(self, "_primary_build_prints", None)
        if built is None:
            built = self._primary_build_prints = {}
        live = {*self.adapters, *(getattr(self, "_failed_platforms", None) or {})}
        plan = plan_platform_changes({p: bp for p, (_c, bp) in wanted.items()}, built, live)
        for platform in plan.forget:
            built.pop(platform, None)
        # Runner-read settings (the home channel) are live at once, whatever the adapter plan says.
        publish_runner_owned_settings(self.config.platforms, fresh.platforms)
        if not plan:
            return plan, installed
        self._primary_platform_plan = plan  # what a ``pending`` verb answer reports while this runs
        logger.info("Platform reconcile (%s): start=%s restart=%s stop=%s", reason,
                    [p.value for p in plan.start], [p.value for p in plan.restart], [p.value for p in plan.stop])
        for platform in (*plan.stop, *plan.restart):
            await self._retire_primary_platform(platform)
        for platform in plan.stop:
            built.pop(platform, None)
            if platform in fresh.platforms:
                self.config.platforms[platform] = fresh.platforms[platform]
            else:
                self.config.platforms.pop(platform, None)
        await self._hot_start_primary_platforms([(p, *wanted[p]) for p in (*plan.restart, *plan.start)])
        return plan, installed

    async def _hot_start_primary_platforms(self, items: list) -> None:
        """Boot's prepare -> connect -> publish-or-queue path for platforms started mid-run. Under multiplex
        the whole create/connect runs in the default profile's scope: its adapters read their secrets
        there, and ``os.environ`` still holds the boot-time values."""
        from contextlib import nullcontext
        from gateway.run import _async_profile_runtime_scope, _dispose_unused_adapter
        scope = (_async_profile_runtime_scope(self._primary_platform_home()) if self._multiplex_on()
                 else nullcontext())
        async with scope:
            pending = []
            for platform, platform_config, build_print in items:
                self.config.platforms[platform] = platform_config
                adapter = await self._prepare_primary_adapter(platform, platform_config, build_print)
                if adapter is None:
                    continue
                if self._primary_claim_refused(platform, adapter):
                    await _dispose_unused_adapter(adapter)
                    continue
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
        """Take one primary platform down: drop its reconnect slot (an attempt already in flight finds
        its slot gone and discards what it built), let the turns this adapter answers finish, then tear
        the adapter down. Nothing else is touched."""
        (getattr(self, "_failed_platforms", None) or {}).pop(platform, None)
        adapter = self.adapters.get(platform)
        if adapter is None:
            return
        await self._await_adapter_turns(platform, adapter)
        if self.adapters.get(platform) is adapter:
            self.adapters.pop(platform, None)
        await self._bounded_adapter_teardown(adapter, platform)
        self._update_platform_runtime_status(platform.value, platform_state="disabled",
                                             error_code=None, error_message=None)

    def _note_secondary_build(self, profile_name: str, platform, build_print: BuildPrint) -> None:
        prints = getattr(self, "_profile_build_prints", None)
        if prints is None:
            prints = self._profile_build_prints = {}
        prints.setdefault(profile_name, {})[platform] = build_print

    def _secondary_build_print(self, profile_home: Path, platform, platform_config) -> BuildPrint:
        """Blocking; call inside the profile's scope (its plugin registry decides ``registered``)."""
        return platform_build_print(platform, platform_config, profile_home)

    async def _retire_stale_profile_platforms(self, profile_name: str, profile_home: Path, current,
                                              claimed: dict) -> None:
        """Secondary rescan: retire live adapters whose platform is no longer wanted or whose build
        print changed; ``_start_one_profile_adapters`` then builds whatever is missing (the restart's
        second half). A retired adapter's claims leave ``claimed``: its own rebuild is not a duplicate."""
        from contextvars import copy_context
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        profile_map = self._profile_adapters.get(profile_name) or {}
        queued = (getattr(self, "_profile_failed_platforms", None) or {}).get(profile_name) or {}
        built = (getattr(self, "_profile_build_prints", None) or {}).setdefault(profile_name, {})
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            wanted_cfg = {p: c for p, c in current.platforms.items()
                          if c.enabled and _platform_has_bot_credential(p, c)}
            ctx = copy_context()
        wanted = await asyncio.to_thread(
            ctx.run, lambda: {p: self._secondary_build_print(profile_home, p, c) for p, c in wanted_cfg.items()})
        plan = plan_platform_changes(wanted, built, {*profile_map, *queued})
        for platform in plan.forget:
            built.pop(platform, None)
        plans = getattr(self, "_profile_platform_plans", None)
        if plans is None:
            plans = self._profile_platform_plans = {}
        plans[profile_name] = plan
        for platform in (*plan.stop, *plan.restart):
            await self._retire_secondary_platform(profile_name, profile_home, platform, queued, claimed)
            built.pop(platform, None)
            logger.info("Platform reconcile: %s %s (profile: %s)", platform.value,
                        "restarting" if platform in plan.restart else "stopped", profile_name)

    async def _retire_secondary_platform(self, profile_name, profile_home, platform, queued, claimed) -> None:
        from gateway.run import _profile_runtime_scope
        task = queued.pop(platform, None)
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()  # awaited: its claim hold must be gone before the rebuild below
            _done, unsettled = await asyncio.wait([task], timeout=self._adapter_disconnect_timeout_secs())
            if unsettled:
                # Still releasing (its hold blocks the rebuild below): rebuild once it settles.
                logger.info("Platform reconcile: %s reconnect still settling; rebuilding when it does "
                            "(profile: %s)", platform.value, profile_name)
                task.add_done_callback(
                    lambda _t: self._schedule_platform_reconcile(profile_name, "reconnect-settled"))
        profile_map = self._profile_adapters.get(profile_name) or {}
        adapter = profile_map.get(platform)
        if adapter is None:
            return
        await self._await_adapter_turns(platform, adapter)
        if profile_map.get(platform) is adapter:
            profile_map.pop(platform, None)
        claims = self._adapter_claims(platform, adapter)
        for claim in claims:
            if claim is not None and claimed.get(claim) == profile_name:
                del claimed[claim]
        self._hold_secondary_claims(profile_name, platform, adapter, claims)
        try:
            with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                await self._bounded_adapter_teardown(adapter, platform, profile=profile_name)
        finally:
            self._release_secondary_claims(profile_name, platform, adapter)
        self._update_platform_runtime_status(f"{profile_name}:{platform.value}", platform_state="disabled",
                                             error_code=None, error_message=None)

    async def _await_adapter_turns(self, platform, adapter) -> None:
        """Wait until no in-flight turn is answered by ``adapter`` (the bot that received it, whatever
        profile it routed to), bounded by ``agent.restart_drain_timeout``: a retired adapter delivers the
        reply it owes. A turn with no known source counts when it runs on the same platform."""
        from hermes_cli.gateway import _get_restart_drain_timeout
        deadline = time.monotonic() + await asyncio.to_thread(_get_restart_drain_timeout)

        def _answers(key: str) -> bool:
            source = self._get_cached_session_source(key)
            answering = self._delivery_adapter_for(source) if source is not None else None
            if answering is not None:
                return answering is adapter
            return _turn_platform(key) == platform.value

        def _busy() -> list:
            return [key for key in self._snapshot_running_agents() if _answers(key)]

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
        states = {p.value: ("connected" if p in self.adapters else "retrying" if p in failed
                            else "disabled" if not c.enabled
                            else "unavailable" if self._primary_platform_wanted(p, c) else "not_configured")
                  for p, c in self.config.platforms.items()}
        for name in getattr(self, "_primary_blocked_platforms", None) or {}:
            states.setdefault(name, "unavailable")
        return states

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
            # Still connecting (or draining a turn): name what it is doing, so callers can say so.
            plan = runner.platform_plan_in_flight(name) or PlatformPlan((), (), ())
            return {"applied": True, "pending": True, "starting": [p.value for p in plan.start],
                    "restarting": [p.value for p in plan.restart], "stopping": [p.value for p in plan.stop],
                    "installing": name is None and bool(getattr(runner, "_primary_installing", False))}

    return _handler


def _recover_beside_start() -> asyncio.Future:
    """The left-core / catalog plugin recovery for the homes the runner loads first, on its own thread."""
    from agent.memory_provider import spawn_context_thread
    from gateway.run_startup import recover_left_core_at_gateway_start
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _settle(outcome, error) -> None:
        if not future.done():
            future.set_exception(error) if error is not None else future.set_result(outcome)

    def _run() -> None:
        try:
            outcome, error = recover_left_core_at_gateway_start(), None
        except Exception as exc:  # health: allow BLE001 -- handed to the awaiting start, never lost
            outcome, error = None, exc
        if not loop.is_closed():
            loop.call_soon_threadsafe(_settle, outcome, error)

    # A daemon thread, not the loop's executor: a clone that hangs must not hold a gateway stop.
    spawn_context_thread(_run, name="gateway-start-plugin-recovery").start()
    return future


async def recover_plugins_for_start() -> tuple:
    """Gateway start: run the plugin recovery beside the start instead of ahead of it, so an install (or a
    wait on another installer's lock) never holds back the platforms that are ready. ``(installed, rest)``:
    ``installed`` True when it finished within :data:`_RECOVERY_HEAD_START_SECS` and installed a platform
    plugin (the runner's config must be re-read); ``rest`` the recovery still running, else None (hand it
    to :func:`start_recovered_platforms`)."""
    recovery = _recover_beside_start()
    done, _pending = await asyncio.wait({recovery}, timeout=_RECOVERY_HEAD_START_SECS)
    if done:
        return bool(recovery.result()), None
    logger.info("Plugin install for this profile's platforms continues beside the start; "
                "the platforms it brings start when it lands")
    return False, recovery


def start_recovered_platforms(runner: Any, recovery: Optional[asyncio.Future]) -> None:
    """Once the runner runs: when the still-running recovery installs a platform plugin, reconcile the
    launch profile, which starts the platform the plugin registers (a runner not running ignores it)."""
    if recovery is None:
        return

    def _done(future: asyncio.Future) -> None:
        if future.cancelled():
            return
        if future.exception() is not None:
            logger.warning("gateway-start plugin recovery failed: %s", future.exception())
        elif future.result():
            runner._schedule_platform_reconcile(None, "plugin-installed")
    recovery.add_done_callback(_done)
