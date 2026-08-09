"""CronScheduler provider interface (Axis B — the trigger).

⚠️ EXPERIMENTAL — this interface is validated by exactly ONE consumer (the
built-in) until an external provider (Chronos, Phase 4) shakes it out. Until
then the module path, method signatures, and start() kwargs MAY change without
a deprecation cycle. Once a second provider validates the shape it becomes
stable. Any growth MUST be additive (new optional method with a default), never
a changed signature on start() or a new abstractmethod.

A CronScheduler decides *when* a due job fires. It does NOT decide what firing
means: execution + delivery stay in cron.scheduler.run_job / _deliver_result,
shared by all providers. Providers must never reimplement agent construction or
delivery.

The built-in InProcessCronScheduler runs the historical 60s daemon-thread
ticker. Alternative providers (e.g. Chronos, a NAS-mediated managed-cron
provider for scale-to-zero deployments) live under plugins/cron_providers/<name>/ and are
selected via the `cron.provider` config key (empty = built-in).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any


class CronScheduler(ABC):
    """Axis-B trigger provider. Decides WHEN a due cron job fires.

    Required surface is intentionally minimal: ``name`` + ``start``. ``stop``
    and ``is_available`` carry safe defaults. The three Phase-4 hooks
    (``on_jobs_changed`` / ``fire_due`` / ``reconcile``) are added later as
    NON-abstract methods so the built-in keeps satisfying the ABC without
    overriding them — see ``test_abc_growth_stays_additive``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'builtin', 'chronos'."""

    def is_available(self) -> bool:
        """Whether this provider can run in the current environment.

        MUST NOT make network calls. The built-in is always available; an
        external provider checks for configured endpoint/credentials. When a
        named provider returns False, the resolver falls back to the built-in.
        """
        return True

    @abstractmethod
    def start(
        self,
        stop_event: threading.Event,
        *,
        adapters: Any = None,
        loop: Any = None,
        interval: int = 60,
    ) -> None:
        """Begin firing due jobs.

        For the built-in this BLOCKS in the 60s loop until stop_event is set
        (it is run inside a daemon thread by the caller, exactly as today).
        An external provider may register a schedule/webhook and return
        immediately; in that case it must still honor stop_event for teardown.
        """

    def stop(self) -> None:
        """Optional eager teardown hook. Default no-op; setting the stop_event
        is the primary stop signal. Override for providers holding external
        resources (queue consumers, HTTP servers)."""
        return None

    # --- Optional hooks for external providers (added Phase 4). --------------
    # All default-safe so the built-in inherits working behavior without
    # overriding. Keep these NON-abstract — see test_abc_growth_stays_additive.

    def on_jobs_changed(self) -> None:
        """Called after a successful store mutation (create/update/remove/
        pause/resume). External providers reconcile their registry here (e.g.
        Chronos re-provisions/cancels the affected one-shot via NAS).
        Built-in: no-op (it re-reads jobs.json on every tick)."""
        return None

    def register_job(self, job: dict[str, Any]) -> None:
        """Register the first external trigger for one newly persisted job.

        The built-in provider reads the local store on every tick, so its
        default is a no-op. External providers override this when creating a
        job requires a remote registration before callers can honestly report
        that the job is scheduled.
        """
        return None

    def recover_interrupted(self) -> int:
        """Run profile-local attempt recovery for every provider lifecycle."""
        from cron.executions import recover_interrupted_executions

        return recover_interrupted_executions()

    def resume_pending_deliveries(self, *, adapters=None, loop=None) -> int:
        """Resume only durable agent-complete deliveries that never began."""
        from cron.scheduler import resume_pending_contextual_deliveries

        return resume_pending_contextual_deliveries(adapters=adapters, loop=loop)

    def fire_due(self, job_id: str, *, adapters: Any = None, loop: Any = None) -> bool:
        """Run a single job NOW via the existing external-provider path.

        Contextual jobs cannot be created while an external provider is
        configured, so this compatibility path remains isolated-only in V1.
        """
        from cron.jobs import claim_job_for_fire, get_job
        from cron.executions import create_execution
        from cron.scheduler import run_one_job

        if not claim_job_for_fire(job_id):
            return False
        job = get_job(job_id)
        if job is None:
            return False
        job["execution_id"] = create_execution(job_id, source=self.name)["id"]
        return run_one_job(job, adapters=adapters, loop=loop)

    def reconcile(self) -> None:
        """Converge the external registry toward jobs.json (the desired state):
        arm missing one-shots, cancel orphaned ones, re-arm changed times.
        Built-in: no-op."""
        return None


def resolve_cron_scheduler() -> "CronScheduler":
    """Return the active cron scheduler provider.

    Reads ``cron.provider`` from config. Empty/absent → built-in. A named
    provider that is missing, fails to load, or reports ``is_available() ==
    False`` falls back to the built-in with a warning — cron must never be left
    without a trigger.
    """
    import logging

    logger = logging.getLogger("cron.scheduler_provider")

    name = ""
    try:
        from hermes_cli.config import cfg_get, load_config
        name = (cfg_get(load_config(), "cron", "provider", default="") or "").strip()
    except Exception:
        pass

    if not name or name in ("builtin", "in-process", "inprocess"):
        return InProcessCronScheduler()

    try:
        from plugins.cron_providers import load_cron_scheduler
        provider = load_cron_scheduler(name)
        if provider is None:
            logger.warning("cron.provider '%s' not found; using built-in ticker", name)
            return InProcessCronScheduler()
        if not provider.is_available():
            logger.warning("cron.provider '%s' not available; using built-in ticker", name)
            return InProcessCronScheduler()
        logger.info("Using cron scheduler provider: %s", provider.name)
        return provider
    except Exception as e:
        logger.warning(
            "Failed to load cron.provider '%s' (%s); using built-in ticker", name, e
        )
        return InProcessCronScheduler()


class InProcessCronScheduler(CronScheduler):
    """Default provider: the historical in-process 60s ticker.

    ``start()`` blocks in the tick loop until ``stop_event`` is set, identical
    to the pre-refactor ``_start_cron_ticker`` core loop. The caller runs it in
    a daemon thread. ``can_dispatch`` is an optional synchronous gate supplied
    by GatewayRunner during external drain; skipped ticks leave due jobs intact
    for the next allowed tick.
    """

    @property
    def name(self) -> str:
        return "builtin"

    def start(
        self,
        stop_event,
        *,
        adapters=None,
        loop=None,
        interval=60,
        can_dispatch=None,
        profile_homes=None,
        profile_adapters=None,
        contextual_transcript_recover=None,
    ):
        import logging
        from cron.scheduler import tick as cron_tick
        from cron.jobs import (
            clear_ticker_error,
            record_ticker_error,
            record_ticker_heartbeat,
        )

        logger = logging.getLogger("cron.scheduler_provider")
        logger.info("In-process cron scheduler started (interval=%ds)", interval)

        # ── Multiplex profiles ────────────────────────────────────────────
        # When profile_homes is set (multiplex_profiles on), tick EACH profile's
        # cron store on every tick cycle so secondary-profile jobs actually fire
        # instead of languishing in a store no ticker owns (#69377). Without this,
        # only the process-global HERMES_HOME (the default profile) is ticked.
        # Heartbeats and recovery are also scoped per profile so `hermes cron
        # status` reflects liveness for every profile independently.
        if profile_homes:
            self._start_multiplex(
                stop_event,
                profile_homes=profile_homes,
                profile_adapters=profile_adapters,
                adapters=adapters,
                loop=loop,
                interval=interval,
                can_dispatch=can_dispatch,
                contextual_transcript_recover=contextual_transcript_recover,
            )
            return

        # ── Single-profile (legacy) path ──────────────────────────────────
        recovered = self.recover_interrupted()
        if recovered:
            logger.warning(
                "Marked %d interrupted cron execution(s) unknown after restart",
                recovered,
            )
        if contextual_transcript_recover is not None:
            try:
                contextual_transcript_recover()
            except BaseException:
                logger.error(
                    "Contextual transcript startup recovery deferred",
                    exc_info=True,
                )
        resumed = self.resume_pending_deliveries(adapters=adapters, loop=loop)
        if resumed:
            logger.info("Resumed %d pending contextual cron delivery(s)", resumed)
        # Heartbeat once before the first sleep so `hermes cron status` sees a
        # live ticker immediately after startup, not only after the first tick.
        record_ticker_heartbeat()
        while not stop_event.is_set():
            ok = False
            try:
                if contextual_transcript_recover is not None:
                    contextual_transcript_recover()
                    self.resume_pending_deliveries(adapters=adapters, loop=loop)
                if can_dispatch is not None and not can_dispatch():
                    logger.debug("Cron dispatch paused while gateway drains existing work")
                else:
                    cron_tick(
                        verbose=False,
                        adapters=adapters,
                        loop=loop,
                        sync=False,
                        can_dispatch=can_dispatch,
                    )
                ok = True
            except BaseException as e:
                # Catch BaseException (not just Exception) so a SystemExit from
                # a misbehaving provider SDK / agent retry path does not kill
                # the ticker thread silently (#32612). KeyboardInterrupt is
                # intentionally caught here too — gateway shutdown is driven by
                # stop_event (set by the main thread's signal handler), not by
                # an exception in this daemon thread, so swallowing it and
                # re-checking stop_event keeps shutdown clean.
                logger.error("Cron tick error: %s", e, exc_info=True)
                # Persist the failure reason next to the heartbeat markers so
                # `hermes cron status`/`list` (separate processes) can show
                # WHY ticks fail, not just that the success marker is stale —
                # e.g. a root-rewritten jobs.json locking out the ticker's
                # uid went unnoticed for ~14h with the reason buried in the
                # gateway log (#68483).
                record_ticker_error(f"{type(e).__name__}: {e}")
            # Record liveness every iteration; bump the success marker only on a
            # clean tick, so status can tell "alive but failing every tick" from
            # "actually firing jobs" (#32612, #32895).
            record_ticker_heartbeat(success=ok)
            if ok:
                clear_ticker_error()
            stop_event.wait(interval)

    def _start_multiplex(
        self,
        stop_event,
        *,
        profile_homes,
        profile_adapters=None,
        adapters=None,
        loop=None,
        interval=60,
        can_dispatch=None,
        contextual_transcript_recover=None,
    ):
        """Tick every served profile's cron store when multiplex_profiles is on.

        Each profile uses ``set_hermes_home_override()`` + ``use_cron_store()``
        to scope its tick, heartbeat, recovery, lock file, config/.env, and
        agent execution to that profile's home — mirroring how
        ``_profile_runtime_scope`` scopes the multiplexed inbound path and
        ``web_server.py`` scopes per-profile cron API calls.
        """
        import logging
        from cron.scheduler import tick as cron_tick
        from cron.jobs import (
            clear_ticker_error,
            record_ticker_error,
            record_ticker_heartbeat,
            use_cron_store,
        )
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        logger = logging.getLogger("cron.scheduler_provider")
        logger.info(
            "Multiplex cron scheduler started for %d profile(s): %s",
            len(profile_homes),
            [p[0] if isinstance(p, tuple) else p for p in profile_homes],
        )

        adapter_maps = profile_adapters or {}
        if profile_adapters is None and adapters:
            logger.error(
                "Multiplex cron received only a process-wide adapter map; "
                "live delivery is disabled rather than risking cross-profile transport reuse"
            )

        def _entry_parts(entry):
            if isinstance(entry, tuple):
                return str(entry[0]), entry[1]
            return str(entry), entry

        # Recovery + initial heartbeat are isolated per profile. One corrupt
        # profile must not prevent later profiles from recovering or ticking.
        for entry in profile_homes:
            profile_name, home = _entry_parts(entry)
            live_adapters = adapter_maps.get(profile_name, {})
            home_token = set_hermes_home_override(str(home))
            try:
                with use_cron_store(home):
                    try:
                        recovered = self.recover_interrupted()
                        if recovered:
                            logger.warning(
                                "Marked %d interrupted cron execution(s) for profile %s",
                                recovered,
                                profile_name,
                            )
                        if contextual_transcript_recover is not None:
                            contextual_transcript_recover()
                        resumed = self.resume_pending_deliveries(
                            adapters=live_adapters,
                            loop=loop,
                        )
                        if resumed:
                            logger.info(
                                "Resumed %d pending contextual delivery(s) for profile %s",
                                resumed,
                                profile_name,
                            )
                        record_ticker_heartbeat(success=True)
                        clear_ticker_error()
                    except BaseException as exc:
                        logger.error(
                            "Cron recovery failed for profile %s: %s",
                            profile_name,
                            exc,
                            exc_info=True,
                        )
                        try:
                            record_ticker_heartbeat(success=False)
                            record_ticker_error(f"{type(exc).__name__}: {exc}")
                        except BaseException:
                            logger.error(
                                "Could not record cron recovery failure for profile %s",
                                profile_name,
                                exc_info=True,
                            )
            except BaseException as exc:
                logger.error(
                    "Could not open cron store for profile %s during recovery: %s",
                    profile_name,
                    exc,
                    exc_info=True,
                )
            finally:
                reset_hermes_home_override(home_token)

        while not stop_event.is_set():
            dispatch_allowed = can_dispatch is None or can_dispatch()
            if not dispatch_allowed:
                logger.debug("Cron dispatch paused while gateway drains existing work")

            for entry in profile_homes:
                profile_name, home = _entry_parts(entry)
                live_adapters = adapter_maps.get(profile_name, {})
                home_token = set_hermes_home_override(str(home))
                try:
                    with use_cron_store(home):
                        ok = False
                        tick_error = None
                        try:
                            if contextual_transcript_recover is not None:
                                contextual_transcript_recover()
                                self.resume_pending_deliveries(
                                    adapters=live_adapters,
                                    loop=loop,
                                )
                            if dispatch_allowed:
                                cron_tick(
                                    verbose=False,
                                    adapters=live_adapters,
                                    loop=loop,
                                    sync=False,
                                    can_dispatch=can_dispatch,
                                )
                            ok = True
                        except BaseException as exc:
                            tick_error = f"{type(exc).__name__}: {exc}"
                            logger.error(
                                "Cron tick failed for profile %s: %s",
                                profile_name,
                                exc,
                                exc_info=True,
                            )
                        try:
                            record_ticker_heartbeat(success=ok)
                            if ok:
                                clear_ticker_error()
                            elif tick_error:
                                record_ticker_error(tick_error)
                        except BaseException:
                            logger.error(
                                "Could not record cron heartbeat for profile %s",
                                profile_name,
                                exc_info=True,
                            )
                except BaseException as exc:
                    logger.error(
                        "Could not open cron store for profile %s during tick: %s",
                        profile_name,
                        exc,
                        exc_info=True,
                    )
                finally:
                    reset_hermes_home_override(home_token)
            stop_event.wait(interval)