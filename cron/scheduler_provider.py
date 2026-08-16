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

import inspect
import threading
from abc import ABC, abstractmethod
from typing import Any

# Cap for the exponential tick backoff applied while consecutive ticks fail
# with fd exhaustion (EMFILE/ENFILE, #87644).  Base is the tick interval
# (60s by default); each consecutive EMFILE failure doubles the wait, capped
# here so a still-alive-but-exhausted gateway never sleeps longer than this
# between recovery attempts.
_EMFILE_BACKOFF_MAX_SECONDS = 15 * 60  # 15 minutes


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

    @property
    def supports_force_fire(self) -> bool:
        """Whether ``fire_due`` accepts the additive ``force`` keyword.

        Signature detection keeps providers written before ``force`` was added
        source-compatible. Providers accepting ``**kwargs`` are compatible.
        """
        return provider_supports_force_fire(self)

    def fire_due(
        self,
        job_id: str,
        *,
        adapters: Any = None,
        loop: Any = None,
        force: bool = False,
    ) -> bool:
        """Run a single job NOW via the shared orchestrator. Called by the
        inbound fire webhook when an external scheduler signals a job is due.

        The default claims the job with a store-level compare-and-set
        (multi-machine at-most-once), then runs it via the shared
        ``run_one_job`` body. Built-in never calls this (it has its own tick
        loop); an external provider routes its inbound fire here.

        Returns True if THIS caller claimed and processed the attempt, even if
        the job itself failed. Returns False only if the claim was lost
        (another machine/retry won it) or the job no longer exists.
        """
        claimed_job = self.claim_fire(job_id, force=force)
        if claimed_job is None:
            return False
        return self.fire_claimed(claimed_job, adapters=adapters, loop=loop)

    def claim_fire(self, job_id: str, *, force: bool = False) -> dict | None:
        """Durably claim one fire and create its audit attempt before dispatch.

        Webhook transports call this synchronously before acknowledging the
        external scheduler, then pass the exact owner-bearing snapshot to
        ``fire_claimed`` in tracked background work.
        """
        from cron.executions import create_execution, finish_execution
        from cron.jobs import claim_job_for_fire

        execution = create_execution(job_id, source=self.name)
        claim_kwargs = {"return_job": True}
        if force:
            claim_kwargs["force"] = True
        try:
            claimed_job = claim_job_for_fire(job_id, **claim_kwargs)
        except BaseException as exc:
            finish_execution(
                execution["id"],
                success=False,
                error=f"Fire claim failed before dispatch: {type(exc).__name__}: {exc}",
            )
            raise
        if not isinstance(claimed_job, dict):
            finish_execution(
                execution["id"],
                success=False,
                error="Fire claim was not acquired",
            )
            return None
        claimed_job["execution_id"] = execution["id"]
        return claimed_job

    def fire_claimed(
        self,
        claimed_job: dict,
        *,
        adapters: Any = None,
        loop: Any = None,
        cancel_event: Any = None,
    ) -> bool:
        """Run an exact snapshot returned by ``claim_fire``.

        ``cancel_event``: optional transport-owned ``threading.Event`` (or
        compatible) that lets the caller stop this execution cooperatively
        — e.g. the dashboard lifespan drain signalling pending webhook
        fires before the event loop shuts down.
        """
        from cron.scheduler import run_one_job

        run_one_job(
            claimed_job,
            adapters=adapters,
            loop=loop,
            cancel_event=cancel_event,
        )
        return True

    def reconcile(self) -> None:
        """Converge the external registry toward jobs.json (the desired state):
        arm missing one-shots, cancel orphaned ones, re-arm changed times.
        Built-in: no-op."""
        return None


def provider_supports_force_fire(provider: Any) -> bool:
    """Return whether a provider can safely receive ``fire_due(force=...)``."""
    try:
        parameters = inspect.signature(provider.fire_due).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "force"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


def provider_supports_split_fire(provider: Any) -> bool:
    """Return whether a provider implements the two-phase fire contract.

    The webhook admission path uses ``claim_fire`` + ``fire_claimed`` so the
    202 response is backed by a durable, owner-fenced claim. A legacy
    third-party provider that overrides the documented single-phase
    ``fire_due`` hook (custom claim/re-arm/telemetry behavior) but inherits
    the base ``claim_fire`` must keep being driven through its own
    ``fire_due`` — silently routing around its override would drop that
    behavior. Providers that customize ``claim_fire`` itself are already
    split-aware and keep the two-phase path.
    """
    cls = type(provider)
    fire_due_impl = getattr(cls, "fire_due", None)
    claim_fire_impl = getattr(cls, "claim_fire", None)
    fire_claimed_impl = getattr(cls, "fire_claimed", None)
    if claim_fire_impl is not None and claim_fire_impl is not CronScheduler.claim_fire:
        return True
    # Overriding the second phase is also proof of split-awareness (the
    # provider composes with the inherited claim path) — e.g. Chronos keeps
    # its re-arm logic in ``fire_claimed`` only.
    if fire_claimed_impl is not None and fire_claimed_impl is not CronScheduler.fire_claimed:
        return True
    if fire_due_impl is None or fire_due_impl is CronScheduler.fire_due:
        return True
    return False


def provider_supports_fire_cancel(provider: Any) -> bool:
    """Return whether ``fire_claimed`` accepts a ``cancel_event`` kwarg."""
    try:
        parameters = inspect.signature(provider.fire_claimed).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "cancel_event"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


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


def scheduler_for_profile_mode(
    provider: "CronScheduler", *, multiplex_profiles: bool
) -> "CronScheduler":
    """Return a scheduler that can safely serve the gateway's profile mode.

    External providers currently own one unscoped remote registry/client and
    therefore cannot safely reconcile several profile stores from one process.
    Fail closed to the built-in multiplex ticker until the provider API carries
    explicit profile identity through lifecycle and webhook calls.
    """
    if not multiplex_profiles or isinstance(provider, InProcessCronScheduler):
        return provider

    import logging

    logging.getLogger("cron.scheduler_provider").warning(
        "cron.provider '%s' does not support multiplex_profiles; using built-in ticker",
        provider.name,
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
    ):
        import logging
        from cron.scheduler import tick as cron_tick
        from cron.scheduler import (
            _is_fd_exhaustion,
            _reclaim_fds_best_effort,
        )
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
                adapters=adapters,
                loop=loop,
                interval=interval,
                can_dispatch=can_dispatch,
            )
            return

        # ── Single-profile (legacy) path ──────────────────────────────────
        recovered = self.recover_interrupted()
        if recovered:
            logger.warning(
                "Marked %d interrupted cron execution(s) unknown after restart",
                recovered,
            )
        # Heartbeat once before the first sleep so `hermes cron status` sees a
        # live ticker immediately after startup, not only after the first tick.
        record_ticker_heartbeat()
        # Exponential backoff for consecutive tick failures — most importantly
        # fd exhaustion (EMFILE/ENFILE, #87644).  While FDs stay exhausted the
        # ticker must NOT hammer the store every 60s; once they free (leak
        # fixed, reclamation ran) the next tick succeeds and the backoff
        # resets, so the scheduler self-heals without a gateway restart.
        consecutive_failures = 0
        while not stop_event.is_set():
            ok = False
            try:
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
                if _is_fd_exhaustion(e):
                    # EMFILE: try to free descriptors (gc.collect + raise the
                    # soft nofile limit) so the NEXT tick can succeed; back off
                    # exponentially so the exhausted process stops hammering
                    # the store while it has no chance of making progress
                    # (#87644).
                    _reclaim_fds_best_effort()
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            # Record liveness every iteration; bump the success marker only on a
            # clean tick, so status can tell "alive but failing every tick" from
            # "actually firing jobs" (#32612, #32895).
            record_ticker_heartbeat(success=ok)
            if ok:
                clear_ticker_error()
                consecutive_failures = 0
            wait_seconds = interval
            if consecutive_failures > 0:
                wait_seconds = min(
                    interval * (2 ** (consecutive_failures - 1)),
                    _EMFILE_BACKOFF_MAX_SECONDS,
                )
            stop_event.wait(wait_seconds)

    def _start_multiplex(
        self,
        stop_event,
        *,
        profile_homes,
        adapters=None,
        loop=None,
        interval=60,
        can_dispatch=None,
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
        from cron.scheduler import (
            _is_fd_exhaustion,
            _reclaim_fds_best_effort,
        )
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

        # Recovery + initial heartbeat for every profile.
        for entry in profile_homes:
            home = entry[1] if isinstance(entry, tuple) else entry
            home_token = set_hermes_home_override(str(home))
            try:
                with use_cron_store(home):
                    recovered = self.recover_interrupted()
                    if recovered:
                        logger.warning(
                            "Marked %d interrupted cron execution(s) for profile at %s",
                            recovered,
                            home,
                        )
                    record_ticker_heartbeat()
            finally:
                reset_hermes_home_override(home_token)

        consecutive_failures = 0
        while not stop_event.is_set():
            ok = False
            _tick_error = None
            try:
                if can_dispatch is not None and not can_dispatch():
                    logger.debug("Cron dispatch paused while gateway drains existing work")
                else:
                    for entry in profile_homes:
                        home = entry[1] if isinstance(entry, tuple) else entry
                        home_token = set_hermes_home_override(str(home))
                        try:
                            with use_cron_store(home):
                                cron_tick(
                                    verbose=False,
                                    adapters=adapters,
                                    loop=loop,
                                    sync=False,
                                    can_dispatch=can_dispatch,
                                )
                        finally:
                            reset_hermes_home_override(home_token)
                ok = True
            except BaseException as e:
                logger.error("Cron tick error: %s", e, exc_info=True)
                _tick_error = f"{type(e).__name__}: {e}"
                if _is_fd_exhaustion(e):
                    # EMFILE: attempt fd reclamation so the next tick can
                    # succeed, and back off exponentially (#87644).
                    _reclaim_fds_best_effort()
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            else:
                _tick_error = None
            # Record per-profile heartbeat after each tick cycle.
            for entry in profile_homes:
                home = entry[1] if isinstance(entry, tuple) else entry
                home_token = set_hermes_home_override(str(home))
                try:
                    with use_cron_store(home):
                        record_ticker_heartbeat(success=ok)
                        # Surface the failure reason (or clear it) per profile
                        # so `hermes cron status` can show WHY ticks fail
                        # (#68483).
                        if ok:
                            clear_ticker_error()
                        elif _tick_error:
                            record_ticker_error(_tick_error)
                finally:
                    reset_hermes_home_override(home_token)
            if ok:
                consecutive_failures = 0
            wait_seconds = interval
            if consecutive_failures > 0:
                wait_seconds = min(
                    interval * (2 ** (consecutive_failures - 1)),
                    _EMFILE_BACKOFF_MAX_SECONDS,
                )
            stop_event.wait(wait_seconds)
