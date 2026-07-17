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

import io
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any


logger = logging.getLogger("cron.scheduler_provider")

_CRON_SCHEDULER_RUNTIME_OWNERS = frozenset({"gateway", "desktop"})


def _read_scheduler_owner_config_strict() -> dict[str, Any] | None:
    """Read scheduler ownership without the normal tolerant config fallback."""
    from hermes_cli import managed_scope
    from hermes_cli.config import get_config_path
    from utils import fast_safe_load

    def _read_mapping(path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return {}
        meaningful = [
            line.split("#", 1)[0].strip()
            for line in raw.splitlines()
            if line.split("#", 1)[0].strip() not in {"", "---", "..."}
        ]
        if not meaningful:
            return {}
        parsed = fast_safe_load(io.StringIO(raw))
        if not isinstance(parsed, dict):
            raise ValueError("config root must be a mapping")
        return parsed

    try:
        user_config = _read_mapping(get_config_path())
        managed_dir = managed_scope.get_managed_dir()
        managed_config = (
            _read_mapping(managed_dir / "config.yaml")
            if managed_dir is not None
            else {}
        )
        user_cron = user_config.get("cron", {})
        managed_cron = managed_config.get("cron", {})
        if "cron" in user_config and not isinstance(user_cron, dict):
            raise ValueError("cron section must be a mapping")
        if "cron" in managed_config and not isinstance(managed_cron, dict):
            raise ValueError("managed cron section must be a mapping")
        effective_cron = dict(user_cron)
        effective_cron.update(managed_cron)
        return {"cron": effective_cron}
    except Exception:
        logger.error(
            "Unable to read a valid cron.scheduler_owner; scheduler startup disabled. "
            "Set it to auto, gateway, or desktop in config.yaml."
        )
        return None


def resolve_cron_scheduler_owner(*, config: dict[str, Any] | None = None) -> str | None:
    """Resolve the single automatic scheduler owner, failing closed on errors."""
    if config is None:
        config = _read_scheduler_owner_config_strict()
        if config is None:
            return None
    if not isinstance(config, dict):
        logger.error("Invalid configuration root; scheduler startup disabled.")
        return None
    cron_config = config.get("cron", {})
    if "cron" in config and not isinstance(cron_config, dict):
        logger.error("Invalid cron configuration shape; scheduler startup disabled.")
        return None
    raw_owner = cron_config.get("scheduler_owner", "auto")
    owner = raw_owner.strip().lower() if isinstance(raw_owner, str) else None
    if owner == "auto":
        return "gateway"
    if owner in _CRON_SCHEDULER_RUNTIME_OWNERS:
        return owner
    logger.error(
        "Invalid cron.scheduler_owner; scheduler startup disabled. "
        "Use auto, gateway, or desktop in config.yaml."
    )
    return None


def should_start_cron_scheduler(
    runtime_owner: str, *, config: dict[str, Any] | None = None
) -> bool:
    """Return whether this long-running runtime owns automatic dispatch."""
    if runtime_owner not in _CRON_SCHEDULER_RUNTIME_OWNERS:
        raise ValueError("Unknown cron scheduler runtime owner")
    return resolve_cron_scheduler_owner(config=config) == runtime_owner


def resolve_owned_cron_scheduler(
    runtime_owner: str, *, config: dict[str, Any] | None = None
) -> "CronScheduler | None":
    """Resolve the provider only after the shared ownership gate succeeds."""
    if not should_start_cron_scheduler(runtime_owner, config=config):
        return None
    return resolve_cron_scheduler()


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

    def fire_due(self, job_id: str, *, adapters: Any = None, loop: Any = None) -> bool:
        """Run a single job NOW via the shared orchestrator. Called by the
        inbound fire webhook when an external scheduler signals a job is due.

        The default claims the job with a store-level compare-and-set
        (multi-machine at-most-once), then runs it via the shared
        ``run_one_job`` body. Built-in never calls this (it has its own tick
        loop); an external provider routes its inbound fire here.

        Returns True if THIS caller claimed and ran the job, False if the claim
        was lost (another machine/retry won it) or the job no longer exists.
        """
        from cron.jobs import claim_job_for_fire, get_job
        from cron.scheduler import run_one_job

        if not claim_job_for_fire(job_id):
            return False  # another machine already claimed this fire
        job = get_job(job_id)
        if job is None:
            return False  # job removed (e.g. repeat-N exhausted) between arm and fire
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

    def start(self, stop_event, *, adapters=None, loop=None, interval=60, can_dispatch=None):
        import logging
        from cron.scheduler import tick as cron_tick
        from cron.jobs import record_ticker_heartbeat

        logger = logging.getLogger("cron.scheduler_provider")
        logger.info("In-process cron scheduler started (interval=%ds)", interval)
        # Heartbeat once before the first sleep so `hermes cron status` sees a
        # live ticker immediately after startup, not only after the first tick.
        record_ticker_heartbeat()
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
            # Record liveness every iteration; bump the success marker only on a
            # clean tick, so status can tell "alive but failing every tick" from
            # "actually firing jobs" (#32612, #32895).
            record_ticker_heartbeat(success=ok)
            stop_event.wait(interval)
