"""Unified policy, lease, provider, and drain lifecycle for cron runtimes."""
from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, cast

from cron.scheduler_lease import SchedulerOwnershipLease

logger = logging.getLogger("cron.scheduler_runtime")
RuntimeOwner = Literal["gateway", "desktop"]
OwnershipMode = Literal["auto", "gateway", "desktop"]
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUILTIN_NAMES = frozenset({"", "builtin", "in-process", "inprocess"})
_ACTIVE_PROVIDERS_LOCK = threading.Condition()
_NEXT_GENERATION = 0


@dataclass
class _ActiveProviderGeneration:
    provider: Any
    generation: int
    accepting: bool = True
    reservations: int = 0


_ACTIVE_PROVIDERS: dict[str, _ActiveProviderGeneration] = {}


@dataclass(frozen=True)
class SchedulerOwnershipPolicy:
    mode: OwnershipMode
    configured_provider: str


def _read_mapping(path: Path) -> dict[str, Any]:
    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate mapping key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
    )

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    meaningful = [
        line.split("#", 1)[0].strip()
        for line in raw.splitlines()
        if line.split("#", 1)[0].strip() not in {"", "---", "..."}
    ]
    if not meaningful:
        return {}
    parsed = yaml.load(raw, Loader=_UniqueKeyLoader)
    if not isinstance(parsed, dict):
        raise ValueError("config root must be a mapping")
    return parsed


def _read_effective_cron_config_strict() -> dict[str, Any] | None:
    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars, get_config_path

    try:
        user_config = cast(dict[str, Any], _expand_env_vars(_read_mapping(get_config_path())))
        managed_dir = managed_scope.get_managed_dir()
        managed_config = (
            cast(
                dict[str, Any],
                _expand_env_vars(_read_mapping(managed_dir / "config.yaml")),
            )
            if managed_dir is not None
            else {}
        )
        user_cron = user_config.get("cron", {})
        managed_cron = managed_config.get("cron", {})
        if "cron" in user_config and not isinstance(user_cron, dict):
            raise ValueError("cron section must be a mapping")
        if "cron" in managed_config and not isinstance(managed_cron, dict):
            raise ValueError("managed cron section must be a mapping")
        effective = dict(user_cron)
        effective.update(managed_cron)
        return {"cron": effective}
    except Exception:
        logger.error(
            "Unable to read valid cron scheduler policy; automatic scheduler startup disabled."
        )
        return None


def read_scheduler_ownership_policy_strict(
    config: dict[str, Any] | None = None,
) -> SchedulerOwnershipPolicy | None:
    """Read ownership and provider together, failing closed on malformed input."""
    if config is None:
        config = _read_effective_cron_config_strict()
        if config is None:
            return None
    if not isinstance(config, dict):
        logger.error("Invalid configuration root; scheduler startup disabled.")
        return None
    cron = config.get("cron", {})
    if "cron" in config and not isinstance(cron, dict):
        logger.error("Invalid cron configuration shape; scheduler startup disabled.")
        return None

    raw_mode = cron.get("scheduler_owner", "auto")
    mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else ""
    if mode not in {"auto", "gateway", "desktop"}:
        logger.error(
            "Invalid cron.scheduler_owner; scheduler startup disabled. Use auto, gateway, or desktop."
        )
        return None

    raw_provider = cron.get("provider", "")
    if raw_provider is None:
        raw_provider = ""
    if not isinstance(raw_provider, str):
        logger.error("Invalid cron.provider; automatic scheduler startup disabled.")
        return None
    provider = raw_provider.strip().lower()
    if provider in _BUILTIN_NAMES:
        provider = "builtin"
    elif not _PROVIDER_NAME_RE.fullmatch(provider):
        logger.error("Invalid cron.provider; automatic scheduler startup disabled.")
        return None
    if mode == "desktop" and provider != "builtin":
        logger.error(
            "Invalid cron scheduler policy: external providers require Gateway ownership; scheduler startup disabled."
        )
        return None
    return SchedulerOwnershipPolicy(cast(OwnershipMode, mode), provider)


def scheduler_runtime_is_eligible(
    policy: SchedulerOwnershipPolicy,
    *,
    runtime: RuntimeOwner,
    same_home_gateway_running: bool,
) -> bool:
    if runtime not in {"gateway", "desktop"}:
        raise ValueError("Unknown cron scheduler runtime owner")
    if policy.configured_provider != "builtin" and runtime != "gateway":
        return False
    if policy.mode == "gateway":
        return runtime == "gateway"
    if policy.mode == "desktop":
        return runtime == "desktop"
    if runtime == "gateway":
        return True
    return policy.configured_provider == "builtin" and not same_home_gateway_running


def _home_key(home: Path | None = None) -> str:
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    return str(home.expanduser().resolve())


class SchedulerProviderReservation:
    """A callback reservation pinned to one accepting leased generation."""

    def __init__(self, state: _ActiveProviderGeneration) -> None:
        self._state = state
        self.provider = state.provider
        self.generation = state.generation
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        with _ACTIVE_PROVIDERS_LOCK:
            if self._released:
                return
            self._released = True
            self._state.reservations -= 1
            _ACTIVE_PROVIDERS_LOCK.notify_all()

    def __enter__(self) -> "SchedulerProviderReservation":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def reserve_active_scheduler_provider(
    *, hermes_home: Path | None = None
) -> SchedulerProviderReservation | None:
    """Reserve the currently accepting leased generation, or fail closed."""
    with _ACTIVE_PROVIDERS_LOCK:
        state = _ACTIVE_PROVIDERS.get(_home_key(hermes_home))
        if state is None or not state.accepting:
            return None
        state.reservations += 1
        return SchedulerProviderReservation(state)


@contextlib.contextmanager
def reserved_active_scheduler_provider(
    *, hermes_home: Path | None = None
) -> Iterator[SchedulerProviderReservation | None]:
    reservation = reserve_active_scheduler_provider(hermes_home=hermes_home)
    try:
        yield reservation
    finally:
        if reservation is not None:
            reservation.release()


def _publish_active_provider(home: Path, provider: Any) -> _ActiveProviderGeneration:
    global _NEXT_GENERATION
    key = _home_key(home)
    with _ACTIVE_PROVIDERS_LOCK:
        _NEXT_GENERATION += 1
        state = _ActiveProviderGeneration(provider=provider, generation=_NEXT_GENERATION)
        _ACTIVE_PROVIDERS[key] = state
        return state


def _stop_accepting_provider(
    home: Path, generation: int
) -> _ActiveProviderGeneration | None:
    key = _home_key(home)
    with _ACTIVE_PROVIDERS_LOCK:
        state = _ACTIVE_PROVIDERS.get(key)
        if state is None or state.generation != generation:
            return None
        state.accepting = False
        _ACTIVE_PROVIDERS.pop(key, None)
        _ACTIVE_PROVIDERS_LOCK.notify_all()
        return state


class OwnedSchedulerRuntime:
    """Blocking scheduler supervisor intended to run in one daemon thread."""

    def __init__(
        self,
        runtime_owner: RuntimeOwner,
        *,
        adapters: Any = None,
        loop: Any = None,
        can_dispatch: Callable[[], bool] | None = None,
        gateway_is_running: Callable[[], bool] | None = None,
        interval: int = 60,
        poll_interval: float = 0.5,
        drain_timeout: float = 65.0,
        hermes_home: Path | None = None,
    ) -> None:
        if runtime_owner not in {"gateway", "desktop"}:
            raise ValueError("Unknown cron scheduler runtime owner")
        self.runtime_owner: RuntimeOwner = cast(RuntimeOwner, runtime_owner)
        self.adapters = adapters
        self.loop = loop
        self.can_dispatch = can_dispatch
        self.gateway_is_running = gateway_is_running or (lambda: False)
        self.interval = interval
        self.poll_interval = poll_interval
        self.drain_timeout = drain_timeout
        self.hermes_home = Path(hermes_home) if hermes_home is not None else None
        self._active_provider: Any | None = None
        self._lease: SchedulerOwnershipLease | None = None
        self._provider_thread: threading.Thread | None = None
        self._provider_stop: threading.Event | None = None
        self._provider_failed = threading.Event()
        self._state_lock = threading.Lock()
        self._active_policy: SchedulerOwnershipPolicy | None = None
        self._active_generation: int | None = None

    @property
    def active_provider(self) -> Any | None:
        with self._state_lock:
            return self._active_provider

    def _current_home(self) -> Path:
        if self.hermes_home is not None:
            return self.hermes_home.expanduser().resolve()
        from hermes_constants import get_hermes_home

        return get_hermes_home().expanduser().resolve()

    def _gateway_running(self) -> bool:
        if self.runtime_owner != "desktop":
            return False
        try:
            return bool(self.gateway_is_running())
        except Exception:
            logger.exception("Unable to inspect gateway runtime lock; Desktop cron yields closed")
            return True

    def _eligible(self, policy: SchedulerOwnershipPolicy | None) -> bool:
        return policy is not None and scheduler_runtime_is_eligible(
            policy,
            runtime=self.runtime_owner,
            same_home_gateway_running=self._gateway_running(),
        )

    def _provider_target(self, provider: Any, stop_event: threading.Event) -> None:
        from cron.scheduler_provider import InProcessCronScheduler

        kwargs: dict[str, Any] = {
            "adapters": self.adapters,
            "loop": self.loop,
            "interval": self.interval,
        }
        if isinstance(provider, InProcessCronScheduler) and self.can_dispatch is not None:
            kwargs["can_dispatch"] = self.can_dispatch
        try:
            provider.start(stop_event, **kwargs)
        except BaseException:
            self._provider_failed.set()
            logger.exception("Cron scheduler provider failed")

    def _start_active(self, policy: SchedulerOwnershipPolicy, home: Path) -> bool:
        lease = SchedulerOwnershipLease.try_acquire(
            hermes_home=home,
            owner=self.runtime_owner,
            provider=policy.configured_provider,
        )
        if lease is None:
            return False

        # Close policy/gateway-presence check-to-acquire races before provider
        # construction can produce side effects.
        fresh = read_scheduler_ownership_policy_strict()
        if not self._eligible(fresh) or fresh != policy:
            lease.release()
            return False
        assert fresh is not None

        from cron.scheduler_provider import resolve_cron_scheduler_runtime_strict

        provider = resolve_cron_scheduler_runtime_strict(fresh.configured_provider)
        if provider is None:
            lease.release()
            return False

        provider_stop = threading.Event()
        self._provider_failed.clear()
        thread = threading.Thread(
            target=self._provider_target,
            args=(provider, provider_stop),
            daemon=True,
            name=f"{self.runtime_owner}-cron-provider",
        )
        with self._state_lock:
            self._lease = lease
            self._active_provider = provider
            self._provider_stop = provider_stop
            self._provider_thread = thread
            self._active_policy = fresh
        try:
            thread.start()
        except BaseException:
            with self._state_lock:
                self._lease = None
                self._active_provider = None
                self._provider_stop = None
                self._provider_thread = None
                self._active_policy = None
                self._active_generation = None
            lease.release()
            raise
        generation_state = _publish_active_provider(home, provider)
        with self._state_lock:
            self._active_generation = generation_state.generation
        logger.info(
            "%s acquired cron scheduler ownership (provider=%s)",
            self.runtime_owner.capitalize(),
            provider.name,
        )
        return True

    @staticmethod
    def _jobs_drained() -> bool:
        try:
            from cron.scheduler import get_running_job_ids

            return not get_running_job_ids()
        except Exception:
            logger.exception("Unable to inspect in-flight cron jobs; retaining scheduler lease")
            return False

    def _drain_active(self, stop_event: threading.Event) -> None:
        with self._state_lock:
            provider = self._active_provider
            provider_stop = self._provider_stop
            thread = self._provider_thread
            lease = self._lease
            generation = self._active_generation
        if provider is None or lease is None:
            return

        # Fence callbacks first: no new reservations may capture this generation
        # after provider shutdown begins.
        generation_state = (
            _stop_accepting_provider(self._current_home(), generation)
            if generation is not None
            else None
        )
        if provider_stop is not None:
            provider_stop.set()
        try:
            provider.stop()
        except BaseException:
            logger.exception("Cron scheduler provider stop() failed; retaining lease until drained")

        deadline = time.monotonic() + max(0.0, self.drain_timeout)
        warned = False
        while True:
            with _ACTIVE_PROVIDERS_LOCK:
                reservations_drained = (
                    generation_state is None or generation_state.reservations == 0
                )
            if (
                (thread is None or not thread.is_alive())
                and self._jobs_drained()
                and reservations_drained
            ):
                break
            if not warned and time.monotonic() >= deadline:
                warned = True
                logger.warning(
                    "Cron scheduler did not drain within %.0fs; retaining ownership until drain or process death",
                    self.drain_timeout,
                )
            # stop_event is already set during shutdown, so waiting on it would
            # busy-spin. The condition paces polling and wakes on reservation release.
            delay = min(self.poll_interval, 0.1) if warned else self.poll_interval
            with _ACTIVE_PROVIDERS_LOCK:
                _ACTIVE_PROVIDERS_LOCK.wait(timeout=delay)

        with self._state_lock:
            self._active_provider = None
            self._provider_stop = None
            self._provider_thread = None
            self._lease = None
            self._active_policy = None
            self._active_generation = None
        lease.release()
        logger.info("%s released cron scheduler ownership", self.runtime_owner.capitalize())

    def run(self, stop_event: threading.Event) -> None:
        """Supervise policy and ownership until stopped and fully drained."""
        home = self._current_home()
        while not stop_event.is_set():
            policy = read_scheduler_ownership_policy_strict()
            with self._state_lock:
                active = self._active_provider
                active_policy = self._active_policy
            if active is not None and (
                not self._eligible(policy)
                or policy != active_policy
                or self._provider_failed.is_set()
            ):
                self._drain_active(stop_event)
                if self._provider_failed.is_set():
                    stop_event.wait(self.poll_interval)
                continue
            if active is None and self._eligible(policy):
                assert policy is not None
                self._start_active(policy, home)
            stop_event.wait(self.poll_interval)

        self._drain_active(stop_event)
