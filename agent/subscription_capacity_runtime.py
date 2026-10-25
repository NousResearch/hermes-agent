"""Mandatory capacity guard for explicitly pinned Kanban subscription workers.

Admission happens in the dispatcher before ``Popen``.  A non-secret lease envelope is inherited by
the child, checked immediately before every provider request, and released by the turn facade (with
an ``atexit`` backstop).  Provider aliases are intentionally outside this integration.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent.account_usage import AccountUsageSnapshot, fetch_account_usage
from agent.subscription_capacity import (
    CapacityLease,
    CapacityRequest,
    ProviderPolicy,
    SubscriptionCapacityRouter,
)

CAPACITY_LEASE_ENV = "HERMES_SUBSCRIPTION_CAPACITY_LEASE"
_NATIVE_SUBSCRIPTION_PROVIDERS = frozenset({"openai-codex", "anthropic"})


class SubscriptionCapacityUnavailable(RuntimeError):
    """The selected subscription route cannot safely start or cross a model-call boundary."""


def is_native_subscription_task(task: Any) -> bool:
    return str(getattr(task, "provider_override", None) or "") in _NATIVE_SUBSCRIPTION_PROVIDERS


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 < parsed < 100 else None


def _capacity_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    kanban = config.get("kanban") if isinstance(config, Mapping) else None
    raw = kanban.get("subscription_capacity") if isinstance(kanban, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def _policy(provider: str, identity: str, config: Mapping[str, Any]) -> ProviderPolicy:
    tasks = frozenset({"builder", "reviewer", "repair", "analysis"})
    if provider == "openai-codex":
        return ProviderPolicy(
            provider, 10.0, tasks, ("gpt-", "codex-"), frozenset({"usage_api"}),
            expected_identity=identity,
        )
    if provider == "anthropic":
        floor = _positive_float(_capacity_config(config).get("claude_floor_percent"))
        if floor is None:
            raise SubscriptionCapacityUnavailable(
                "native Anthropic subscription workers require a configured positive "
                "kanban.subscription_capacity.claude_floor_percent"
            )
        return ProviderPolicy(
            provider, floor, tasks, ("claude-",), frozenset({"oauth_usage_api"}),
            expected_identity=identity,
        )
    raise SubscriptionCapacityUnavailable(f"unsupported native subscription provider {provider!r}")


def _lease_dict(lease: CapacityLease) -> dict[str, Any]:
    data = asdict(lease)
    data["expires_at"] = lease.expires_at.isoformat()
    return data


def _policy_dict(policy: ProviderPolicy) -> dict[str, Any]:
    data = asdict(policy)
    for key in ("eligible_tasks", "allowed_sources"):
        data[key] = sorted(data[key])
    data["model_prefixes"] = list(data["model_prefixes"])
    return data


def _policy_from_dict(data: Mapping[str, Any]) -> ProviderPolicy:
    return ProviderPolicy(
        provider=str(data["provider"]),
        protected_floor_percent=float(data["protected_floor_percent"]),
        eligible_tasks=frozenset(str(v) for v in data["eligible_tasks"]),
        model_prefixes=tuple(str(v) for v in data["model_prefixes"]),
        allowed_sources=frozenset(str(v) for v in data["allowed_sources"]),
        expected_identity=str(data["expected_identity"]),
        max_bounded_leases=int(data.get("max_bounded_leases", 1)),
    )


def _lease_from_dict(data: Mapping[str, Any]) -> CapacityLease:
    return CapacityLease(
        lease_id=str(data["lease_id"]), request_id=str(data["request_id"]),
        provider=str(data["provider"]), task=str(data["task"]), model=str(data["model"]),
        owner=str(data["owner"]), reserved_percent=data.get("reserved_percent"),
        expires_at=datetime.fromisoformat(str(data["expires_at"])),
    )


def _profile_scope(profile_home: Optional[str]):
    """Install the same profile/secret scope used by the Kanban worker launch."""
    from contextlib import contextmanager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import (
        build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope,
    )

    @contextmanager
    def scoped():
        home_token = set_hermes_home_override(profile_home) if profile_home else None
        secret_token = (
            set_secret_scope(build_profile_secret_scope(Path(profile_home)))
            if profile_home and is_multiplex_active() else None
        )
        try:
            yield
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
            if home_token is not None:
                reset_hermes_home_override(home_token)

    return scoped()


def prepare_kanban_worker_capacity(
    task: Any, *, shared_root: Path, profile_home: Optional[str], config: Optional[Mapping[str, Any]] = None,
    fetcher: Callable[[str], Optional[AccountUsageSnapshot]] = fetch_account_usage,
) -> Optional[str]:
    """Acquire before spawn and return the child envelope, or ``None`` for non-native routes."""
    provider = str(getattr(task, "provider_override", None) or "")
    model = str(getattr(task, "model_override", None) or "")
    if provider not in _NATIVE_SUBSCRIPTION_PROVIDERS:
        return None
    if not model:
        raise SubscriptionCapacityUnavailable("native subscription worker requires an explicit model")

    with _profile_scope(profile_home):
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        try:
            snapshot = fetcher(provider)
        except Exception as exc:
            raise SubscriptionCapacityUnavailable(f"subscription usage unavailable: {exc}") from exc
        identity = str(getattr(snapshot, "identity", "") or "").strip() if snapshot else ""
        if not identity:
            raise SubscriptionCapacityUnavailable("subscription account identity unavailable")
        policy = _policy(provider, identity, config)
        scope = f"{provider}:{identity}"
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
        db_path = Path(shared_root) / "kanban" / "subscription_capacity" / f"{digest}.sqlite3"
        max_runtime = int(getattr(task, "max_runtime_seconds", 0) or 0)
        ttl = timedelta(seconds=max(1800, max_runtime + 300))
        router = SubscriptionCapacityRouter(
            db_path, (policy,), coordination_scope=scope,
            fetcher=lambda candidate: snapshot if candidate == provider else None,
            lease_ttl=ttl,
        )
        run_id = getattr(task, "current_run_id", None)
        if run_id is None:
            raise SubscriptionCapacityUnavailable("claimed subscription worker has no run id")
        step = str(getattr(task, "current_step_key", None) or "").lower()
        task_role = (
            "repair" if "repair" in step else
            "reviewer" if "sdlc-review" in (getattr(task, "skills", None) or ()) else
            "builder"
        )
        owner = f"kanban:{getattr(task, 'id')}:{run_id}"
        result = router.acquire(
            CapacityRequest(owner, task_role, owner), candidate_models={provider: model},
        )
        if result.lease is None:
            reasons = "; ".join(a.reason for a in result.assessments) or "capacity denied"
            raise SubscriptionCapacityUnavailable(reasons)
        if not router.authorize_dispatcher_process(result.lease):
            # An idempotent replay may refer to a lease held by a still-live earlier dispatcher.
            # Never delete that durable reservation merely because this process lacks authority.
            raise SubscriptionCapacityUnavailable("capacity lease belongs to another dispatcher process")

    payload = {
        "version": 1, "db_path": str(db_path), "coordination_scope": scope,
        "policy": _policy_dict(policy), "lease": _lease_dict(result.lease),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def release_prepared_capacity(payload: Optional[str]) -> bool:
    """Release a dispatcher-acquired lease when the child could not be launched."""
    if not payload:
        return False
    state = json.loads(payload)
    lease = _lease_from_dict(state["lease"])
    router = SubscriptionCapacityRouter(
        Path(state["db_path"]), (_policy_from_dict(state["policy"]),),
        coordination_scope=str(state["coordination_scope"]),
    )
    return router.release(lease.lease_id, owner=lease.owner)


def bind_prepared_capacity(payload: Optional[str], worker_pid: int) -> bool:
    """Dispatcher-only handoff: bind an acquired lease to the exact PID returned by Popen."""
    if not payload:
        return False
    try:
        state = json.loads(payload)
        if state.get("version") != 1:
            return False
        lease = _lease_from_dict(state["lease"])
        router = SubscriptionCapacityRouter(
            Path(state["db_path"]), (_policy_from_dict(state["policy"]),),
            coordination_scope=str(state["coordination_scope"]),
        )
    except Exception:
        return False
    return router.register_worker_process(lease, worker_pid)


def release_recovered_kanban_capacity(shared_root: Path, task_id: str, run_id: Any) -> int:
    """Release an exact dead/finalized worker owner during dispatcher recovery.

    The coordination scope lives in each database's metadata, while the owner is globally stable
    for a claimed task run.  This bounded scan is used only by recovery paths and never admits work.
    """
    owner = f"kanban:{task_id}:{run_id}"
    directory = Path(shared_root) / "kanban" / "subscription_capacity"
    if not directory.is_dir():
        return 0
    import sqlite3
    released = 0
    for db_path in directory.glob("*.sqlite3"):
        try:
            with sqlite3.connect(db_path, timeout=2, isolation_level=None) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS subscription_capacity_bindings "
                    "(lease_id TEXT PRIMARY KEY, worker_pid INTEGER NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS subscription_capacity_authorities "
                    "(lease_id TEXT PRIMARY KEY, dispatcher_pid INTEGER NOT NULL)"
                )
                cursor = conn.execute(
                    "DELETE FROM subscription_capacity_leases WHERE owner = ?", (owner,),
                )
                conn.execute(
                    "DELETE FROM subscription_capacity_bindings WHERE lease_id NOT IN "
                    "(SELECT lease_id FROM subscription_capacity_leases)"
                )
                conn.execute(
                    "DELETE FROM subscription_capacity_authorities WHERE lease_id NOT IN "
                    "(SELECT lease_id FROM subscription_capacity_leases)"
                )
                conn.commit()
                released += max(0, cursor.rowcount)
        except (OSError, sqlite3.Error):
            continue
    return released


_runtime_lock = threading.Lock()
_runtime: Optional[tuple[SubscriptionCapacityRouter, CapacityLease]] = None
_runtime_released = False


def _load_runtime() -> Optional[tuple[SubscriptionCapacityRouter, CapacityLease]]:
    global _runtime
    payload = os.environ.get(CAPACITY_LEASE_ENV, "").strip()
    if not payload:
        return None
    with _runtime_lock:
        if _runtime is None:
            try:
                state = json.loads(payload)
                if state.get("version") != 1:
                    raise ValueError("unsupported capacity lease envelope")
                policy = _policy_from_dict(state["policy"])
                lease = _lease_from_dict(state["lease"])
                router = SubscriptionCapacityRouter(
                    Path(state["db_path"]), (policy,),
                    coordination_scope=str(state["coordination_scope"]),
                )
            except Exception as exc:
                raise SubscriptionCapacityUnavailable(f"invalid subscription capacity lease: {exc}") from exc
            _runtime = (router, lease)
        return _runtime


def _owns_worker_context(lease: CapacityLease) -> bool:
    from agent.delegation_context import is_dispatcher_owned_worker_context
    if not is_dispatcher_owned_worker_context():
        return False
    return (
        (os.environ.get("HERMES_KANBAN_TASK") or "") == lease.owner.split(":", 2)[1]
        and (os.environ.get("HERMES_KANBAN_RUN_ID") or "") == lease.owner.rsplit(":", 1)[1]
    )


def _verify_owning_process(router: SubscriptionCapacityRouter, lease: CapacityLease) -> None:
    if not _owns_worker_context(lease) or not router.worker_process_matches(lease, os.getpid()):
        raise SubscriptionCapacityUnavailable(
            "subscription capacity lease is not owned by this Kanban worker process"
        )


def check_worker_capacity_phase(agent: Any) -> None:
    """Fail before a consequential provider call unless the inherited lease remains valid."""
    runtime = _load_runtime()
    if runtime is None:
        return
    router, lease = runtime
    _verify_owning_process(router, lease)
    if str(getattr(agent, "provider", "") or "") != lease.provider or str(
        getattr(agent, "model", "") or ""
    ) != lease.model:
        raise SubscriptionCapacityUnavailable(
            "guarded worker route changed from its admitted subscription provider/model"
        )
    assessment = router.check_phase(lease)
    if not assessment.admitted:
        raise SubscriptionCapacityUnavailable(f"subscription capacity phase denied: {assessment.reason}")


def release_worker_capacity() -> bool:
    """Idempotently release the child lease on every turn/process finalization path."""
    global _runtime_released
    runtime = _load_runtime()
    if runtime is None:
        return False
    router, lease = runtime
    if not _owns_worker_context(lease) or not router.worker_process_matches(lease, os.getpid()):
        return False
    with _runtime_lock:
        if _runtime_released:
            return False
        released = router.release(lease.lease_id, owner=lease.owner)
        _runtime_released = True
        return released


atexit.register(release_worker_capacity)
