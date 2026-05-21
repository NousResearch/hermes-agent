"""Worker lane registry for Kanban dispatcher assignee resolution.

The registry is intentionally small: it maps a Kanban ``assignee`` string
to a trusted spawn function.  The dispatcher still owns task lifecycle,
claiming, PID tracking, stale-heartbeat handling, max-runtime enforcement,
and retry accounting.  Worker lanes only decide how a claimed task's worker
process is started.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SpawnFn = Callable[..., Optional[int]]

_LANE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_REGISTRY_LOCK = threading.RLock()
_WORKER_LANES: dict[str, "WorkerLane"] = {}

_ALLOWED_LANE_TYPES = {"codex_cli"}
_ALLOWED_CODEX_MODELS = {
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2",
}
_ALLOWED_SANDBOXES = {"read-only", "workspace-write"}
_ALLOWED_APPROVALS = {"never", "on-request", "on-failure", "untrusted"}
_ALLOWED_SUCCESS_POLICIES = {"block_for_review"}
_MAX_LANE_CONCURRENCY = 8
_FORBIDDEN_REQUEST_KEYS = {"command", "cmd", "shell", "argv", "executable"}


@dataclass
class WorkerLane:
    """A trusted external worker lane registered with Kanban."""

    name: str
    kind: str
    description: str
    spawn_fn: SpawnFn
    success_policy: str = "block_for_review"
    max_concurrency: Optional[int] = None
    source: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = normalize_lane_name(self.name)
        if not self.kind or not str(self.kind).strip():
            raise ValueError("worker lane kind is required")
        self.kind = str(self.kind).strip()
        self.description = str(self.description or "")
        if not callable(self.spawn_fn):
            raise ValueError("worker lane spawn_fn must be callable")
        self.success_policy = str(self.success_policy or "block_for_review").strip()
        if not self.success_policy:
            raise ValueError("worker lane success_policy is required")
        if self.max_concurrency is not None:
            mc = int(self.max_concurrency)
            if mc < 1:
                raise ValueError("worker lane max_concurrency must be >= 1")
            self.max_concurrency = mc


@dataclass(frozen=True)
class WorkerAssigneeResolution:
    """Result of resolving a Kanban assignee."""

    name: str
    kind: str
    lane: Optional[WorkerLane] = None
    profile: Optional[str] = None

    @property
    def spawnable(self) -> bool:
        return self.kind in {"worker_lane", "hermes_profile"}


def normalize_lane_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    lane = name.strip().lower()
    if not lane:
        raise ValueError("worker lane name cannot be empty")
    if not _LANE_ID_RE.match(lane):
        raise ValueError(
            f"invalid worker lane name {name!r}: must match "
            "[a-z0-9][a-z0-9_-]{0,63}"
        )
    return lane


def register_worker_lane(lane: WorkerLane, *, replace: bool = False) -> WorkerLane:
    """Register ``lane`` by name.

    Duplicate names are rejected by default.  Internal config refreshes may
    pass ``replace=True`` after clearing prior config-sourced lanes.
    """
    if not isinstance(lane, WorkerLane):
        raise TypeError("register_worker_lane expects a WorkerLane instance")
    name = normalize_lane_name(lane.name)
    with _REGISTRY_LOCK:
        existing = _WORKER_LANES.get(name)
        if existing is not None and not replace:
            raise ValueError(
                f"worker lane {name!r} already registered "
                f"(kind={existing.kind}, source={existing.source or 'unknown'})"
            )
        _WORKER_LANES[name] = lane
    logger.debug("Registered worker lane %s (kind=%s source=%s)", name, lane.kind, lane.source)
    return lane


def get_worker_lane(name: str) -> Optional[WorkerLane]:
    try:
        key = normalize_lane_name(name)
    except ValueError:
        return None
    with _REGISTRY_LOCK:
        return _WORKER_LANES.get(key)


def list_worker_lanes() -> list[WorkerLane]:
    with _REGISTRY_LOCK:
        return [_WORKER_LANES[name] for name in sorted(_WORKER_LANES)]


def clear_worker_lanes(
    *,
    source: Optional[str] = None,
    source_prefix: Optional[str] = None,
) -> None:
    """Clear registered lanes.

    This is primarily used by tests and plugin/config rediscovery.  With no
    filters it clears the whole registry.
    """
    with _REGISTRY_LOCK:
        if source is None and source_prefix is None:
            _WORKER_LANES.clear()
            return
        for name, lane in list(_WORKER_LANES.items()):
            if source is not None and lane.source == source:
                _WORKER_LANES.pop(name, None)
            elif source_prefix is not None and lane.source.startswith(source_prefix):
                _WORKER_LANES.pop(name, None)


def is_worker_lane_assignee(name: str) -> bool:
    return get_worker_lane(name) is not None


def resolve_worker_assignee(
    name: str,
    *,
    refresh_config: bool = True,
) -> WorkerAssigneeResolution:
    """Resolve an assignee in dispatcher order.

    Order:
      1. registered worker lane
      2. Hermes profile
      3. skipped_nonspawnable
    """
    lane_name = normalize_lane_name(name)
    if refresh_config:
        register_configured_worker_lanes()
    lane = get_worker_lane(lane_name)
    if lane is not None:
        return WorkerAssigneeResolution(name=lane_name, kind="worker_lane", lane=lane)

    try:
        from hermes_cli.profiles import profile_exists
        if profile_exists(lane_name):
            return WorkerAssigneeResolution(
                name=lane_name,
                kind="hermes_profile",
                profile=lane_name,
            )
    except Exception:
        # Preserve the pre-registry degraded behavior: if profile discovery
        # itself is unavailable, assume the assignee is spawnable and let the
        # existing Hermes profile spawn path surface any concrete error.
        return WorkerAssigneeResolution(
            name=lane_name,
            kind="hermes_profile",
            profile=lane_name,
        )
    return WorkerAssigneeResolution(name=lane_name, kind="skipped_nonspawnable")


def validate_worker_lane_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate a skill-generated lane request without executing it.

    The output is a sanitized config fragment that may be registered only by a
    trusted control path.  Arbitrary shell commands are not accepted.
    """
    if not isinstance(request, dict):
        raise ValueError("worker_lane_request must be an object")
    forbidden = sorted(_FORBIDDEN_REQUEST_KEYS.intersection(request.keys()))
    if forbidden:
        raise ValueError(
            "worker_lane_request may not include executable command fields: "
            + ", ".join(forbidden)
        )

    lane_type = str(request.get("type") or "").strip()
    if lane_type not in _ALLOWED_LANE_TYPES:
        raise ValueError(
            f"worker lane type {lane_type!r} is not allowed; "
            f"allowed: {sorted(_ALLOWED_LANE_TYPES)}"
        )

    name = normalize_lane_name(str(request.get("name") or ""))
    model = request.get("model")
    if model is not None:
        model = str(model).strip()
        if model not in _ALLOWED_CODEX_MODELS:
            raise ValueError(
                f"model {model!r} is not allowed; allowed: {sorted(_ALLOWED_CODEX_MODELS)}"
            )

    sandbox = str(request.get("sandbox") or "workspace-write").strip()
    if sandbox not in _ALLOWED_SANDBOXES:
        raise ValueError(
            f"sandbox {sandbox!r} is not allowed; allowed: {sorted(_ALLOWED_SANDBOXES)}"
        )

    approval = str(request.get("approval") or "never").strip()
    if approval not in _ALLOWED_APPROVALS:
        raise ValueError(
            f"approval {approval!r} is not allowed; allowed: {sorted(_ALLOWED_APPROVALS)}"
        )

    success_policy = str(request.get("success_policy") or "block_for_review").strip()
    if success_policy not in _ALLOWED_SUCCESS_POLICIES:
        raise ValueError(
            f"success_policy {success_policy!r} is not allowed; "
            f"allowed: {sorted(_ALLOWED_SUCCESS_POLICIES)}"
        )

    raw_max = request.get("max_concurrency", 1)
    max_concurrency = 1 if raw_max is None else int(raw_max)
    if max_concurrency < 1 or max_concurrency > _MAX_LANE_CONCURRENCY:
        raise ValueError(
            f"max_concurrency must be between 1 and {_MAX_LANE_CONCURRENCY}"
        )

    out: dict[str, Any] = {
        "name": name,
        "type": lane_type,
        "sandbox": sandbox,
        "approval": approval,
        "max_concurrency": max_concurrency,
        "success_policy": success_policy,
    }
    if model:
        out["model"] = model
    if request.get("reason"):
        out["reason"] = str(request["reason"])
    if request.get("timeout_seconds") is not None:
        timeout = int(request["timeout_seconds"])
        if timeout < 1:
            raise ValueError("timeout_seconds must be >= 1")
        out["timeout_seconds"] = timeout
    return out


def _lane_from_validated_config(
    config: dict[str, Any],
    *,
    source: str,
) -> WorkerLane:
    lane_type = config["type"]
    if lane_type == "codex_cli":
        from hermes_cli.codex_worker import make_codex_worker_lane
        return make_codex_worker_lane(config, source=source)
    raise ValueError(f"unsupported worker lane type {lane_type!r}")


def enable_worker_lane_request(
    request: dict[str, Any],
    *,
    persist: bool = False,
    replace: bool = False,
    config: Optional[dict[str, Any]] = None,
) -> WorkerLane:
    """Validate and enable a skill-generated worker lane request.

    Model output is accepted only as a request object. This helper runs the
    deterministic validator, instantiates the adapter from the sanitized
    config, registers it in the trusted in-process registry, and optionally
    persists the sanitized config under ``kanban.worker_lanes``.
    """
    valid = validate_worker_lane_request(request)
    source = "config" if persist else "lane_request"
    lane = _lane_from_validated_config(valid, source=source)
    existing = get_worker_lane(lane.name)
    if existing is not None and not replace:
        raise ValueError(
            f"worker lane {lane.name!r} already registered "
            f"(kind={existing.kind}, source={existing.source or 'unknown'})"
        )
    if persist:
        try:
            from hermes_cli.config import load_config, save_config

            cfg = load_config() if config is None else config
            kanban_cfg = cfg.setdefault("kanban", {})
            if not isinstance(kanban_cfg, dict):
                raise ValueError("config key 'kanban' must be a mapping")
            lanes_cfg = kanban_cfg.setdefault("worker_lanes", {})
            if not isinstance(lanes_cfg, dict):
                raise ValueError("config key 'kanban.worker_lanes' must be a mapping")
            existing_cfg = lanes_cfg.get(lane.name)
            if existing_cfg is not None and not replace:
                raise ValueError(f"worker lane {lane.name!r} already exists in config")
            stored = {k: v for k, v in valid.items() if k not in {"name", "reason"}}
            lanes_cfg[lane.name] = stored
            if config is None:
                save_config(cfg)
        except Exception:
            # Do not leave a memory-only lane enabled when the caller asked
            # for persistence and config write/validation failed.
            raise
    return register_worker_lane(lane, replace=replace)


def register_configured_worker_lanes(config: Optional[dict[str, Any]] = None) -> None:
    """Register lanes from ``kanban.worker_lanes`` in config.yaml.

    Invalid config entries are skipped and logged.  Existing plugin lanes keep
    their names; a config lane cannot silently override a plugin lane.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception as exc:
            logger.debug("Could not load config for worker lanes: %s", exc)
            config = {}
    lanes_cfg = ((config or {}).get("kanban") or {}).get("worker_lanes") or {}
    if not isinstance(lanes_cfg, dict):
        logger.warning("kanban.worker_lanes must be a mapping; got %s", type(lanes_cfg).__name__)
        lanes_cfg = {}

    with _REGISTRY_LOCK:
        config_names: set[str] = set()
        for name, raw in lanes_cfg.items():
            if not isinstance(raw, dict):
                continue
            try:
                config_names.add(normalize_lane_name(name))
            except ValueError:
                # The per-entry registration loop below logs the concrete
                # validation error; keep cleanup best-effort here so one bad
                # key cannot disable every configured lane.
                continue
        for name, lane in list(_WORKER_LANES.items()):
            if lane.source == "config" and name not in config_names:
                _WORKER_LANES.pop(name, None)

    for raw_name, raw_cfg in lanes_cfg.items():
        if not isinstance(raw_cfg, dict):
            logger.warning("Skipping worker lane %r: config must be a mapping", raw_name)
            continue
        req = dict(raw_cfg)
        req.setdefault("name", raw_name)
        try:
            valid = validate_worker_lane_request(req)
            lane = _lane_from_validated_config(valid, source="config")

            existing = get_worker_lane(lane.name)
            if existing is not None and existing.source != "config":
                logger.warning(
                    "Skipping configured worker lane %s: name already registered by %s",
                    lane.name,
                    existing.source or existing.kind,
                )
                continue
            register_worker_lane(lane, replace=True)
        except Exception as exc:
            logger.warning("Skipping worker lane %r: %s", raw_name, exc)
