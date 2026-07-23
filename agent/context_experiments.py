"""Session-scoped prompt context experiments.

Context experiments let operators A/B test different project-context files and
preloaded skills without changing the prompt mid-conversation. Assignment is
resolved once per session id and then reused, preserving prefix-cache stability
within that session.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

fcntl: Any | None
try:  # pragma: no cover - platform branch
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None

msvcrt: Any | None
try:  # pragma: no cover - platform branch
    import msvcrt as _msvcrt

    msvcrt = _msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

_STATE_LOCK = threading.RLock()
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ContextExperimentAssignment:
    """Resolved experiment arm for a session."""

    experiment_name: str
    arm_name: str
    arm: dict[str, Any]


def _safe_state_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("-", name.strip()).strip(".-")
    return safe or "context-experiment"


def _state_path(experiment_name: str) -> Path:
    return (
        get_hermes_home()
        / "context_experiments"
        / f"{_safe_state_name(experiment_name)}.json"
    )


@contextlib.contextmanager
def _locked_state(path: Path) -> Iterator[None]:
    """Serialize round-robin assignment across gateway/CLI processes."""

    with _STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_fd = None
        lock_held = False
        try:
            if fcntl is not None:
                lock_fd = open(lock_path, "a+", encoding="utf-8")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            elif msvcrt is not None:
                # msvcrt.locking() locks from the current file position and
                # requires that byte to already exist.  Seed the lock file
                # before reopening it in r+ mode so the lock always starts at
                # byte zero, including on the first use.
                with open(lock_path, "ab") as seed_fd:
                    if seed_fd.tell() == 0:
                        seed_fd.write(b"\x00")
                lock_fd = open(lock_path, "r+", encoding="utf-8")
                lock_fd.seek(0)
                getattr(msvcrt, "locking")(
                    lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1
                )
            else:
                raise OSError("context experiment state locking unavailable")
            lock_held = True
        except (OSError, IOError) as exc:
            logger.warning("context experiment state lock unavailable: %s", exc)
            if lock_fd is not None:
                try:
                    lock_fd.close()
                except OSError:
                    pass
            raise
        try:
            yield
        finally:
            if lock_fd is not None:
                try:
                    if lock_held and fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    elif lock_held and msvcrt is not None:
                        lock_fd.seek(0)
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                        )
                except (OSError, IOError):
                    pass
                try:
                    lock_fd.close()
                except OSError:
                    pass


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"assignments": {}, "next_index": 0}
    if not isinstance(data, dict):
        return {"assignments": {}, "next_index": 0}
    assignments = data.get("assignments")
    if not isinstance(assignments, dict):
        data["assignments"] = {}
    if not isinstance(data.get("next_index"), int):
        data["next_index"] = 0
    return data


def _arm_names(spec: Mapping[str, Any]) -> list[str]:
    arms = spec.get("arms")
    if not isinstance(arms, dict):
        return []
    configured_order = spec.get("arm_order")
    if isinstance(configured_order, list):
        ordered = [str(name) for name in configured_order if str(name) in arms]
        if ordered:
            return ordered
    return [str(name) for name in arms.keys()]


def _session_allowed(spec: Mapping[str, Any], platform: str | None) -> bool:
    unit = str(spec.get("unit") or "session").strip().lower()
    if unit != "session":
        return False

    platform_key = (platform or "").strip().lower()
    only = spec.get("platforms")
    if isinstance(only, list) and only:
        allowed = {str(item).strip().lower() for item in only if str(item).strip()}
        if platform_key not in allowed:
            return False
    excluded = spec.get("exclude_platforms")
    if isinstance(excluded, list) and excluded:
        denied = {str(item).strip().lower() for item in excluded if str(item).strip()}
        if platform_key in denied:
            return False
    return True


def _iter_enabled_experiments(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    experiments = config.get("context_experiments")
    if not isinstance(experiments, dict) or not experiments:
        return []

    # Shorthand for a single unnamed experiment:
    # context_experiments: {enabled: true, arms: {...}}
    if isinstance(experiments.get("arms"), dict):
        return [("default", experiments)] if experiments.get("enabled") else []

    active = experiments.get("active")
    if isinstance(active, str) and active in experiments:
        spec = experiments.get(active)
        if isinstance(spec, dict) and spec.get("enabled"):
            return [(active, spec)]
        return []

    enabled: list[tuple[str, Mapping[str, Any]]] = []
    for name, spec in experiments.items():
        if name == "active" or not isinstance(spec, dict):
            continue
        if spec.get("enabled"):
            enabled.append((str(name), spec))
    return enabled


def _hash_arm(experiment_name: str, session_id: str, arms: list[str]) -> str:
    digest = hashlib.sha256(f"{experiment_name}:{session_id}".encode("utf-8")).digest()
    return arms[int.from_bytes(digest[:8], "big") % len(arms)]


def _round_robin_arm(experiment_name: str, session_id: str, arms: list[str]) -> str:
    path = _state_path(experiment_name)
    with _locked_state(path):
        state = _load_state(path)
        assignments = state.setdefault("assignments", {})
        existing = assignments.get(session_id)
        if existing in arms:
            return str(existing)

        idx = int(state.get("next_index", 0)) % len(arms)
        arm = arms[idx]
        state["next_index"] = idx + 1
        assignments[session_id] = arm
        atomic_json_write(path, state, mode=0o600)
        return arm


def resolve_context_experiment_assignment(
    *,
    session_id: str | None,
    platform: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> ContextExperimentAssignment | None:
    """Return the active context-experiment arm for *session_id*, if configured."""

    sid = (session_id or "").strip()
    if not sid:
        return None

    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            return None
    if not isinstance(config, Mapping):
        return None

    for experiment_name, spec in _iter_enabled_experiments(config):
        if not _session_allowed(spec, platform):
            continue
        arms_cfg = spec.get("arms")
        if not isinstance(arms_cfg, dict):
            continue
        names = _arm_names(spec)
        if not names:
            continue
        strategy = str(spec.get("assignment") or "hash").strip().lower()
        if strategy in {"round_robin", "round-robin", "sequential"}:
            arm_name = _round_robin_arm(experiment_name, sid, names)
        else:
            arm_name = _hash_arm(experiment_name, sid, names)
        arm = arms_cfg.get(arm_name)
        if isinstance(arm, dict):
            return ContextExperimentAssignment(
                experiment_name=experiment_name,
                arm_name=arm_name,
                arm=dict(arm),
            )
    return None


__all__ = [
    "ContextExperimentAssignment",
    "resolve_context_experiment_assignment",
]
