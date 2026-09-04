"""Owner-aware workspace mutation admission for delegated children.

When a delegated child times out or is cancelled but stays live after the
existing tool-interrupt grace window, later known built-in mutations in the
same trusted local workspace are denied until every stale owner actually
finishes. Reads, unknown plugin/MCP tools, and other workspace domains stay
available. Release is driven by real completion, not elapsed time alone.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Set

from agent.tool_result_classification import FILE_MUTATING_TOOL_NAMES

logger = logging.getLogger(__name__)

# Mirrors agent.tool_executor's cooperative abandon wait after interrupt.
DEFAULT_QUARANTINE_GRACE_SECONDS = 3.0

KNOWN_BUILTIN_MUTATING_TOOLS = FILE_MUTATING_TOOL_NAMES


@dataclass
class _Owner:
    owner_id: str
    domain: Path
    mutation_gate_open: bool = True
    stale: bool = False
    is_live: Callable[[], bool] = field(default=lambda: False)
    quarantine_timer: Optional[threading.Timer] = None


_lock = threading.Lock()
_owners: Dict[str, _Owner] = {}
_quarantined: Dict[Path, Set[str]] = {}
_grace_seconds = DEFAULT_QUARANTINE_GRACE_SECONDS
_CURRENT_FENCE_OWNER: ContextVar[Optional[str]] = ContextVar(
    "hermes_workspace_fence_owner",
    default=None,
)


@contextmanager
def owning_delegated_child(owner_id: str) -> Iterator[None]:
    """Mark the current delegated child as the fence owner for this stack."""
    token = _CURRENT_FENCE_OWNER.set(str(owner_id) if owner_id else None)
    try:
        yield
    finally:
        _CURRENT_FENCE_OWNER.reset(token)


def reset_for_tests() -> None:
    """Drop all fence state. Tests only."""
    with _lock:
        for owner in _owners.values():
            timer = owner.quarantine_timer
            if timer is not None:
                timer.cancel()
        _owners.clear()
        _quarantined.clear()


def set_grace_seconds_for_tests(seconds: float) -> None:
    """Override the post-interrupt grace used before domain quarantine."""
    global _grace_seconds
    _grace_seconds = max(0.0, float(seconds))


def resolve_workspace_domain(
    raw: Optional[str] = None,
    *,
    child: Any = None,
) -> Path:
    """Resolve the trusted local workspace domain for a delegated child."""
    explicit = getattr(child, "_delegate_workspace_domain", None) if child is not None else None
    candidate = explicit or raw
    if not candidate:
        candidate = os.environ.get("TERMINAL_CWD") or os.getcwd()
    path = Path(str(candidate)).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def bind_owner(
    owner_id: str,
    domain: Path,
    *,
    is_live: Optional[Callable[[], bool]] = None,
) -> None:
    """Register a delegated child as an owner of ``domain``."""
    if not owner_id:
        return
    owner = _Owner(
        owner_id=str(owner_id),
        domain=Path(domain),
        is_live=is_live or (lambda: False),
    )
    with _lock:
        _owners[owner.owner_id] = owner


def set_owner_liveness(owner_id: str, is_live: Callable[[], bool]) -> None:
    with _lock:
        owner = _owners.get(owner_id)
        if owner is not None:
            owner.is_live = is_live


def close_mutation_gate(owner_id: str) -> None:
    """Stop the named child from starting further known mutations."""
    with _lock:
        owner = _owners.get(owner_id)
        if owner is None:
            return
        owner.mutation_gate_open = False
        owner.stale = True


def schedule_quarantine_if_still_live(owner_id: str) -> None:
    """After the existing grace window, quarantine the domain if the child is live."""
    delay = _grace_seconds
    if delay <= 0:
        _quarantine_if_still_live(owner_id)
        return
    timer = threading.Timer(delay, _quarantine_if_still_live, args=(owner_id,))
    timer.daemon = True
    with _lock:
        owner = _owners.get(owner_id)
        if owner is None:
            return
        if owner.quarantine_timer is not None:
            owner.quarantine_timer.cancel()
        owner.quarantine_timer = timer
    timer.start()


def mark_timeout_or_cancel(owner_id: str) -> None:
    """Close the child's mutation gate and arm domain quarantine after grace."""
    close_mutation_gate(owner_id)
    schedule_quarantine_if_still_live(owner_id)


def release_owner(owner_id: str) -> None:
    """Release on real completion. Lifts quarantine when no stale live owners remain."""
    if not owner_id:
        return
    with _lock:
        owner = _owners.pop(owner_id, None)
        if owner is None:
            return
        if owner.quarantine_timer is not None:
            owner.quarantine_timer.cancel()
        holders = _quarantined.get(owner.domain)
        if holders is not None:
            holders.discard(owner_id)
            if not holders:
                _quarantined.pop(owner.domain, None)


def mutation_target_path(function_name: str, function_args: Optional[dict]) -> Optional[Path]:
    if function_name not in KNOWN_BUILTIN_MUTATING_TOOLS:
        return None
    args = function_args or {}
    raw = args.get("path") or args.get("file")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve()
    except OSError:
        return path


def path_in_domain(path: Path, domain: Path) -> bool:
    try:
        path.relative_to(domain)
        return True
    except ValueError:
        return False


def deny_delegated_mutation(
    function_name: str,
    function_args: Optional[dict] = None,
) -> Optional[str]:
    """Return a deny message when a later delegated mutation is fenced.

    Non-delegated callers, read-only tools, and unknown plugin/MCP tools
    are never fenced.
    """
    if function_name not in KNOWN_BUILTIN_MUTATING_TOOLS:
        return None
    try:
        from agent.delegation_context import is_delegated_child_context
    except Exception:
        return None
    if not is_delegated_child_context():
        return None

    target = mutation_target_path(function_name, function_args)
    current_id = _current_owner_id()

    with _lock:
        owner = _owners.get(current_id) if current_id else None
        if owner is not None and not owner.mutation_gate_open:
            if target is None or path_in_domain(target, owner.domain):
                return (
                    "Delegated mutation denied: this child was timed out or "
                    "cancelled and its mutation gate is closed."
                )

        for domain, holders in _quarantined.items():
            if not holders:
                continue
            in_domain = (
                path_in_domain(target, domain)
                if target is not None
                else (owner is not None and owner.domain == domain)
            )
            if not in_domain:
                continue
            other_holders = set(holders)
            if current_id:
                other_holders.discard(current_id)
            if other_holders:
                return _quarantine_message(domain)
    return None


def _quarantine_message(domain: Path) -> str:
    return (
        "Delegated mutation denied: a timed-out or cancelled child is still "
        f"live in workspace {domain}. Reads remain available; wait until "
        "that child actually exits."
    )


def _current_owner_id() -> Optional[str]:
    bound = _CURRENT_FENCE_OWNER.get()
    if bound:
        return bound
    try:
        from gateway.session_context import get_current_session_id
    except Exception:
        return None
    try:
        session_id = get_current_session_id()
    except Exception:
        return None
    return str(session_id) if session_id else None


def _quarantine_if_still_live(owner_id: str) -> None:
    with _lock:
        owner = _owners.get(owner_id)
        if owner is None or not owner.stale:
            return
        try:
            still_live = bool(owner.is_live())
        except Exception:
            still_live = True
        if not still_live:
            return
        holders = _quarantined.setdefault(owner.domain, set())
        holders.add(owner_id)
        logger.warning(
            "Quarantined workspace %s until stale delegated owner %s exits",
            owner.domain,
            owner_id,
        )
