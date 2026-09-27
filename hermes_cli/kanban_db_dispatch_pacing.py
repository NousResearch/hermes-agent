"""Pacing-governor wiring for kanban worker spawns (sibling of kanban_db_dispatch.py).

``_worker_argv()`` calls :func:`resolve_worker_provider` immediately before building
``--provider`` so a NEW kanban worker spawn is actually routed by the pacing governor's
pace-line decision (t_31b1338d, follow-up to the ``aos/pacing_governor.py`` library landed
in t_1eb32e10 / qwickapps/aos#457). Before this module, the governor's
``resolve_provider_for_new_spawn()`` existed only as a library call nobody made.

``aos`` is a separate repo/venv and is not on hermes-agent's runtime ``sys.path`` today (no
installed package, no PYTHONPATH entry in the systemd worker units) — checked directly
against the deployed ``/opt/hermes`` venv before writing this module. This module therefore
tries the real import first (so it picks up ``aos`` for free the day it becomes an installed
dependency) and otherwise falls back to :func:`_vendored_resolve_provider_for_new_spawn`, a
tiny mirror of the same on-disk state-file contract. The vendored copy is a stopgap, not a
fork: keep it in lockstep with ``aos/pacing_governor.py``'s ``decide_next_provider`` /
``resolve_provider_for_new_spawn`` if either changes.

Provider ids: kanban's actual runtime ``--provider`` ids are ``claude-subscription`` and
``openai-codex`` (see ``plugins/model-providers/*/plugin.yaml``), which is exactly what
``aos.pacing_governor.USAGE_PROVIDER_KEY`` already maps from — the poll tick that keeps the
governor's state file fresh MUST be started with
``HERMES_PACING_CHAIN=claude-subscription,openai-codex`` (its own default,
``anthropic,codex``, does not match any state-file entry kanban ever passes as
``default_provider`` and every lookup would silently miss).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# Kanban workers are never the priority lane -- that's chat bots / prime (see
# aos/pacing_governor.py's RESERVED_LANE_PCT rule 4).
RESERVED_LANE = False

DEFAULT_STATE_PATH = Path(os.environ.get(
    "HERMES_PACING_STATE_PATH", "~/.qwickapps/state/provider_pace.json")).expanduser()
DEFAULT_DECISION_LOG_PATH = Path(os.environ.get(
    "HERMES_PACING_DECISION_LOG", "~/.qwickapps/state/provider_pace_decisions.jsonl")).expanduser()

# Mirrors aos.pacing_governor.RESERVED_LANE_PCT; only used by the vendored fallback below.
_RESERVED_LANE_PCT = 15.0

try:
    from aos.pacing_governor import resolve_provider_for_new_spawn as _aos_resolve_provider_for_new_spawn
except ImportError:
    _aos_resolve_provider_for_new_spawn = None


def _vendored_over_pace(used: Optional[float], allowed: Optional[float], *, reserve_pct: float) -> bool:
    if used is None or allowed is None:
        return False
    cap = max(0.0, min(allowed, 100.0 - reserve_pct))
    return used > cap


def _vendored_has_data(state: Mapping[str, Any]) -> bool:
    return state.get("error") is None and (
        state.get("five_hour_used_pct") is not None or state.get("weekly_used_pct") is not None
    )


def _vendored_decide_next_provider(
    chain: list[str], states: Mapping[str, Mapping[str, Any]], *, reserved_lane: bool = False
) -> Optional[str]:
    """Mirror of ``aos.pacing_governor.decide_next_provider``'s routing rule, reading the
    already-evaluated pace state straight out of the on-disk JSON doc (this module never
    re-derives pace math -- that stays owned by the poll tick / aos). Keep in lockstep with
    the real function."""
    reserve = 0.0 if reserved_lane else _RESERVED_LANE_PCT
    over_pace: dict[str, bool] = {}
    for provider in chain:
        state = states.get(provider) or {}
        over_pace[provider] = bool(
            state
            and _vendored_has_data(state)
            and (
                _vendored_over_pace(state.get("five_hour_used_pct"), state.get("five_hour_allowed_pct"), reserve_pct=reserve)
                or _vendored_over_pace(state.get("weekly_used_pct"), state.get("weekly_allowed_pct"), reserve_pct=reserve)
            )
        )
    for provider in chain:
        state = states.get(provider)
        if not state or not _vendored_has_data(state):
            continue  # no usable reading for this provider; skip rather than trust a blank
        if not over_pace[provider]:
            return provider
    return chain[0] if chain else None


def _vendored_resolve_provider_for_new_spawn(
    default_provider: str,
    *,
    state_path: Path,
    sticky_provider: Optional[str],
    reserved_lane: bool,
    decision_log_path: Optional[Path],
    context: str,
) -> str:
    """Stopgap mirror of ``aos.pacing_governor.resolve_provider_for_new_spawn``, used only
    while ``aos`` is not importable from hermes-agent's runtime. Not a fork: re-reads the
    exact same on-disk state-file contract the real function does, and fails open to
    ``default_provider`` on a missing/unreadable/empty state file -- a governor outage must
    never block or alter a spawn.
    """
    if sticky_provider:
        return sticky_provider
    try:
        doc = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if not doc:
        return default_provider
    effective_chain = list(doc.get("chain") or [default_provider])
    if default_provider not in effective_chain:
        effective_chain = [default_provider, *effective_chain]
    states = doc.get("providers") or {}
    provider = _vendored_decide_next_provider(effective_chain, states, reserved_lane=reserved_lane) or default_provider
    if decision_log_path is not None:
        try:
            decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "decided_at": time.time(),
                "provider": provider,
                "context": context or "spawn",
                "vendored": True,
            }
            with decision_log_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            logger.debug("pacing governor (vendored): decision log append failed", exc_info=True)
    return provider


def _resolve_profile_active_provider(hermes_home: Optional[str]) -> Optional[str]:
    """The assignee profile's own currently-active provider -- what a worker would resolve
    to with no ``--provider`` flag at all (the pacing governor's ``default_provider``
    baseline, never an override). Scoped to ``hermes_home`` the same way
    ``kanban_db_dispatch._resolve_worker_cli_toolsets`` scopes its own config read: a
    context-local override, never a mutation of ``os.environ`` (shared by every thread in
    the dispatcher process)."""
    if not hermes_home:
        return None
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.auth import get_active_provider

    token = set_hermes_home_override(hermes_home)
    try:
        return get_active_provider()
    finally:
        reset_hermes_home_override(token)


def resolve_worker_provider(task: Any, hermes_home: Optional[str], *, sticky_provider: Optional[str] = None) -> Optional[str]:
    """Provider to pin via ``--provider`` for a kanban worker spawn, or ``None`` to leave the
    worker on the profile's own configured provider (byte-for-byte pre-governor behavior).

    ``task.provider_override`` -- an explicit pin by the task creator -- always wins and
    skips the governor entirely; checked here (not just by the caller) so every caller of
    this function gets the same override contract.

    ``sticky_provider``: the provider of an ALREADY-RUNNING session for this task/worker,
    never for a genuine new spawn. Kanban's own dispatcher never has one to pass:
    ``claim_task``/``claim_review_task`` CAS a task from ready/review to ``running``
    exclusively, so ``_worker_argv`` (this function's only kanban caller) is never reached
    while a worker for the task is already alive. The parameter exists so this integration
    point still honors the governor's "never reassign a running session" contract if a
    future caller (or a direct unit test of this function) ever has one to pass.

    Fail-open, always: a missing/unreadable state file, an unresolved profile provider, or
    ANY exception here returns ``None`` -- no ``--provider`` flag, identical to dispatch
    before the governor existed -- rather than blocking or altering the spawn.
    """
    provider_override = getattr(task, "provider_override", None)
    if provider_override:
        return provider_override
    if sticky_provider:
        return sticky_provider
    task_id = getattr(task, "id", "?")
    try:
        default_provider = _resolve_profile_active_provider(hermes_home)
        if not default_provider:
            return None
        context = f"kanban:{task_id}"
        if _aos_resolve_provider_for_new_spawn is not None:
            return _aos_resolve_provider_for_new_spawn(
                default_provider,
                state_path=DEFAULT_STATE_PATH,
                sticky_provider=None,
                reserved_lane=RESERVED_LANE,
                decision_log_path=DEFAULT_DECISION_LOG_PATH,
                context=context,
            ) or default_provider
        return _vendored_resolve_provider_for_new_spawn(
            default_provider,
            state_path=DEFAULT_STATE_PATH,
            sticky_provider=None,
            reserved_lane=RESERVED_LANE,
            decision_log_path=DEFAULT_DECISION_LOG_PATH,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 -- a spawn must never fail because the governor did
        logger.debug("kanban worker: pacing governor spawn routing skipped for %s (%s)", task_id, exc)
        return None
