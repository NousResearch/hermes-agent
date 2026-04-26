"""FleetCoordinator — the orchestration brain for Telegram Fleet Mode.

Responsibilities:

* Hold the parsed roster in memory and persist mutations.
* Talk to the manager bot via :class:`FleetApiClient` (token rotation,
  managed-bot identity, child-bot send).
* Mint pending entries when the agent calls ``telegram_spawn_bot`` and
  return the deep link for the user to tap.
* Promote a pending entry to active when a ``managed_bot`` update arrives
  (the gateway feeds these updates in via :meth:`absorb_managed_bot`).
* Fan out a swarm: assign sub-tasks to active "pool" children, run them in
  parallel via ``delegate_task``, post status updates to each child's chat
  so the user can watch live, then aggregate findings.

Single-instance per process, fetched via :func:`get_coordinator`.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from gateway.telegram_fleet.api import (
    BotApiError,
    FleetApiClient,
    ManagedBotInfo,
    build_managed_bot_deep_link,
)
from gateway.telegram_fleet.audit import audit_event
from gateway.telegram_fleet.guardrails import (
    FleetGuardrailError,
    check_can_spawn,
    check_rate_limit,
)
from gateway.telegram_fleet.roster import (
    ChildBot,
    FleetRoster,
    PENDING_TTL_SECONDS,
    RosterError,
    load_roster,
    save_roster,
)

logger = logging.getLogger(__name__)


# ── Spawn / orchestration result types ────────────────────────────────


class FleetApprovalRequired(FleetGuardrailError):
    """Raised when ``orchestrate_swarm`` needs the user to approve the plan.

    Carries the proposed bindings + objective so the tool layer can render
    a structured plan back to the agent (which then surfaces it via
    ``clarify`` for the user to approve / adjust / reject).
    """

    def __init__(
        self,
        *,
        objective: str,
        bindings: List[Dict[str, Any]],
        report_chat_id: Optional[str],
    ):
        bots = [b["bot_username"] for b in bindings]
        super().__init__(
            "Telegram fleet swarm requires user approval before posting as "
            f"{len(bots)} named bots ({', '.join('@' + b for b in bots)}).  "
            "Use the `clarify` tool to confirm, then re-call with "
            "user_approved=true.  To bypass: pin every subtask to a "
            "specific bot_username, or set telegram_fleet.auto_approve: "
            "true in config.yaml."
        )
        self.objective = objective
        self.bindings = bindings
        self.report_chat_id = report_chat_id

    def to_plan_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "report_chat_id": self.report_chat_id,
            "workers": [
                {
                    "bot_username": b["bot_username"],
                    "persona": b.get("persona", ""),
                    "goal": b["goal"],
                }
                for b in self.bindings
            ],
        }


def _is_by_name_request(subtasks: List[Dict[str, Any]]) -> bool:
    """Return True when every subtask pinned a ``bot_username``.

    A by-name request means the user already named who they want — that
    counts as explicit consent and bypasses the approval gate.
    """
    if not subtasks:
        return False
    for raw in subtasks:
        if not isinstance(raw, dict):
            return False
        pin = str(raw.get("bot_username") or "").strip()
        if not pin:
            return False
    return True


def _auto_approve_enabled() -> bool:
    """Return True when the operator opted out of the approval prompt.

    Resolution order: ``TELEGRAM_FLEET_AUTO_APPROVE`` env var, then
    ``telegram_fleet.auto_approve`` in ``~/.hermes/config.yaml``.  Default
    is False — the gate stays on unless the operator explicitly disables it.
    """
    if (os.environ.get("TELEGRAM_FLEET_AUTO_APPROVE") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return True
    try:
        from hermes_constants import get_config_path

        config_path = get_config_path()
        if not config_path.exists():
            return False
        import yaml  # local import — avoid hard dep at module-load time

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # malformed YAML, missing yaml, IO error
        logger.debug("could not read auto_approve from config.yaml: %s", e)
        return False
    if not isinstance(parsed, dict):
        return False
    cfg = parsed.get("telegram_fleet") or {}
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("auto_approve", False))


@dataclass
class SpawnResult:
    """Returned by :meth:`FleetCoordinator.spawn_bot`."""

    suggested_username: str
    deep_link: str
    nonce: str
    qr_text: Optional[str] = None  # Caller can render it; coordinator does not.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggested_username": self.suggested_username,
            "deep_link": self.deep_link,
            "nonce": self.nonce,
        }


@dataclass
class SwarmTaskResult:
    """One sub-task's outcome inside an orchestrate_swarm run."""

    persona: str
    bot_username: str
    goal: str
    response: str
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "persona": self.persona,
            "bot_username": self.bot_username,
            "goal": self.goal,
            "response": self.response,
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.error:
            out["error"] = self.error
        return out


# ── Coordinator ───────────────────────────────────────────────────────


class FleetCoordinator:
    """Single-process orchestrator for the Telegram fleet."""

    def __init__(
        self,
        *,
        manager_token: Optional[str] = None,
        manager_username: Optional[str] = None,
        api_client: Optional[FleetApiClient] = None,
        roster_path=None,
        delegate_fn: Optional[Callable[..., str]] = None,
    ):
        self._lock = threading.RLock()
        self._roster_path = roster_path
        self._roster: FleetRoster = load_roster(path=roster_path)
        # Drop stale pending entries immediately so callers don't see them.
        if self._roster.prune_expired_pending():
            save_roster(self._roster, path=roster_path)
        self._manager_token = manager_token or os.getenv("TELEGRAM_FLEET_MANAGER_TOKEN")
        if manager_username:
            self._roster.manager_bot_username = manager_username.lstrip("@")
        self._api = api_client
        self._delegate_fn = delegate_fn  # Injectable for tests; falls back to delegate_tool.

    # ── Roster access ────────────────────────────────────────────────

    @property
    def roster(self) -> FleetRoster:
        with self._lock:
            return self._roster

    def list_children(self, *, status: Optional[str] = None) -> List[ChildBot]:
        with self._lock:
            children = list(self._roster.children)
        if status:
            children = [c for c in children if c.status == status]
        return children

    def find(self, username: str) -> Optional[ChildBot]:
        with self._lock:
            return self._roster.find(username)

    def reload(self) -> None:
        """Re-read the roster from disk (e.g. after an external edit)."""
        with self._lock:
            self._roster = load_roster(path=self._roster_path)
            self._roster.prune_expired_pending()

    # ── Manager API client ───────────────────────────────────────────

    def _require_api(self) -> FleetApiClient:
        if self._api is not None:
            return self._api
        if not self._manager_token:
            raise FleetGuardrailError(
                "no manager bot token configured.  Set TELEGRAM_FLEET_MANAGER_TOKEN "
                "in ~/.hermes/.env or pass manager_token to FleetCoordinator."
            )
        self._api = FleetApiClient(self._manager_token)
        return self._api

    def manager_username(self) -> str:
        with self._lock:
            cached = self._roster.manager_bot_username
        if cached:
            return cached
        try:
            api = self._require_api()
            me = api.get_me()
        except (FleetGuardrailError, BotApiError) as e:
            raise FleetGuardrailError(f"could not resolve manager bot: {e}") from e
        username = str(me.get("username") or "")
        if not username:
            raise FleetGuardrailError("manager bot has no username — set one in @BotFather")
        with self._lock:
            self._roster.manager_bot_username = username
            save_roster(self._roster, path=self._roster_path)
        return username

    # ── Spawn ────────────────────────────────────────────────────────

    def spawn_bot(
        self,
        suggested_username: str,
        *,
        persona: str = "",
        display_name: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        toolset: Optional[List[str]] = None,
        rate_limit_per_min: int = 30,
        daily_budget_usd: Optional[float] = None,
        notes: str = "",
    ) -> SpawnResult:
        """Mint a pending child entry and return the deep link to confirm.

        The bot only enters ``active`` status once a ``managed_bot`` update
        arrives via :meth:`absorb_managed_bot`.
        """
        suggested = _normalize_username(suggested_username)
        if not suggested:
            raise FleetGuardrailError("suggested_username must be non-empty")
        with self._lock:
            check_can_spawn(self._roster)
            if self._roster.find(suggested) is not None:
                raise FleetGuardrailError(
                    f"a bot named @{suggested} is already in the roster"
                )
            nonce = secrets.token_hex(8)
            child = ChildBot(
                username=suggested,
                persona=persona,
                model=model,
                profile=profile,
                toolset=toolset,
                rate_limit_per_min=rate_limit_per_min,
                daily_budget_usd=daily_budget_usd,
                status="pending",
                nonce=nonce,
                notes=notes,
            )
            self._roster.upsert(child)
            save_roster(self._roster, path=self._roster_path)

        manager = self.manager_username()
        deep_link = build_managed_bot_deep_link(
            manager, suggested, name=display_name or persona[:40] or suggested
        )
        audit_event(
            "spawn_requested",
            bot_username=suggested,
            persona_chars=len(persona),
            has_model_override=bool(model),
            has_toolset_override=bool(toolset),
            nonce=_redact_nonce(nonce),
        )
        return SpawnResult(
            suggested_username=suggested, deep_link=deep_link, nonce=nonce
        )

    def absorb_pending_updates(
        self, *, max_polls: int = 1, poll_timeout: int = 0
    ) -> List[ChildBot]:
        """Drain ``managed_bot`` updates from Telegram and absorb them.

        Each poll call (``max_polls`` total) issues one ``getUpdates``
        request with the given ``poll_timeout`` (0 = non-blocking, >0 =
        long-poll seconds — Telegram caps this at ~50).  Any matching
        pending roster entries are promoted to active.  Returns the list
        of newly-active children.
        """
        api = self._require_api()
        absorbed: List[ChildBot] = []
        offset: Optional[int] = None
        for _ in range(max(1, int(max_polls))):
            try:
                infos, offset = api.drain_managed_bot_events(
                    offset=offset, timeout=poll_timeout
                )
            except BotApiError as e:
                logger.warning("getUpdates failed: %s", e)
                break
            for info in infos:
                child = self.absorb_managed_bot(info)
                if child is not None:
                    absorbed.append(child)
            if not infos and poll_timeout == 0:
                # Non-blocking poll exhausted the queue.
                break
        return absorbed

    def adopt_existing_bot(
        self,
        token: str,
        *,
        persona: str = "",
        model: Optional[str] = None,
        profile: Optional[str] = None,
        toolset: Optional[List[str]] = None,
        rate_limit_per_min: int = 30,
        daily_budget_usd: Optional[float] = None,
        notes: str = "",
    ) -> ChildBot:
        """Adopt a bot the user *already* created in BotFather into the fleet.

        Skips the Managed Bots ``getManagedBotToken`` ceremony entirely —
        validates the token by calling ``getMe`` against it and writes the
        result straight into the roster as ``status="active"``.  Use this
        when ``hermes fleet add`` (the deep-link spawn flow) hits a
        "username already taken" error because the bot was created manually.

        Trade-off: bots adopted this way cannot be rotated via
        ``replaceManagedBotToken`` — that endpoint only works on bots that
        were *created* through the manager flow.  Rotation for adopted bots
        means generating a fresh token in BotFather and re-adopting.
        """
        if not token or ":" not in token:
            raise FleetGuardrailError(
                "token doesn't look like a Telegram bot token (expected "
                "'<id>:<rest>')"
            )
        # Validate against Telegram + read identity.  Re-import so test
        # patches against ``gateway.telegram_fleet.api.FleetApiClient`` take
        # effect (the module-level import is bound at coordinator import
        # time and would not see monkeypatches).
        from gateway.telegram_fleet import api as _api_module
        client = _api_module.FleetApiClient(token)
        try:
            me = client.get_me()
        except BotApiError as e:
            raise FleetGuardrailError(
                f"Telegram rejected this token: {e}"
            ) from e
        username = _normalize_username(str(me.get("username") or ""))
        bot_id = int(me.get("id") or 0)
        if not username or not bot_id:
            raise FleetGuardrailError(
                f"getMe returned an unusable identity: {me!r}"
            )

        with self._lock:
            check_can_spawn(self._roster)
            existing = self._roster.find(username)
            child = ChildBot(
                username=username,
                persona=persona,
                bot_id=bot_id,
                token=token,
                model=model,
                profile=profile,
                toolset=toolset,
                status="active",
                rate_limit_per_min=rate_limit_per_min,
                daily_budget_usd=daily_budget_usd,
                notes=notes or (existing.notes if existing else ""),
                last_rotated_at=_now_iso(),
            )
            self._roster.upsert(child)
            save_roster(self._roster, path=self._roster_path)
        audit_event(
            "adopted",
            bot_username=username,
            bot_id=bot_id,
            replaced_existing=existing is not None,
        )
        return child

    def absorb_managed_bot(self, info: ManagedBotInfo) -> Optional[ChildBot]:
        """Promote a pending child to active when the user confirms the spawn.

        Called by the gateway when it receives a ``managed_bot`` update
        and resolves the token via ``getManagedBotToken``.  Matches by
        username (we generate unique suggested usernames per spawn).
        Returns the updated ``ChildBot`` or None if no pending entry was
        waiting for it (e.g. the user created a bot directly in BotFather).
        """
        username = _normalize_username(info.bot_username)
        with self._lock:
            child = self._roster.find(username)
            if child is None:
                logger.info(
                    "managed_bot update for @%s with no pending roster entry; ignoring",
                    username,
                )
                return None
            child.bot_id = info.bot_id
            child.token = info.token
            child.status = "active"
            child.last_rotated_at = _now_iso()
            self._roster.upsert(child)
            save_roster(self._roster, path=self._roster_path)
        audit_event(
            "spawn_confirmed",
            bot_username=username,
            bot_id=info.bot_id,
        )
        return child

    # ── Token rotation / decommission ────────────────────────────────

    def rotate_token(self, username: str) -> ChildBot:
        with self._lock:
            child = self._roster.find(username)
            if child is None or child.status != "active":
                raise FleetGuardrailError(
                    f"@{username} is not an active fleet member"
                )
            if child.bot_id is None:
                raise FleetGuardrailError(
                    f"@{username} has no bot_id recorded; cannot rotate"
                )
            bot_id = child.bot_id
        api = self._require_api()
        info = api.replace_managed_bot_token(bot_id)
        with self._lock:
            child = self._roster.find(username)
            if child is None:  # raced with a decommission
                raise FleetGuardrailError(f"@{username} disappeared mid-rotation")
            child.token = info.token
            child.last_rotated_at = _now_iso()
            self._roster.upsert(child)
            save_roster(self._roster, path=self._roster_path)
            updated = child
        audit_event("token_rotated", bot_username=username, bot_id=bot_id)
        return updated

    def decommission(self, username: str) -> bool:
        """Mark a child decommissioned (kept in roster for audit) and zero its token."""
        with self._lock:
            child = self._roster.find(username)
            if child is None:
                return False
            child.token = None
            child.status = "decommissioned"
            self._roster.upsert(child)
            save_roster(self._roster, path=self._roster_path)
        audit_event("decommissioned", bot_username=username)
        return True

    # ── Direct delegation (one bot speaks via another) ───────────────

    def delegate_message(
        self,
        target_username: str,
        chat_id: str,
        text: str,
        *,
        reply_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Have the named child bot post *text* into *chat_id*."""
        with self._lock:
            child = self._roster.find(target_username)
            if child is None or not child.is_active():
                raise FleetGuardrailError(
                    f"@{target_username} is not an active fleet member"
                )
            token = child.token
            rate = child.rate_limit_per_min
            username = child.username
        if not check_rate_limit(username, per_minute=rate):
            raise FleetGuardrailError(
                f"@{username} exceeded its {rate}/min rate limit"
            )
        api = self._require_api()
        result = api.send_message_as(token, chat_id, text, reply_to=reply_to)
        audit_event(
            "delegate_message",
            bot_username=username,
            chat_id=str(chat_id),
            text_chars=len(text),
        )
        return result

    # ── The orchestration primitive ──────────────────────────────────

    def orchestrate_swarm(
        self,
        objective: str,
        subtasks: List[Dict[str, Any]],
        *,
        report_chat_id: Optional[str] = None,
        max_parallel: int = 8,
        per_task_timeout_s: float = 600.0,
        parent_agent: Any = None,
        user_approved: bool = False,
    ) -> Dict[str, Any]:
        """Fan *subtasks* across active fleet members, aggregate results.

        Each entry of *subtasks* is::

            {
                "goal": "...",                # required
                "persona": "...",             # optional override
                "bot_username": "@worker_3",  # optional explicit pin
                "context": "...",             # optional extra context
                "toolsets": ["web", ...],     # optional override
            }

        Workflow:

        1. Bind each subtask to an active fleet member (explicit pin first,
           otherwise round-robin across remaining active children).
        2. Run all of them in parallel via the injected ``delegate_fn``
           (defaults to :func:`tools.delegate_tool.delegate_task`).
        3. If a ``report_chat_id`` is configured, each child posts a
           one-line "starting / done" status as itself so the operator can
           watch progress live.
        4. Collect results, write a ``swarm_started`` / ``swarm_completed``
           audit pair.
        """
        if not objective.strip():
            raise FleetGuardrailError("objective must be non-empty")
        if not subtasks:
            raise FleetGuardrailError("subtasks must contain at least one entry")
        if max_parallel < 1:
            raise FleetGuardrailError(f"max_parallel must be ≥ 1, got {max_parallel}")
        if per_task_timeout_s <= 0:
            raise FleetGuardrailError(
                f"per_task_timeout_s must be positive, got {per_task_timeout_s}"
            )
        with self._lock:
            active = [c for c in self._roster.children if c.is_active()]
        if not active:
            raise FleetGuardrailError(
                "no active fleet members.  Spawn at least one bot via "
                "telegram_spawn_bot first."
            )

        bindings = self._bind_subtasks(subtasks, active)

        # Approval gate (Hermes house style — mirrors terminal_tool's
        # ``force=True`` pattern).  The Telegram swarm has visible side-effects
        # (named bots posting messages); gate it behind explicit consent.
        # Two ways to bypass:
        #   (a) ``user_approved=True`` — caller explicitly confirms.
        #   (b) Every subtask pinned ``bot_username`` — by-name request.
        # Operators can disable the gate entirely via config flag
        # ``telegram_fleet.auto_approve: true``.
        if not user_approved and not _is_by_name_request(subtasks):
            if not _auto_approve_enabled():
                # Record the request so operators can see swarm-attempt patterns
                # without leaking subtask content or bot tokens.
                audit_event(
                    "swarm_approval_required",
                    objective_chars=len(objective),
                    workers=len(bindings),
                    bots=[b["bot_username"] for b in bindings],
                    report_chat_id=str(report_chat_id) if report_chat_id else None,
                )
                raise FleetApprovalRequired(
                    objective=objective,
                    bindings=bindings,
                    report_chat_id=report_chat_id,
                )

        delegate_fn = self._delegate_fn or _resolve_delegate()

        run_id = secrets.token_hex(6)
        audit_event(
            "swarm_started",
            run_id=run_id,
            objective_chars=len(objective),
            tasks=len(bindings),
            bots=[b["bot_username"] for b in bindings],
            report_chat_id=str(report_chat_id) if report_chat_id else None,
        )

        # Optional per-task status posts back to a results chat.
        if report_chat_id:
            for b in bindings:
                self._safe_post(
                    b["bot_username"],
                    report_chat_id,
                    f"▶︎ starting: {b['goal'][:120]}",
                )

        results: List[SwarmTaskResult] = []
        wall_clock_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(bindings))) as ex:
            futures = {
                ex.submit(
                    self._run_subtask,
                    binding=b,
                    objective=objective,
                    delegate_fn=delegate_fn,
                    parent_agent=parent_agent,
                ): b
                for b in bindings
            }
            # per_task_timeout_s is a per-task budget; multiply by task count for
            # the overall wall-clock cap so a single slow task doesn't starve all others.
            overall_timeout = per_task_timeout_s * len(bindings)
            try:
                pending = dict(futures)
                for fut in as_completed(pending, timeout=overall_timeout):
                    b = pending[fut]
                    try:
                        result = fut.result()
                    except Exception as e:  # pragma: no cover - delegate failure
                        result = SwarmTaskResult(
                            persona=b.get("persona", ""),
                            bot_username=b["bot_username"],
                            goal=b["goal"],
                            response="",
                            duration_seconds=0.0,
                            error=f"{type(e).__name__}: {e}",
                        )
                    results.append(result)
                    if report_chat_id:
                        summary = result.error or _truncate(result.response, 280)
                        self._safe_post(
                            result.bot_username,
                            report_chat_id,
                            f"✅ done: {summary}" if not result.error else f"❌ failed: {summary}",
                        )
            except FuturesTimeout:
                for fut, b in list(futures.items()):
                    if not fut.done():
                        fut.cancel()
                        results.append(
                            SwarmTaskResult(
                                persona=b.get("persona", ""),
                                bot_username=b["bot_username"],
                                goal=b["goal"],
                                response="",
                                duration_seconds=per_task_timeout_s,
                                error="timed out",
                            )
                        )

        # Critical Path metric — wall-clock per stage = max(workers), not sum.
        # Single-stage fan-out so critical_path == max(durations).
        durations = [r.duration_seconds for r in results]
        critical_path = max(durations) if durations else 0.0
        total_serial = sum(durations)
        speedup = (total_serial / critical_path) if critical_path > 0 else 1.0

        audit_event(
            "swarm_completed",
            run_id=run_id,
            tasks=len(results),
            failures=sum(1 for r in results if r.error),
            critical_path_seconds=round(critical_path, 3),
            parallel_speedup=round(speedup, 2),
        )
        aggregate = _aggregate(objective, results)
        return {
            "run_id": run_id,
            "objective": objective,
            "results": [r.to_dict() for r in results],
            "summary": aggregate,
            "metrics": {
                "workers": len(results),
                "failures": sum(1 for r in results if r.error),
                "critical_path_seconds": round(critical_path, 3),
                "total_serial_seconds": round(total_serial, 3),
                "wall_clock_seconds": round(time.monotonic() - wall_clock_start, 3),
                "parallel_speedup": round(speedup, 2),
            },
        }

    # ── internals ────────────────────────────────────────────────────

    def _bind_subtasks(
        self, subtasks: List[Dict[str, Any]], active: List[ChildBot]
    ) -> List[Dict[str, Any]]:
        """Resolve each subtask to a concrete bot username + persona."""
        bindings: List[Dict[str, Any]] = []
        rotation = list(active)
        rr_idx = 0
        for raw in subtasks:
            if not isinstance(raw, dict):
                raise FleetGuardrailError(f"subtask must be a mapping, got {type(raw).__name__}")
            goal = str(raw.get("goal") or "").strip()
            if not goal:
                raise FleetGuardrailError("subtask is missing 'goal'")
            pin = _normalize_username(str(raw.get("bot_username") or ""))
            if pin:
                child = self._roster.find(pin)
                if child is None or not child.is_active():
                    raise FleetGuardrailError(
                        f"subtask requested @{pin} but it is not an active fleet member"
                    )
                bot = child
            else:
                bot = rotation[rr_idx % len(rotation)]
                rr_idx += 1
            persona = str(raw.get("persona") or bot.persona or "")
            raw_toolsets = raw.get("toolsets")
            if raw_toolsets is not None and not isinstance(raw_toolsets, list):
                raise FleetGuardrailError(
                    f"subtask 'toolsets' must be a list of strings, got "
                    f"{type(raw_toolsets).__name__!r}"
                )
            toolsets = (raw_toolsets if raw_toolsets is not None else list(bot.toolset or [])) or None
            bindings.append(
                {
                    "goal": goal,
                    "context": str(raw.get("context") or ""),
                    "toolsets": toolsets,
                    "bot_username": bot.username,
                    "persona": persona,
                }
            )
        return bindings

    def _run_subtask(
        self,
        *,
        binding: Dict[str, Any],
        objective: str,
        delegate_fn: Callable[..., str],
        parent_agent: Any,
    ) -> SwarmTaskResult:
        start = time.monotonic()
        persona = binding.get("persona") or ""
        context_parts = [
            f"Swarm objective: {objective}",
        ]
        if persona:
            context_parts.append(f"Your persona for this task: {persona}")
        if binding.get("context"):
            context_parts.append(str(binding["context"]))
        context = "\n\n".join(context_parts)
        try:
            response = delegate_fn(
                goal=binding["goal"],
                context=context,
                toolsets=binding.get("toolsets"),
                role="leaf",
                parent_agent=parent_agent,
            )
        except Exception as e:
            return SwarmTaskResult(
                persona=persona,
                bot_username=binding["bot_username"],
                goal=binding["goal"],
                response="",
                duration_seconds=time.monotonic() - start,
                error=f"{type(e).__name__}: {e}",
            )
        return SwarmTaskResult(
            persona=persona,
            bot_username=binding["bot_username"],
            goal=binding["goal"],
            response=str(response),
            duration_seconds=time.monotonic() - start,
        )

    def _safe_post(self, username: str, chat_id: str, text: str) -> None:
        try:
            self.delegate_message(username, chat_id, text)
        except Exception as e:
            logger.debug("could not post status from @%s: %s", username, e)


# ── Module-level singleton ─────────────────────────────────────────────


_singleton_lock = threading.Lock()
_singleton: Optional[FleetCoordinator] = None


def get_coordinator(*, refresh: bool = False) -> FleetCoordinator:
    """Return the per-process FleetCoordinator (lazy-init).

    Pass ``refresh=True`` to discard the cached instance and re-read the
    roster from disk — useful after the user edits ``telegram_fleet.yaml``
    by hand.
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None or refresh:
            _singleton = FleetCoordinator()
        return _singleton


def reset_coordinator() -> None:
    """Drop the cached coordinator.  Test helper."""
    global _singleton
    with _singleton_lock:
        _singleton = None


# ── helpers ───────────────────────────────────────────────────────────


def _resolve_delegate() -> Callable[..., str]:
    """Lazy import so the fleet package doesn't pull in tools at import time."""
    from tools.delegate_tool import delegate_task

    return delegate_task


def _normalize_username(value: str) -> str:
    return (value or "").strip().lstrip("@").lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _redact_nonce(nonce: str) -> str:
    if not nonce or len(nonce) < 8:
        return "***"
    return f"{nonce[:4]}…"


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _aggregate(objective: str, results: List[SwarmTaskResult]) -> str:
    """Compose a human-readable summary of the swarm run."""
    lines = [f"Swarm objective: {objective}", ""]
    successes = [r for r in results if not r.error]
    failures = [r for r in results if r.error]
    lines.append(f"Workers: {len(results)}  ·  ok: {len(successes)}  ·  failed: {len(failures)}")
    lines.append("")
    for r in results:
        header = f"— @{r.bot_username}"
        if r.persona:
            header += f" ({r.persona[:60]})"
        header += f"  [{r.duration_seconds:.1f}s]"
        lines.append(header)
        lines.append(r.error if r.error else _truncate(r.response, 600))
        lines.append("")
    return "\n".join(lines).rstrip()


# Re-export for callers
__all__ = [
    "FleetCoordinator",
    "SpawnResult",
    "SwarmTaskResult",
    "get_coordinator",
    "reset_coordinator",
    "PENDING_TTL_SECONDS",
]
