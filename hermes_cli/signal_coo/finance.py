"""Finance worker for Torben's Signal COO operator."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_ledger import ActionLedger, ActionRecord
from .briefs import ScopeBrief

DEFAULT_FINANCE_MIN_SCORE = 0.70
DEFAULT_MAX_FINANCE_ITEMS = 2


def _compact(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


class FinanceSlice:
    """Stage hard-limited trading and personal-finance proposals."""

    def __init__(self, ledger: ActionLedger):
        self.ledger = ledger

    def generate_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> ScopeBrief:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        trade_signals = _list(evidence.get("trade_signals") or evidence.get("market_signals"))
        personal_finance = _list(evidence.get("personal_finance_signals") or evidence.get("monarch_signals"))

        if trade_signals:
            return self._trade_brief(_dict(trade_signals[0]), now=now)
        if personal_finance:
            return self._personal_finance_brief(_dict(personal_finance[0]), now=now)
        return ScopeBrief(
            scope="finance",
            title="Finance",
            status="quiet",
            priority="low",
            text="Finance: no trade or Monarch signal supplied. No capital is at risk.",
        )

    def _trade_brief(self, signal: dict[str, Any], *, now: datetime) -> ScopeBrief:
        catalyst = _compact(signal.get("catalyst") or signal.get("event"), "market catalyst")
        thesis = _compact(signal.get("thesis"), "the trade needs a clearer thesis before execution")
        expression = _compact(signal.get("expression") or signal.get("instrument"), "supported option or marginable equity")
        max_loss = _compact(signal.get("max_loss") or signal.get("premium") or signal.get("risk_cap"), "unset")
        exit_rule = _compact(signal.get("exit_rule"), "exit rule required before execution")
        expected_payoff = _compact(signal.get("expected_payoff"), "payoff model required before execution")
        evidence_ids = [str(item) for item in signal.get("evidence_ids") or []]

        action = self.ledger.add_action(
            scope="FIN",
            summary=f"Review live-trading setup: {expression}",
            evidence_ids=evidence_ids,
            allowed_next_actions=["revise", "approve_trade_review", "discard"],
            status="approval_required",
            risk_class="critical",
            now=now,
            executor_state={
                "mutation_type": "broker_order",
                "provider": "robinhood-agentic-mcp",
                "mutation_status": "review_only",
                "auth_required": True,
                "execution_blocked_until": [
                    "broker_auth",
                    "account_eligibility_check",
                    "risk_policy_limits",
                    "explicit_signal_approval",
                ],
                "requires_options_margin_review": True,
            },
        )

        lines = [
            "Finance: I staged a trade review, not an order.",
            "",
            f"Catalyst: {catalyst}.",
            f"Thesis: {thesis}.",
            f"Expression: {expression}.",
            f"Max loss: {max_loss}.",
            f"Expected payoff: {expected_payoff}.",
            f"Exit rule: {exit_rule}.",
            "",
            f"[{action.handle}] Approve trade review or ask for a smaller risk version.",
        ]
        return ScopeBrief(
            scope="finance",
            title="Finance",
            text="\n".join(lines),
            priority="high",
            actions=[action],
            evidence_ids=evidence_ids,
        )

    def _personal_finance_brief(self, signal: dict[str, Any], *, now: datetime) -> ScopeBrief:
        summary = _compact(signal.get("summary") or signal.get("opportunity"), "personal finance opportunity")
        recommendation = _compact(signal.get("recommendation"), "review the expense and decide whether to cut it")
        evidence_ids = [str(item) for item in signal.get("evidence_ids") or []]
        action = self.ledger.add_action(
            scope="FIN",
            summary=f"Review Monarch finance action: {summary}",
            evidence_ids=evidence_ids,
            allowed_next_actions=["revise", "approve_note", "discard"],
            status="staged",
            risk_class="medium",
            now=now,
            executor_state={
                "mutation_type": "monarch_review",
                "provider": "monarch-money-mcp",
                "mutation_status": "draft_only",
                "external_change_blocked_until": "explicit_signal_approval",
            },
        )
        lines = [
            "Finance: I found a Monarch action to review.",
            "",
            f"Signal: {summary}.",
            f"Recommendation: {recommendation}.",
            "",
            f"[{action.handle}] Review the finance action. Nothing is changed in Monarch.",
        ]
        return ScopeBrief(
            scope="finance",
            title="Finance",
            text="\n".join(lines),
            priority="normal",
            actions=[action],
            evidence_ids=evidence_ids,
        )


def build_torben_finance_radar_adapter(
    ratatosk_run: dict[str, Any],
    *,
    ledger: ActionLedger,
    state_path: str | Path,
    min_score: float = DEFAULT_FINANCE_MIN_SCORE,
    max_items: int = DEFAULT_MAX_FINANCE_ITEMS,
    now: datetime | None = None,
    mark_delivered: bool = True,
    stage_actions: bool = True,
    force_wake: bool = False,
) -> dict[str, Any]:
    """Adapt a Ratatosk Robinhood v0.1 run into Torben FIN actions.

    The Ratatosk run is research/staging evidence only. This adapter never
    places, cancels, modifies, or approves broker orders.
    """

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state_file = Path(state_path)
    state = _load_state(state_file)
    delivered = state.get("delivered_candidates") if isinstance(state.get("delivered_candidates"), dict) else {}
    candidates = _candidate_rows(ratatosk_run)
    unsafe_counts = _unsafe_mutation_counts(ratatosk_run)
    if unsafe_counts["broker_orders_submitted"] > 0 or unsafe_counts["external_mutations"] > 0:
        return _unsafe_mutation_payload(
            ratatosk_run=ratatosk_run,
            candidates=candidates,
            counts=unsafe_counts,
            min_score=min_score,
            now=now,
        )
    selected = _select_candidate_rows(candidates, min_score=min_score, max_items=max_items, force_wake=force_wake)
    fresh = [candidate for candidate in selected if _candidate_key(ratatosk_run, candidate) not in delivered]

    if not fresh:
        payload = {
            "task": "torben_finance_radar",
            "wakeAgent": False,
            "generated_at": _iso(now),
            "reason": _silent_reason(ratatosk_run=ratatosk_run, candidates=candidates, min_score=min_score),
            "ratatosk_status": ratatosk_run.get("status"),
            "ratatosk_phase": ratatosk_run.get("phase"),
            "ratatosk_run_id": _ratatosk_run_id(ratatosk_run),
            "candidate_count": len(candidates),
            "selected_count": 0,
            "suppressed_duplicate_count": len(selected),
            "min_score": min_score,
            "market_regime": _llm_result(ratatosk_run).get("market_regime"),
            "no_trade_reason": _llm_result(ratatosk_run).get("no_trade_reason"),
            "llm_judge": _llm_audit(ratatosk_run),
            "public_actions_taken": 0,
            "external_mutations": 0,
            "orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
            "broker_orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
            "text": "",
        }
        return payload

    actions = [
        _stage_finance_action(
            ledger=ledger,
            ratatosk_run=ratatosk_run,
            candidate=candidate,
            rank=rank,
            now=now,
            min_score=min_score,
        )
        if stage_actions
        else _preview_finance_action(
            ratatosk_run=ratatosk_run,
            candidate=candidate,
            rank=rank,
            now=now,
            min_score=min_score,
        )
        for rank, candidate in enumerate(fresh, start=1)
    ]
    text = render_torben_finance_radar_text(
        ratatosk_run=ratatosk_run,
        candidates=fresh,
        actions=actions,
        min_score=min_score,
        now=now,
    )
    payload = {
        "task": "torben_finance_radar",
        "wakeAgent": True,
        "generated_at": _iso(now),
        "ratatosk_status": ratatosk_run.get("status"),
        "ratatosk_phase": ratatosk_run.get("phase"),
        "ratatosk_run_id": _ratatosk_run_id(ratatosk_run),
        "candidate_count": len(candidates),
        "selected_count": len(fresh),
        "suppressed_duplicate_count": max(0, len(selected) - len(fresh)),
        "min_score": min_score,
        "market_regime": _llm_result(ratatosk_run).get("market_regime"),
        "no_trade_reason": _llm_result(ratatosk_run).get("no_trade_reason"),
        "candidates": fresh,
        "actions": [action.to_dict() for action in actions],
        "llm_judge": _llm_audit(ratatosk_run),
        "text": text,
        "public_actions_taken": 0,
        "external_mutations": 0,
        "orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
        "broker_orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
        "delivery": {
            "surface": "signal",
            "operator": "torben",
            "source": "ratatosk_robinhood_v01",
            "delivery_mode": "adapter_text",
        },
    }
    if mark_delivered and stage_actions:
        _mark_delivered(state_file, state, ratatosk_run=ratatosk_run, candidates=fresh, now=now)
    return payload


def render_torben_finance_radar_text(
    *,
    ratatosk_run: dict[str, Any],
    candidates: list[dict[str, Any]],
    actions: list[ActionRecord],
    min_score: float,
    now: datetime,
) -> str:
    lines = [
        f"Torben / Finance Radar / {now:%Y-%m-%d %H:%M UTC}",
        "",
        (
            f"Ratatosk ran the Robinhood v0.1 {ratatosk_run.get('phase') or 'market'} analysis "
            f"and found {len(candidates)} candidate(s) above the {min_score:.2f} review bar."
        ),
        "This is a stage-only finance review. No order was placed, cancelled, modified, or approved.",
        "Research can create candidates; it cannot trade directly.",
        "",
    ]
    regime = _compact(_llm_result(ratatosk_run).get("market_regime"), "")
    if regime:
        lines.append(f"Market read: {regime}.")
    no_trade_reason = _compact(_llm_result(ratatosk_run).get("no_trade_reason"), "")
    if no_trade_reason:
        lines.append(f"Guard note: {no_trade_reason}")
    if regime or no_trade_reason:
        lines.append("")

    for idx, (candidate, action) in enumerate(zip(candidates, actions), start=1):
        symbol = _compact(candidate.get("symbol"), "unknown")
        score = _score(candidate)
        instrument = _compact(candidate.get("instrument_type") or candidate.get("asset_class"), "instrument")
        direction = _compact(candidate.get("direction") or candidate.get("action"), "review")
        note = _compact(candidate.get("research_note") or candidate.get("thesis"), "No thesis provided.")
        constraints = [str(item) for item in candidate.get("constraints") or [] if str(item).strip()]
        lines.extend(
            [
                f"{idx}. {symbol} {direction} {instrument} candidate, score {score:.2f}.",
                f"Why: {note}",
            ]
        )
        if constraints:
            lines.append(f"Constraints: {', '.join(constraints[:8])}.")
        lines.append(f"[{action.handle}] Review the thesis, ask for a smaller-risk version, or hold.")
        lines.append("")
    lines.append("Live trading remains blocked until the finance mandate, consent, kill switch, and guard all pass.")
    return "\n".join(lines).rstrip() + "\n"


def write_finance_radar_artifacts(
    payload: dict[str, Any],
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(text_output, str(payload.get("text") or ""))


def _stage_finance_action(
    *,
    ledger: ActionLedger,
    ratatosk_run: dict[str, Any],
    candidate: dict[str, Any],
    rank: int,
    now: datetime,
    min_score: float,
) -> ActionRecord:
    symbol = _compact(candidate.get("symbol"), "unknown")
    score = _score(candidate)
    evidence_ids = [_ratatosk_run_id(ratatosk_run), _candidate_key(ratatosk_run, candidate)]
    source_refs = [str(item) for item in candidate.get("source_refs") or [] if str(item).strip()]
    evidence_ids.extend(source_refs[:5])
    return ledger.add_action(
        scope="FIN",
        summary=f"Review Ratatosk Robinhood candidate {rank}: {symbol} score {score:.2f}",
        evidence_ids=evidence_ids,
        allowed_next_actions=["review_thesis", "smaller_risk", "hold"],
        status="staged",
        risk_class="critical",
        ttl_hours=18,
        now=now,
        executor_state={
            "mutation_type": "broker_order_candidate",
            "mutation_status": "stage_only_not_ordered",
            "provider": "ratatosk_robinhood_v01",
            "source": "ratatosk_robinhood_v01_cron_tick",
            "ratatosk_run_id": _ratatosk_run_id(ratatosk_run),
            "ratatosk_phase": ratatosk_run.get("phase"),
            "radar_rank": rank,
            "candidate_fingerprint": _candidate_key(ratatosk_run, candidate),
            "candidate": candidate,
            "llm_judged": _llm_result(ratatosk_run) != {},
            "llm_score": score,
            "min_score": min_score,
            "can_place_order_directly": bool(candidate.get("can_place_order_directly")),
            "research_signal_can_trade_directly": False,
            "order_tools_available": False,
            "orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
            "external_mutations": int(ratatosk_run.get("external_mutations") or 0),
            "execution_blocked_until": [
                "TBC-DECIDE-LIVE-FINANCE",
                "written_mandate",
                "human_consent",
                "kill_switch_allows",
                "pre_trade_guard_passes",
                "reconciliation_canary_passes",
            ],
            "reply_actions": ["review", "smaller_risk", "hold"],
            "reply_aliases": [f"review finance {rank}", f"smaller risk {rank}", f"hold finance {rank}"],
        },
    )


def _preview_finance_action(
    *,
    ratatosk_run: dict[str, Any],
    candidate: dict[str, Any],
    rank: int,
    now: datetime,
    min_score: float,
) -> ActionRecord:
    symbol = _compact(candidate.get("symbol"), "unknown")
    score = _score(candidate)
    return ActionRecord(
        handle=f"FIN-{now:%Y%m%d}-{rank:03d}",
        scope="fin",
        summary=f"Preview Ratatosk Robinhood candidate {rank}: {symbol} score {score:.2f}",
        evidence_ids=[_ratatosk_run_id(ratatosk_run), _candidate_key(ratatosk_run, candidate)],
        allowed_next_actions=["review_thesis", "smaller_risk", "hold"],
        status="staged",
        risk_class="critical",
        created_at=now,
        user_visible_summary=f"Preview Ratatosk Robinhood candidate {rank}: {symbol} score {score:.2f}",
        executor_state={
            "mutation_type": "broker_order_candidate",
            "mutation_status": "preview_only_not_ordered",
            "provider": "ratatosk_robinhood_v01",
            "candidate": candidate,
            "llm_score": score,
            "min_score": min_score,
            "can_place_order_directly": bool(candidate.get("can_place_order_directly")),
            "research_signal_can_trade_directly": False,
            "order_tools_available": False,
            "orders_submitted": int(ratatosk_run.get("orders_submitted") or 0),
            "external_mutations": int(ratatosk_run.get("external_mutations") or 0),
            "execution_blocked_until": [
                "TBC-DECIDE-LIVE-FINANCE",
                "written_mandate",
                "human_consent",
                "kill_switch_allows",
                "pre_trade_guard_passes",
                "reconciliation_canary_passes",
            ],
        },
    )


def _llm_result(ratatosk_run: dict[str, Any]) -> dict[str, Any]:
    value = ratatosk_run.get("llm_result")
    return value if isinstance(value, dict) else {}


def _candidate_rows(ratatosk_run: dict[str, Any]) -> list[dict[str, Any]]:
    result = _llm_result(ratatosk_run)
    rows = result.get("candidates")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _select_candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    min_score: float,
    max_items: int,
    force_wake: bool,
) -> list[dict[str, Any]]:
    sorted_rows = sorted(candidates, key=_score, reverse=True)
    if force_wake:
        return sorted_rows[: max(0, max_items)]
    selected = []
    for candidate in sorted_rows:
        if _score(candidate) < min_score:
            continue
        if bool(candidate.get("can_place_order_directly")):
            continue
        selected.append(candidate)
        if len(selected) >= max_items:
            break
    return selected


def _score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score") or candidate.get("confidence") or 0)
    except (TypeError, ValueError):
        return 0.0


def _ratatosk_run_id(ratatosk_run: dict[str, Any]) -> str:
    llm_run = ratatosk_run.get("llm_run") if isinstance(ratatosk_run.get("llm_run"), dict) else {}
    for key in ("run_id", "cron_tick_id"):
        value = str(llm_run.get(key) or "").strip()
        if value:
            return value
    material = json.dumps(ratatosk_run, sort_keys=True, default=str)[:2000]
    return f"ratatosk-run-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _candidate_key(ratatosk_run: dict[str, Any], candidate: dict[str, Any]) -> str:
    material = {
        "run_id": _ratatosk_run_id(ratatosk_run),
        "phase": ratatosk_run.get("phase"),
        "symbol": candidate.get("symbol"),
        "score": candidate.get("score"),
        "instrument_type": candidate.get("instrument_type") or candidate.get("asset_class"),
        "direction": candidate.get("direction") or candidate.get("action"),
        "research_note": candidate.get("research_note") or candidate.get("thesis"),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"ratatosk-candidate-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _silent_reason(*, ratatosk_run: dict[str, Any], candidates: list[dict[str, Any]], min_score: float) -> str:
    status = str(ratatosk_run.get("status") or "")
    if status == "no_due_tick":
        return "no due Robinhood v0.1 market phase"
    if not candidates:
        return "Ratatosk produced no research candidates"
    return f"no fresh Ratatosk candidate reached the {min_score:.2f} review threshold"


def _unsafe_mutation_counts(ratatosk_run: dict[str, Any]) -> dict[str, int]:
    return {
        "external_mutations": _int_count(ratatosk_run.get("external_mutations")),
        "broker_orders_submitted": max(
            _int_count(ratatosk_run.get("broker_orders_submitted")),
            _int_count(ratatosk_run.get("orders_submitted")),
        ),
    }


def _unsafe_mutation_payload(
    *,
    ratatosk_run: dict[str, Any],
    candidates: list[dict[str, Any]],
    counts: dict[str, int],
    min_score: float,
    now: datetime,
) -> dict[str, Any]:
    broker_orders = counts["broker_orders_submitted"]
    external_mutations = counts["external_mutations"]
    text = (
        f"Torben / Finance Radar / {now:%Y-%m-%d %H:%M UTC}\n\n"
        "Ratatosk reported a mutation during a stage-only finance run.\n"
        f"Broker orders submitted: {broker_orders}.\n"
        f"External mutations reported: {external_mutations}.\n"
        "I did not stage a FIN review card because stage-only finance must never report a trade as safe.\n"
        "Live trading remains blocked until the finance mandate, consent, kill switch, and guard all pass.\n"
    )
    return {
        "task": "torben_finance_radar",
        "wakeAgent": True,
        "generated_at": _iso(now),
        "status": "fail_closed",
        "reason": "ratatosk_reported_stage_only_mutation",
        "ratatosk_status": ratatosk_run.get("status"),
        "ratatosk_phase": ratatosk_run.get("phase"),
        "ratatosk_run_id": _ratatosk_run_id(ratatosk_run),
        "candidate_count": len(candidates),
        "selected_count": 0,
        "suppressed_duplicate_count": 0,
        "min_score": min_score,
        "market_regime": _llm_result(ratatosk_run).get("market_regime"),
        "no_trade_reason": _llm_result(ratatosk_run).get("no_trade_reason"),
        "actions": [],
        "llm_judge": _llm_audit(ratatosk_run),
        "public_actions_taken": 0,
        "external_mutations": external_mutations,
        "orders_submitted": broker_orders,
        "broker_orders_submitted": broker_orders,
        "text": text,
    }


def _int_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _llm_audit(ratatosk_run: dict[str, Any]) -> dict[str, Any]:
    llm_run = ratatosk_run.get("llm_run") if isinstance(ratatosk_run.get("llm_run"), dict) else {}
    result = _llm_result(ratatosk_run)
    return {
        "invoked": result != {},
        "status": "completed" if result else str(ratatosk_run.get("status") or "unknown"),
        "phase": ratatosk_run.get("phase"),
        "run_id": llm_run.get("run_id"),
        "cron_tick_id": llm_run.get("cron_tick_id"),
        "token_budget": llm_run.get("token_budget"),
        "order_tools_available": bool(llm_run.get("order_tools_available")),
        "llm_error": ratatosk_run.get("llm_error"),
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "delivered_candidates": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "delivered_candidates": {}}
    return payload if isinstance(payload, dict) else {"version": 1, "delivered_candidates": {}}


def _mark_delivered(
    path: Path,
    state: dict[str, Any],
    *,
    ratatosk_run: dict[str, Any],
    candidates: list[dict[str, Any]],
    now: datetime,
) -> None:
    delivered = state.get("delivered_candidates") if isinstance(state.get("delivered_candidates"), dict) else {}
    for candidate in candidates:
        delivered[_candidate_key(ratatosk_run, candidate)] = {
            "delivered_at": _iso(now),
            "ratatosk_run_id": _ratatosk_run_id(ratatosk_run),
            "phase": ratatosk_run.get("phase"),
            "symbol": candidate.get("symbol"),
            "score": _score(candidate),
        }
    state.update({"version": 1, "delivered_candidates": delivered, "updated_at": _iso(now)})
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
