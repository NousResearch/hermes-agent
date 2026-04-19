import hashlib
import json
import math
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import AgentRun, AgentRunStep, IdempotencyRecord, QueryLog, TenantPolicy, User
from .policy import get_policy_pack


def utc_day_string(dt: datetime | None = None) -> str:
    now = dt or datetime.utcnow()
    return now.strftime("%Y-%m-%d")


def json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def canonical_request_hash(payload: dict[str, Any]) -> str:
    raw = json_dumps(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    # coarse estimate, enough for budgeting
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    # fixed local estimate: $0.20 / 1M input, $0.60 / 1M output
    return round((input_tokens * 0.20 + output_tokens * 0.60) / 1_000_000, 8)


def _apply_pack_defaults(policy: TenantPolicy) -> None:
    pack = get_policy_pack(policy.policy_pack)
    policy.min_confidence = float(pack["min_confidence"])
    policy.max_citations = int(pack["max_citations"])
    policy.daily_query_budget = int(pack["daily_query_budget"])
    policy.daily_run_budget = int(pack["daily_run_budget"])
    policy.daily_cost_budget_usd = float(pack["daily_cost_budget_usd"])
    policy.max_top_k = int(pack["max_top_k"])
    policy.max_question_chars = int(pack["max_question_chars"])
    if getattr(policy, "daily_external_api_budget", None) is None:
        policy.daily_external_api_budget = 200
    if getattr(policy, "external_api_timeout_cap_seconds", None) is None:
        policy.external_api_timeout_cap_seconds = 8
    if getattr(policy, "public_api_allowlist_json", None) is None:
        policy.public_api_allowlist_json = "[]"


def get_or_create_tenant_policy(db: Session, tenant_id: int) -> TenantPolicy:
    p = db.query(TenantPolicy).filter(TenantPolicy.tenant_id == tenant_id).first()
    if p:
        if not p.policy_pack:
            p.policy_pack = "balanced"
        patched = False
        if getattr(p, "daily_external_api_budget", None) is None:
            p.daily_external_api_budget = 200
            patched = True
        if getattr(p, "external_api_timeout_cap_seconds", None) is None:
            p.external_api_timeout_cap_seconds = 8
            patched = True
        if getattr(p, "public_api_allowlist_json", None) in (None, ""):
            p.public_api_allowlist_json = "[]"
            patched = True
        if patched:
            db.add(p)
            db.flush()
        return p

    p = TenantPolicy(tenant_id=tenant_id, policy_pack="balanced")
    _apply_pack_defaults(p)
    db.add(p)
    db.flush()
    return p


def usage_today(db: Session, tenant_id: int) -> dict[str, Any]:
    day = utc_day_string()

    queries_today = (
        db.query(func.count(QueryLog.id))
        .filter(QueryLog.tenant_id == tenant_id, func.date(QueryLog.created_at) == day)
        .scalar()
        or 0
    )
    runs_today = (
        db.query(func.count(AgentRun.id))
        .filter(AgentRun.tenant_id == tenant_id, func.date(AgentRun.started_at) == day)
        .scalar()
        or 0
    )
    cost_today = (
        db.query(func.sum(QueryLog.estimated_cost_usd))
        .filter(QueryLog.tenant_id == tenant_id, func.date(QueryLog.created_at) == day)
        .scalar()
        or 0.0
    )

    return {
        "day_utc": day,
        "queries_today": int(queries_today),
        "runs_today": int(runs_today),
        "estimated_cost_today_usd": round(float(cost_today), 8),
    }


def enforce_budget(db: Session, user: User, extra_estimated_cost_usd: float = 0.0) -> dict[str, Any]:
    policy = get_or_create_tenant_policy(db, user.tenant_id)
    usage = usage_today(db, user.tenant_id)

    query_remaining = max(0, int(policy.daily_query_budget) - usage["queries_today"])
    run_remaining = max(0, int(policy.daily_run_budget) - usage["runs_today"])
    cost_remaining = max(0.0, float(policy.daily_cost_budget_usd) - (usage["estimated_cost_today_usd"] + extra_estimated_cost_usd))

    return {
        **usage,
        "query_budget_remaining": query_remaining,
        "run_budget_remaining": run_remaining,
        "cost_budget_remaining_usd": round(cost_remaining, 8),
    }


def assert_budget_available(usage_snapshot: dict[str, Any]) -> None:
    if usage_snapshot.get("query_budget_remaining", 0) <= 0:
        raise HTTPException(status_code=429, detail="Daily query budget exceeded")
    if usage_snapshot.get("run_budget_remaining", 0) <= 0:
        raise HTTPException(status_code=429, detail="Daily run budget exceeded")
    if float(usage_snapshot.get("cost_budget_remaining_usd", 0.0)) <= 0.0:
        raise HTTPException(status_code=429, detail="Daily cost budget exceeded")


def find_idempotent_response(
    db: Session,
    *,
    tenant_id: int,
    endpoint: str,
    idempotency_key: str,
    request_hash: str,
) -> tuple[dict[str, Any] | None, int | None, int | None]:
    if not idempotency_key:
        return None, None, None

    row = (
        db.query(IdempotencyRecord)
        .filter(
            IdempotencyRecord.tenant_id == tenant_id,
            IdempotencyRecord.endpoint == endpoint,
            IdempotencyRecord.idempotency_key == idempotency_key,
        )
        .first()
    )
    if not row:
        return None, None, None

    if row.request_hash != request_hash:
        raise HTTPException(status_code=409, detail="Idempotency key already used with a different payload")

    if not row.response_json:
        return None, row.run_id, row.status_code

    try:
        body = json.loads(row.response_json)
    except Exception:
        body = None
    return body, row.run_id, row.status_code


def upsert_idempotent_response(
    db: Session,
    *,
    tenant_id: int,
    user_id: int,
    endpoint: str,
    idempotency_key: str,
    request_hash: str,
    response: dict[str, Any],
    status_code: int,
    run_id: int | None,
) -> None:
    if not idempotency_key:
        return

    row = (
        db.query(IdempotencyRecord)
        .filter(
            IdempotencyRecord.tenant_id == tenant_id,
            IdempotencyRecord.endpoint == endpoint,
            IdempotencyRecord.idempotency_key == idempotency_key,
        )
        .first()
    )

    now = datetime.utcnow()
    if row is None:
        row = IdempotencyRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            endpoint=endpoint,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            response_json=json_dumps(response),
            status_code=int(status_code),
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )
    else:
        row.request_hash = request_hash
        row.response_json = json_dumps(response)
        row.status_code = int(status_code)
        row.run_id = run_id
        row.updated_at = now

    db.add(row)


def create_run(
    db: Session,
    *,
    tenant_id: int,
    user_id: int,
    endpoint: str,
    idempotency_key: str,
    request_payload: dict[str, Any],
    replay_of_run_id: int | None = None,
) -> AgentRun:
    row = AgentRun(
        tenant_id=tenant_id,
        user_id=user_id,
        endpoint=endpoint,
        status="running",
        idempotency_key=idempotency_key,
        replay_of_run_id=replay_of_run_id,
        request_json=json_dumps(request_payload),
        response_json="{}",
        error="",
        estimated_input_tokens=0,
        estimated_output_tokens=0,
        estimated_cost_usd=0.0,
        started_at=datetime.utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def finish_run(
    db: Session,
    *,
    run: AgentRun,
    status: str,
    response_payload: dict[str, Any] | None,
    error: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    now = datetime.utcnow()
    run.status = status
    run.response_json = json_dumps(response_payload or {})
    run.error = error[:4000] if error else ""
    run.estimated_input_tokens = int(max(0, input_tokens))
    run.estimated_output_tokens = int(max(0, output_tokens))
    run.estimated_cost_usd = estimate_cost_usd(run.estimated_input_tokens, run.estimated_output_tokens)
    run.finished_at = now
    run.duration_ms = max(0, int((now - run.started_at).total_seconds() * 1000))
    db.add(run)


def add_step(db: Session, *, run_id: int, step_order: int, name: str, status: str, metadata: dict[str, Any]) -> AgentRunStep:
    row = AgentRunStep(
        run_id=run_id,
        step_order=step_order,
        name=name,
        status=status,
        metadata_json=json_dumps(metadata),
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        duration_ms=0,
    )
    db.add(row)
    db.flush()
    return row
