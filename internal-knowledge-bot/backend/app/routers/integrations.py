import json
from datetime import datetime
from typing import Any
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..audit_utils import record_audit
from ..database import get_db
from ..deps import get_current_user, require_manager_or_admin
from ..hermes_bridge import get_catalog as get_hermes_catalog
from ..hermes_bridge import invoke_tool as invoke_hermes_tool
from ..models import AgentRun, IntegrationConnection, PublicApiProvider, User
from ..run_ledger import (
    add_step,
    assert_budget_available,
    canonical_request_hash,
    create_run,
    enforce_budget,
    estimate_tokens,
    find_idempotent_response,
    finish_run,
    get_or_create_tenant_policy,
    upsert_idempotent_response,
    utc_day_string,
)
from ..schemas import (
    HermesToolInvokeRequest,
    HermesToolInvokeResponse,
    IntegrationConnectRequest,
    PublicApiFetchRequest,
    PublicApiFetchResponse,
    PublicApiProviderOut,
)

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

SUPPORTED = {
    "slack": "Slack Workspace",
    "notion": "Notion Workspace",
    "gdrive": "Google Drive",
    "webhook": "Custom Webhook",
    "hermes": "Hermes Tool Bridge",
    "public_api": "Public API Registry",
}

# Curated from references/public-apis-curated.md (high-utility no-auth/HTTPS providers)
PUBLIC_PROVIDER_SEEDS: list[dict[str, Any]] = [
    {
        "name": "open_meteo",
        "category": "weather",
        "base_url": "https://api.open-meteo.com/v1",
        "auth_type": "none",
        "docs_url": "https://open-meteo.com/",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 5,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {"latitude": "60.98", "longitude": "25.66", "current": "temperature_2m"},
    },
    {
        "name": "spaceflight_news",
        "category": "news",
        "base_url": "https://api.spaceflightnewsapi.net/v4",
        "auth_type": "none",
        "docs_url": "https://spaceflightnewsapi.net",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 5,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {"limit": "10"},
    },
    {
        "name": "frankfurter",
        "category": "currency-exchange",
        "base_url": "https://api.frankfurter.app",
        "auth_type": "none",
        "docs_url": "https://www.frankfurter.app/docs",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 5,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "dict_single",
        "sample_query_json": {"from": "EUR", "to": "USD"},
    },
    {
        "name": "exchangerate_host",
        "category": "currency-exchange",
        "base_url": "https://api.exchangerate.host",
        "auth_type": "none",
        "docs_url": "https://exchangerate.host",
        "cors": "unknown",
        "tenant_scope": "global",
        "default_timeout_seconds": 5,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "dict_single",
        "sample_query_json": {"base": "EUR", "symbols": "USD,GBP"},
    },
    {
        "name": "geojs",
        "category": "geocoding",
        "base_url": "https://get.geojs.io/v1",
        "auth_type": "none",
        "docs_url": "https://www.geojs.io/",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 5,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "dict_single",
        "sample_query_json": {},
    },
    {
        "name": "nominatim",
        "category": "geocoding",
        "base_url": "https://nominatim.openstreetmap.org",
        "auth_type": "none",
        "docs_url": "https://nominatim.org/release-docs/latest/api/Overview/",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 6,
        "rate_limit_hint": "respect usage policy",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {"q": "Lahti Finland", "format": "json", "limit": "5"},
    },
    {
        "name": "sec_edgar",
        "category": "finance",
        "base_url": "https://data.sec.gov",
        "auth_type": "none",
        "docs_url": "https://www.sec.gov/edgar/sec-api-documentation",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 7,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {},
    },
    {
        "name": "fiscaldata_treasury",
        "category": "finance",
        "base_url": "https://api.fiscaldata.treasury.gov/services/api/fiscal_service",
        "auth_type": "none",
        "docs_url": "https://fiscaldata.treasury.gov/api-documentation/",
        "cors": "unknown",
        "tenant_scope": "global",
        "default_timeout_seconds": 6,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {"page[size]": "5"},
    },
    {
        "name": "apisetu",
        "category": "open-data",
        "base_url": "https://api.apisetu.gov.in",
        "auth_type": "none",
        "docs_url": "https://www.apisetu.gov.in/",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 6,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {},
    },
    {
        "name": "opensanctions",
        "category": "open-data",
        "base_url": "https://api.opensanctions.org",
        "auth_type": "none",
        "docs_url": "https://www.opensanctions.org/docs/api/",
        "cors": "yes",
        "tenant_scope": "global",
        "default_timeout_seconds": 7,
        "rate_limit_hint": "public/no-auth",
        "normalization_strategy": "results_or_data",
        "sample_query_json": {"q": "example", "limit": "10"},
    },
]


def _load_list(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return [str(x).strip().lower() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def _normalize_items(payload: Any) -> list[dict[str, Any]]:
    source: list[Any]
    if isinstance(payload, list):
        source = payload
    elif isinstance(payload, dict):
        source = []
        for key in ("results", "data", "items", "entries", "articles", "features"):
            val = payload.get(key)
            if isinstance(val, list):
                source = val
                break
        if not source:
            source = [payload]
    else:
        source = [payload]

    out: list[dict[str, Any]] = []
    for item in source[:200]:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({"value": item})
    return out


def _build_url(base_url: str, path: str, query: dict[str, Any]) -> str:
    path = (path or "").strip()
    if path.startswith("http://") or path.startswith("https://"):
        raise HTTPException(status_code=400, detail="Path must be relative, not an absolute URL")

    url = (base_url or "").rstrip("/")
    if path:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{url}{path}"

    if query:
        qp: dict[str, str] = {}
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, bool):
                qp[str(key)] = "true" if value else "false"
            else:
                qp[str(key)] = str(value)
        if qp:
            url = f"{url}?{url_parse.urlencode(qp, doseq=True)}"

    return url


def _fetch_json(url: str, *, timeout_seconds: int, retries: int) -> tuple[int, Any]:
    tries = max(1, min(3, retries + 1))
    headers = {
        "Accept": "application/json, text/plain;q=0.9,*/*;q=0.8",
        "User-Agent": "InternalKnowledgeBot/2.2",
    }

    for attempt in range(tries):
        try:
            req = url_request.Request(url=url, method="GET", headers=headers)
            with url_request.urlopen(req, timeout=timeout_seconds) as resp:
                status = int(getattr(resp, "status", resp.getcode()))
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {"raw_text": raw}

                if status >= 500 and attempt < tries - 1:
                    continue
                return status, parsed
        except url_error.HTTPError as exc:
            status = int(exc.code or 502)
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(exc)

            if status >= 500 and attempt < tries - 1:
                continue

            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"error": body[:2000]}
            return status, parsed
        except Exception as exc:  # noqa: BLE001
            if attempt < tries - 1:
                continue
            raise HTTPException(status_code=502, detail=f"Provider request failed: {type(exc).__name__}") from exc

    raise HTTPException(status_code=502, detail="Provider request failed")


def _ensure_public_provider_seeded(db: Session) -> None:
    existing = db.query(func.count(PublicApiProvider.id)).scalar() or 0
    if int(existing) > 0:
        return

    now = datetime.utcnow()
    for seed in PUBLIC_PROVIDER_SEEDS:
        row = PublicApiProvider(
            name=seed["name"],
            category=seed["category"],
            base_url=seed["base_url"],
            auth_type=seed["auth_type"],
            docs_url=seed["docs_url"],
            cors=seed["cors"],
            enabled=True,
            tenant_scope=seed["tenant_scope"],
            default_timeout_seconds=int(seed["default_timeout_seconds"]),
            rate_limit_hint=seed["rate_limit_hint"],
            normalization_strategy=seed["normalization_strategy"],
            sample_query_json=json.dumps(seed["sample_query_json"]),
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    db.flush()


def _external_calls_today(db: Session, tenant_id: int) -> int:
    return (
        db.query(func.count(AgentRun.id))
        .filter(
            AgentRun.tenant_id == tenant_id,
            AgentRun.endpoint == "/api/integrations/public/fetch",
            func.date(AgentRun.started_at) == utc_day_string(),
        )
        .scalar()
        or 0
    )


@router.get("")
def list_integrations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(IntegrationConnection).filter(IntegrationConnection.tenant_id == user.tenant_id).order_by(IntegrationConnection.created_at.desc()).all()
    return {
        "supported": [{"provider": k, "label": v} for k, v in SUPPORTED.items()],
        "connected": [
            {
                "id": r.id,
                "provider": r.provider,
                "display_name": r.display_name,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }


@router.post("/connect")
def connect(payload: IntegrationConnectRequest, db: Session = Depends(get_db), actor: User = Depends(require_manager_or_admin)):
    if payload.provider not in SUPPORTED:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    row = IntegrationConnection(
        tenant_id=actor.tenant_id,
        provider=payload.provider,
        display_name=payload.display_name,
        status="connected",
        config_json=json.dumps(payload.config),
    )
    db.add(row)
    db.flush()

    record_audit(
        db,
        tenant_id=actor.tenant_id,
        actor_user_id=actor.id,
        action="integrations.connect",
        resource_type="integration",
        resource_id=str(row.id),
        metadata={"provider": payload.provider},
    )

    db.commit()
    db.refresh(row)
    return {"success": True, "id": row.id}


@router.post("/{integration_id}/sync")
def sync(integration_id: int, db: Session = Depends(get_db), actor: User = Depends(require_manager_or_admin)):
    row = db.query(IntegrationConnection).filter(IntegrationConnection.id == integration_id, IntegrationConnection.tenant_id == actor.tenant_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found")

    row.status = "sync_queued"
    db.add(row)

    record_audit(
        db,
        tenant_id=actor.tenant_id,
        actor_user_id=actor.id,
        action="integrations.sync",
        resource_type="integration",
        resource_id=str(row.id),
        metadata={},
    )

    db.commit()
    return {"success": True, "integration_id": integration_id, "message": "Sync queued"}


@router.get("/hermes/tools")
def list_hermes_tools(actor: User = Depends(require_manager_or_admin)):
    catalog = get_hermes_catalog()
    return {
        "success": True,
        "tenant_id": actor.tenant_id,
        **catalog,
    }


@router.post("/hermes/invoke", response_model=HermesToolInvokeResponse)
def invoke_hermes(
    payload: HermesToolInvokeRequest,
    db: Session = Depends(get_db),
    actor: User = Depends(require_manager_or_admin),
):
    out = invoke_hermes_tool(
        tool_name=payload.tool_name,
        args=payload.args,
        tenant_id=actor.tenant_id,
        user_id=actor.id,
        task_id=payload.task_id,
    )

    record_audit(
        db,
        tenant_id=actor.tenant_id,
        actor_user_id=actor.id,
        action="integrations.hermes.invoke",
        resource_type="integration",
        resource_id="hermes",
        metadata={
            "tool_name": payload.tool_name,
            "task_id": out["task_id"],
        },
    )
    db.commit()

    return HermesToolInvokeResponse(**out)


@router.get("/public/providers")
def list_public_providers(
    db: Session = Depends(get_db),
    actor: User = Depends(require_manager_or_admin),
):
    _ensure_public_provider_seeded(db)

    policy = get_or_create_tenant_policy(db, actor.tenant_id)
    allowlist = set(_load_list(policy.public_api_allowlist_json))

    rows = (
        db.query(PublicApiProvider)
        .filter(PublicApiProvider.enabled.is_(True))
        .order_by(PublicApiProvider.category.asc(), PublicApiProvider.name.asc())
        .all()
    )

    providers: list[PublicApiProviderOut] = []
    for row in rows:
        try:
            sample_query = json.loads(row.sample_query_json or "{}")
            if not isinstance(sample_query, dict):
                sample_query = {}
        except Exception:
            sample_query = {}

        allowed = True if not allowlist else row.name.lower() in allowlist
        providers.append(
            PublicApiProviderOut(
                id=row.id,
                name=row.name,
                category=row.category,
                base_url=row.base_url,
                auth_type=row.auth_type,
                docs_url=row.docs_url,
                cors=row.cors,
                enabled=row.enabled,
                tenant_scope=row.tenant_scope,
                default_timeout_seconds=row.default_timeout_seconds,
                rate_limit_hint=row.rate_limit_hint,
                normalization_strategy=row.normalization_strategy,
                sample_query=sample_query,
                allowed_for_tenant=allowed,
            )
        )

    db.commit()
    return {
        "success": True,
        "tenant_id": actor.tenant_id,
        "allowlist": sorted(allowlist),
        "providers": [p.model_dump() for p in providers],
    }


@router.post("/public/fetch", response_model=PublicApiFetchResponse)
def fetch_public_api(
    payload: PublicApiFetchRequest,
    db: Session = Depends(get_db),
    actor: User = Depends(require_manager_or_admin),
):
    _ensure_public_provider_seeded(db)

    policy = get_or_create_tenant_policy(db, actor.tenant_id)
    allowlist = set(_load_list(policy.public_api_allowlist_json))

    request_payload = {
        "provider": payload.provider,
        "path": payload.path,
        "query": payload.query,
        "timeout_seconds": payload.timeout_seconds,
        "retries": payload.retries,
    }
    request_hash = canonical_request_hash(request_payload)
    idempotency_key = (payload.idempotency_key or "").strip()

    prior_response, prior_run_id, _ = find_idempotent_response(
        db,
        tenant_id=actor.tenant_id,
        endpoint="/api/integrations/public/fetch",
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )
    if prior_response is not None:
        replay = dict(prior_response)
        replay["run_id"] = replay.get("run_id") or prior_run_id
        replay["run_status"] = "replayed"
        return PublicApiFetchResponse(**replay)

    # Existing global tenant budgets still apply
    usage_snapshot = enforce_budget(db, actor)
    assert_budget_available(usage_snapshot)

    provider_name = payload.provider.strip().lower()
    provider = (
        db.query(PublicApiProvider)
        .filter(PublicApiProvider.name == provider_name, PublicApiProvider.enabled.is_(True))
        .first()
    )
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found or disabled")

    if allowlist and provider_name not in allowlist:
        raise HTTPException(status_code=403, detail=f"Provider '{provider_name}' is not allowlisted for this tenant")

    calls_today = int(_external_calls_today(db, actor.tenant_id))
    external_budget = max(1, int(policy.daily_external_api_budget))
    if calls_today >= external_budget:
        raise HTTPException(status_code=429, detail="Daily external API call budget exceeded")

    timeout_cap = max(1, int(policy.external_api_timeout_cap_seconds))
    requested_timeout = payload.timeout_seconds or provider.default_timeout_seconds or 5
    timeout_seconds = max(1, min(timeout_cap, int(requested_timeout)))
    retries = max(0, min(2, int(payload.retries)))

    url = _build_url(provider.base_url, payload.path, payload.query)

    run = create_run(
        db,
        tenant_id=actor.tenant_id,
        user_id=actor.id,
        endpoint="/api/integrations/public/fetch",
        idempotency_key=idempotency_key,
        request_payload=request_payload,
    )

    try:
        add_step(
            db,
            run_id=run.id,
            step_order=1,
            name="provider_call.start",
            status="running",
            metadata={
                "provider": provider_name,
                "url": url,
                "timeout_seconds": timeout_seconds,
                "retries": retries,
            },
        )

        status_code, raw_payload = _fetch_json(url, timeout_seconds=timeout_seconds, retries=retries)
        if status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Provider returned HTTP {status_code}")

        items = _normalize_items(raw_payload)
        response = PublicApiFetchResponse(
            success=True,
            provider=provider_name,
            url=url,
            status_code=status_code,
            item_count=len(items),
            items=items,
            raw=raw_payload,
            run_id=run.id,
            run_status="completed",
            budget_enforced=True,
            external_api_budget_remaining=max(0, external_budget - (calls_today + 1)),
        )

        add_step(
            db,
            run_id=run.id,
            step_order=2,
            name="provider_call.success",
            status="completed",
            metadata={"provider": provider_name, "status_code": status_code, "item_count": len(items)},
        )

        request_tokens = estimate_tokens(json.dumps(request_payload, ensure_ascii=False))
        response_tokens = estimate_tokens(json.dumps(raw_payload, ensure_ascii=False, default=str))

        finish_run(
            db,
            run=run,
            status="completed",
            response_payload=response.model_dump(),
            error="",
            input_tokens=request_tokens,
            output_tokens=response_tokens,
        )

        upsert_idempotent_response(
            db,
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            endpoint="/api/integrations/public/fetch",
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            response=response.model_dump(),
            status_code=200,
            run_id=run.id,
        )

        record_audit(
            db,
            tenant_id=actor.tenant_id,
            actor_user_id=actor.id,
            action="integrations.public.fetch",
            resource_type="integration",
            resource_id=provider_name,
            metadata={
                "run_id": run.id,
                "provider": provider_name,
                "status_code": status_code,
                "item_count": len(items),
                "timeout_seconds": timeout_seconds,
                "retries": retries,
            },
        )

        db.commit()
        return response
    except HTTPException as http_exc:
        add_step(
            db,
            run_id=run.id,
            step_order=2,
            name="provider_call.failed",
            status="failed",
            metadata={"provider": provider_name, "detail": str(http_exc.detail)},
        )
        finish_run(
            db,
            run=run,
            status="failed",
            response_payload={"detail": http_exc.detail},
            error=str(http_exc.detail),
            input_tokens=estimate_tokens(json.dumps(request_payload, ensure_ascii=False)),
            output_tokens=0,
        )
        record_audit(
            db,
            tenant_id=actor.tenant_id,
            actor_user_id=actor.id,
            action="integrations.public.fetch.failed",
            resource_type="integration",
            resource_id=provider_name,
            metadata={"run_id": run.id, "provider": provider_name, "detail": str(http_exc.detail)},
        )
        db.commit()
        raise
