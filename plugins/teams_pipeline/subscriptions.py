"""Microsoft Graph subscription helpers for the Teams pipeline plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from plugins.teams_pipeline.models import GraphSubscription
from plugins.teams_pipeline.store import TeamsPipelineStore, resolve_teams_pipeline_store_path
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
from tools.microsoft_graph_client import MicrosoftGraphClient


def build_graph_client() -> MicrosoftGraphClient:
    provider = MicrosoftGraphTokenProvider.from_env()
    return MicrosoftGraphClient(provider)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_store_path(path: str | None) -> str:
    return str(resolve_teams_pipeline_store_path(path))


def build_store(path: str | None = None) -> TeamsPipelineStore:
    return TeamsPipelineStore(resolve_store_path(path))


def sync_graph_subscription_record(
    store: TeamsPipelineStore,
    subscription_payload: dict[str, Any],
    *,
    status: str | None = None,
    renewed: bool = False,
) -> dict[str, Any]:
    normalized = GraphSubscription.from_dict(subscription_payload).to_dict()
    expiration = _parse_datetime(normalized.get("expiration_datetime"))
    effective_status = status
    if effective_status is None:
        effective_status = "expired" if expiration and expiration <= _utc_now() else "active"
    normalized["status"] = effective_status
    if renewed:
        normalized["latest_renewal_at"] = _utc_now_iso()
    return store.upsert_subscription(normalized["subscription_id"], normalized)


def expected_client_state(raw: str | None = None) -> str | None:
    if raw is None:
        from os import getenv

        raw = getenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "")
    value = str(raw or "").strip()
    return value or None


def is_managed_subscription(
    store: TeamsPipelineStore,
    subscription_payload: dict[str, Any],
    *,
    expected_client_state_value: str | None,
) -> bool:
    subscription_id = str(
        subscription_payload.get("subscription_id") or subscription_payload.get("id") or ""
    ).strip()
    if subscription_id and store.get_subscription(subscription_id):
        return True

    if expected_client_state_value:
        candidate_state = str(
            subscription_payload.get("client_state") or subscription_payload.get("clientState") or ""
        ).strip()
        if candidate_state and candidate_state == expected_client_state_value:
            return True

    return False


async def maintain_graph_subscriptions(
    *,
    client: MicrosoftGraphClient,
    store: TeamsPipelineStore,
    renew_within_hours: int = 24,
    extend_hours: int = 24,
    dry_run: bool = False,
    client_state: str | None = None,
) -> dict[str, Any]:
    threshold_hours = max(1, int(renew_within_hours))
    extend_hours = max(1, int(extend_hours))
    managed_client_state = expected_client_state(client_state)
    now = _utc_now()

    remote_subscriptions = await client.collect_paginated("/subscriptions")
    remote_ids: set[str] = set()
    synced = 0
    renewed: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw in remote_subscriptions:
        if not isinstance(raw, dict):
            continue
        subscription_id = str(raw.get("id") or "").strip()
        if not subscription_id:
            continue
        managed = is_managed_subscription(
            store,
            raw,
            expected_client_state_value=managed_client_state,
        )
        if not managed:
            skipped.append(
                {
                    "subscription_id": subscription_id,
                    "reason": "not_managed_by_teams_pipeline",
                }
            )
            continue

        remote_ids.add(subscription_id)
        try:
            sync_graph_subscription_record(store, raw)
            synced += 1
        except Exception as exc:
            skipped.append(
                {
                    "subscription_id": subscription_id,
                    "reason": f"failed_to_sync_local_store: {exc}",
                }
            )
            continue

        expiration = _parse_datetime(raw.get("expirationDateTime"))
        if expiration is None:
            skipped.append({"subscription_id": subscription_id, "reason": "missing_expiration"})
            continue

        seconds_until_expiry = int((expiration - now).total_seconds())
        if seconds_until_expiry < 0:
            store.upsert_subscription(
                subscription_id,
                {
                    "status": "expired",
                    "expiration_datetime": expiration.isoformat().replace("+00:00", "Z"),
                },
            )
            skipped.append(
                {
                    "subscription_id": subscription_id,
                    "reason": "already_expired",
                    "expiration_datetime": expiration.isoformat().replace("+00:00", "Z"),
                }
            )
            continue

        if seconds_until_expiry > threshold_hours * 3600:
            skipped.append(
                {
                    "subscription_id": subscription_id,
                    "reason": "not_due",
                    "expires_in_seconds": seconds_until_expiry,
                }
            )
            continue

        new_expiration = (max(now, expiration) + timedelta(hours=extend_hours)).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
        candidate = {
            "subscription_id": subscription_id,
            "resource": raw.get("resource"),
            "current_expiration": expiration.isoformat().replace("+00:00", "Z"),
            "new_expiration": new_expiration,
        }
        candidates.append(candidate)
        if dry_run:
            continue

        patched = await client.patch_json(
            f"/subscriptions/{subscription_id}",
            json_body={"expirationDateTime": new_expiration},
        )
        merged = {**raw, **(patched or {}), "id": subscription_id, "expirationDateTime": new_expiration}
        sync_graph_subscription_record(store, merged, status="active", renewed=True)
        renewed.append({**candidate, "result": patched})

    for subscription_id in store.list_subscriptions():
        if subscription_id in remote_ids:
            continue
        store.upsert_subscription(
            subscription_id,
            {
                "status": "missing_remote",
                "last_seen_missing_remote_at": _utc_now_iso(),
            },
        )

    return {
        "success": True,
        "dry_run": bool(dry_run),
        "store_path": str(store.path),
        "remote_subscription_count": len(remote_subscriptions),
        "synced_subscription_count": synced,
        "candidate_count": len(candidates),
        "renewed_count": len(renewed),
        "threshold_hours": threshold_hours,
        "extend_hours": extend_hours,
        "candidates": candidates,
        "renewed": renewed,
        "skipped": skipped,
    }
