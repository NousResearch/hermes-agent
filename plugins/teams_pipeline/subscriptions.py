"""Microsoft Graph subscription helpers for the Teams pipeline plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from os import getenv
from typing import Any

from plugins.teams_pipeline.models import GraphSubscription, _parse_datetime
from plugins.teams_pipeline.models import _serialize_datetime as _iso_z
from plugins.teams_pipeline.store import TeamsPipelineStore
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
from tools.microsoft_graph_client import MicrosoftGraphClient


def build_graph_client() -> MicrosoftGraphClient:
    return MicrosoftGraphClient(MicrosoftGraphTokenProvider.from_env())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(hours_from_now: int = 0, *, base: datetime | None = None) -> str:
    """Second-precision UTC ISO timestamp with a ``Z`` suffix (Graph's expirationDateTime format)."""
    return _iso_z(((base or _utc_now()) + timedelta(hours=hours_from_now)).replace(microsecond=0))


def sync_graph_subscription_record(
    store: TeamsPipelineStore, subscription_payload: dict[str, Any], *, status: str | None = None, renewed: bool = False,
) -> dict[str, Any]:
    normalized = GraphSubscription.from_dict(subscription_payload).to_dict()
    if status is None:
        expiration = _parse_datetime(normalized.get("expiration_datetime"))
        status = "expired" if expiration and expiration <= _utc_now() else "active"
    normalized["status"] = status
    if renewed:
        normalized["latest_renewal_at"] = utc_timestamp()
    return store.upsert_subscription(normalized["subscription_id"], normalized)


def expected_client_state(raw: str | None = None) -> str | None:
    if raw is None:
        raw = getenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "")
    return str(raw or "").strip() or None


def is_managed_subscription(store: TeamsPipelineStore, subscription_payload: dict[str, Any], *, expected_client_state_value: str | None) -> bool:
    """A subscription is ours if the store knows it or its clientState matches the configured one."""
    subscription_id = str(subscription_payload.get("subscription_id") or subscription_payload.get("id") or "").strip()
    if subscription_id and store.get_subscription(subscription_id):
        return True
    candidate_state = str(subscription_payload.get("client_state") or subscription_payload.get("clientState") or "").strip()
    return bool(expected_client_state_value and candidate_state == expected_client_state_value)


async def maintain_graph_subscriptions(
    *, client: MicrosoftGraphClient, store: TeamsPipelineStore, renew_within_hours: int = 24, extend_hours: int = 24,
    dry_run: bool = False, client_state: str | None = None,
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

        def skip(reason: str, **extra: Any) -> None:
            skipped.append({"subscription_id": subscription_id, "reason": reason, **extra})
        if not is_managed_subscription(store, raw, expected_client_state_value=managed_client_state):
            skip("not_managed_by_teams_pipeline")
            continue
        remote_ids.add(subscription_id)
        try:
            sync_graph_subscription_record(store, raw)
            synced += 1
        except Exception as exc:
            skip(f"failed_to_sync_local_store: {exc}")
            continue
        expiration = _parse_datetime(raw.get("expirationDateTime"))
        if expiration is None:
            skip("missing_expiration")
            continue
        seconds_until_expiry = int((expiration - now).total_seconds())
        if seconds_until_expiry < 0:
            store.upsert_subscription(subscription_id, {"status": "expired", "expiration_datetime": _iso_z(expiration)})
            skip("already_expired", expiration_datetime=_iso_z(expiration))
            continue
        if seconds_until_expiry > threshold_hours * 3600:
            skip("not_due", expires_in_seconds=seconds_until_expiry)
            continue
        new_expiration = utc_timestamp(extend_hours, base=max(now, expiration))
        candidate = {"subscription_id": subscription_id, "resource": raw.get("resource"),
                     "current_expiration": _iso_z(expiration), "new_expiration": new_expiration}
        candidates.append(candidate)
        if dry_run:
            continue
        patched = await client.patch_json(f"/subscriptions/{subscription_id}", json_body={"expirationDateTime": new_expiration})
        merged = {**raw, **(patched or {}), "id": subscription_id, "expirationDateTime": new_expiration}
        sync_graph_subscription_record(store, merged, status="active", renewed=True)
        renewed.append({**candidate, "result": patched})
    # Locally-known subscriptions Graph no longer reports are flagged, never deleted.
    for subscription_id in store.list_subscriptions():
        if subscription_id not in remote_ids:
            store.upsert_subscription(subscription_id, {"status": "missing_remote", "last_seen_missing_remote_at": utc_timestamp()})
    return {
        "success": True, "dry_run": bool(dry_run), "store_path": str(store.path),
        "remote_subscription_count": len(remote_subscriptions), "synced_subscription_count": synced,
        "candidate_count": len(candidates), "renewed_count": len(renewed),
        "threshold_hours": threshold_hours, "extend_hours": extend_hours,
        "candidates": candidates, "renewed": renewed, "skipped": skipped,
    }


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path  # noqa: F401,E402

def resolve_store_path(path: str | None) -> str:
    return str(resolve_teams_pipeline_store_path(path))

def build_store(path: str | None = None) -> TeamsPipelineStore:
    return TeamsPipelineStore(resolve_store_path(path))


_PLUGIN_COMPAT_LAZY = {
    'resolve_teams_pipeline_store_path': ('plugins.teams_pipeline.store', 'resolve_teams_pipeline_store_path'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
