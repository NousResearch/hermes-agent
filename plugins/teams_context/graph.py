"""Microsoft Graph operations for Teams chat context."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from plugins.teams_context.models import TeamsChatMessage, parse_graph_datetime
from plugins.teams_context.store import TeamsContextStore
from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClient


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc_timestamp(hours_from_now: int = 24) -> str:
    return (_utc_now() + timedelta(hours=hours_from_now)).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


class TeamsContextGraph:
    def __init__(
        self,
        *,
        client: MicrosoftGraphClient,
        store: TeamsContextStore,
        tenant_id: str | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.tenant_id = tenant_id

    async def backfill_chat(
        self,
        chat_id: str,
        *,
        since_days: int = 30,
        limit: int | None = None,
    ) -> dict[str, Any]:
        since = _utc_now() - timedelta(days=max(0, int(since_days)))
        path = f"/chats/{quote(chat_id, safe='')}/messages"
        count = 0
        scanned = 0
        async for page in self.client.iterate_pages(path, params={"$top": 50}):
            for raw in page.get("value") or []:
                if not isinstance(raw, dict):
                    continue
                scanned += 1
                created_at = parse_graph_datetime(raw.get("createdDateTime"))
                if created_at and created_at < since:
                    continue
                message = TeamsChatMessage.from_graph(
                    chat_id,
                    raw,
                    tenant_id=self.tenant_id,
                )
                self.store.upsert_message(message)
                count += 1
                if limit and count >= limit:
                    return {"chat_id": chat_id, "scanned": scanned, "stored": count}
        return {"chat_id": chat_id, "scanned": scanned, "stored": count}

    async def ingest_notification(self, notification: dict[str, Any]) -> dict[str, Any]:
        subscription_id = str(notification.get("subscriptionId") or "").strip()
        resource = str(notification.get("resource") or "").strip()
        from plugins.teams_context.models import parse_chat_resource

        chat_id, message_id = parse_chat_resource(resource)
        if not chat_id and subscription_id:
            chat_id = self.store.subscription_chat_id(subscription_id)
        resource_data = notification.get("resourceData")
        if not message_id and isinstance(resource_data, dict):
            message_id = str(resource_data.get("id") or "").strip() or None
        if not chat_id or not message_id:
            return {"ignored": True, "reason": "missing_chat_or_message_id", "resource": resource}

        change_type = str(notification.get("changeType") or "").lower()
        if "deleted" in change_type:
            self.store.mark_deleted(chat_id, message_id)
            return {"deleted": True, "chat_id": chat_id, "message_id": message_id}

        path = f"/chats/{quote(chat_id, safe='')}/messages/{quote(message_id, safe='')}"
        try:
            payload = await self.client.get_json(path)
        except MicrosoftGraphAPIError as exc:
            if exc.status_code == 404:
                self.store.mark_deleted(chat_id, message_id)
                return {"deleted": True, "chat_id": chat_id, "message_id": message_id}
            raise
        if not isinstance(payload, dict):
            return {"ignored": True, "reason": "message_payload_not_object"}
        message = TeamsChatMessage.from_graph(chat_id, payload, tenant_id=self.tenant_id)
        self.store.upsert_message(message)
        return {"stored": True, "chat_id": chat_id, "message_id": message_id}

    async def create_subscription(
        self,
        *,
        chat_id: str,
        notification_url: str,
        client_state: str | None = None,
        expiration: str | None = None,
        change_type: str = "created,updated,deleted",
        lifecycle_notification_url: str | None = None,
    ) -> dict[str, Any]:
        resource = f"chats/{chat_id}/messages"
        payload: dict[str, Any] = {
            "changeType": change_type,
            "notificationUrl": notification_url,
            "resource": resource,
            "expirationDateTime": expiration or iso_utc_timestamp(1),
            "latestSupportedTlsVersion": "v1_2",
        }
        if client_state:
            payload["clientState"] = client_state
        if lifecycle_notification_url:
            payload["lifecycleNotificationUrl"] = lifecycle_notification_url
        result = await self.client.post_json("/subscriptions", json_body=payload)
        if isinstance(result, dict):
            self.store.upsert_subscription(result, chat_id=chat_id)
        return result

    async def renew_due_subscriptions(
        self,
        *,
        renew_within_hours: int = 24,
        extend_hours: int = 24,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        now = _utc_now()
        renewed: list[dict[str, Any]] = []
        due: list[dict[str, Any]] = []
        for record in self.store.list_subscriptions():
            exp = parse_graph_datetime(record.get("expiration_datetime"))
            if exp is None:
                continue
            if (exp - now).total_seconds() > max(1, renew_within_hours) * 3600:
                continue
            new_exp = (max(now, exp) + timedelta(hours=max(1, extend_hours))).replace(
                microsecond=0
            ).isoformat().replace("+00:00", "Z")
            item = {
                "subscription_id": record["subscription_id"],
                "chat_id": record["chat_id"],
                "current_expiration": record.get("expiration_datetime"),
                "new_expiration": new_exp,
            }
            due.append(item)
            if dry_run:
                continue
            result = await self.client.patch_json(
                f"/subscriptions/{quote(record['subscription_id'], safe='')}",
                json_body={"expirationDateTime": new_exp},
            )
            merged = {**record, **(result or {}), "id": record["subscription_id"], "expirationDateTime": new_exp}
            self.store.upsert_subscription(merged, chat_id=record["chat_id"])
            renewed.append(item)
        return {"dry_run": dry_run, "due": due, "renewed": renewed}

