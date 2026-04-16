from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, Optional
from urllib.parse import quote

from gateway.platforms.msteams_mentions import strip_leading_teams_mentions, strip_teams_mentions
from gateway.platforms.msteams_state import ConversationRef

logger = logging.getLogger(__name__)

BOTFRAMEWORK_SCOPE = "https://api.botframework.com/.default"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
GRAPH_BETA_BASE_URL = "https://graph.microsoft.com/beta"
USER_SELECT_FIELDS = "id,displayName,mail,jobTitle,userPrincipalName,officeLocation"
THREAD_CONTEXT_HEADER = "[Thread context — prior messages in this thread (not yet in conversation history):]"
THREAD_CONTEXT_FOOTER = "[End of thread context]"
RECENT_CONTEXT_HEADER = "[Recent Teams context — prior messages not yet in conversation history:]"
RECENT_CONTEXT_FOOTER = "[End of recent Teams context]"


class _OAuthClientCredentialsMixin:
    def __init__(self, app_id: str, app_password: str, tenant_id: str, session) -> None:
        self._app_id = app_id
        self._app_password = app_password
        self._tenant_id = tenant_id
        self._session = session
        self._token_cache: dict[str, tuple[str, float]] = {}
        self._token_lock = asyncio.Lock()

    async def _get_token_for_scope(self, scope: str) -> str:
        async with self._token_lock:
            cached = self._token_cache.get(scope)
            now = asyncio.get_running_loop().time()
            if cached and cached[1] > now + 60:
                return cached[0]

            data = {
                "grant_type": "client_credentials",
                "client_id": self._app_id,
                "client_secret": self._app_password,
                "scope": scope,
            }
            async with self._session.post(
                TOKEN_URL_TMPL.format(tenant_id=self._tenant_id),
                data=data,
            ) as resp:
                payload = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"MSTeams token request failed ({resp.status}): {payload}")
                token = str(payload["access_token"])
                expires_in = int(payload.get("expires_in", 3600))
                self._token_cache[scope] = (token, now + expires_in)
                return token


class MSTeamsGraphClient(_OAuthClientCredentialsMixin):
    """Microsoft Graph helper for Teams user and history enrichments."""

    async def _graph_request(
        self,
        method_name: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Any = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = await self._get_token_for_scope(GRAPH_SCOPE)
        merged_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)
        if content_type:
            merged_headers["Content-Type"] = content_type
        url = f"{GRAPH_BASE_URL}{path}"
        method = getattr(self._session, method_name)
        request_kwargs = {"params": params, "headers": merged_headers}
        if json_body is not None:
            request_kwargs["json"] = json_body
        if data is not None:
            request_kwargs["data"] = data
        async with method(url, **request_kwargs) as resp:
            try:
                payload = await resp.json(content_type=None)
            except Exception:
                payload = {} if resp.status < 400 else await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"MSTeams Graph request failed ({resp.status}): {payload}")
            if not isinstance(payload, dict):
                return {"raw": payload}
            return payload

    async def _graph_get(self, path: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return await self._graph_request("get", path, params=params, headers=headers)

    async def _graph_post(self, path: str, *, json_body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._graph_request("post", path, json_body=json_body)

    async def _graph_put_bytes(self, path: str, data: bytes, *, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        return await self._graph_request("put", path, data=data, content_type=content_type)

    async def _graph_beta_post(self, path: str, *, json_body: Dict[str, Any]) -> Dict[str, Any]:
        token = await self._get_token_for_scope(GRAPH_SCOPE)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"{GRAPH_BETA_BASE_URL}{path}"
        async with self._session.post(url, json=json_body, headers=headers) as resp:
            try:
                payload = await resp.json(content_type=None)
            except Exception:
                payload = {} if resp.status < 400 else await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"MSTeams Graph beta request failed ({resp.status}): {payload}")
            return payload if isinstance(payload, dict) else {"raw": payload}

    async def _graph_get_bytes(self, path: str) -> bytes:
        token = await self._get_token_for_scope(GRAPH_SCOPE)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "*/*",
        }
        url = f"{GRAPH_BASE_URL}{path}"
        async with self._session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                try:
                    payload = await resp.json(content_type=None)
                except Exception:
                    payload = await resp.text()
                raise RuntimeError(f"MSTeams Graph bytes request failed ({resp.status}): {payload}")
            return await resp.read()

    @staticmethod
    def _normalize_user(user: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(user.get("id") or "").strip(),
            "displayName": str(user.get("displayName") or "").strip(),
            "mail": str(user.get("mail") or "").strip(),
            "jobTitle": str(user.get("jobTitle") or "").strip(),
            "userPrincipalName": str(user.get("userPrincipalName") or "").strip(),
            "officeLocation": str(user.get("officeLocation") or "").strip(),
        }

    async def get_member_info(self, user_id: str) -> Dict[str, Any]:
        payload = await self._graph_get(f"/users/{user_id}", params={"$select": USER_SELECT_FIELDS})
        return {"user": self._normalize_user(payload)}

    async def search_users(self, query: str, *, limit: int = 5) -> list[Dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        params: Dict[str, Any] = {
            "$select": USER_SELECT_FIELDS,
            "$top": max(1, min(int(limit or 5), 25)),
        }
        headers: Dict[str, str] = {}
        if "@" in normalized_query:
            escaped = normalized_query.replace("'", "''")
            params["$filter"] = f"mail eq '{escaped}' or userPrincipalName eq '{escaped}'"
        else:
            params["$search"] = f'"{normalized_query}"'
            headers["ConsistencyLevel"] = "eventual"
        payload = await self._graph_get("/users", params=params, headers=headers)
        values = payload.get("value") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return []
        return [self._normalize_user(item) for item in values if isinstance(item, dict)]

    @staticmethod
    def _extract_message_text(message: Dict[str, Any]) -> str:
        body = message.get("body") if isinstance(message, dict) else None
        if isinstance(body, dict):
            content = str(body.get("content") or "").strip()
            if content:
                return strip_teams_mentions(strip_leading_teams_mentions(content))
        return str(message.get("summary") or message.get("subject") or "").strip()

    @staticmethod
    def _extract_message_sender(message: Dict[str, Any]) -> tuple[str, str, bool]:
        from_payload = message.get("from") if isinstance(message, dict) else None
        if not isinstance(from_payload, dict):
            return "", "unknown", False
        user = from_payload.get("user") if isinstance(from_payload.get("user"), dict) else {}
        app = from_payload.get("application") if isinstance(from_payload.get("application"), dict) else {}
        sender_id = str(user.get("id") or app.get("id") or "").strip()
        sender_name = str(user.get("displayName") or app.get("displayName") or "unknown").strip() or "unknown"
        is_bot = bool(app) or str(message.get("messageType") or "").lower() == "systemeventmessage"
        return sender_id, sender_name, is_bot

    @staticmethod
    def _format_recent_context(prefix: str, messages: list[tuple[str, str]]) -> str:
        if not messages:
            return ""
        lines = [f"{name}: {text}" for name, text in messages if text]
        if not lines:
            return ""
        return prefix + "\n" + "\n".join(lines) + "\n" + RECENT_CONTEXT_FOOTER + "\n\n"

    async def get_channel_message(self, team_id: str, channel_id: str, message_id: str) -> Dict[str, Any]:
        return await self._graph_get(f"/teams/{team_id}/channels/{channel_id}/messages/{message_id}")

    async def list_channel_thread_replies(self, team_id: str, channel_id: str, message_id: str, *, limit: int = 20) -> list[Dict[str, Any]]:
        payload = await self._graph_get(
            f"/teams/{team_id}/channels/{channel_id}/messages/{message_id}/replies",
            params={"$top": max(1, min(int(limit or 20), 50))},
        )
        values = payload.get("value") if isinstance(payload, dict) else None
        return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []

    async def list_channel_messages(self, team_id: str, channel_id: str, *, limit: int = 20) -> list[Dict[str, Any]]:
        payload = await self._graph_get(
            f"/teams/{team_id}/channels/{channel_id}/messages",
            params={"$top": max(1, min(int(limit or 20), 50))},
        )
        values = payload.get("value") if isinstance(payload, dict) else None
        return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []

    async def list_chat_messages(self, chat_id: str, *, limit: int = 20) -> list[Dict[str, Any]]:
        payload = await self._graph_get(
            f"/chats/{quote(chat_id, safe='')}/messages",
            params={"$top": max(1, min(int(limit or 20), 50))},
        )
        values = payload.get("value") if isinstance(payload, dict) else None
        return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []

    async def build_thread_context(self, team_id: str, channel_id: str, message_id: str, *, limit: int = 20) -> str:
        parent = await self.get_channel_message(team_id, channel_id, message_id)
        replies = await self.list_channel_thread_replies(team_id, channel_id, message_id, limit=limit)
        parts: list[str] = []

        parent_text = self._extract_message_text(parent)
        if parent_text:
            parent_from = parent.get("from") if isinstance(parent, dict) else None
            parent_name = "unknown"
            if isinstance(parent_from, dict):
                user = parent_from.get("user") if isinstance(parent_from.get("user"), dict) else {}
                parent_name = str(user.get("displayName") or parent_from.get("application", {}).get("displayName") or "unknown")
            parts.append(f"[thread parent] {parent_name}: {parent_text}")

        for reply in replies:
            reply_id = str(reply.get("id") or "")
            if reply_id == str(message_id):
                continue
            reply_text = self._extract_message_text(reply)
            if not reply_text:
                continue
            reply_from = reply.get("from") if isinstance(reply, dict) else None
            reply_name = "unknown"
            if isinstance(reply_from, dict):
                user = reply_from.get("user") if isinstance(reply_from.get("user"), dict) else {}
                reply_name = str(user.get("displayName") or reply_from.get("application", {}).get("displayName") or "unknown")
            parts.append(f"{reply_name}: {reply_text}")

        if not parts:
            return ""
        return THREAD_CONTEXT_HEADER + "\n" + "\n".join(parts) + "\n" + THREAD_CONTEXT_FOOTER + "\n\n"

    async def build_recent_channel_context(
        self,
        team_id: str,
        channel_id: str,
        *,
        current_message_id: str | None = None,
        limit: int = 20,
        allowed_sender_ids: set[str] | None = None,
    ) -> str:
        messages = await self.list_channel_messages(team_id, channel_id, limit=limit)
        normalized_current_id = str(current_message_id or "").strip()
        items: list[tuple[str, str]] = []
        for message in reversed(messages):
            message_id = str(message.get("id") or "").strip()
            if normalized_current_id and message_id == normalized_current_id:
                continue
            text = self._extract_message_text(message)
            if not text:
                continue
            sender_id, sender_name, is_bot = self._extract_message_sender(message)
            if is_bot:
                continue
            if allowed_sender_ids is not None and sender_id not in allowed_sender_ids:
                continue
            items.append((sender_name, text))
        return self._format_recent_context(RECENT_CONTEXT_HEADER, items)

    async def build_recent_chat_context(
        self,
        chat_id: str,
        *,
        current_message_id: str | None = None,
        limit: int = 20,
        allowed_sender_ids: set[str] | None = None,
        user_turns_only: bool = False,
    ) -> str:
        messages = await self.list_chat_messages(chat_id, limit=limit)
        normalized_current_id = str(current_message_id or "").strip()
        items: list[tuple[str, str]] = []
        for message in reversed(messages):
            message_id = str(message.get("id") or "").strip()
            if normalized_current_id and message_id == normalized_current_id:
                continue
            text = self._extract_message_text(message)
            if not text:
                continue
            sender_id, sender_name, is_bot = self._extract_message_sender(message)
            if user_turns_only and is_bot:
                continue
            if allowed_sender_ids is not None and sender_id not in allowed_sender_ids:
                continue
            items.append((sender_name, text))
        return self._format_recent_context(RECENT_CONTEXT_HEADER, items)

    async def upload_file_to_sharepoint(
        self,
        site_id: str,
        file_name: str,
        data: bytes,
        *,
        folder: str = "HermesShared",
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        normalized_site_id = str(site_id or "").strip()
        normalized_name = str(file_name or "").strip()
        if not normalized_site_id:
            raise RuntimeError("SharePoint site ID is required")
        if not normalized_name:
            raise RuntimeError("File name is required")
        encoded_folder = "/".join(quote(part, safe="") for part in folder.split("/") if part)
        encoded_name = quote(normalized_name, safe="")
        path = f"/sites/{normalized_site_id}/drive/root:/{encoded_folder}/{encoded_name}:/content"
        return await self._graph_put_bytes(path, data, content_type=content_type)

    async def create_sharepoint_link(
        self,
        site_id: str,
        item_id: str,
        *,
        scope: str = "organization",
        link_type: str = "view",
    ) -> Dict[str, Any]:
        normalized_site_id = str(site_id or "").strip()
        normalized_item_id = str(item_id or "").strip()
        if not normalized_site_id or not normalized_item_id:
            raise RuntimeError("SharePoint site ID and item ID are required to create a sharing link")
        return await self._graph_post(
            f"/sites/{normalized_site_id}/drive/items/{normalized_item_id}/createLink",
            json_body={"type": link_type, "scope": scope},
        )

    @staticmethod
    def _conversation_base_path(ref: ConversationRef) -> str:
        if ref.chat_type == "channel":
            if not ref.team_id or not ref.channel_id:
                raise RuntimeError("Teams channel conversation requires team_id and channel_id")
            return f"/teams/{quote(ref.team_id, safe='')}/channels/{quote(ref.channel_id, safe='')}"
        return f"/chats/{quote(ref.conversation_id, safe='')}"

    @classmethod
    def _message_path(cls, ref: ConversationRef, message_id: str) -> str:
        return f"{cls._conversation_base_path(ref)}/messages/{quote(message_id, safe='')}"

    def build_message_url_candidates(self, activity: Dict[str, Any]) -> list[str]:
        conversation = activity.get("conversation") if isinstance(activity.get("conversation"), dict) else {}
        conversation_type = str(conversation.get("conversationType") or "").strip().lower()
        conversation_id = str(conversation.get("id") or "").strip()
        reply_to_id = str(activity.get("replyToId") or "").strip()
        channel_data = activity.get("channelData") if isinstance(activity.get("channelData"), dict) else {}
        team_id = str(((channel_data.get("team") or {}) if isinstance(channel_data.get("team"), dict) else {}).get("id") or "").strip()
        channel_id = str(((channel_data.get("channel") or {}) if isinstance(channel_data.get("channel"), dict) else {}).get("id") or "").strip()

        candidates: list[str] = []
        for candidate in (
            str(activity.get("id") or "").strip(),
            str(channel_data.get("messageId") or "").strip(),
            str(channel_data.get("teamsMessageId") or "").strip(),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if conversation_type == "personal" and conversation_id.startswith("a:"):
            from_user = activity.get("from") if isinstance(activity.get("from"), dict) else {}
            aad_object_id = str(from_user.get("aadObjectId") or from_user.get("id") or "").strip()
            if aad_object_id and self._app_id:
                conversation_id = f"19:{aad_object_id}_{self._app_id}@unq.gbl.spaces"

        urls: list[str] = []
        if team_id and channel_id:
            for candidate in candidates:
                if reply_to_id:
                    urls.append(f"/teams/{team_id}/channels/{channel_id}/messages/{reply_to_id}/replies/{candidate}")
                urls.append(f"/teams/{team_id}/channels/{channel_id}/messages/{candidate}")
        elif conversation_id:
            for candidate in candidates:
                urls.append(f"/chats/{conversation_id}/messages/{candidate}")

        deduped: list[str] = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        return deduped

    @staticmethod
    def _graph_share_content_path(share_url: str) -> str:
        encoded = base64.urlsafe_b64encode(share_url.encode("utf-8")).decode("ascii").rstrip("=")
        return f"/shares/u!{encoded}/driveItem/content"

    async def download_message_media(self, message_path: str) -> list[Dict[str, Any]]:
        message = await self._graph_get(message_path)
        recovered: list[Dict[str, Any]] = []

        attachments = message.get("attachments") if isinstance(message.get("attachments"), list) else []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            content_type = str(attachment.get("contentType") or "").strip().lower()
            if content_type != "reference":
                continue
            content_url = str(attachment.get("contentUrl") or "").strip()
            if not content_url:
                continue
            recovered.append({
                "kind": "reference",
                "content_url": content_url,
                "name": str(attachment.get("name") or "attachment"),
                "content_type": str(attachment.get("contentType") or "application/octet-stream"),
                "graph_path": self._graph_share_content_path(content_url),
            })

        hosted_contents = await self._graph_get(f"{message_path}/hostedContents")
        for item in hosted_contents.get("value") if isinstance(hosted_contents.get("value"), list) else []:
            if not isinstance(item, dict):
                continue
            content_bytes = item.get("contentBytes")
            if isinstance(content_bytes, str) and content_bytes.strip():
                data = base64.b64decode(content_bytes)
            else:
                hosted_id = str(item.get("id") or "").strip()
                if not hosted_id:
                    continue
                data = await self._graph_get_bytes(f"{message_path}/hostedContents/{hosted_id}/$value")
            recovered.append({
                "kind": "hosted",
                "name": str(item.get("contentType") or item.get("id") or "hosted-content"),
                "content_type": str(item.get("contentType") or "application/octet-stream"),
                "data": data,
            })

        return recovered


    async def get_message(self, ref: ConversationRef, message_id: str) -> Dict[str, Any]:
        payload = await self._graph_get(self._message_path(ref, message_id))
        return {
            "id": str(payload.get("id") or message_id),
            "text": (((payload.get("body") or {}) if isinstance(payload.get("body"), dict) else {}).get("content")),
            "from": payload.get("from"),
            "createdAt": payload.get("createdDateTime"),
        }

    async def pin_message(self, ref: ConversationRef, message_id: str) -> Dict[str, Any]:
        base_path = self._conversation_base_path(ref)
        payload = await self._graph_post(f"{base_path}/pinnedMessages", json_body={"message": {"id": message_id}})
        return {"ok": True, "pinnedMessageId": str(payload.get("id") or "").strip() or None}

    async def unpin_message(self, ref: ConversationRef, pinned_message_id: str) -> Dict[str, Any]:
        base_path = self._conversation_base_path(ref)
        await self._graph_request("delete", f"{base_path}/pinnedMessages/{quote(pinned_message_id, safe='')}")
        return {"ok": True}

    async def list_pins(self, ref: ConversationRef) -> Dict[str, Any]:
        base_path = self._conversation_base_path(ref)
        payload = await self._graph_get(f"{base_path}/pinnedMessages", params={"$expand": "message"})
        values = payload.get("value") if isinstance(payload.get("value"), list) else []
        return {
            "pins": [
                {
                    "id": str(item.get("id") or ""),
                    "pinnedMessageId": str(item.get("id") or ""),
                    "messageId": str(((item.get("message") or {}) if isinstance(item.get("message"), dict) else {}).get("id") or "") or None,
                    "text": (((item.get("message") or {}) if isinstance(item.get("message"), dict) else {}).get("body") or {}).get("content"),
                }
                for item in values
                if isinstance(item, dict)
            ]
        }

    async def set_reaction(self, ref: ConversationRef, message_id: str, reaction_type: str) -> Dict[str, Any]:
        path = f"{self._message_path(ref, message_id)}/setReaction"
        await self._graph_beta_post(path, json_body={"reactionType": reaction_type})
        return {"ok": True}

    async def unset_reaction(self, ref: ConversationRef, message_id: str, reaction_type: str) -> Dict[str, Any]:
        path = f"{self._message_path(ref, message_id)}/unsetReaction"
        await self._graph_beta_post(path, json_body={"reactionType": reaction_type})
        return {"ok": True}

    async def list_reactions(self, ref: ConversationRef, message_id: str) -> Dict[str, Any]:
        payload = await self._graph_get(self._message_path(ref, message_id))
        grouped: dict[str, list[Dict[str, str]]] = {}
        for reaction in payload.get("reactions") if isinstance(payload.get("reactions"), list) else []:
            if not isinstance(reaction, dict):
                continue
            rtype = str(reaction.get("reactionType") or "unknown")
            grouped.setdefault(rtype, [])
            user = reaction.get("user") if isinstance(reaction.get("user"), dict) else {}
            if user.get("id"):
                grouped[rtype].append({"id": str(user.get("id")), "displayName": str(user.get("displayName") or "")})
        return {"reactions": [{"reactionType": rtype, "count": len(users), "users": users} for rtype, users in grouped.items()]}

    async def search_messages(self, ref: ConversationRef, query: str, *, from_display_name: Optional[str] = None, limit: int = 25) -> Dict[str, Any]:
        top = max(1, min(int(limit or 25), 50))
        sanitized = str(query or "").replace('"', "").strip()
        params = {"$search": f'"{sanitized}"', "$top": str(top)}
        if from_display_name:
            escaped = str(from_display_name).replace("'", "''")
            params["$filter"] = f"from/user/displayName eq '{escaped}'"
        payload = await self._graph_get(f"{self._conversation_base_path(ref)}/messages", params=params, headers={"ConsistencyLevel": "eventual"})
        values = payload.get("value") if isinstance(payload.get("value"), list) else []
        return {
            "messages": [
                {
                    "id": str(msg.get("id") or ""),
                    "text": (((msg.get("body") or {}) if isinstance(msg.get("body"), dict) else {}).get("content")),
                    "from": msg.get("from"),
                    "createdAt": msg.get("createdDateTime"),
                }
                for msg in values
                if isinstance(msg, dict)
            ]
        }

    async def list_channels(self, team_id: str) -> Dict[str, Any]:
        next_path = f"/teams/{quote(team_id, safe='')}/channels"
        params = {"$select": "id,displayName,description,membershipType"}
        collected = []
        pages = 0
        while next_path and pages < 10:
            payload = await self._graph_get(next_path, params=params if pages == 0 else None)
            values = payload.get("value") if isinstance(payload.get("value"), list) else []
            collected.extend(item for item in values if isinstance(item, dict))
            next_link = str(payload.get("@odata.nextLink") or "").strip()
            next_path = next_link.replace(f"{GRAPH_BASE_URL}", "") if next_link else ""
            pages += 1
            params = None
        return {"channels": [{"id": str(ch.get("id") or ""), "displayName": str(ch.get("displayName") or ""), "description": str(ch.get("description") or ""), "membershipType": str(ch.get("membershipType") or "")} for ch in collected], "truncated": bool(next_path)}

    async def get_channel_info(self, team_id: str, channel_id: str) -> Dict[str, Any]:
        payload = await self._graph_get(f"/teams/{quote(team_id, safe='')}/channels/{quote(channel_id, safe='')}", params={"$select": "id,displayName,description,membershipType,webUrl,createdDateTime"})
        return {"channel": {"id": str(payload.get("id") or ""), "displayName": str(payload.get("displayName") or ""), "description": str(payload.get("description") or ""), "membershipType": str(payload.get("membershipType") or ""), "webUrl": str(payload.get("webUrl") or ""), "createdDateTime": str(payload.get("createdDateTime") or "")}}


class MSTeamsBotClient(_OAuthClientCredentialsMixin):
    """Small Bot Framework client for Teams adapter outbound sends."""

    async def _get_token(self) -> str:
        return await self._get_token_for_scope(BOTFRAMEWORK_SCOPE)

    async def _request_activity(
        self,
        method_name: str,
        ref: ConversationRef,
        payload: Dict[str, Any],
        *,
        activity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = await self._get_token()
        base = ref.service_url.rstrip("/")
        if activity_id:
            url = f"{base}/v3/conversations/{ref.conversation_id}/activities/{activity_id}"
        else:
            url = f"{base}/v3/conversations/{ref.conversation_id}/activities"

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        method = getattr(self._session, method_name)
        async with method(url, json=payload, headers=headers) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"MSTeams send failed ({resp.status}): {body}")
            return body if isinstance(body, dict) else {"raw": body}

    async def _send_activity(
        self,
        ref: ConversationRef,
        payload: Dict[str, Any],
        *,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._request_activity("post", ref, payload, activity_id=reply_to)

    @staticmethod
    def _apply_text_payload(payload: Dict[str, Any], content: str) -> None:
        if not content:
            return
        # Follow OpenClaw's Teams send path for normal text: preserve markdown as-is
        # and let Teams render its supported markdown subset. Mention tokens are already
        # converted upstream into <at> tags + entities when needed.
        payload["text"] = content

    async def update_message(
        self,
        ref: ConversationRef,
        message_id: str,
        content: str,
        *,
        entities: Optional[list[dict[str, Any]]] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": "message", "id": message_id}
        self._apply_text_payload(payload, content)
        if entities:
            payload["entities"] = entities
        if attachments:
            payload["attachments"] = attachments
        return await self._request_activity("put", ref, payload, activity_id=message_id)

    async def delete_message(self, ref: ConversationRef, message_id: str) -> None:
        await self._request_activity("delete", ref, {}, activity_id=message_id)

    async def send_message(
        self,
        ref: ConversationRef,
        content: str,
        *,
        reply_to: Optional[str] = None,
        entities: Optional[list[dict[str, Any]]] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": "message"}
        self._apply_text_payload(payload, content)
        if entities:
            payload["entities"] = entities
        if attachments:
            payload["attachments"] = attachments
        return await self._send_activity(ref, payload, reply_to=reply_to)

    async def send_typing(self, ref: ConversationRef) -> Dict[str, Any]:
        return await self._send_activity(ref, {"type": "typing"})
