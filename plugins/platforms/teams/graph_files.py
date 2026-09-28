"""Microsoft Graph helpers for Teams channel/group SharePoint files.

Outbound: Bot Framework document attachments 400 and FileConsent is personal-scope
only, so the adapter PUTs into the conversation's ``filesFolder`` drive and posts
the resulting webUrl / org sharing link.

Inbound: channel file drops often omit ``file.download.info`` entirely (Bot Framework
sends only a ``text/html`` body mirror). ``list_graph_message_file_refs`` GETs the
Graph channel/chat message and ``resolve_inbound_file_download_url`` then fetches
SharePoint bytes via filesFolder / the shares API using the same app-only credentials.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlparse

from tools.microsoft_graph_auth import GraphCredentials
from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClientError


# Simple PUT to ``:/content`` is capped at 4 MiB; larger files use an upload session.
SIMPLE_UPLOAD_MAX_BYTES = 4 * 1024 * 1024
# Application permission that covers filesFolder + content PUT + createLink + inbound download.
GRAPH_CHANNEL_FILE_PERMISSION = "Files.ReadWrite.All"
# GET /teams/{id}/channels/{id}/messages/{id} when Bot Framework only sent text/html.
GRAPH_CHANNEL_MESSAGE_PERMISSION = "ChannelMessage.Read.All"
GRAPH_CHANNEL_MESSAGE_RSC = "ChannelMessage.Read.Group"
GRAPH_CHAT_MESSAGE_PERMISSION = "Chat.Read.All"
GRAPH_CHAT_MESSAGE_RSC = "ChatMessage.Read.Chat"
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class GraphFileTarget:
    """SharePoint destination resolved from a Bot Framework conversation."""

    conversation_type: str  # "channel" | "groupChat"
    team_id: str = ""
    channel_id: str = ""
    chat_id: str = ""

    def files_folder_path(self) -> str:
        if self.team_id and self.channel_id:
            return (
                f"/teams/{quote(self.team_id, safe='')}"
                f"/channels/{quote(self.channel_id, safe='')}/filesFolder"
            )
        if self.chat_id:
            return f"/chats/{quote(self.chat_id, safe='')}/filesFolder"
        raise GraphFileTargetError(
            "Graph file upload needs team_id+channel_id or chat_id."
        )


@dataclass(frozen=True)
class GraphUploadedFile:
    """Drive item created by the upload (plus optional org sharing link)."""

    name: str
    web_url: str
    share_url: str = ""
    drive_id: str = ""
    item_id: str = ""
    size: int = 0

    @property
    def link(self) -> str:
        return self.share_url or self.web_url


class GraphFileTargetError(ValueError):
    """The conversation cannot be mapped to a Graph files folder."""


class GraphFileUploadError(RuntimeError):
    """Upload failed after the Graph client was constructed."""


def _field(value: Any, *names: str) -> Any:
    """Read a field from an SDK model or a dict (channelData uses both)."""
    if isinstance(value, dict):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
        return None
    for name in names:
        got = getattr(value, name, None)
        if got is not None:
            return got
    return None


def _as_text(value: Any) -> str:
    """IDs must be real strings — never ``str(MagicMock)`` / SDK stubs."""
    return value.strip() if isinstance(value, str) else ""


def _is_mock(value: Any) -> bool:
    return type(value).__module__.startswith("unittest.mock")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or _is_mock(value):
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        for kwargs in ({"by_alias": True}, {}):
            try:
                dumped = dump(**kwargs) if kwargs else dump()
            except TypeError:
                continue
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped
            break
    result: dict[str, Any] = {}
    for key in (
        "team", "channel", "tenant", "id",
        "aadGroupId", "aad_group_id",
        "teamsTeamId", "teamsChannelId", "teams_team_id", "teams_channel_id",
        "teamAadGroupId", "team_aad_group_id",
        "conversation_type", "conversationType",
    ):
        got = getattr(value, key, None)
        if _is_mock(got):
            continue
        if isinstance(got, (str, dict)):
            result[key] = got
        elif got is not None and callable(getattr(got, "model_dump", None)):
            result[key] = got
    return result


def safe_graph_filename(name: str) -> str:
    """Basename safe for a driveItem path segment (no path separators / reserved chars)."""
    base = os.path.basename((name or "").replace("\\", "/")).strip() or "file"
    base = _UNSAFE_FILENAME.sub("_", base).strip(" .") or "file"
    return base[:200]


def markdown_file_link(name: str, url: str) -> str:
    """Clickable markdown link; neutralize `]` so a hostile filename cannot break the href."""
    safe_name = (name or "file").replace("[", "(").replace("]", ")").replace("\n", " ").strip() or "file"
    return f"[{safe_name}]({url})"


def resolve_graph_credentials(
    environ: dict[str, str] | None = None,
    *,
    teams_tenant_id: str = "",
    teams_client_id: str = "",
    teams_client_secret: str = "",
) -> GraphCredentials | None:
    """MSGRAPH_* first; otherwise the Teams bot app (same Entra registration, if consented)."""
    if environ is None:
        from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
        environ = {
            name: (_get_scoped_secret(name, "") or "").strip()
            for name in (
                "MSGRAPH_TENANT_ID", "MSGRAPH_CLIENT_ID", "MSGRAPH_CLIENT_SECRET",
                "MSGRAPH_SCOPE", "MSGRAPH_AUTHORITY_URL",
            )
        }
    creds = GraphCredentials.from_env(environ, required=False)
    if creds is not None:
        return creds
    tenant, client, secret = (
        (teams_tenant_id or "").strip(),
        (teams_client_id or "").strip(),
        (teams_client_secret or "").strip(),
    )
    if tenant and client and secret:
        return GraphCredentials(tenant, client, secret)
    return None


def extract_graph_file_target(activity: Any) -> GraphFileTarget | None:
    """Pull Graph team/channel/chat ids out of a Bot Framework activity's channelData."""
    conv = getattr(activity, "conversation", None)
    conv_type = _as_text(_field(conv, "conversation_type", "conversationType"))
    conv_id = _as_text(_field(conv, "id"))
    raw_channel_data = _field(activity, "channel_data", "channelData")
    if _is_mock(raw_channel_data):
        raw_channel_data = None
    channel_data = _as_mapping(raw_channel_data)
    team = _as_mapping(_field(channel_data, "team") or channel_data.get("team"))
    channel = _as_mapping(_field(channel_data, "channel") or channel_data.get("channel"))
    team_id = _as_text(
        _field(team, "aadGroupId", "aad_group_id")
        or _field(channel_data, "teamAadGroupId", "team_aad_group_id")
    )
    channel_id = _as_text(
        _field(channel, "id")
        or _field(channel_data, "teamsChannelId", "teams_channel_id")
        or (conv_id if conv_type == "channel" else "")
    )
    if conv_type == "channel" or team_id:
        if team_id and channel_id:
            return GraphFileTarget(
                conversation_type="channel", team_id=team_id, channel_id=channel_id)
        return None
    if conv_type == "groupChat" and conv_id:
        return GraphFileTarget(conversation_type="groupChat", chat_id=conv_id)
    return None


def resolve_graph_file_target(
    chat_id: str,
    *,
    conv_type: str | None = None,
    cached: GraphFileTarget | None = None,
    extra_team_id: str = "",
    extra_channel_id: str = "",
) -> GraphFileTarget | None:
    """Combine inbound stash, conversation type, and optional TEAMS_TEAM_ID fallback."""
    kind = str((cached.conversation_type if cached else None) or conv_type or "").strip()
    team_id = (extra_team_id or (cached.team_id if cached else "") or "").strip()
    channel_id = (extra_channel_id or (cached.channel_id if cached else "") or "").strip()
    graph_chat_id = ((cached.chat_id if cached else "") or "").strip()
    chat_id = (chat_id or "").strip()

    if kind == "groupChat" or (graph_chat_id and not team_id and kind != "channel"):
        cid = graph_chat_id or (chat_id if kind == "groupChat" else "")
        if cid:
            return GraphFileTarget(conversation_type="groupChat", chat_id=cid)
    if kind == "channel" or team_id:
        ch = channel_id or chat_id
        if team_id and ch:
            return GraphFileTarget(conversation_type="channel", team_id=team_id, channel_id=ch)
        return None
    if kind == "groupChat" and chat_id:
        return GraphFileTarget(conversation_type="groupChat", chat_id=chat_id)
    return None


def encode_graph_share_id(sharing_url: str) -> str:
    """Graph sharing token: ``u!`` + base64url(url) without padding."""
    raw = base64.urlsafe_b64encode(sharing_url.encode("utf-8")).decode("ascii").rstrip("=")
    return f"u!{raw}"


def graph_item_download_url(item: Any) -> str:
    """Preauthenticated ``@microsoft.graph.downloadUrl`` from a driveItem payload."""
    if not isinstance(item, dict):
        return ""
    return _as_text(
        item.get("@microsoft.graph.downloadUrl")
        or item.get("@microsoft.graph.download_url")
        or item.get("downloadUrl")
        or item.get("download_url")
    )


def _looks_like_sharepoint_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https" or parsed.port not in (None, 443):
        return False
    host = (parsed.hostname or "").lower()
    return host in {"sharepoint.com", "onedrive.com", "1drv.com"} or host.endswith(
        (".sharepoint.com", ".sharepoint-df.com", ".onedrive.com", ".1drv.com")
    )


def _strip_item_id(value: str) -> str:
    return (value or "").strip().strip("{}")


async def resolve_inbound_file_download_url(
    graph: Any,
    *,
    target: GraphFileTarget | None = None,
    content: dict[str, Any] | None = None,
    content_url: str = "",
    filename: str = "",
) -> str:
    """Resolve a preauthenticated download URL when the activity omitted ``downloadUrl``.

    Channel ``file.download.info`` attachments often carry ``uniqueId`` / a SharePoint
    ``contentUrl`` without ``downloadUrl``. Same app-only Graph credentials as outbound
    filesFolder upload (``Files.ReadWrite.All``).
    """
    content = content or {}
    unique_id = _strip_item_id(_as_text(
        _field(content, "uniqueId", "unique_id", "itemId", "item_id")
    ))
    sharing_url = _as_text(
        _field(content, "webUrl", "web_url", "fileUrl", "file_url", "contentUrl", "content_url")
        or content_url
    )
    if unique_id and target is not None:
        url = await _download_url_via_files_folder(
            graph, target, unique_id=unique_id, filename="")
        if url:
            return url
    if filename and target is not None:
        url = await _download_url_via_files_folder(
            graph, target, unique_id="", filename=filename)
        if url:
            return url
    if _looks_like_sharepoint_url(sharing_url):
        url = await _download_url_via_share(graph, sharing_url)
        if url:
            return url
    return ""


def graph_message_path(target: GraphFileTarget, message_id: str) -> str:
    """Graph path for one channel or chat message (inbound HTML-only file fallback)."""
    mid = _safe_graph_id(message_id, label="message id")
    if target.team_id and target.channel_id:
        return (
            f"/teams/{quote(target.team_id, safe='')}"
            f"/channels/{quote(target.channel_id, safe='')}"
            f"/messages/{quote(mid, safe='')}"
        )
    if target.chat_id:
        return f"/chats/{quote(target.chat_id, safe='')}/messages/{quote(mid, safe='')}"
    raise GraphFileTargetError("Graph message lookup needs team_id+channel_id or chat_id.")


def extract_graph_message_file_refs(payload: Any) -> list[dict[str, str]]:
    """SharePoint file refs from a Graph chatMessage (skip HTML body / cards)."""
    if not isinstance(payload, dict):
        return []
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for att in payload.get("attachments") or []:
        name = _as_text(_field(att, "name"))
        content_url = _as_text(_field(att, "contentUrl", "content_url"))
        content_type = _as_text(_field(att, "contentType", "content_type")).lower()
        unique_id = _strip_item_id(_as_text(_field(att, "uniqueId", "unique_id")))
        if content_type in ("text/html", "text/plain") or content_type.startswith(
            "application/vnd.microsoft.card"
        ):
            continue
        if content_type in ("messageReference", "forwardedMessageReference"):
            continue
        if not name and not content_url:
            continue
        key = (name, content_url)
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "name": name,
            "contentUrl": content_url,
            "uniqueId": unique_id,
            "contentType": content_type,
        })
    return refs


async def list_graph_message_file_refs(
    graph: Any, target: GraphFileTarget, message_id: str,
) -> list[dict[str, str]]:
    """GET the Graph channel/chat message and return downloadable file refs."""
    payload = await graph.get_json(graph_message_path(target, message_id))
    return extract_graph_message_file_refs(payload)


async def _download_url_via_files_folder(
    graph: Any,
    target: GraphFileTarget,
    *,
    unique_id: str,
    filename: str,
) -> str:
    try:
        folder = await graph.get_json(target.files_folder_path())
    except (MicrosoftGraphAPIError, MicrosoftGraphClientError):
        return ""
    if not isinstance(folder, dict):
        return ""
    parent = folder.get("parentReference") if isinstance(folder.get("parentReference"), dict) else {}
    try:
        drive_id = _safe_graph_id(str(parent.get("driveId") or "").strip(), label="drive id")
        folder_id = _safe_graph_id(str(folder.get("id") or "").strip(), label="folder id")
    except GraphFileUploadError:
        return ""
    paths: list[str] = []
    if unique_id:
        try:
            item_id = _safe_graph_id(unique_id, label="item id")
        except GraphFileUploadError:
            item_id = ""
        if item_id:
            paths.append(f"/drives/{quote(drive_id, safe='')}/items/{quote(item_id, safe='')}")
    if filename:
        encoded = quote(safe_graph_filename(filename), safe="._-")
        paths.append(
            f"/drives/{quote(drive_id, safe='')}/items/{quote(folder_id, safe='')}:/{encoded}"
        )
    for path in paths:
        try:
            item = await graph.get_json(path)
        except (MicrosoftGraphAPIError, MicrosoftGraphClientError):
            continue
        url = graph_item_download_url(item)
        if url:
            return url
        web = _as_text((item or {}).get("webUrl") or (item or {}).get("web_url")) if isinstance(item, dict) else ""
        if _looks_like_sharepoint_url(web):
            via_share = await _download_url_via_share(graph, web)
            if via_share:
                return via_share
    return ""


async def _download_url_via_share(graph: Any, sharing_url: str) -> str:
    share_id = encode_graph_share_id(sharing_url)
    path = f"/shares/{quote(share_id, safe='!')}/driveItem"
    try:
        item = await graph.get_json(path)
    except (MicrosoftGraphAPIError, MicrosoftGraphClientError):
        return ""
    return graph_item_download_url(item)


def _safe_graph_id(value: str, *, label: str) -> str:
    """Reject ids that would break a Graph path (slash, query, fragment)."""
    text = (value or "").strip()
    if not text or any(ch in text for ch in "/\\?#") or ".." in text:
        raise GraphFileUploadError(f"Graph {label} contained unexpected characters.")
    return text


def _drive_item_content_path(drive_id: str, folder_id: str, filename: str) -> str:
    drive_id = _safe_graph_id(drive_id, label="drive id")
    folder_id = _safe_graph_id(folder_id, label="folder id")
    encoded = quote(filename, safe="._-")
    return f"/drives/{drive_id}/items/{folder_id}:/{encoded}:"


async def put_preauthenticated_upload(url: str, data: bytes) -> Any:
    """PUT file bytes to a Graph upload-session URL (the URL is already authorized)."""
    from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
    from gateway.platforms.base import _ssrf_redirect_guard
    from plugins.platforms.teams.adapter import _is_allowed_onedrive_upload_url

    if not url or not _is_allowed_onedrive_upload_url(str(url)) or not is_safe_url(str(url)):
        raise GraphFileUploadError("Graph upload session URL failed host/SSRF checks.")
    size = len(data)
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
        "Content-Range": f"bytes 0-{max(size - 1, 0)}/{size}",
    }
    async with create_ssrf_safe_async_client(
        timeout=60.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
    ) as client:
        response = await client.put(url, content=data, headers=headers)
        response.raise_for_status()
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}


async def upload_conversation_file(
    graph: Any,
    target: GraphFileTarget,
    *,
    file_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    put_upload_session: Callable[[str, bytes], Awaitable[Any]] | None = None,
) -> GraphUploadedFile:
    """Upload ``data`` into the conversation's SharePoint folder and return URLs."""
    name = safe_graph_filename(file_name)
    folder = await graph.get_json(target.files_folder_path())
    if not isinstance(folder, dict):
        raise GraphFileUploadError("Graph filesFolder response was not an object.")
    parent = folder.get("parentReference") if isinstance(folder.get("parentReference"), dict) else {}
    drive_id = str(parent.get("driveId") or "").strip()
    folder_id = str(folder.get("id") or "").strip()
    if not drive_id or not folder_id:
        raise GraphFileUploadError("Graph filesFolder did not include driveId/id.")
    item_root = _drive_item_content_path(drive_id, folder_id, name)
    if len(data) <= SIMPLE_UPLOAD_MAX_BYTES:
        item = await graph.put_bytes(
            f"{item_root}/content",
            content=data,
            content_type=content_type,
            params={"@microsoft.graph.conflictBehavior": "rename"},
        )
    else:
        session = await graph.post_json(
            f"{item_root}/createUploadSession",
            json_body={"item": {"@microsoft.graph.conflictBehavior": "rename", "name": name}},
        )
        upload_url = str((session or {}).get("uploadUrl") or "").strip()
        if not upload_url:
            raise GraphFileUploadError("Graph createUploadSession did not return uploadUrl.")
        putter = put_upload_session or put_preauthenticated_upload
        item = await putter(upload_url, data)
    if not isinstance(item, dict):
        item = {}
    item_id = str(item.get("id") or "").strip()
    web_url = str(item.get("webUrl") or item.get("web_url") or "").strip()
    final_name = str(item.get("name") or name)
    share_url = ""
    if item_id:
        share_url = await _create_org_view_link(graph, drive_id, item_id)
    if not web_url and not share_url:
        raise GraphFileUploadError("Graph upload succeeded but returned no webUrl.")
    size = item.get("size")
    try:
        size_int = int(size) if size is not None else len(data)
    except (TypeError, ValueError):
        size_int = len(data)
    return GraphUploadedFile(
        name=final_name, web_url=web_url or share_url, share_url=share_url,
        drive_id=drive_id, item_id=item_id, size=size_int,
    )


async def _create_org_view_link(graph: Any, drive_id: str, item_id: str) -> str:
    """Best-effort organization view link; empty when createLink is denied."""
    try:
        payload = await graph.post_json(
            f"/drives/{drive_id}/items/{quote(item_id, safe='')}/createLink",
            json_body={"type": "view", "scope": "organization"},
        )
    except (MicrosoftGraphAPIError, MicrosoftGraphClientError):
        return ""
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    link = payload.get("link") if isinstance(payload.get("link"), dict) else payload
    return str((link or {}).get("webUrl") or (link or {}).get("web_url") or "").strip()
