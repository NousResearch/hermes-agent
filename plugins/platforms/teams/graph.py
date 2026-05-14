"""Microsoft Graph client wrapper for Teams file upload/download.

A narrow façade over :class:`msgraph.GraphServiceClient` that exposes only
the operations the Teams plugin actually needs:

* :meth:`GraphClient.download_hosted_content` — fallback retrieval of inline
  hosted attachments on channel messages, when the Bot Framework activity
  does not include the bytes itself.
* :meth:`GraphClient.upload_to_sharepoint` — outbound channel/group file
  shares: PUT bytes to a SharePoint document library and return the
  ``webUrl`` for inclusion in a FileInfo card.

The Microsoft Graph SDK (``msgraph-sdk``) and Azure SDK (``azure-identity``,
``azure-core``) are **lazy-imported** inside method bodies so this module is
cheap to import even when those packages are not yet installed — the
plugin's lazy-deps mechanism installs them on first use.

:class:`_HermesTokenCredential` adapts our synchronous-ish
:class:`~plugins.platforms.teams.auth_graph.GraphTokenProvider` to the
``AsyncTokenCredential`` contract the Graph SDK expects.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from plugins.platforms.teams.auth_graph import AuthError, GraphTokenProvider

__all__ = ["GraphClient", "GRAPH_SCOPE", "_HermesTokenCredential"]

log = logging.getLogger(__name__)

GRAPH_SCOPE = "https://graph.microsoft.com/.default"


# ---------------------------------------------------------------------------
# Credential adapter — bridges our async provider to azure.core.credentials
# ---------------------------------------------------------------------------

class _HermesTokenCredential:
    """Adapt :class:`GraphTokenProvider` to ``AsyncTokenCredential``.

    The Graph SDK calls ``await cred.get_token(*scopes, **kwargs)`` and
    expects a named tuple ``AccessToken(token, expires_on)``. We delegate
    to the underlying provider and report an approximate ``expires_on``
    — our provider already manages real expiry/refresh internally, and
    the Graph SDK only uses ``expires_on`` as a best-effort refresh hint
    because it always re-requests on a 401.
    """

    def __init__(self, provider: GraphTokenProvider) -> None:
        self._provider = provider

    async def get_token(self, *scopes: str, **kwargs: Any):
        # Local import keeps this module importable when azure-identity /
        # azure-core haven't been installed yet (lazy-deps fetches them on
        # first plugin use).
        from azure.core.credentials import AccessToken

        if not scopes:
            raise AuthError(
                "AsyncTokenCredential.get_token requires at least one scope"
            )
        token = await self._provider.get_token(scopes[0])
        return AccessToken(token, int(time.time()) + 3000)

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Error-tolerant helpers
# ---------------------------------------------------------------------------

async def _safe(call, *, action: str, default: Any) -> Any:
    """Run *call()* and return ``default`` on Graph errors after logging.

    Centralises error handling so every public method on
    :class:`GraphClient` stays short. ``action`` is a free-form label
    that appears in log lines and makes Graph failures identifiable.
    """
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001 — broad on purpose
        log.warning("teams.graph error action=%s err=%s", action, exc)
        return default


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute access that also tolerates dicts and ``None``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# GraphClient
# ---------------------------------------------------------------------------

class GraphClient:
    """Narrow façade over :class:`msgraph.GraphServiceClient`.

    The Teams adapter constructs one instance per connect() cycle and
    shares it across inbound handlers and outbound senders. The Graph
    SDK builds its HTTPS session lazily on the first call so a
    GraphClient with no in-flight calls has no real network footprint.
    """

    def __init__(self, provider: GraphTokenProvider) -> None:
        self._provider = provider
        # Built lazily on first .client access — msgraph-sdk pulls a lot
        # of generated kiota modules and is expensive to import.
        self._client: Any = None

    def _build_client(self) -> Any:
        # Local import: only pay the cost when somebody actually makes a
        # Graph call. Tests monkeypatch this method to inject a fake.
        from msgraph import GraphServiceClient

        credential = _HermesTokenCredential(self._provider)
        return GraphServiceClient(
            credentials=credential,
            scopes=[GRAPH_SCOPE],
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # ------------------------------------------------------------------
    # Hosted content (inline attachments inside channel messages)
    # ------------------------------------------------------------------

    async def download_hosted_content(
        self,
        team_id: str,
        channel_id: str,
        message_id: str,
        hosted_content_id: str,
    ) -> Optional[bytes]:
        """Fetch the raw bytes of an inline image / attachment.

        Returns ``None`` if the call fails — the adapter logs the miss
        and passes only the URL reference through to the agent.
        """

        async def _call() -> Optional[bytes]:
            return await (
                self.client.teams.by_team_id(team_id)
                .channels.by_channel_id(channel_id)
                .messages.by_chat_message_id(message_id)
                .hosted_contents.by_chat_message_hosted_content_id(hosted_content_id)
                .content.get()
            )

        return await _safe(
            _call,
            action=f"download_hosted_content({message_id}/{hosted_content_id})",
            default=None,
        )

    # ------------------------------------------------------------------
    # SharePoint upload (channel / group file attachments)
    # ------------------------------------------------------------------

    async def upload_to_sharepoint(
        self,
        site_id: str,
        folder_path: str,
        filename: str,
        content: bytes,
    ) -> Optional[str]:
        """Upload *content* to a SharePoint document library folder.

        Returns the ``webUrl`` the adapter can attach to a Teams message
        so recipients see a proper file card. Graph's simple-upload
        endpoint (PUT /content) caps at 4 MB per request — callers with
        larger payloads should chunk via createUploadSession, which is
        out of scope for the MVP.
        """
        clean_folder = folder_path.strip("/")
        path_prefix = f"/{clean_folder}/" if clean_folder else "/"
        # Graph drive item paths use colon-delimited "path" syntax:
        # /sites/{site}/drive/root:/folder/file.png:/content
        encoded_path = f"{path_prefix}{filename}"

        async def _call() -> Optional[str]:
            drive = self.client.sites.by_site_id(site_id).drive
            item = drive.items.by_drive_item_id(f"root:{encoded_path}:")
            result = await item.content.put(content)
            return _attr(result, "web_url") or _attr(result, "_raw_url")

        return await _safe(
            _call,
            action=f"upload_to_sharepoint({site_id}/{filename})",
            default=None,
        )
