"""Thin async wrapper around the Carbon Voice REST endpoints we use.

Methods raise on HTTP/network errors so callers can map them to their own
result types (the adapter wraps them into ``SendResult``; ``standalone_send``
catches everything and returns a dict).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from .constants import (
    AGENT_ID_HEADER,
    DEFAULT_BASE_URL,
    HTTP_TIMEOUT,
    TRANSIENT_RETRY_ATTEMPTS,
    TRANSIENT_RETRY_BACKOFF_S,
    TRANSIENT_STATUS,
    USER_AGENT,
)
from .parse import client_headers, first_str

logger = logging.getLogger(__name__)


class CarbonVoiceAPI:
    """Stateless REST client. Open with ``await api.open()`` before use."""

    def __init__(self, pat: str, base_url: str = DEFAULT_BASE_URL):
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed")
        self._pat = pat
        self._base_url = base_url.rstrip("/")
        self._client: Optional["httpx.AsyncClient"] = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=client_headers(self._pat),
                timeout=HTTP_TIMEOUT,
            )

    def set_agent_id(self, user_guid: str) -> None:
        """Tag all subsequent requests with the bot's own id (from /whoami).

        Goes into both ``agent-id`` and the User-Agent — the backend's
        request logger only captures the ua field today, so that's where
        the id must live for per-agent grouping. Only the few bootstrap
        calls made before whoami resolves go out without it.
        """
        if self._client is not None and user_guid:
            self._client.headers[AGENT_ID_HEADER] = user_guid
            self._client.headers["user-agent"] = (
                f"{USER_AGENT} (agent-id: {user_guid})"
            )

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _require_client(self) -> "httpx.AsyncClient":
        if self._client is None:
            raise RuntimeError("CarbonVoiceAPI used before open()")
        return self._client

    async def _request_retrying(self, method: str, url: str, **kwargs):
        """Issue a request, retrying ONLY on transient 5xx (502/503/504) and
        network errors, with short backoff. For idempotent calls only — never
        wrap a send/POST that creates a message (a retry could duplicate it).

        CV's gateway returns 502s in bursts; without this a transient hiccup
        on a latency-critical read (e.g. v5 enrichment, a reaction) stalls
        until the next ~5s poll tick. Retries recover in <1s. Returns the
        final response (the caller still inspects status); raises the last
        network error if every attempt failed to connect.
        """
        client = self._require_client()
        last_exc: Optional[Exception] = None
        for attempt in range(TRANSIENT_RETRY_ATTEMPTS + 1):
            try:
                resp = await client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt >= TRANSIENT_RETRY_ATTEMPTS:
                    raise
            else:
                if (
                    resp.status_code in TRANSIENT_STATUS
                    and attempt < TRANSIENT_RETRY_ATTEMPTS
                ):
                    logger.debug(
                        "carbonvoice: %s %s → %s, retry %d/%d",
                        method, url, resp.status_code,
                        attempt + 1, TRANSIENT_RETRY_ATTEMPTS,
                    )
                else:
                    return resp
            await asyncio.sleep(TRANSIENT_RETRY_BACKOFF_S * (attempt + 1))
        # Exhausted retries on repeated network errors.
        if last_exc is not None:
            raise last_exc
        return resp  # pragma: no cover - loop always returns or raises

    async def whoami(self) -> "tuple[Optional[str], Optional[str]]":
        """Return ``(user_guid, owner_id)`` for the bot account.

        - ``user_guid`` — the agent's own id (for the self-loop guard).
        - ``owner_id`` — ``user.created_by``, the user who *created* the bot
          account. That's the deny-by-default owner: always authorized, and
          auto-detected here so no manual setup is needed. Either may be
          None when not parseable.
        """
        client = self._require_client()
        resp = await client.get("/whoami")
        resp.raise_for_status()
        data = resp.json() or {}
        user = data.get("user") or {}
        return (
            first_str(user.get("user_guid"), user.get("_id"), user.get("id")),
            first_str(user.get("created_by")),
        )

    async def fetch_recent(
        self,
        since_iso: str,
        direction: str = "newer",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        client = self._require_client()
        # ``use_last_updated: True`` filters by ``updated_at`` instead
        # of ``created_at``. Required for voice messages with picker
        # tags: ``created_at`` fires when the audio bytes land (no
        # transcript, no tagged_user_ids), but the backend updates the
        # message ~10–15 s later when STT and the tag-resolution job
        # finish, bumping ``updated_at`` and emitting cv-api's
        # ``message:updated`` socket event. With the old
        # ``created_at`` filter, the polling/catch-up after that socket
        # event missed the message entirely — its ``created_at`` was
        # already older than the cursor that advanced past the empty
        # ``message:created`` window. SeenCache (TTL 5 min) handles the
        # extra fan-out from messages that update multiple times in
        # the lookback window.
        body = {
            "date": since_iso,
            "direction": direction,
            "limit": limit,
            "use_last_updated": True,
        }
        resp = await client.post("/v3/messages/recent", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def send_message(
        self,
        channel_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v3/messages/start — legacy. Prefer ``send_text_v5``.

        Kept for compatibility with code paths that still pass through the v3
        contract. New code should use ``send_text_v5`` which accepts
        ``reply_to_message_id`` directly (the server resolves the thread
        root, no reply-anchor resolution required) and uses
        ``idempotency_key`` instead of the deprecated ``unique_client_id``.
        """
        client = self._require_client()
        body: Dict[str, Any] = {
            "unique_client_id": str(uuid.uuid4()),
            "transcript": content,
            "is_text_message": True,
            "is_streaming": False,
            "channel_id": channel_id.strip(),
        }
        if reply_to:
            body["reply_to_message_id"] = str(reply_to)
        resp = await client.post("/v3/messages/start", json=body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── v5 transport ────────────────────────────────────────────────────
    #
    # The v5 endpoints replace the v3 contract with cleaner naming
    # (``reply_to_message_id`` as the reply field, ``idempotency_key``
    # in place of ``unique_client_id``) and split create paths by media
    # kind: ``/text``, ``/audio`` (multipart), and ``/attachment`` (URLs).
    #
    # Threading contract (cv-api PR #277 / CV-13155, cv-contracts 4.0.1):
    # the v5 *conversation* create routes accept ``reply_to_message_id``
    # — the id of the message being replied to. The backend resolves the
    # thread *root* automatically (``resolveRootParentMessageId``): pass
    # a root and it stays the root; pass a reply and the server attaches
    # to that reply's root instead of rejecting it (the old
    # "You cannot reply to a message that is a reply" 400 is gone). The
    # only remaining reply error is cross-conversation (400).
    #
    # NOTE: the earlier ``thread_id`` input field was *renamed* to
    # ``reply_to_message_id`` and ``thread_id`` is now in the v5
    # reject-deprecated-keys pipe — sending it returns a 400. Callers
    # pass the thread root from the inbound message
    # (``ConversationTracker.thread_id_of(msg)``) as
    # ``reply_to_message_id``; root-resolves-to-itself keeps threading
    # correct with no reply-anchor lookup.

    async def send_text_v5(
        self,
        conversation_id: str,
        transcript: str,
        reply_to_message_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """POST /v5/messages/text — create a text message in a conversation.

        Returns the created MessageV5 dict on 2xx. ``reply_to_message_id``
        threads the message (see module docstring — the backend resolves
        the thread root automatically); pass ``None`` for a new top-level
        post. Sending the old ``thread_id`` key is rejected with 400 by
        the v5 deprecated-fields pipe (cv-api PR #277).

        ``attachments`` is an optional list of
        ``V5RequestAttachmentPayload`` dicts (same shape used by
        :meth:`send_attachment_v5`). When the agent wants text + an
        attached file in a single bubble (e.g. "here's the report" + a
        .md file), pass both fields together — the server enforces a
        non-empty ``transcript`` on this route, so use
        :meth:`send_attachment_v5` for the attachment-only case.
        """
        client = self._require_client()
        body: Dict[str, Any] = {
            "conversation_id": conversation_id.strip(),
            "transcript": transcript,
            "idempotency_key": idempotency_key or str(uuid.uuid4()),
        }
        if reply_to_message_id:
            body["reply_to_message_id"] = str(reply_to_message_id)
        if attachments:
            body["attachments"] = attachments
        resp = await client.post("/v5/messages/text", json=body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def send_audio_v5(
        self,
        conversation_id: str,
        audio_path: str,
        reply_to_message_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /v5/messages/audio — multipart upload of an audio file.

        Sends two parts: ``payload`` (JSON with conversation_id /
        reply_to_message_id / idempotency_key / duration) and
        ``audio_file`` (the raw bytes of the file at ``audio_path``). The
        server transcribes and threads the resulting message; returns the
        created MessageV5 dict on 2xx. This is the *conversation* audio
        route (``messages/audio``), which accepts ``reply_to_message_id``
        — only the ``voicememos/audio`` route forbids it (cv-api PR #277).

        For Hermes' ``send_voice`` adapter override.
        """
        import json as _json
        from pathlib import Path as _Path

        client = self._require_client()
        path = _Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")

        payload: Dict[str, Any] = {
            "conversation_id": conversation_id.strip(),
            "idempotency_key": idempotency_key or str(uuid.uuid4()),
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = str(reply_to_message_id)
        if duration_ms is not None:
            payload["duration"] = int(duration_ms)

        with path.open("rb") as fh:
            files = {
                "payload": (None, _json.dumps(payload), "application/json"),
                "audio_file": (path.name, fh.read(), "application/octet-stream"),
            }
            resp = await client.post("/v5/messages/audio", files=files)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def send_attachment_v5(
        self,
        conversation_id: str,
        attachments: List[Dict[str, Any]],
        reply_to_message_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v5/messages/attachment — create a message with link attachments.

        ``attachments`` is a list of ``V5RequestAttachmentPayload`` dicts
        (``{type, link, idempotency_key?, ...}``). The CV API expects
        each attachment to reference an already-hosted resource by URL;
        binary uploads via this endpoint are not supported (use
        ``send_audio_v5`` for audio, or host the file elsewhere first and
        pass the URL here).

        For Hermes' ``send_image`` / ``send_document`` adapter overrides
        when the caller passes a URL.
        """
        client = self._require_client()
        body: Dict[str, Any] = {
            "conversation_id": conversation_id.strip(),
            "attachments": attachments,
            "idempotency_key": idempotency_key or str(uuid.uuid4()),
        }
        if reply_to_message_id:
            body["reply_to_message_id"] = str(reply_to_message_id)
        resp = await client.post("/v5/messages/attachment", json=body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Local-file attachment flow (v3 signed-URL + S3 + status) ────────
    #
    # CV's v5 attachment endpoint is URL-based — it expects ``link`` to
    # point to an already-hosted file. To send a *local* file (the
    # agent's generated .md, an audio clip, a PDF) we follow the same
    # four-step pattern the Flutter client uses:
    #
    #   1. ``get_signed_upload_urls`` → pre-signed S3 PUT URLs
    #   2. ``upload_to_s3``           → PUT the raw bytes (no Bearer)
    #   3. ``send_text_v5`` /
    #      ``send_attachment_v5``     → create the message with
    #                                    ``type: "file"`` referencing the
    #                                    canonical S3 URL (the signed URL
    #                                    minus its query string)
    #   4. ``update_attachment``      → flip status from ``Initializing``
    #                                    to ``Uploaded`` / ``Failed`` so
    #                                    the recipient's UI reflects
    #                                    completion
    #
    # The PR #251 backend change (commit d209c472) exposes ``status`` and
    # ``percent_complete`` on the attachment response, which is what
    # makes step 4 visible to clients.

    async def get_signed_upload_urls(
        self,
        files: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """POST /v3/attachments/signedurl — get pre-signed S3 upload URLs.

        ``files`` is a list of ``{"filename": ..., "mimetype": ...}``
        dicts (the server's ``CreateAttachmentUrls`` DTO). Returns the
        ``AttachmentUrl`` list ``[{"url", "filename", "mimetype"}, ...]``
        in the same order, where each ``url`` is a short-lived S3
        pre-signed PUT URL. The canonical attachment ``link`` we hand
        back to CV is this URL with the query string stripped (the
        bucket's ACL is public-read for the rendered path).
        """
        if not files:
            return []
        client = self._require_client()
        resp = await client.post(
            "/v3/attachments/signedurl",
            json={"files": files},
            headers={"x-api-version": "3"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def upload_to_s3(
        self,
        signed_url: str,
        file_path: str,
        mime_type: str,
    ) -> None:
        """PUT the file at *file_path* to *signed_url* directly.

        The signed URL embeds its own AWS credentials in the query
        string, so we must NOT send our ``Authorization: Bearer …``
        header on this request — that's why we go via a one-shot
        ``httpx.AsyncClient`` instead of ``self._client``. Raises on
        non-2xx so the caller can flip the attachment status to
        ``Failed``.
        """
        from pathlib import Path as _Path

        path = _Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as plain:
            with path.open("rb") as fh:
                resp = await plain.put(
                    signed_url,
                    content=fh.read(),
                    headers={"Content-Type": mime_type},
                )
        resp.raise_for_status()

    async def update_attachment(
        self,
        message_id: str,
        attachment_id: str,
        body: Dict[str, Any],
    ) -> None:
        """PUT /messages/{message_id}/attachment/{attachment_id}.

        Used to flip ``status`` (``Initializing`` → ``Uploading`` →
        ``Uploaded`` / ``Failed``) and ``percent_complete`` on an
        attachment after the S3 upload settles. ``body`` should carry
        the full attachment row the server expects — ``type``, ``link``,
        ``filename``, ``mime_type``, ``status``, ``percent_complete``.
        """
        client = self._require_client()
        resp = await client.put(
            f"/messages/{message_id}/attachment/{attachment_id}",
            json=body,
        )
        resp.raise_for_status()

    # ── Inbound attachment download (PR 7) ──────────────────────────────
    #
    # CV's inbound messages carry ``attachments[]`` entries whose
    # ``link`` is the canonical S3 URL — but that URL requires AWS
    # auth (returns 403 to unauthenticated requests). To consume the
    # file we ask CV for a short-lived pre-signed GET URL via
    # ``GET /attachments/signedurl/:attachment_id`` (authenticated with
    # our Bearer), then download the bytes from S3 with no auth header
    # (the signature lives in the query string).

    async def get_attachment_download_url(self, attachment_id: str) -> str:
        """GET /attachments/signedurl/:attachment_id — pre-signed S3 GET URL.

        Returns the URL as a plain string (CV's controller returns the
        URL as the bare response body, no JSON envelope). The signature
        in the query string makes the URL self-authenticating for the
        S3 GET that follows; do NOT send our Bearer header on that
        request (S3 would 400 on the unexpected auth).
        """
        client = self._require_client()
        resp = await client.get(f"/attachments/signedurl/{attachment_id}")
        resp.raise_for_status()
        return self._unquote_url(resp.text)

    @staticmethod
    def _unquote_url(text: str) -> str:
        # CV returns signed URLs either as a plain string or wrapped in
        # quotes (JSON string). Strip leading/trailing quotes either way.
        url = text.strip()
        if len(url) >= 2 and url[0] == url[-1] and url[0] in ('"', "'"):
            url = url[1:-1]
        return url

    async def download_attachment(
        self,
        attachment_id: str,
        dest_dir: "Path",
        *,
        filename: Optional[str] = None,
        max_bytes: Optional[int] = None,
    ) -> "Path":
        """Resolve the attachment's signed URL and stream bytes to disk.

        ``dest_dir`` is created if missing. ``filename`` overrides the
        on-disk name (default: the attachment_id with no extension —
        callers that know the filename should pass it). ``max_bytes``
        rejects responses whose ``Content-Length`` exceeds the cap so
        we don't bloat the agent context with multi-MB uploads.

        Raises:
            ValueError: if ``max_bytes`` is set and the response is
                larger.
            httpx.HTTPStatusError: on the signed-URL fetch or the S3
                download.
        """
        signed_url = await self.get_attachment_download_url(attachment_id)
        return await self._download_from_signed_url(
            signed_url,
            attachment_id,
            dest_dir,
            filename=filename,
            max_bytes=max_bytes,
        )

    async def _download_from_signed_url(
        self,
        signed_url: str,
        attachment_id: str,
        dest_dir: "Path",
        *,
        filename: Optional[str] = None,
        max_bytes: Optional[int] = None,
    ) -> "Path":
        """Stream a pre-signed S3 GET URL to ``dest_dir``; shared by the
        regular and share-link attachment download paths."""
        from pathlib import Path as _Path

        dest_dir = _Path(dest_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_name = filename or f"{attachment_id}.bin"
        out_path = dest_dir / out_name

        # S3 GET — use a fresh client without our Bearer header, since
        # the signed URL carries its own credentials in the query.
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as plain:
            async with plain.stream("GET", signed_url) as resp:
                resp.raise_for_status()
                if max_bytes is not None:
                    cl = resp.headers.get("content-length")
                    if cl and int(cl) > max_bytes:
                        raise ValueError(
                            f"attachment {attachment_id} too large: "
                            f"{cl} bytes > limit {max_bytes}"
                        )
                with out_path.open("wb") as fh:
                    written = 0
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                        written += len(chunk)
                        if max_bytes is not None and written > max_bytes:
                            # Truncate + raise — partial file gets
                            # cleaned up by the caller's exception
                            # handler when it falls out of scope.
                            fh.close()
                            out_path.unlink(missing_ok=True)
                            raise ValueError(
                                f"attachment {attachment_id} exceeded "
                                f"limit {max_bytes} mid-stream"
                            )
        return out_path

    # ── Forwarded messages (share links) ────────────────────────────────
    #
    # Forwarding a message in CV creates a *share link* and stamps its id
    # on the new wrapper message as ``share_link_id``. The wrapper carries
    # only the forwarder's optional comment — the original content
    # (transcript + attachments) lives behind the share link. Same flow
    # cv-claude-channels uses. Note the attachment signed-URL route is
    # share-link-scoped (NOT the regular ``/attachments/signedurl/:id``):
    # the bot may have access to the forward without having access to the
    # original message's channel, and the share-link route authorizes via
    # the link itself.

    async def get_share_link(
        self, share_link_id: str
    ) -> Optional[Dict[str, Any]]:
        """GET /v3/message-sharelinks/{id} — share-link + shared_message.

        Returns the share-link dict (with ``shared_message`` carrying the
        original message's ``creator_id`` / ``text_models`` /
        ``attachments``) or None on 4xx (revoked, expired, no access).
        Retries transient 5xx — this read sits on the latency-critical
        inbound path.
        """
        resp = await self._request_retrying(
            "GET", f"/v3/message-sharelinks/{share_link_id}"
        )
        if resp.status_code >= 400 or not resp.content:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None

    async def get_share_link_attachment_download_url(
        self, share_link_id: str, attachment_id: str
    ) -> str:
        """GET /message-sharelinks/{sl}/attachments/signedurl/{att}.

        Pre-signed S3 GET URL for an attachment on the *shared* (original)
        message, authorized through the share link. Plain-string response,
        same contract as :meth:`get_attachment_download_url`.
        """
        client = self._require_client()
        resp = await client.get(
            f"/message-sharelinks/{share_link_id}"
            f"/attachments/signedurl/{attachment_id}"
        )
        resp.raise_for_status()
        return self._unquote_url(resp.text)

    async def download_share_link_attachment(
        self,
        share_link_id: str,
        attachment_id: str,
        dest_dir: "Path",
        *,
        filename: Optional[str] = None,
        max_bytes: Optional[int] = None,
    ) -> "Path":
        """Download an attachment of a forwarded message to ``dest_dir``.

        Same semantics as :meth:`download_attachment` (size cap,
        ValueError on overflow) but resolves the signed URL through the
        share-link-scoped route.
        """
        signed_url = await self.get_share_link_attachment_download_url(
            share_link_id, attachment_id
        )
        return await self._download_from_signed_url(
            signed_url,
            attachment_id,
            dest_dir,
            filename=filename,
            max_bytes=max_bytes,
        )

    async def get_message_v5(self, message_id: str) -> Optional[Dict[str, Any]]:
        """GET /v5/messages/{id} — returns the flat MessageV5 dict or None.

        The v5 single-GET wraps its payload in a ``{"message": {...}}``
        envelope (unlike ``GET /v3/messages/{id}``, which is flat). We
        unwrap it here so callers receive the flat shape that the
        ``extract_*`` helpers and the mention gate expect:
        ``tagged_user_ids``, ``parent_message_id``, and ``transcript``
        live on the message object, not the envelope.

        Returning the envelope unchanged hid every field behind the
        ``message`` key — that was the bug that silently dropped
        @-mentions in group channels: the enriched payload's
        ``tagged_user_ids`` was invisible to ``is_user_mentioned``
        (DMs masked it, since the gate passes them regardless).

        ``parent_message_id`` is the canonical public thread field
        (cv-contracts 4.0.1 / cv-api PR #277 removed the short-lived
        ``thread_id`` field). The unwrap is defensive: if the endpoint
        ever returns a flat body, that is passed through unchanged.
        """
        resp = await self._request_retrying("GET", f"/v5/messages/{message_id}")
        if resp.status_code >= 400 or not resp.content:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        inner = data.get("message")
        return inner if isinstance(inner, dict) else data

    async def get_messages_by_ids_v5(
        self, conversation_id: str, message_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """POST /v5/messages/by-ids — batch fetch of multiple MessageV5s.

        Used by the thread-context fetch path (see
        ``adapter._fetch_thread_context``) to pull the transcripts for the
        message ids that ``list_channel_message_index`` identified as
        belonging to the thread, in a single round-trip.

        The endpoint requires BOTH ``conversation_id`` and ``message_ids``
        — it rejects the message-id list alone with a 400
        ("conversation_id should not be empty"). The returned items are
        flat MessageV5 dicts (no ``{"message": …}`` envelope, unlike the
        single ``GET /v5/messages/{id}``), so ``extract_*`` helpers work on
        them directly.
        """
        if not message_ids:
            return []
        client = self._require_client()
        resp = await client.post(
            "/v5/messages/by-ids",
            json={"conversation_id": conversation_id.strip(), "message_ids": message_ids},
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("messages", []) if isinstance(data, dict) else []

    async def list_channel_message_index(
        self,
        channel_id: str,
        *,
        limit: int = 200,
        direction: str = "older",
    ) -> List[Dict[str, Any]]:
        """GET /messages/<channel_id>/index — lightweight list-by-channel.

        Returns ``MessageIndex`` items (``message_id`` /
        ``parent_message_id`` / ``status`` / ``created_at`` / ...) without
        transcripts. Used by the thread-context path: we list the channel's
        recent messages, filter client-side by ``parent_message_id ==
        thread_id`` (CV is flat — every reply's ``parent_message_id`` is
        the true root, see DEVELOPMENT.md §4), then batch-fetch the
        identified ids via :meth:`get_messages_by_ids_v5`.

        ``direction='older'`` defaults to "the last <limit> messages in the
        channel" — what we want for thread context. The caller passes
        ``limit`` sized for typical active-channel volume (200 is a
        reasonable upper bound for a 30-message thread cap).

        This is a v3-only endpoint today; the cv-api roadmap may add a
        more direct ``GET /v5/channels/:id/threads/:thread_id/messages``
        in the future, at which point the workaround here collapses to a
        single call.
        """
        client = self._require_client()
        params: Dict[str, Any] = {
            "limit": int(limit),
            "direction": direction,
        }
        resp = await client.get(f"/messages/{channel_id}/index", params=params)
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results")
        if isinstance(results, list):
            return results
        return data if isinstance(data, list) else []

    async def fetch_reactions(self) -> List[Dict[str, Any]]:
        """GET /reactions — returns the workspace's available reactions."""
        client = self._require_client()
        resp = await client.get("/reactions")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def react(self, reaction_id: str, message_id: str) -> None:
        """POST /reactions/{reaction_id}/{message_id} — empty body.

        Retries transient 5xx: re-reacting is idempotent server-side (the same
        reaction by the same user is a no-op), so a retry can't duplicate, and
        the visual ack shouldn't be lost to a one-off CV 502.
        """
        resp = await self._request_retrying(
            "POST", f"/reactions/{reaction_id}/{message_id}"
        )
        resp.raise_for_status()

    async def mark_read(self, channel_id: str, message_id: str) -> None:
        """DELETE /notifications/{channel}/{message} — clears the unread badge."""
        client = self._require_client()
        resp = await client.delete(
            f"/notifications/{channel_id}/{message_id}",
            params={"type": "message", "notification_removal_mode": "hard"},
        )
        resp.raise_for_status()

    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """GET /v3/messages/{message_id} — returns the message dict or None on 4xx.

        Same payload shape as inbound Socket.IO / fetch_recent messages, so
        parse helpers (``extract_transcript``, ``extract_creator_id``, etc.)
        work unchanged. Used to resolve the text of a parent message when an
        inbound reply carries ``parent_message_id`` — gives the agent the
        thread context it would otherwise have to guess at.
        """
        client = self._require_client()
        resp = await client.get(f"/v3/messages/{message_id}")
        if resp.status_code >= 400:
            return None
        return resp.json() if resp.content else None

    async def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """GET /channel/{id} — returns the PersonalizedChannel dict or None on 4xx.

        Carbon Voice's response exposes ``type`` (directMessage |
        customerConversation | namedConversation | asyncMeeting) and
        ``dm_hash`` (null for non-DMs) — both usable to discriminate DM
        vs group conversation when gating the agent's behavior.
        """
        client = self._require_client()
        resp = await client.get(f"/channel/{channel_id}")
        if resp.status_code >= 400:
            return None
        return resp.json() if resp.content else None


async def standalone_send(
    pat: str,
    base_url: str,
    channel_id: str,
    content: str,
) -> Dict[str, Any]:
    """One-shot send for out-of-process delivery (cron). No persistent client."""
    if not HTTPX_AVAILABLE:
        return {"success": False, "error": "httpx not installed"}
    body = {
        "unique_client_id": str(uuid.uuid4()),
        "transcript": content,
        "is_text_message": True,
        "is_streaming": False,
        "channel_id": channel_id.strip(),
    }
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=client_headers(pat),
        timeout=HTTP_TIMEOUT,
    ) as client:
        try:
            resp = await client.post("/v3/messages/start", json=body)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            return {
                "success": True,
                "message_id": first_str(data.get("message_id"), data.get("id")),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}
