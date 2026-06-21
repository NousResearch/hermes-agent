"""
Weixin QR login session manager for web-based onboarding.

Mirrors the Telegram onboarding pattern in ``hermes_cli/web_server.py``:
an in-memory dict of active QR sessions, each backed by a background
asyncio task that polls Tencent's iLink Bot API for scan status.

The CLI wizard (``gateway/platforms/weixin.py::qr_login``) remains the
canonical interactive path. This module exposes the same iLink flow as
start/poll/cancel primitives so the dashboard can render a "Set up with
QR" button that works entirely in the browser — no TTY required.

Design notes:
- Sessions are process-global and short-lived (default 8 min).
- One background task per session polls iLink every ~1s.
- QR refresh (iLink ``expired`` status) is handled inside the task,
  up to 3 refreshes — same limit as the CLI wizard.
- Credentials are saved to ``~/.hermes/weixin/accounts/<id>.json`` via
  ``save_weixin_account`` on confirmation, matching the CLI path.
- The web layer (``web_server.py``) is responsible for additionally
  writing ``WEIXIN_ACCOUNT_ID`` / ``WEIXIN_TOKEN`` / ``WEIXIN_BASE_URL``
  to ``.env`` and toggling ``platforms.weixin.enabled`` in config.yaml
  — see ``apply_weixin_onboarding``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Session lifetime — matches the CLI wizard default (480s).
DEFAULT_SESSION_TIMEOUT_SECONDS = 480
# Poll interval for iLink status checks.
_POLL_INTERVAL_SECONDS = 1.0
# Max QR refreshes before giving up (matches CLI wizard).
_MAX_QR_REFRESHES = 3


@dataclass
class WeixinQRSession:
    """In-memory state for a single QR login session."""

    session_id: str
    hermes_home: str
    bot_type: str = "3"
    created_at_ts: float = field(default_factory=time.time)
    expires_at_ts: float = 0.0
    # Mutable state updated by the background task.
    state: str = "starting"  # starting | waiting | scanned | confirmed | failed | expired | cancelled
    qr_payload: str = ""  # The scannable liteapp URL (preferred) or hex token.
    qr_image_base64: str = ""  # PNG base64 for <img src="data:image/png;base64,..."> rendering.
    error_message: str = ""
    # Populated on confirmed.
    credentials: Optional[Dict[str, str]] = None
    # Internal — the background task handle.
    _task: Optional[asyncio.Task] = None
    _qrcode_value: str = ""  # Hex token used for status polling.
    _current_base_url: str = ""  # May change on scaned_but_redirect.
    _refresh_count: int = 0
    _aiohttp_session: Any = None  # aiohttp.ClientSession, owned by the task.

    @property
    def expires_at_iso(self) -> str:
        return time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.expires_at_ts)
        )

    def to_status_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable status snapshot for the polling endpoint."""
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "state": self.state,
            "expires_at": self.expires_at_iso,
        }
        if self.qr_image_base64 and self.state in {"waiting", "scanned"}:
            payload["qr_image_base64"] = self.qr_image_base64
        if self.error_message:
            payload["error"] = self.error_message
        if self.credentials and self.state == "confirmed":
            # Only expose non-secret fields to the browser. The token itself
            # is saved server-side; the dashboard just needs to know it worked.
            payload["account_id"] = self.credentials.get("account_id", "")
            payload["base_url"] = self.credentials.get("base_url", "")
        return payload


class WeixinQRSessionManager:
    """Process-global manager for concurrent Weixin QR login sessions.

    Thread-safe for start/get/cancel from sync web handlers; the poll loop
    itself runs on the dashboard's asyncio event loop.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, WeixinQRSession] = {}
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the manager to the dashboard's event loop.

        Called once at web_server startup. The poll tasks are scheduled on
        this loop via ``asyncio.run_coroutine_threadsafe`` from the sync
        ``start_session`` entry point.
        """
        self._loop = loop

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _prune_expired(self) -> None:
        """Evict expired sessions. Caller must hold ``self._lock``."""
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.expires_at_ts <= now and session.state not in {
                "confirmed",
                "failed",
                "cancelled",
            }
        ]
        for sid in expired:
            session = self._sessions.pop(sid, None)
            if session and session._task and not session._task.done():
                session._task.cancel()

    def start_session(
        self,
        hermes_home: str,
        *,
        bot_type: str = "3",
        timeout_seconds: int = DEFAULT_SESSION_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Start a new QR login session.

        Returns the initial status dict (state="starting"). The caller
        should poll ``get_status`` until state transitions to ``waiting``
        (QR ready to scan) or ``failed``.
        """
        with self._lock:
            self._prune_expired()
            session_id = uuid.uuid4().hex
            now = time.time()
            session = WeixinQRSession(
                session_id=session_id,
                hermes_home=hermes_home,
                bot_type=bot_type,
                created_at_ts=now,
                expires_at_ts=now + timeout_seconds,
            )
            self._sessions[session_id] = session

        # Schedule the background poll task on the dashboard event loop.
        loop = self._get_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._run_session(session), loop
        )
        # Stash the task so cancel() can abort it. We can't get the actual
        # asyncio.Task from run_coroutine_threadsafe, but the future is enough
        # for cancellation via future.cancel().
        session._task = future  # type: ignore[assignment]

        return session.to_status_dict()

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the current status of a session, or None if not found."""
        with self._lock:
            self._prune_expired()
            session = self._sessions.get(session_id)
            if not session:
                return None
            return session.to_status_dict()

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a session. Returns True if a session was found and cancelled."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if not session:
            return False
        session.state = "cancelled"
        task = session._task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
        return True

    def get_credentials(self, session_id: str) -> Optional[Dict[str, str]]:
        """Return confirmed credentials for a session, or None.

        Used by the apply endpoint to retrieve the token before saving
        to .env. The session is NOT removed here — the caller removes it
        via ``cancel_session`` after a successful apply.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.state != "confirmed":
                return None
            return dict(session.credentials) if session.credentials else None

    # ------------------------------------------------------------------
    # Background task implementation
    # ------------------------------------------------------------------

    async def _run_session(self, session: WeixinQRSession) -> None:
        """Background coroutine: fetch QR, poll for scan, save credentials."""
        try:
            from gateway.platforms.weixin import (
                AIOHTTP_AVAILABLE,
                ILINK_BASE_URL,
                EP_GET_BOT_QR,
                EP_GET_QR_STATUS,
                QR_TIMEOUT_MS,
                _api_get,
                _make_ssl_connector,
                save_weixin_account,
            )

            if not AIOHTTP_AVAILABLE:
                session.state = "failed"
                session.error_message = "aiohttp is required for Weixin QR login"
                return

            import aiohttp

            connector = _make_ssl_connector()
            async with aiohttp.ClientSession(
                trust_env=True, connector=connector
            ) as aiohttp_session:
                session._aiohttp_session = aiohttp_session
                session._current_base_url = ILINK_BASE_URL

                # ── Initial QR fetch ────────────────────────────────────
                if not await self._fetch_qr(session, aiohttp_session):
                    return  # _fetch_qr sets state=failed on error

                session.state = "waiting"

                # ── Poll loop ───────────────────────────────────────────
                deadline = session.expires_at_ts
                while time.time() < deadline:
                    if session.state == "cancelled":
                        return
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                    if session.state == "cancelled":
                        return

                    try:
                        status_resp = await _api_get(
                            aiohttp_session,
                            base_url=session._current_base_url,
                            endpoint=f"{EP_GET_QR_STATUS}?qrcode={session._qrcode_value}",
                            timeout_ms=QR_TIMEOUT_MS,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.warning("weixin QR session %s: poll error: %s", session.session_id, exc)
                        continue

                    status = str(status_resp.get("status") or "wait")
                    if status == "wait":
                        # Stay in waiting; QR still valid.
                        if session.state != "waiting":
                            session.state = "waiting"
                    elif status == "scaned":
                        session.state = "scanned"
                    elif status == "scaned_but_redirect":
                        redirect_host = str(status_resp.get("redirect_host") or "")
                        if redirect_host:
                            session._current_base_url = f"https://{redirect_host}"
                        # Stay in scanned state until confirmed.
                    elif status == "expired":
                        session._refresh_count += 1
                        if session._refresh_count > _MAX_QR_REFRESHES:
                            session.state = "expired"
                            session.error_message = "QR code expired after multiple refreshes. Start a new setup."
                            return
                        # Refresh: fetch a new QR, stay in waiting.
                        if not await self._fetch_qr(session, aiohttp_session):
                            return
                        session.state = "waiting"
                    elif status == "confirmed":
                        await self._handle_confirmed(session, status_resp)
                        return
                    else:
                        logger.warning(
                            "weixin QR session %s: unknown status %r",
                            session.session_id,
                            status,
                        )

                # Deadline reached.
                if session.state not in {"confirmed", "cancelled"}:
                    session.state = "expired"
                    session.error_message = "QR login timed out. Start a new setup."

        except asyncio.CancelledError:
            # Graceful cancellation from cancel_session().
            if session.state != "cancelled":
                session.state = "cancelled"
        except Exception as exc:
            logger.exception("weixin QR session %s: unexpected error", session.session_id)
            session.state = "failed"
            session.error_message = f"Unexpected error: {exc}"

    async def _fetch_qr(
        self, session: WeixinQRSession, aiohttp_session: Any
    ) -> bool:
        """Fetch a fresh QR from iLink and update session state.

        Returns True on success, False on failure (session.state set to failed).
        """
        try:
            from gateway.platforms.weixin import (
                ILINK_BASE_URL,
                EP_GET_BOT_QR,
                QR_TIMEOUT_MS,
                _api_get,
            )

            qr_resp = await _api_get(
                aiohttp_session,
                base_url=ILINK_BASE_URL,
                endpoint=f"{EP_GET_BOT_QR}?bot_type={session.bot_type}",
                timeout_ms=QR_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.error("weixin QR session %s: failed to fetch QR: %s", session.session_id, exc)
            session.state = "failed"
            session.error_message = f"Failed to fetch QR code from iLink: {exc}"
            return False

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            session.state = "failed"
            session.error_message = "iLink returned an empty QR code."
            return False

        session._qrcode_value = qrcode_value
        # WeChat must scan the full liteapp URL, not the raw hex token.
        session.qr_payload = qrcode_url if qrcode_url else qrcode_value

        # Render QR as base64 PNG for inline <img> rendering.
        session.qr_image_base64 = _render_qr_png_base64(session.qr_payload)
        return True

    async def _handle_confirmed(
        self, session: WeixinQRSession, status_resp: Dict[str, Any]
    ) -> None:
        """Extract credentials from confirmed status and save to disk."""
        try:
            from gateway.platforms.weixin import (
                ILINK_BASE_URL,
                save_weixin_account,
            )
        except ImportError as exc:
            session.state = "failed"
            session.error_message = f"Failed to import weixin adapter: {exc}"
            return

        account_id = str(status_resp.get("ilink_bot_id") or "")
        token = str(status_resp.get("bot_token") or "")
        base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL)
        user_id = str(status_resp.get("ilink_user_id") or "")

        if not account_id or not token:
            session.state = "failed"
            session.error_message = "QR confirmed but credential payload was incomplete."
            return

        # Save to ~/.hermes/weixin/accounts/<account_id>.json (same path as CLI wizard).
        try:
            save_weixin_account(
                session.hermes_home,
                account_id=account_id,
                token=token,
                base_url=base_url,
                user_id=user_id,
            )
        except Exception as exc:
            session.state = "failed"
            session.error_message = f"Failed to save WeChat credentials: {exc}"
            return

        session.credentials = {
            "account_id": account_id,
            "token": token,
            "base_url": base_url,
            "user_id": user_id,
        }
        session.state = "confirmed"
        logger.info(
            "weixin QR session %s: confirmed account_id=%s",
            session.session_id,
            account_id,
        )


def _render_qr_png_base64(data: str) -> str:
    """Render a QR code as a base64-encoded PNG string.

    Falls back to empty string if the ``qrcode`` package is unavailable.
    The frontend will then display the raw URL as a fallback link.
    """
    try:
        import qrcode

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("weixin QR: failed to render PNG: %s", exc)
        return ""


# Process-global singleton, mirroring _telegram_onboarding_pairings.
_weixin_qr_sessions = WeixinQRSessionManager()


def get_weixin_qr_session_manager() -> WeixinQRSessionManager:
    """Return the process-global WeixinQRSessionManager."""
    return _weixin_qr_sessions
