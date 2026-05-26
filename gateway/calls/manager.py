import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from .browser_room import BrowserRoomProvider
from .models import CallError, CallSession, CallState
from .tokens import CallTokenService


@dataclass(frozen=True)
class CallCommandResult:
    ok: bool
    message: str
    code: str | None = None
    session: CallSession | None = None


def _platform_name(source) -> str:
    platform = getattr(source, "platform", "")
    return str(getattr(platform, "value", platform) or "")


def _session_key(source) -> tuple[str, str, str]:
    return (
        _platform_name(source),
        str(getattr(source, "chat_id", "") or ""),
        str(getattr(source, "user_id", "") or ""),
    )


class CallManager:
    def __init__(
        self,
        *,
        browser_provider: BrowserRoomProvider,
        token_service: CallTokenService,
        now: Callable[[], datetime] | None = None,
        ttl_seconds: int = 600,
    ):
        self.browser_provider = browser_provider
        self.token_service = token_service
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.ttl_seconds = int(ttl_seconds)
        self._sessions: dict[tuple[str, str, str], CallSession] = {}

    async def start_browser_call(self, source) -> CallCommandResult:
        if str(getattr(source, "chat_type", "dm") or "dm").lower() != "dm":
            return CallCommandResult(
                ok=False,
                code="call_private_chat_required",
                message="Calls are private-only. DM me /call to create a private room.",
            )
        key = _session_key(source)
        existing = self._sessions.get(key)
        if existing and existing.state not in {CallState.ENDED, CallState.FAILED}:
            return CallCommandResult(
                ok=False,
                code="call_already_active",
                message="A call is already active for this chat. Use /call status or /call end.",
                session=existing,
            )
        created_at = self.now()
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        expires_at = created_at + timedelta(seconds=self.ttl_seconds)
        call_id = f"call_{uuid.uuid4().hex}"
        token = self.token_service.mint(
            platform=key[0],
            chat_id=key[1],
            user_id=key[2],
            call_id=call_id,
            now=created_at,
            ttl_seconds=self.ttl_seconds,
        )
        try:
            room_url = self.browser_provider.create_room_url(call_id, token)
        except CallError as exc:
            return CallCommandResult(ok=False, code=exc.code, message=exc.message)
        session = CallSession(
            call_id=call_id,
            platform=key[0],
            chat_id=key[1],
            user_id=key[2],
            mode="browser",
            state=CallState.WAITING,
            room_url=room_url,
            created_at=created_at,
            expires_at=expires_at,
        )
        self._sessions[key] = session
        return CallCommandResult(
            ok=True,
            session=session,
            message=(
                "Private call room ready:\n"
                f"{room_url}\n\n"
                f"This Tailnet-only link expires in {self.ttl_seconds // 60} minutes."
            ),
        )

    def record_native_call(
        self,
        source,
        call_id: str,
        state: str = CallState.CONNECTING.value,
    ) -> CallSession:
        key = _session_key(source)
        existing = self._sessions.get(key)
        if existing and existing.state not in {CallState.ENDED, CallState.FAILED}:
            return existing
        created_at = self.now()
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        expires_at = created_at + timedelta(seconds=self.ttl_seconds)
        try:
            call_state = CallState(state)
        except ValueError:
            call_state = CallState.CONNECTING
        session = CallSession(
            call_id=call_id,
            platform=key[0],
            chat_id=key[1],
            user_id=key[2],
            mode="simplex_native",
            state=call_state,
            room_url=None,
            created_at=created_at,
            expires_at=expires_at,
        )
        self._sessions[key] = session
        return session

    async def status(self, source) -> CallCommandResult:
        session = self._sessions.get(_session_key(source))
        if not session or session.state in {CallState.ENDED, CallState.FAILED}:
            return CallCommandResult(ok=True, message="No active call.")
        if session.room_url is None:
            return CallCommandResult(
                ok=True,
                session=session,
                message=f"Call {session.call_id} is {session.state.value}.",
            )
        return CallCommandResult(
            ok=True,
            session=session,
            message=(
                f"Call {session.call_id} is {session.state.value}. "
                f"Link expires at {session.expires_at.isoformat()}."
            ),
        )

    async def end(self, source) -> CallCommandResult:
        key = _session_key(source)
        session = self._sessions.get(key)
        if not session or session.state in {CallState.ENDED, CallState.FAILED}:
            return CallCommandResult(ok=True, message="No active call.")
        ended = CallSession(
            call_id=session.call_id,
            platform=session.platform,
            chat_id=session.chat_id,
            user_id=session.user_id,
            mode=session.mode,
            state=CallState.ENDED,
            room_url=session.room_url,
            created_at=session.created_at,
            expires_at=session.expires_at,
            ended_at=self.now(),
        )
        self._sessions[key] = ended
        return CallCommandResult(ok=True, session=ended, message="Call ended.")
