"""Best-effort provisional and semantic titles for Slack assistant DM threads."""

import logging
import re

logger = logging.getLogger("plugins.platforms.slack.adapter")

# rename does not create a session. These are the codes Slack returns when the
# thread has no agent session yet (a real thread, or a ts that is not one).
_MISSING_SESSION = frozenset({"session_not_found", "not_authorized"})


def _slack_error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("error") or "")
    return ""


def _agent_sessions_enabled() -> bool:
    try:
        from plugins.platforms.slack.adapter import _sdk_supports_agent_sessions
    except ImportError:  # flat plugin-directory import
        from adapter import _sdk_supports_agent_sessions
    return bool(_sdk_supports_agent_sessions())


def _agent_session_lifecycle_status(status: str) -> str:
    """Map a legacy typing phrase onto ``agents.sessions.setStatus``.

    That method rejects free text and an empty clear. ``processing`` is the
    working state; ``active`` is the ready state that replaced an empty clear.
    """
    return "processing" if str(status or "").strip() else "active"


class AssistantThreadTitles:
    def _assistant_thread_title_enabled(self) -> bool:
        raw = self.config.extra.get("assistant_thread_titles", True)
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "no", "off"}
        return bool(raw)

    async def _write_session_title(self, client, channel_id: str, thread_ts: str, title: str) -> None:
        """Rename the session, creating it first when Slack has not opened one yet."""
        rename = self._session_title_method(client)
        try:
            await rename(channel_id=channel_id, thread_ts=thread_ts, title=title)
            return
        except Exception as exc:
            if not _agent_sessions_enabled() or _slack_error_code(exc) not in _MISSING_SESSION:
                raise
        # setStatus creates the session. Its title applies only on create, so rename
        # still runs for a session that appeared between the two calls.
        await self._session_status_method(client)(
            channel_id=channel_id, thread_ts=thread_ts, status="active", title=title)
        await rename(channel_id=channel_id, thread_ts=thread_ts, title=title)

    async def _set_thread_status(
        self, chat_id: str, team_id: str, thread_ts: str, status: str, fail_label: str) -> None:
        """Set or clear the thread status. Agent Sessions uses the lifecycle enum."""
        if _agent_sessions_enabled():
            status = _agent_session_lifecycle_status(status)
        try:
            setter = self._session_status_method(self._get_client(chat_id, team_id=team_id))
            await setter(channel_id=chat_id, thread_ts=thread_ts, status=status)
        except Exception as exc:
            logger.warning(
                "[Slack] session status %s: %s",
                fail_label, _slack_error_code(exc) or type(exc).__name__,
                exc_info=True,
            )

    async def _set_assistant_thread_title(
        self, channel_id: str, thread_ts: str, title_source: str, *, team_id: str = "", semantic: bool = False) -> None:
        """Best-effort title for visible Slack AI DM threads."""
        if (
            not self._app or not channel_id or not thread_ts or not title_source
            or not self._assistant_thread_title_enabled()):
            return
        key = self._workspace_thread_key(team_id, channel_id, thread_ts)
        if not key or (
            key in self._titled_assistant_threads
            and (not semantic or key in self._semantically_titled_assistant_threads)
        ):
            return
        title = re.sub(r"\s+", " ", title_source).strip()
        if not title or title.startswith("/"):
            return
        title = title[:77].rstrip() + "..." if len(title) > 80 else title
        try:
            await self._write_session_title(
                self._get_client(channel_id, team_id=team_id), channel_id, thread_ts, title)
        except Exception as exc:
            logger.warning(
                "[Slack] session title set failed: %s",
                _slack_error_code(exc) or type(exc).__name__,
                exc_info=True,
            )
            return
        self._titled_assistant_threads.add(key)
        if semantic:
            self._semantically_titled_assistant_threads.add(key)
        # Evict oldest thread_ts first so recently titled threads keep their guard.
        self._evict_oldest_by_ts(
            self._titled_assistant_threads, self._TITLED_ASSISTANT_THREADS_MAX, lambda e: e[2])
        self._evict_oldest_by_ts(
            self._semantically_titled_assistant_threads, self._TITLED_ASSISTANT_THREADS_MAX, lambda e: e[2])
