"""Best-effort provisional and semantic titles for Slack assistant DM threads."""

import logging
import re

logger = logging.getLogger("plugins.platforms.slack.adapter")

# rename does not create a session. session_not_found is the no-session code.
# not_authorized means the app is not allowed to rename an existing session.
_MISSING_SESSION = frozenset({"session_not_found"})


def _slack_error_code(exc: BaseException) -> str:
    """Structured Slack ``error`` field, including mapping-like SDK responses."""
    response = getattr(exc, "response", None)
    getter = getattr(response, "get", None)
    if not callable(getter):
        return ""
    try:
        code = getter("error")
    except (AttributeError, TypeError, KeyError):
        return ""
    return str(code or "")


def _response_title(response) -> str:
    """Title echoed by a Slack response, including mapping-like SDK bodies."""
    getter = getattr(response, "get", None)
    title = None
    if callable(getter):
        try:
            title = getter("title")
        except (AttributeError, TypeError, KeyError):
            title = None
    if isinstance(title, str):
        return title
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        nested = data.get("title")
        if isinstance(nested, str):
            return nested
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
        # setStatus creates the session and applies the title only then. A failed
        # follow-up rename keeps the guard only when Slack echoed that title.
        created = await self._session_status_method(client)(
            channel_id=channel_id, thread_ts=thread_ts, status="active", title=title)
        try:
            await rename(channel_id=channel_id, thread_ts=thread_ts, title=title)
        except Exception as exc:
            logger.warning(
                "[Slack] session title follow-up rename failed: %s",
                _slack_error_code(exc) or type(exc).__name__,
                exc_info=True,
            )
            # setStatus applies title only on create. Keep the guard only when
            # Slack echoed that title; otherwise a later message must be able to retry.
            if _response_title(created) != title:
                raise

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
