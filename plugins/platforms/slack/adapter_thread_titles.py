"""Best-effort provisional and semantic titles for Slack assistant DM threads."""

import logging
import re

logger = logging.getLogger("plugins.platforms.slack.adapter")


class AssistantThreadTitles:
    def _assistant_thread_title_enabled(self) -> bool:
        raw = self.config.extra.get("assistant_thread_titles", True)
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "no", "off"}
        return bool(raw)

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
            _set_title = self._session_title_method(self._get_client(channel_id, team_id=team_id))
            await _set_title(channel_id=channel_id, thread_ts=thread_ts, title=title)
        except Exception as e:
            logger.debug("[Slack] session title set failed: %s", e, exc_info=True)
            return
        self._titled_assistant_threads.add(key)
        if semantic:
            self._semantically_titled_assistant_threads.add(key)
        # Evict oldest thread_ts first so recently titled threads keep their guard.
        self._evict_oldest_by_ts(
            self._titled_assistant_threads, self._TITLED_ASSISTANT_THREADS_MAX, lambda e: e[2])
        self._evict_oldest_by_ts(
            self._semantically_titled_assistant_threads, self._TITLED_ASSISTANT_THREADS_MAX, lambda e: e[2])
