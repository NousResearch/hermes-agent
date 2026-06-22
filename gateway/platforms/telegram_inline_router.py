"""Dispatch layer for Telegram inline queries.

Loads ``inline_tools.yaml`` from the Hermes config directory to determine which
executor handles each query.  The framework ships the dispatch surface only —
concrete tool implementations are user-space concerns.

Registry format (inline_tools.yaml)::

    version: 1
    tools:
      - id: my_tool
        type: direct          # "direct" skips LLM; "llm" routes through a model
        executor: my_executor # name passed to TelegramInlineRouter.register_executor()
        match:
          - pattern: "https?://example\\.com/"
            type: url          # url | prefix | search (catch-all)
            priority: 0        # lower = higher precedence
          - pattern: ".*"
            type: search
            priority: 99
        timeout_sec: 25
        cache:
          backend: memory
          ttl_sec: 3600
        staging_chat_env: TELEGRAM_INLINE_STAGING_CHAT  # env var naming the staging chat id
        enabled: true

``staging_chat_env`` names an environment variable that holds the chat ID where
the executor should stage files to obtain a ``file_id``.  The recommended env var
name is ``TELEGRAM_INLINE_STAGING_CHAT``; set it to any chat or channel the inline
bot has access to.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_registry_path() -> str:
    try:
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home() / "inline_tools.yaml")
    except Exception:
        return os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "inline_tools.yaml")


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

class InlineToolRegistry:
    """Loads and caches inline_tools.yaml; provides pattern matching."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._tools: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            import yaml
            with open(self._path) as fh:
                data = yaml.safe_load(fh) or {}
            self._tools = data.get("tools", [])
            logger.info("[inline_router] loaded %d tools from %s", len(self._tools), self._path)
        except FileNotFoundError:
            logger.warning("[inline_router] registry not found at %s — all inline queries will be dropped", self._path)
        except Exception as exc:
            logger.error("[inline_router] registry load error: %s", exc)

    def match(self, query: str) -> Optional[Dict[str, Any]]:
        """Return the first enabled tool whose match patterns fit query, or None."""
        enabled = [t for t in self._tools if t.get("enabled", False)]

        def _min_prio(tool: Dict[str, Any]) -> int:
            return min((m.get("priority", 0) for m in tool.get("match", [])), default=0)

        for tool in sorted(enabled, key=_min_prio):
            for matcher in tool.get("match", []):
                mtype = matcher.get("type", "search")
                pattern = matcher.get("pattern", "")
                try:
                    if mtype == "url" and re.search(pattern, query):
                        return tool
                    elif mtype == "prefix" and re.match(pattern, query):
                        return tool
                    elif mtype == "search":
                        return tool  # catch-all
                except re.error:
                    pass
        return None


# ---------------------------------------------------------------------------
# Executor interface
# ---------------------------------------------------------------------------

class InlineExecutor(ABC):
    """Base class for inline query executors.

    Subclass this in user-space and register with
    ``TelegramInlineRouter.register_executor()``.

    ``execute()`` must return within the adapter's ``INLINE_RESPONSE_DEADLINE``
    (default 6.5 s) or it will be cancelled.  Executors that need longer may
    return a stub result (e.g. ``InlineQueryResultArticle("Processing...")``)
    immediately and manage their own background state.
    """

    @abstractmethod
    async def execute(self, user_id: int, query: str) -> List[Any]:
        """Run the tool and return a list of InlineQueryResult objects.

        Args:
            user_id: Telegram user ID of the query sender.
            query: The full inline query text after tool matching.

        Returns:
            List of ``InlineQueryResult`` objects passed directly to
            ``answerInlineQuery``.  Return an empty list to show nothing.

        Raises:
            Exception: Propagated to the adapter, which answers with an empty
                result list and logs a warning.
        """
        ...


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TelegramInlineRouter:
    """Dispatch router for Telegram inline queries.  One instance per adapter.

    Usage::

        router = TelegramInlineRouter(bot)
        router.register_executor("my_executor", MyExecutor)

    Executors are looked up by the ``executor`` field in inline_tools.yaml.
    The framework ships no concrete executors; implementations are user-space.
    Result lifecycle (caching, stubs, retries) is entirely the executor's concern.
    """

    def __init__(self, bot: Any, registry_path: Optional[str] = None) -> None:
        self._bot = bot
        _path = registry_path if registry_path is not None else _default_registry_path()
        self._registry = InlineToolRegistry(_path)
        self._executors: Dict[str, Callable[[Dict[str, Any], Any], InlineExecutor]] = {}

    def register_executor(
        self,
        name: str,
        factory: Callable[[Dict[str, Any], Any], InlineExecutor],
    ) -> None:
        """Register an executor factory for tools that declare ``executor: <name>``.

        The factory is called as ``factory(tool_config, bot)`` and must return an
        ``InlineExecutor`` instance.  Call this before the first inline query arrives
        (e.g. right after ``TelegramInlineRouter.__init__``).

        Example::

            router.register_executor("audio_dl", MyAudioDownloader)
        """
        self._executors[name] = factory

    async def dispatch(self, user_id: int, query: str) -> List[Any]:
        """Match query to a tool, call its executor, return InlineQueryResult list."""
        tool = self._registry.match(query)
        if tool is None:
            logger.debug("[inline_router] no enabled tool matched %r", query[:60])
            return []

        executor_name = tool.get("executor", "")
        factory = self._executors.get(executor_name)
        if factory is None:
            logger.warning("[inline_router] no executor registered for %r", executor_name)
            return []

        executor = factory(tool, self._bot)
        return await executor.execute(user_id, query)
