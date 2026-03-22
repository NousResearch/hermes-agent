"""HindsightSessionManager: auto-retain and auto-recall for the agent loop.

This mirrors the pattern used by HonchoSessionManager:

  - After each turn, the user + assistant messages are queued for async retain
    (background thread, never blocks the response).
  - Before each turn, recall() is prefetched in a background thread so context
    is ready to inject into the system prompt or user message.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_client import Hindsight
    from hindsight_integration.client import HindsightClientConfig

logger = logging.getLogger(__name__)


class HindsightSessionManager:
    """Manages async retain + recall prefetch for a Hindsight bank."""

    def __init__(self, client: Hindsight, config: HindsightClientConfig) -> None:
        self._client = client
        self._config = config
        self._bank_id = config.bank_id
        self._budget = config.budget

        # ── Async retain worker ──
        self._retain_queue: queue.Queue[tuple[str, str | None] | None] = queue.Queue()
        self._retain_thread = threading.Thread(
            target=self._retain_worker, daemon=True, name="hindsight-retain"
        )
        self._retain_thread.start()

        # ── Recall prefetch cache ──
        # Protected by _prefetch_lock; _prefetch_result is set by the bg thread.
        self._prefetch_lock = threading.Lock()
        self._prefetch_result: str | None = None
        self._prefetch_thread: threading.Thread | None = None

    # ── Retain ────────────────────────────────────────────────────────────────

    def _retain_worker(self) -> None:
        """Background thread: drains the retain queue."""
        while True:
            item = self._retain_queue.get()
            if item is None:
                self._retain_queue.task_done()
                break
            content, context = item
            try:
                self._client.retain(bank_id=self._bank_id, content=content, context=context)
                logger.debug("Hindsight retained content (%d chars)", len(content))
            except Exception as e:
                logger.warning("Hindsight retain failed: %s", e)
            finally:
                self._retain_queue.task_done()

    def retain_async(self, content: str, context: str | None = None) -> None:
        """Queue content for async retain — returns immediately."""
        if content and content.strip():
            self._retain_queue.put((content, context))

    def flush(self) -> None:
        """Block until all queued retain calls complete."""
        self._retain_queue.join()

    # ── Recall / prefetch ─────────────────────────────────────────────────────

    def _do_recall(self, query: str) -> str:
        """Run recall synchronously; return formatted string or empty."""
        try:
            resp = self._client.recall(
                bank_id=self._bank_id, query=query, budget=self._budget
            )
            if not resp.results:
                return ""
            return "\n".join(r.text for r in resp.results if r.text)
        except Exception as e:
            logger.debug("Hindsight recall failed: %s", e)
            return ""

    def prefetch_recall(self, query: str) -> None:
        """Start a background recall for the given query.

        Call pop_recall_result() on the next turn to consume the result.
        """
        def _run() -> None:
            result = self._do_recall(query)
            with self._prefetch_lock:
                self._prefetch_result = result

        # Wait for any in-flight prefetch before starting a new one
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0)  # non-blocking; stale result is fine

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="hindsight-prefetch"
        )
        self._prefetch_thread.start()

    def pop_recall_result(self) -> str:
        """Return the cached prefetch result and clear it.

        Waits up to 3 s for a running prefetch to finish.
        """
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result or ""
            self._prefetch_result = None
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the retain worker thread gracefully."""
        self._retain_queue.put(None)
        self._retain_thread.join(timeout=5.0)
