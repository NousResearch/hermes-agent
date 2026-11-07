"""Turn-owned whole-file speech, independent of provider-returned filenames."""

import asyncio
from contextlib import AbstractContextManager
import logging
from pathlib import Path
import tempfile
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class GeneratedAudio(AbstractContextManager):
    """Own a unique child of the selected output root until synthesis AND delivery end.

    Returned paths may belong to a provider or another turn. Only this private directory
    is ours to remove; keeping the selected basename preserves format and chunk naming.
    """

    def __init__(self, requested_path: str):
        requested = Path(requested_path)
        self._directory = tempfile.TemporaryDirectory(prefix="tts-", dir=requested.parent)
        self.output_path = str(Path(self._directory.name) / requested.name)
        self._lock = threading.Lock()
        self._closed = False
        self._worker_running = False

    async def synthesize(self, provider: Callable, **kwargs):
        """Copy the active profile scope, but leave cleanup in the actual provider thread."""
        return await asyncio.to_thread(self._run, provider, kwargs)

    def _run(self, provider: Callable, kwargs: dict):
        with self._lock:
            # A queued call may start after its caller has already cancelled.
            if self._closed:
                return None
            self._worker_running = True
        try:
            return provider(output_path=self.output_path, **kwargs)
        finally:
            with self._lock:
                self._worker_running = False
                if self._closed:
                    self._cleanup()

    def _cleanup(self):
        try:
            self._directory.cleanup()
        except OSError:
            # A locked file must not replace the answer or the original cancellation.
            logger.warning("Could not remove generated speech files", exc_info=True)

    def __exit__(self, *exc):
        with self._lock:
            self._closed = True
            # Cancellation cannot stop a running provider. It retains this object and
            # cleans in its finally, even after the originating loop has closed.
            if not self._worker_running:
                self._cleanup()
