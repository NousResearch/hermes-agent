"""Per-provider-response buffering for long-turn terminal drafts.

The gate owns only already-scrubbed visible text from one provider response.
It never suppresses status, tool, guardrail, or other direct callbacks.  A
tool-bearing response is drained as interim prose; a terminal response is held
until turn finalization has produced and persisted the canonical closer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


MAX_BUFFERED_VISIBLE_BYTES = 256 * 1024


@dataclass
class ProviderResponseGate:
    """Bounded buffer for one provider response's scrubbed visible deltas."""

    discard: bool = False
    max_bytes: int = MAX_BUFFERED_VISIBLE_BYTES
    _chunks: list[str] = field(default_factory=list)
    _byte_count: int = 0
    overflowed: bool = False

    @property
    def has_text(self) -> bool:
        return bool(self._chunks)

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def capture(self, text: str) -> tuple[bool, list[str]]:
        """Capture *text* and return ``(consumed, fail_open_chunks)``.

        Once the hard byte cap is crossed, all buffered bytes plus the current
        delta are returned for one fail-open publication.  Later deltas pass
        through normally, so overflow cannot duplicate the prefix.
        """
        if not isinstance(text, str) or not text:
            return True, []
        if self.discard:
            return True, []
        if self.overflowed:
            return False, []

        encoded_size = len(text.encode("utf-8", errors="replace"))
        if self._byte_count + encoded_size > self.max_bytes:
            chunks = [*self._chunks, text]
            self._chunks.clear()
            self._byte_count = 0
            self.overflowed = True
            return True, chunks

        self._chunks.append(text)
        self._byte_count += encoded_size
        return True, []

    def drain(self) -> list[str]:
        chunks = self._chunks
        self._chunks = []
        self._byte_count = 0
        return chunks

    def clear(self) -> None:
        self._chunks.clear()
        self._byte_count = 0


__all__ = ["MAX_BUFFERED_VISIBLE_BYTES", "ProviderResponseGate"]
