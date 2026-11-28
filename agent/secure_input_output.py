"""Streaming exact-value redaction before output storage, not after truncation."""
from __future__ import annotations

import re
import threading

from agent import secure_input_broker as broker

REDACTED = '[REDACTED SECURE INPUT]'
EXPIRED = '[Secure input TTL expired: process output blocked]'


def _trailing_prefix_length(data: str, value: str) -> int:
    """Longest undecidable suffix in linear time, without retaining a matcher."""
    if len(value) <= 1:
        return 0
    prefix = [0] * len(value)
    for index in range(1, len(value)):
        matched = prefix[index - 1]
        while matched and value[index] != value[matched]:
            matched = prefix[matched - 1]
        if value[index] == value[matched]:
            matched += 1
        prefix[index] = matched
    matched = 0
    for char in data[-(len(value) - 1):]:
        while matched and char != value[matched]:
            matched = prefix[matched - 1]
        if char == value[matched]:
            matched += 1
    return matched


class SecureOutputFilter:
    """Keep only an undecidable trailing fragment until a later chunk or EOF."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self._pending = ''
        self._lock = threading.Lock()
        self._expired = False
        self._finished = False
        self._timer = threading.Timer(max(0, deadline - broker._now()), self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _expire(self) -> None:
        with self._lock:
            self._expired = True
            self._pending = ''

    def _expire_if_due(self) -> bool:
        self._expired = self._expired or broker._now() >= self.deadline
        if self._expired:
            self._pending = ''
        return self._expired

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expire_if_due()

    def feed(self, text: str, *, final: bool = False) -> str:
        with self._lock:
            if final:
                self._timer.cancel()
            if self._finished:
                return ''
            self._finished = final
            if self._expire_if_due():
                return ''
            with broker._lock:
                broker._purge_expired_locked()
                values = sorted(broker._redaction_values, key=len, reverse=True)
            if not values:
                self._expired = True
                self._pending = ''
                return ''
            data = self._pending + text
            hold = max(_trailing_prefix_length(data, value) for value in values)
            cut = len(data) if final else max(0, len(data) - hold)
            pattern = re.compile('|'.join(re.escape(value) for value in values)) if values else None
            if pattern is not None:
                for match in pattern.finditer(data):
                    if match.start() < cut < match.end():
                        cut = match.start()
                        break
            self._pending = data[cut:]
            safe = pattern.sub(REDACTED, data[:cut]) if pattern is not None else data[:cut]
            return '' if self._expire_if_due() else safe


_DEADLINE_ATTR = '_hermes_secure_output_deadline'


def attach_output_deadline(proc, deadline: float) -> None:
    """Mark a spawned process whose output must pass the secure-output filter until *deadline*."""
    setattr(proc, _DEADLINE_ATTR, deadline)


def process_output_filter(proc):
    """A foreground Popen carries its filter through adoption into background."""
    deadline = getattr(proc, _DEADLINE_ATTR, None)
    if deadline is None:
        return None
    output_filter = getattr(proc, '_hermes_secure_output_filter', None)
    if output_filter is None:
        output_filter = SecureOutputFilter(deadline)
        proc._hermes_secure_output_filter = output_filter
    return output_filter
