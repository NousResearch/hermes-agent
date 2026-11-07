"""Runaway-loop signal from a local vLLM server's speculative-decoding counters.

A model stuck in an exact loop is predicted perfectly by its own draft head, so every
draft token is accepted, step after step. Measured on DeepSeek V4.1 Flash with MTP K=5:
a real 9-minute reasoning loop held 6.00 tokens/step (all drafts accepted) every minute,
normal reasoning 2-3, and verbatim copying of a 13 KB source file 4.6-6.0 with at most one
5 s stretch fully accepted. ``RunawayStreamWatch`` reads the text and needs ~39K chars of
evidence; this reads the server and cuts after CUT_AFTER_S of full acceptance.

Only local endpoints are polled (``/metrics`` next to ``/v1``). Anything that does not
expose the vLLM counters disables the watch on its first poll.
"""
from __future__ import annotations

import http.client
import math
import re
import time
import urllib.request
from typing import Callable, Optional

from agent.model_metadata import is_local_endpoint

FULL_ACCEPT_RATIO = 0.98
CUT_AFTER_S = 60.0
POLL_S = 2.0
MIN_DRAFT_TOKENS = 20  # per poll: fewer means the server was not really decoding

_COUNTER = re.compile(
    r"^vllm:spec_decode_num_(draft|accepted)_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$", re.MULTILINE)


def read_counters(metrics_text: str) -> Optional[tuple[float, float]]:
    """(draft tokens, accepted tokens) summed over engines, or None without both series."""
    sums: dict[str, float] = {}
    for kind, value in _COUNTER.findall(metrics_text):
        sums[kind] = sums.get(kind, 0.0) + float(value)
    if "draft" not in sums or "accepted" not in sums:
        return None
    return sums["draft"], sums["accepted"]


def metrics_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/metrics"


class DraftAcceptanceWatch:
    def __init__(self, fetch: Callable[[], str], clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch
        self._clock = clock
        self._next = 0.0
        self._prev: Optional[tuple[float, tuple[float, float]]] = None
        self._since: Optional[float] = None

    @classmethod
    def for_base_url(cls, base_url: Optional[str]) -> Optional[DraftAcceptanceWatch]:
        if not base_url or not is_local_endpoint(base_url):
            return None
        url = metrics_url(base_url)
        return cls(lambda: urllib.request.urlopen(url, timeout=0.5).read().decode("utf-8", "replace"))

    def full_accept_seconds(self) -> float:
        return self._clock() - self._since if self._since is not None else 0.0

    def poll(self) -> bool:
        """Call per stream chunk; True once every draft has been accepted for CUT_AFTER_S."""
        now = self._clock()
        if now < self._next:
            return False
        self._next = now + POLL_S
        try:
            counters = read_counters(self._fetch())
        except (OSError, ValueError, http.client.HTTPException):  # this server cannot be watched
            counters = None
        if counters is None:
            self._next = math.inf
            return False
        prev, self._prev = self._prev, (now, counters)
        if prev is None:
            return False
        d_draft = counters[0] - prev[1][0]
        d_acc = counters[1] - prev[1][1]
        if d_draft >= MIN_DRAFT_TOKENS and d_acc >= FULL_ACCEPT_RATIO * d_draft:
            if self._since is None:
                self._since = prev[0]
        else:
            self._since = None
        return self.full_accept_seconds() >= CUT_AFTER_S
