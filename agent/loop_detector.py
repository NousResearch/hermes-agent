"""Streaming degenerate-repetition ("loop") detector for the assistant token stream.

Algorithm-only and dependency-free: it is fed streamed assistant *content* text
incrementally (never reasoning or tool-call args) and reports when the output has
collapsed into a repetition loop. Detection keys on **redundancy, not length**, so
legitimately long agentic responses (tens of thousands of tokens) are never
truncated — only genuinely degenerate output trips it.

Two cheap detectors run incrementally:
  * fast path (per completed line, O(1)): consecutive identical-line counter, with
    a higher bar for short/trivial lines and inside code fences so normal code
    (indentation, ``}``, short imports) never trips;
  * tail path (no-newline loops like ``abab…`` / ``I'll go. I'll go.``): a smallest
    -period (KMP) test over the tail of the current line;
  * heavy path (only every ``check_every_bytes`` ~1 KB): over a bounded sliding
    window, requires BOTH a low distinct-n-gram ratio AND low Shannon entropy
    (or, opt-in, a low zlib compression ratio).

On a positive ``feed()`` the streaming layer aborts via the existing interrupt
path and raises :class:`StreamLoopDetected`; the conversation loop discards the
looped partial (no history poisoning) and re-prompts. See the project plan.
"""

from __future__ import annotations

import math
import os
import zlib
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Optional


class StreamLoopDetected(InterruptedError):
    """Raised by the streaming layer when a repetition loop is confirmed.

    Subclasses :class:`InterruptedError` so the existing ``except InterruptedError``
    handlers in the streaming worker and the conversation loop already route it;
    the conversation loop distinguishes it *by type* to run loop-recovery instead
    of the user-interrupt path.
    """


@dataclass(frozen=True)
class LoopDetectionConfig:
    enabled: bool = True
    window_chars: int = 4000
    consecutive_line_threshold: int = 6
    tail_check_min_len: int = 64
    tail_max_period: int = 64
    tail_min_repeats: int = 5
    ngram_size: int = 48
    distinct_ngram_ratio_threshold: float = 0.15
    entropy_threshold: float = 2.5
    block_min_lines: int = 8
    block_repeat_ratio_threshold: float = 0.5
    allowed_min_len: int = 12
    check_every_bytes: int = 1024
    relax_in_code_fence: bool = True
    use_zlib_ratio: bool = False
    zlib_ratio_threshold: float = 0.10
    max_retries: int = 2


_FALSEY = {"0", "false", "no", "off", ""}


def _env_override_enabled(default: bool) -> bool:
    v = os.environ.get("HERMES_LOOP_DETECTION_ENABLED")
    if v is None:
        return default
    return v.strip().lower() not in _FALSEY


def load_loop_detection_config(config: Optional[dict] = None) -> LoopDetectionConfig:
    """Build a :class:`LoopDetectionConfig` from ``config.yaml`` + env overrides.

    Mirrors the precedence used by ``verification_stop.verify_on_stop_enabled``:
    an explicit ``HERMES_LOOP_DETECTION_ENABLED`` env var wins over the
    ``loop_detection.enabled`` config value, which defaults to ``True``.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    section: dict = {}
    if isinstance(config, dict):
        sec = config.get("loop_detection")
        if isinstance(sec, dict):
            section = sec

    defaults = LoopDetectionConfig()

    def _num(key: str, default, cast):
        try:
            return cast(section.get(key, default))
        except Exception:
            return default

    def _flag(key: str, default: bool) -> bool:
        val = section.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() not in _FALSEY
        return bool(val)

    return LoopDetectionConfig(
        enabled=_env_override_enabled(_flag("enabled", defaults.enabled)),
        window_chars=max(256, _num("window_chars", defaults.window_chars, int)),
        consecutive_line_threshold=max(
            2, _num("consecutive_line_threshold", defaults.consecutive_line_threshold, int)
        ),
        tail_check_min_len=max(16, _num("tail_check_min_len", defaults.tail_check_min_len, int)),
        tail_max_period=max(2, _num("tail_max_period", defaults.tail_max_period, int)),
        tail_min_repeats=max(3, _num("tail_min_repeats", defaults.tail_min_repeats, int)),
        ngram_size=max(8, _num("ngram_size", defaults.ngram_size, int)),
        distinct_ngram_ratio_threshold=_num(
            "distinct_ngram_ratio_threshold", defaults.distinct_ngram_ratio_threshold, float
        ),
        entropy_threshold=_num("entropy_threshold", defaults.entropy_threshold, float),
        block_min_lines=max(4, _num("block_min_lines", defaults.block_min_lines, int)),
        block_repeat_ratio_threshold=_num(
            "block_repeat_ratio_threshold", defaults.block_repeat_ratio_threshold, float
        ),
        allowed_min_len=max(0, _num("allowed_min_len", defaults.allowed_min_len, int)),
        check_every_bytes=max(256, _num("check_every_bytes", defaults.check_every_bytes, int)),
        relax_in_code_fence=_flag("relax_in_code_fence", defaults.relax_in_code_fence),
        use_zlib_ratio=_flag("use_zlib_ratio", defaults.use_zlib_ratio),
        zlib_ratio_threshold=_num("zlib_ratio_threshold", defaults.zlib_ratio_threshold, float),
        max_retries=max(0, _num("max_retries", defaults.max_retries, int)),
    )


def _smallest_period(s: str) -> int:
    """Length of the smallest period of ``s`` via the KMP failure function.

    Returns ``len(s)`` when ``s`` is not (perfectly) periodic.
    """
    n = len(s)
    fail = [0] * n
    k = 0
    for i in range(1, n):
        while k and s[i] != s[k]:
            k = fail[k - 1]
        if s[i] == s[k]:
            k += 1
        fail[i] = k
    period = n - fail[n - 1]
    return period if period and n % period == 0 else n


def _shannon_entropy(s: str) -> float:
    if not s:
        return 8.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


class StreamLoopDetector:
    """Incremental loop detector. Feed assistant ``content`` deltas; ``feed`` returns
    ``True`` once a loop is confirmed (and stays ``True`` thereafter)."""

    def __init__(self, cfg: Optional[LoopDetectionConfig] = None):
        self.cfg = cfg or LoopDetectionConfig()
        self.reset()

    def reset(self) -> None:
        self._buf: deque[str] = deque(maxlen=self.cfg.window_chars)
        self._partial = ""
        self._last_line: Optional[str] = None
        self._streak = 1
        self._in_fence = False
        self._bytes_since_heavy = 0
        self._tripped = False
        self._reason = ""

    def reason(self) -> str:
        return self._reason

    def _trip(self, reason: str) -> bool:
        self._tripped = True
        self._reason = reason
        return True

    def _on_line(self, line: str) -> bool:
        stripped = line.strip()
        # Toggle code-fence state on a fence line; fence lines never count.
        if stripped.startswith("```"):
            self._in_fence = not self._in_fence
            self._last_line = None
            self._streak = 1
            return False
        if not stripped:  # blank lines never count
            self._last_line = None
            self._streak = 1
            return False

        if line == self._last_line:
            self._streak += 1
        else:
            self._last_line = line
            self._streak = 1

        # Higher bar for short/trivial lines and inside code fences so legit
        # repetition (indentation, ``}``, short imports, code) does not trip.
        threshold = self.cfg.consecutive_line_threshold
        if len(stripped) < self.cfg.allowed_min_len or (self._in_fence and self.cfg.relax_in_code_fence):
            threshold *= 2
        if self._streak >= threshold:
            return self._trip(f"consecutive identical line x{self._streak}: {stripped[:40]!r}")
        return False

    def _tail_check(self) -> bool:
        # No-newline loops (``abab…``, ``I'll go. I'll go.``): test the tail of the
        # current (long) line for a short repeating period.
        tail = self._partial[-(self.cfg.tail_max_period * self.cfg.tail_min_repeats):]
        if len(tail) < self.cfg.tail_check_min_len:
            return False
        period = _smallest_period(tail)
        if 2 <= period <= self.cfg.tail_max_period and len(tail) // period >= self.cfg.tail_min_repeats:
            unit = tail[:period]
            if unit.strip():  # ignore pure-whitespace periods
                return self._trip(f"repeating period x{len(tail) // period}: {unit[:40]!r}")
        return False

    def _heavy_check(self) -> bool:
        window = "".join(self._buf)
        if len(window) < self.cfg.window_chars // 2:
            return False
        # Don't run the heavy redundancy check on code-fence-dominated windows.
        if self._in_fence and self.cfg.relax_in_code_fence:
            return False
        # Block/paragraph repetition: a small set of non-trivial lines dominating the
        # window. Catches plan/paragraph cycling that the entropy + n-gram checks miss,
        # because repeated NATURAL PROSE keeps high per-char entropy and (with 2-3
        # blocks cycling) moderate n-gram diversity. Trivial/short lines are excluded
        # so normal code (indentation, braces) doesn't trip it.
        lines = [l.strip() for l in window.split("\n") if len(l.strip()) >= self.cfg.allowed_min_len]
        if len(lines) >= self.cfg.block_min_lines:
            ratio = len(set(lines)) / len(lines)
            if ratio <= self.cfg.block_repeat_ratio_threshold:
                return self._trip(
                    f"block repetition: {len(set(lines))} distinct of {len(lines)} lines (ratio {ratio:.2f})"
                )
        if self.cfg.use_zlib_ratio:
            ratio = len(zlib.compress(window.encode("utf-8", "ignore"), 1)) / max(1, len(window))
            if ratio <= self.cfg.zlib_ratio_threshold:
                return self._trip(f"zlib ratio {ratio:.3f} <= {self.cfg.zlib_ratio_threshold}")
            return False
        n = self.cfg.ngram_size
        if len(window) <= n:
            return False
        total = len(window) - n
        distinct = len({window[i : i + n] for i in range(total)})
        ratio = distinct / total
        if ratio > self.cfg.distinct_ngram_ratio_threshold:
            return False
        entropy = _shannon_entropy(window)
        if entropy <= self.cfg.entropy_threshold:
            return self._trip(
                f"low redundancy: ngram_ratio {ratio:.3f}, entropy {entropy:.2f} bits/char"
            )
        return False

    def feed(self, text: str) -> bool:
        if self._tripped:
            return True
        if not text:
            return False
        for ch in text:
            self._buf.append(ch)
            self._bytes_since_heavy += 1
            if ch == "\n":
                if self._on_line(self._partial):
                    return True
                self._partial = ""
            else:
                self._partial += ch
        # tail (no-newline) check — cheap, bounded slice
        if len(self._partial) >= self.cfg.tail_check_min_len and self._tail_check():
            return True
        # heavy check, throttled to ~once per check_every_bytes
        if self._bytes_since_heavy >= self.cfg.check_every_bytes:
            self._bytes_since_heavy = 0
            if self._heavy_check():
                return True
        return False


def build_stream_loop_detector(agent: Any = None) -> Optional[StreamLoopDetector]:
    """Factory: returns a detector, or ``None`` when disabled (single ``if`` guard
    at the call site -> zero hot-path cost when off).

    Uses a cached ``agent._loop_detection_cfg`` if present (set once at agent init)
    so the streaming hot path never re-reads YAML.
    """
    cfg = getattr(agent, "_loop_detection_cfg", None) if agent is not None else None
    if not isinstance(cfg, LoopDetectionConfig):
        cfg = load_loop_detection_config()
    if not cfg.enabled:
        return None
    return StreamLoopDetector(cfg)


# ── Reasoning-trace loop detection ────────────────────────────────────────────
# The content detector is deliberately never fed reasoning/thinking tokens (they
# are legitimately repetitive). But quantized local reasoners (Qwen3, Gemma) can
# loop for MINUTES inside the <think> trace — which the content detector can't see
# and the thinking-timeout doesn't reliably catch. A SECOND detector, fed reasoning
# deltas with looser (loop-tolerant) thresholds, catches egregious reasoning loops
# while leaving normal reasoning alone. Same algorithm — validated to trip on real
# Qwen reasoning loops (block repetition) and pass on varied reasoning.
REASONING_DEFAULTS = LoopDetectionConfig(
    enabled=True,
    window_chars=8000,                  # reasoning traces run longer
    consecutive_line_threshold=10,      # reasoning revisits ideas; higher bar
    block_min_lines=12,
    block_repeat_ratio_threshold=0.35,  # lower => more repetition required to trip
    check_every_bytes=1024,
)


def _reasoning_env_override_enabled(default: bool) -> bool:
    v = os.environ.get("HERMES_REASONING_LOOP_DETECTION_ENABLED")
    if v is None:
        return default
    return v.strip().lower() not in _FALSEY


def load_reasoning_loop_detection_config(config: Optional[dict] = None) -> LoopDetectionConfig:
    """Config for the reasoning-trace detector: ``loop_detection.reasoning`` applied
    over reasoning-tuned defaults (:data:`REASONING_DEFAULTS`). Env
    ``HERMES_REASONING_LOOP_DETECTION_ENABLED`` wins over the config flag."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    section: dict = {}
    if isinstance(config, dict):
        outer = config.get("loop_detection")
        if isinstance(outer, dict) and isinstance(outer.get("reasoning"), dict):
            section = outer["reasoning"]

    d = REASONING_DEFAULTS

    def _num(key: str, default, cast):
        try:
            return cast(section.get(key, default))
        except Exception:
            return default

    def _flag(key: str, default: bool) -> bool:
        val = section.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() not in _FALSEY
        return bool(val)

    return LoopDetectionConfig(
        enabled=_reasoning_env_override_enabled(_flag("enabled", d.enabled)),
        window_chars=max(256, _num("window_chars", d.window_chars, int)),
        consecutive_line_threshold=max(
            2, _num("consecutive_line_threshold", d.consecutive_line_threshold, int)
        ),
        ngram_size=d.ngram_size,
        distinct_ngram_ratio_threshold=_num(
            "distinct_ngram_ratio_threshold", d.distinct_ngram_ratio_threshold, float
        ),
        entropy_threshold=_num("entropy_threshold", d.entropy_threshold, float),
        block_min_lines=max(4, _num("block_min_lines", d.block_min_lines, int)),
        block_repeat_ratio_threshold=_num(
            "block_repeat_ratio_threshold", d.block_repeat_ratio_threshold, float
        ),
        allowed_min_len=d.allowed_min_len,
        check_every_bytes=max(256, _num("check_every_bytes", d.check_every_bytes, int)),
        max_retries=d.max_retries,
    )


def build_reasoning_loop_detector(agent: Any = None) -> Optional[StreamLoopDetector]:
    """Factory for the reasoning-trace detector; ``None`` when disabled (single
    ``if`` at the call site -> zero hot-path cost when off). Uses a cached
    ``agent._reasoning_loop_detection_cfg`` if present."""
    cfg = getattr(agent, "_reasoning_loop_detection_cfg", None) if agent is not None else None
    if not isinstance(cfg, LoopDetectionConfig):
        cfg = load_reasoning_loop_detection_config()
    if not cfg.enabled:
        return None
    return StreamLoopDetector(cfg)


__all__ = [
    "StreamLoopDetector",
    "StreamLoopDetected",
    "LoopDetectionConfig",
    "load_loop_detection_config",
    "build_stream_loop_detector",
    "REASONING_DEFAULTS",
    "load_reasoning_loop_detection_config",
    "build_reasoning_loop_detector",
]
