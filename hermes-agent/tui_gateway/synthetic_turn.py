"""Synthetic GIL-heavy turn driver for the AC-4 isolation certify harness. The regime under test is
interpreter-wide GIL starvation (serving-process turn threads park the WebSocket loop in ``take_gil``),
so the driver must hold the GIL with sustained pure-Python CPU — a network/sleep stub releases the GIL
and a green off it is fake. Test seam: dead unless ``HERMES_ISO_CERTIFY_SYNTH_TURN=1``; when armed,
``server._make_agent`` returns a :class:`SyntheticHeavyAgent` on both the in-process and compute-host
paths, so the isolation boundary is the only variable. Per-turn intensity rides in the prompt text as
a JSON object; any other prompt falls back to env / built-in defaults."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Any, Callable, Optional

from tui_gateway._env import env_float as _env_float, env_int as _env_int


# Per-turn intensity spec: (key, caster, default source). ``chunk`` = pure-Python ops per interrupt
# check (ms-level interrupt latency, still hot on the GIL); ``delta_interval_s`` = streamed-delta
# cadence (each delta is a loop wakeup marshalling a frame); ``tokens_per_delta`` drives the 100K+-token
# heavy-turn proxy; ``sleep_s`` = optional per-chunk sleep for a mixed regime (0 = pure burn; --dry-run
# shortens duration instead, so it still exercises the real seam).
_SPEC_FIELDS = (
    ("duration_s", float, lambda: _env_float("HERMES_ISO_CERTIFY_DURATION_S", 8.0)),
    ("chunk", int, lambda: _env_int("HERMES_ISO_CERTIFY_CHUNK", 20_000)),
    ("delta_interval_s", float, lambda: _env_float("HERMES_ISO_CERTIFY_DELTA_S", 0.05)),
    ("tokens_per_delta", int, lambda: _env_int("HERMES_ISO_CERTIFY_TPD", 512)),
    ("sleep_s", float, lambda: 0.0),
)


def synth_turn_armed() -> bool:
    """True when the synthetic-turn test seam is armed via env."""
    return os.environ.get("HERMES_ISO_CERTIFY_SYNTH_TURN") == "1"


class SyntheticHeavyAgent:
    """An AIAgent-shaped object whose turn is a GIL-holding CPU burn. Presents only the surface
    ``tui_gateway.server``'s turn path and status helpers read (``run_conversation``/``interrupt``/
    ``clear_interrupt`` plus the ``model``/``provider``/``session_*`` attributes consumed by
    ``_get_usage`` and ``_session_info``). Never opens a socket or spawns a subprocess."""

    def __init__(self, session_id: str, *, model: str = "synthetic-heavy") -> None:
        self.session_id, self.model, self.provider, self.api_mode = session_id, model, "synthetic", "chat_completions"
        self.base_url = self.api_key = self.platform = self._cached_system_prompt = ""
        self.tools: list[Any] = []
        self.reasoning_config = self.service_tier = self.context_compressor = None
        self._config_context_length = 200_000
        # Cumulative session counters (read by _get_usage → status bar).
        self.session_input_tokens = self.session_output_tokens = self.session_prompt_tokens = 0
        self.session_completion_tokens = self.session_reasoning_tokens = self.session_total_tokens = 0
        self.session_api_calls = 0
        self.history: list[dict[str, str]] = []
        self._interrupt = threading.Event()

    # interrupt contract (mirrors AIAgent); close() is a no-op teardown that also stops the burn
    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def interrupt(self) -> None:
        self._interrupt.set()

    close = interrupt

    def _has_stream_consumers(self) -> bool:  # defensive; not used by our loop
        return True

    @staticmethod
    def _parse_spec(message: Any) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if isinstance(message, str) and message.strip().startswith("{"):
            with contextlib.suppress(ValueError, TypeError):
                parsed = json.loads(message.strip())
                spec = parsed if isinstance(parsed, dict) else {}
        return {key: cast(spec.get(key, default())) for key, cast, default in _SPEC_FIELDS}

    def run_conversation(
        self, message: Any, *, conversation_history: Optional[list[dict[str, str]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None, task_id: Optional[str] = None, **_kwargs: Any,
    ) -> dict[str, Any]:
        spec = self._parse_spec(message)
        duration, chunk = max(0.0, spec["duration_s"]), max(1, spec["chunk"])
        interval, tokens_per_delta, sleep_s = max(0.001, spec["delta_interval_s"]), max(0, spec["tokens_per_delta"]), max(0.0, spec["sleep_s"])
        base_history = list(conversation_history if conversation_history is not None else self.history)
        start = last_delta = time.monotonic()
        acc = deltas = 0
        while not (interrupted := self._interrupt.is_set()) and (now := time.monotonic()) - start < duration:
            # A tight integer loop never releases the GIL — the exact contention that starves the serving loop.
            for _ in range(chunk):
                acc = (acc * 1_000_003 + 12_345) & 0xFFFFFFFFFFFFFFFF
            if sleep_s:
                time.sleep(sleep_s)
            if now - last_delta >= interval:
                deltas += 1
                self.session_output_tokens += tokens_per_delta
                self.session_completion_tokens += tokens_per_delta
                self.session_total_tokens += tokens_per_delta
                if stream_callback is not None:
                    stream_callback(f"synthtok-{deltas:05d} ")
                last_delta = now
        self.session_api_calls += 1
        # Fold the checksum into the reply so the loop can't be eliminated and the turn is deterministic.
        final = (
            f"[synthetic heavy turn] deltas={deltas} out_tokens={self.session_output_tokens} "
            f"interrupted={interrupted} checksum={acc & 0xFFFF:04x}"
        )
        self.history = [*base_history, {"role": "user", "content": str(message)[:200]}, {"role": "assistant", "content": final}]
        return {"final_response": final, "messages": self.history, "interrupted": interrupted, "error": None, "last_reasoning": None}


def maybe_build_synthetic_agent(session_id: str, model_override: Any = None) -> SyntheticHeavyAgent | None:
    """Return a :class:`SyntheticHeavyAgent` when the seam is armed, else ``None``. ``model_override``
    (dict or str) only influences the reported ``model`` label; it never changes the compute."""
    if not synth_turn_armed():
        return None
    if isinstance(model_override, dict):
        model_override = str(model_override["model"]) if model_override.get("model") else ""
    return SyntheticHeavyAgent(session_id, model=model_override if isinstance(model_override, str) and model_override else "synthetic-heavy")


__all__ = ["SyntheticHeavyAgent", "maybe_build_synthetic_agent", "synth_turn_armed"]
