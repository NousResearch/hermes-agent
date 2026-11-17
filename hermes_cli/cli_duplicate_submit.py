"""PROOF OF CONCEPT: silently drop duplicate submissions sent while a turn is still unanswered.

Pressing Up and Enter (or repeating a paste) while the agent works routes every copy through
busy_input_mode on its own: each redirect/steer/queue entry starts or feeds another model call
for text the agent is already handling. This mixin drops such a copy before routing: no notice,
the composer clears as if sent, and the text still lands in prompt history once.

Scope, deliberately narrow:
- Only plain text, never images or local dispatch (``/command``, ``!shell``).
- "Unanswered" = the input that started the running turn plus every text already accepted while
  it runs. The set resets when the next turn starts, so resending after the answer is untouched.
- While idle, only an identical submit within ``_DUP_DOUBLE_ENTER_S`` of the previous one (a
  double Enter that races the turn start) is dropped.

Bound after ``CLIBacktrackMixin`` in ``HermesCLI``'s bases, so a pending backtrack edit is
handled first and this wraps the base ``_tui_submit_buffer`` via ``super()``.
"""

from __future__ import annotations

import time

_DUP_DOUBLE_ENTER_S = 2.0


def _dup_key(text: str) -> str:
    return text.strip()


class CLIDuplicateSubmitMixin:
    _dup_unanswered: set[str] | None = None
    _dup_last_idle: tuple[str, float] | None = None

    def _dup_should_drop(self, text: str) -> bool:
        from cli import _looks_like_slash_command
        key = _dup_key(text)
        if not key or self._attached_images or _looks_like_slash_command(key) or key.startswith("!"):
            return False
        if self._agent_running:
            seen = self._dup_unanswered if self._dup_unanswered is not None else set()
            if key in seen:
                return True
            seen.add(key)
            self._dup_unanswered = seen
            return False
        now = time.monotonic()
        last = self._dup_last_idle
        self._dup_last_idle = (key, now)
        return last is not None and last[0] == key and now - last[1] < _DUP_DOUBLE_ENTER_S

    def _tui_submit_buffer(self, event, buf):
        if self._dup_should_drop(buf.text):
            self._inline_pastes(buf)
            buf.reset(append_to_history=True)  # prompt_toolkit skips a repeat of the last entry
            event.app.invalidate()
            return None
        return super()._tui_submit_buffer(event, buf)

    def _tui_process_one_input(self, user_input):
        # A new turn: what was unanswered has been answered. Seed with the text this turn handles.
        text = user_input[0] if isinstance(user_input, tuple) else user_input
        self._dup_unanswered = {_dup_key(text)} if isinstance(text, str) and _dup_key(text) else set()
        return super()._tui_process_one_input(user_input)
