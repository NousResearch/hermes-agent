"""Streamed tool-call delta assembly for the chat-completions stream (and Relay's accumulator)."""

from typing import Optional

from agent.reasoning_carriers import field


class _ToolCallAccumulator:
    """Assemble streamed tool-call deltas into complete ``tool_calls`` entries
    (``acc``: slot index -> entry dict). Ollama-compatible endpoints reuse index 0
    for every call in a parallel batch, distinguishing them only by id, so a new
    id at an already-seen raw index is redirected to a fresh slot."""

    def __init__(self):
        self.acc: dict = {}
        self._notified: set = set()
        self._last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
        self._active_slot_by_idx: dict = {}  # raw_index -> current slot in acc
        # Argument deltas are collected per slot and joined once in ``materialize`` —
        # ``+=`` per chunk rebuilds the whole string every delta (quadratic on big args).
        self._argument_parts: dict[int, list[str]] = {}

    def materialize(self) -> dict:
        """Join buffered argument deltas into each entry's ``arguments``; idempotent. Returns ``acc``."""
        for idx, parts in self._argument_parts.items():
            self.acc[idx]["function"]["arguments"] = "".join(parts)
        return self.acc

    def feed(self, tc_delta) -> Optional[str]:
        """Merge one delta; return the tool name the first time it is complete."""
        raw_idx = getattr(tc_delta, "index", None)
        if raw_idx is None:
            raw_idx = 0
        tc_id = getattr(tc_delta, "id", None)
        delta_id = tc_id or ""
        if isinstance(tc_id, int):  # Poolside sends integer ids
            tc_id = str(tc_id)

        self._active_slot_by_idx.setdefault(raw_idx, raw_idx)
        if delta_id and raw_idx in self._last_id_at_idx and delta_id != self._last_id_at_idx[raw_idx]:
            self._active_slot_by_idx[raw_idx] = max(self.acc, default=-1) + 1
        if delta_id:
            self._last_id_at_idx[raw_idx] = delta_id
        idx = self._active_slot_by_idx[raw_idx]

        entry = self.acc.setdefault(
            idx, {"id": tc_id or "", "type": "function", "function": {"name": "", "arguments": ""}, "extra_content": None},
        )
        parts = self._argument_parts.setdefault(idx, [])
        if tc_id:
            entry["id"] = tc_id
        tc_function = getattr(tc_delta, "function", None)
        if tc_function:
            if getattr(tc_function, "name", None):
                # Assignment, not +=: names arrive complete and some providers (MiniMax via
                # NVIDIA NIM) resend the full name every chunk — += gives "read_fileread_file".
                entry["function"]["name"] = tc_function.name
            if getattr(tc_function, "arguments", None):
                parts.append(tc_function.arguments)
            # Copilot's Gemini 3 variant signs the call inside ``function``.
            if isinstance(fn_sig := field(tc_function, "thought_signature"), str) and fn_sig:
                entry["thought_signature"] = fn_sig
        extra = getattr(tc_delta, "extra_content", None)
        if extra is None and hasattr(tc_delta, "model_extra"):
            extra = (tc_delta.model_extra if isinstance(tc_delta.model_extra, dict) else {}).get("extra_content")
        if extra is not None:
            from agent.chat_completion_helpers import _dump_if_model
            entry["extra_content"] = _dump_if_model(extra)
        name = entry["function"]["name"]
        if name and idx not in self._notified:
            self._notified.add(idx)
            return name
        return None
