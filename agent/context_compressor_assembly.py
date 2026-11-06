"""Head/summary/tail assembly for ordinary and pressure compaction windows."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional

from agent.context_compressor_tail import with_skill_safety_rule
from agent.turn_context import drop_stale_api_content

if TYPE_CHECKING:
    from agent.context_compressor import _HandoffScan


class CompactionAssemblyMixin:
    """Assemble a validated summary while respecting the selected window's protection."""

    def _assemble_head(self, messages: list[dict[str, Any]], compress_start: int) -> list[dict[str, Any]]:
        """Protected head with the compaction note on the system prompt and stale handoffs stripped."""
        from agent.context_compressor import _fresh_compaction_message_copy, _content_text_for_contains, _append_text_to_content

        compressed = []
        for i in range(compress_start):
            # Head handoff already lives in _previous_summary: strip it (standalone dropped, merged
            # keeps prior-tail text). Merged rows hold real user text — never blanket-skip.
            msg = _fresh_compaction_message_copy(messages[i])
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                if self._COMPRESSION_NOTE not in _content_text_for_contains(existing):
                    sep = "\n\n" if isinstance(existing, str) and existing else ""
                    msg["content"] = _append_text_to_content(existing, sep + self._COMPRESSION_NOTE)
            stripped = self._strip_context_summary_handoff_message(msg)
            if stripped is not None:
                compressed.append(stripped)
        return compressed

    def _assemble_tail(
        self, messages: list[dict[str, Any]], compress_end: int, tail_start: int, summary_indices: set,
        *, preserve_from: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Protected tail with already-folded handoff rows dropped and merged handoffs unwrapped."""
        from agent.context_compressor import _fresh_compaction_message_copy

        tail_messages: list[dict[str, Any]] = []
        # Start at tail_start, not compress_end: the rehydration scan may have advanced it (#57835).
        for i in range(max(compress_end, tail_start), len(messages)):
            if preserve_from is not None and i >= preserve_from:
                tail_messages.append(copy.deepcopy(messages[i]))
                continue
            if i in summary_indices and i >= tail_start:
                continue  # already folded into _previous_summary; don't re-emit
            stripped = self._strip_context_summary_handoff_message(_fresh_compaction_message_copy(messages[i]))
            if stripped is not None:
                tail_messages.append(stripped)
        return tail_messages

    @staticmethod
    def _summary_placement(
        compressed: list[dict[str, Any]], tail_messages: list[dict[str, Any]], compress_start: int,
    ) -> tuple[str, bool, bool, Optional[int]]:
        """Pick the summary row's role so template-visible alternation holds.
        Returns ``(summary_role, merge_into_tail, force_user_leading, first_tail_visible_idx)``. Roles
        read the assembled (post-strip) head/tail and are TEMPLATE-VISIBLE: Mistral-strict templates
        skip tool rows for alternation, so alternate against what the template counts."""
        from agent.context_compressor import _last_template_visible_role, _template_visible_role, _content_text_for_contains

        last_head_role: Optional[str] = "user"
        if compressed:
            # None = all-exempt head: the summary opens the visible sequence and must be "user".
            last_head_role = _last_template_visible_role(compressed)
        first_tail_visible_idx, first_tail_role = next(
            ((idx, role) for idx, role in enumerate(map(_template_visible_role, tail_messages)) if role is not None),
            (None, None),
        )
        # System-only head: the summary is the first visible message and Anthropic requires role=user
        # (#52160). Zero-user-turn guard (#58753): if no user row with non-empty TEXT survives, the
        # summary must be role="user" or OpenAI-compatible backends reject. Image-only rows don't count.
        force_user_leading = compress_start == 0 or last_head_role == "system" or not any(
            m.get("role") == "user" and bool(_content_text_for_contains(m.get("content")).strip())
            for m in (*compressed, *tail_messages)
        )
        # Alternate against head first, then tail; None (all-exempt head) means "user".
        summary_role = "user" if last_head_role in {None, "assistant", "tool"} or force_user_leading else "assistant"
        merge_into_tail = False
        # Flip on a tail collision only if that doesn't collide with the head. All-exempt head pins "user";
        # flipping would open the visible sequence with "assistant". Neither alternates: merge into the first tail row.
        if first_tail_role is not None and summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role and last_head_role is not None and not force_user_leading:
                summary_role = flipped
            else:
                merge_into_tail = bool(tail_messages)
        return summary_role, merge_into_tail, force_user_leading, first_tail_visible_idx

    def _merge_summary_into_tail_row(
        self, msg: dict[str, Any], summary: str, summary_role: str, force_user_leading: bool,
    ) -> None:
        """Fold the summary into a carried tail row (in place) when no standalone role alternates."""
        from agent.context_compressor import _append_text_to_content, _SUMMARY_END_MARKER, _MERGED_SUMMARY_DELIMITER, _MERGED_PRIOR_CONTEXT_HEADER, COMPRESSED_SUMMARY_METADATA_KEY, COMPRESSED_SUMMARY_HAS_USER_TURN_KEY

        old_content = msg.get("content", "")
        if force_user_leading and summary_role == "user":
            # Anthropic/Bedrock: summary must lead the first visible message; the real request
            # follows the end marker.
            msg["content"] = _append_text_to_content(old_content, summary + "\n\n" + _SUMMARY_END_MARKER + "\n\n", prepend=True)
        else:
            # Old tail content is kept as delimited reference BEFORE the summary; the end marker goes last.
            suffix = "\n\n" + _MERGED_SUMMARY_DELIMITER + "\n\n" + summary + "\n\n" + _SUMMARY_END_MARKER
            msg["content"] = _append_text_to_content(
                _append_text_to_content(old_content, suffix, prepend=False),
                _MERGED_PRIOR_CONTEXT_HEADER + "\n", prepend=True,
            )
        # Frontends use this to detect a summary-prefixed message.
        msg[COMPRESSED_SUMMARY_METADATA_KEY], msg[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] = True, bool(self._summary_has_user_turn)
        # Rewritten content: drop the stale api_content sidecar so replay can't resend pre-merge bytes.
        drop_stale_api_content(msg)

    def _assemble_compressed(
        self, messages: list[dict[str, Any]], compress_start: int, compress_end: int, scan: _HandoffScan, summary: str,
        *, protect_active: bool = False,
        active_start: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Head + summary + tail from the pruned copy: its tool-result demotions are what let an oversized
        head/tail compress at all (#61932); tool-call arguments are never rewritten by pruning."""
        from agent.context_compressor import _SUMMARY_END_MARKER, COMPRESSED_SUMMARY_METADATA_KEY, COMPRESSED_SUMMARY_HAS_USER_TURN_KEY, MODEL_ONLY_DISPLAY_METADATA_KEY, _COMPACTION_TAIL_MARKER

        compressed = copy.deepcopy(messages[:compress_start]) if protect_active else self._assemble_head(messages, compress_start)
        tail_messages = self._assemble_tail(
            messages, compress_end, scan.tail_start, scan.summary_indices, preserve_from=active_start,
        )
        summary, compressed, tail_messages = with_skill_safety_rule(
            summary, compressed, tail_messages, preserve_tail=protect_active,
        )
        summary_role, merge_into_tail, force_user_leading, first_tail_visible_idx = (
            self._summary_placement(compressed, tail_messages, compress_start)
        )
        bridge = protect_active and merge_into_tail
        if bridge:
            # A summary/ack handoff preserves alternation without rewriting the live user request.
            merge_into_tail = False
        if not merge_into_tail:
            # End marker stops weak models treating the quoted summary as fresh input (#11475) or
            # regurgitating it (#33256).
            compressed.append({
                "role": summary_role, "content": summary + "\n\n" + _SUMMARY_END_MARKER,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: bool(self._summary_has_user_turn),
            })
            if bridge:
                compressed.append({
                    "role": "assistant", "content": "I will continue using this historical context.",
                    "display_metadata": {MODEL_ONLY_DISPLAY_METADATA_KEY: True},
                })
        # Default carrier is tail[0]: an exempt row absorbs the summary invisibly. The forced repair
        # path needs a non-empty role=user row, so it targets the template-visible row.
        merge_target_idx = first_tail_visible_idx if force_user_leading and first_tail_visible_idx is not None else 0
        for tail_idx, msg in enumerate(tail_messages):
            # Tag carried-forward tail rows so archive_and_compact treats their originals as
            # superseded duplicates (#86366).
            if isinstance(msg, dict):
                msg[_COMPACTION_TAIL_MARKER] = True
            if merge_into_tail and tail_idx == merge_target_idx:
                self._merge_summary_into_tail_row(msg, summary, summary_role, force_user_leading)
            compressed.append(msg)
        return compressed

