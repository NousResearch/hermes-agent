"""Owned compaction carrier boundaries, preserving authentic multimodal content."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.context_compressor_continuation import MICRO_SUMMARY_PREFIX


class HandoffContentMixin:
    @staticmethod
    def _summary_text_block(item: Any) -> Optional[str]:
        """Return the textual payload of a content block, if it has one."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                return text
        return None

    @staticmethod
    def _replace_summary_text_block(item: Any, text: str) -> Any:
        """Return *item* with only its text payload replaced."""
        if isinstance(item, str):
            return text
        copied = item.copy()
        copied["text"] = text
        return copied

    @staticmethod
    def _drop_one_leading_line_break(text: str) -> str:
        if text.startswith("\r\n"):
            return text[2:]
        if text.startswith(("\r", "\n")):
            return text[1:]
        return text

    @staticmethod
    def _drop_two_trailing_line_breaks(text: str) -> str:
        """Remove only the two line breaks owned by merged transport."""
        for _ in range(2):
            if text.endswith("\r\n"):
                text = text[:-2]
            elif text.endswith(("\r", "\n")):
                text = text[:-1]
            else:
                break
        return text

    @classmethod
    def _qualified_summary_delimiters(cls, text: str) -> list[int]:
        """Locate delimiters whose suffix actually begins a known summary."""
        from agent.context_compressor import _MERGED_SUMMARY_DELIMITER

        positions: list[int] = []
        cursor = 0
        while True:
            position = text.find(_MERGED_SUMMARY_DELIMITER, cursor)
            if position < 0:
                break
            suffix = text[position + len(_MERGED_SUMMARY_DELIMITER):].lstrip()
            if cls._starts_with_summary_prefix(suffix):
                positions.append(position)
            cursor = position + len(_MERGED_SUMMARY_DELIMITER)
        return positions

    @classmethod
    def _analyze_summary_content(
        cls,
        content: Any,
        *,
        allow_headerless_string: bool = True,
    ) -> Optional[tuple[str, Any, str]]:
        """Classify and split a handoff without flattening authentic content.

        The tuple is ``(kind, surviving_prior_or_live_content, summary_text)``.
        Standalone recognition intentionally wins over delimiter recognition:
        force-user-leading carriers start with a summary and keep their live
        request after the end marker.
        """
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
            _INFLIGHT_TASK_REPLAY_HEADER,
        )

        if isinstance(content, str):
            leading = content.lstrip()
            if cls._starts_with_summary_prefix(leading):
                live_content: Any = None
                marker = leading.find(_SUMMARY_END_MARKER)
                if marker >= 0:
                    remainder = leading[marker + len(_SUMMARY_END_MARKER):].lstrip()
                    if remainder:
                        live_content = remainder
                return "standalone", live_content, leading

            header_led = leading.startswith(_MERGED_PRIOR_CONTEXT_HEADER)
            positions = cls._qualified_summary_delimiters(leading)
            if header_led and positions:
                boundary = positions[-1]
            elif allow_headerless_string and len(positions) == 1:
                boundary = positions[0]
            else:
                return None

            prior = leading[:boundary]
            if header_led:
                prior = prior[len(_MERGED_PRIOR_CONTEXT_HEADER):]
                prior = cls._drop_one_leading_line_break(prior)
            prior = cls._drop_two_trailing_line_breaks(prior)
            summary_text = leading[
                boundary + len(_MERGED_SUMMARY_DELIMITER):
            ].lstrip()
            marker = summary_text.find(_SUMMARY_END_MARKER)
            if marker >= 0:
                replay = summary_text[marker + len(_SUMMARY_END_MARKER):].lstrip()
                if replay:
                    prior = f"{prior}\n\n{replay}" if prior.strip() else replay
            return "merged", prior if prior.strip() else None, summary_text

        if not isinstance(content, list):
            return None

        # Current compaction can append a user-task replay as a new block.
        # Preserve it alongside the genuine content before the summary.
        for index in range(len(content) - 1, -1, -1):
            tail_text = cls._summary_text_block(content[index])
            if tail_text is not None and not tail_text.strip():
                continue
            if tail_text and tail_text.lstrip().startswith(_INFLIGHT_TASK_REPLAY_HEADER):
                original = cls._analyze_summary_content(
                    content[:index], allow_headerless_string=allow_headerless_string,
                )
                if original and original[0] == "merged" and original[2].rstrip().endswith(_SUMMARY_END_MARKER):
                    prior = original[1] or []
                    return "merged", [*prior, *content[index:]], original[2]
            break

        # Standalone list carriers (including force-user-leading) keep the
        # summary in the first non-empty textual block. Later blocks are live
        # content and must remain structurally intact.
        standalone_index: Optional[int] = None
        standalone_text = ""
        for index, item in enumerate(content):
            text = cls._summary_text_block(item)
            if text is None or not text.strip():
                continue
            standalone_index = index
            standalone_text = text.lstrip()
            break
        if (
            standalone_index is not None
            and cls._starts_with_summary_prefix(standalone_text)
        ):
            live_blocks: list[Any] = []
            # Non-text blocks before the first textual summary block are real
            # payload. Whitespace-only text wrappers are transport noise.
            for item in content[:standalone_index]:
                text = cls._summary_text_block(item)
                if text is None:
                    live_blocks.append(item)
                elif text.strip():
                    live_blocks.append(item)

            marker = standalone_text.find(_SUMMARY_END_MARKER)
            if marker >= 0:
                remainder = standalone_text[
                    marker + len(_SUMMARY_END_MARKER):
                ].lstrip()
                if remainder:
                    live_blocks.append(
                        cls._replace_summary_text_block(
                            content[standalone_index], remainder
                        )
                    )
                live_blocks.extend(content[standalone_index + 1:])
            return (
                "standalone",
                live_blocks if live_blocks else None,
                standalone_text,
            )

        # Delimiter-merged lists are only owned when their first textual
        # wrapper is header-led and their suffix lives in the final non-blank
        # text block. A non-text or non-empty block after that suffix makes the
        # boundary ambiguous and therefore unowned.
        header_index: Optional[int] = None
        header_text = ""
        for index, item in enumerate(content):
            text = cls._summary_text_block(item)
            if text is not None:
                header_index = index
                header_text = text.lstrip()
                break
        if (
            header_index is None
            or not header_text.startswith(_MERGED_PRIOR_CONTEXT_HEADER)
        ):
            return None

        suffix_index: Optional[int] = None
        suffix_text = ""
        for index in range(len(content) - 1, -1, -1):
            text = cls._summary_text_block(content[index])
            if text is None:
                return None
            if not text.strip():
                continue
            suffix_index = index
            suffix_text = text
            break
        if suffix_index is None or suffix_index < header_index:
            return None

        positions = cls._qualified_summary_delimiters(suffix_text)
        if not positions:
            return None
        boundary = positions[-1]
        summary_text = suffix_text[
            boundary + len(_MERGED_SUMMARY_DELIMITER):
        ].lstrip()

        prior_blocks: list[Any] = []
        for index, item in enumerate(content[:suffix_index]):
            if index != header_index:
                prior_blocks.append(item)
                continue
            text = cls._summary_text_block(item)
            assert text is not None
            leading = text.lstrip()[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            leading = cls._drop_one_leading_line_break(leading)
            if leading.strip():
                prior_blocks.append(cls._replace_summary_text_block(item, leading))

        suffix_prior = cls._drop_two_trailing_line_breaks(
            suffix_text[:boundary]
        )
        if suffix_index == header_index:
            suffix_prior = suffix_prior.lstrip()[
                len(_MERGED_PRIOR_CONTEXT_HEADER):
            ]
            suffix_prior = cls._drop_one_leading_line_break(suffix_prior)
        if suffix_prior.strip():
            prior_blocks.append(
                cls._replace_summary_text_block(
                    content[suffix_index], suffix_prior
                )
            )
        marker = summary_text.find(_SUMMARY_END_MARKER)
        if marker >= 0:
            replay = summary_text[marker + len(_SUMMARY_END_MARKER):].lstrip()
            if replay:
                prior_blocks.append(cls._replace_summary_text_block(content[suffix_index], replay))
        return "merged", prior_blocks if prior_blocks else None, summary_text

    @classmethod
    def _split_header_led_merged_handoff(
        cls,
        content: Any,
    ) -> Optional[tuple[Any, str]]:
        """Return the owned header-led boundary, excluding compatibility forms."""
        analyzed = cls._analyze_summary_content(
            content,
            allow_headerless_string=False,
        )
        if analyzed is None or analyzed[0] != "merged":
            return None
        return analyzed[1], analyzed[2]

    @classmethod
    def _strip_summary_prefix(
        cls,
        summary: Any,
        *,
        allow_merged_carrier: bool = True,
    ) -> str:
        """Return summary body without the current, legacy, or any historical
        handoff prefix.

        Historical prefixes must be stripped too: a handoff persisted under an
        older prefix can be inherited into a resumed lineage (#35344), and if we
        only re-prepend the current prefix without removing the old one, the
        stale directive it carried stays embedded in the body.
        """
        from agent.context_compressor import (
            SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, _HISTORICAL_SUMMARY_PREFIXES,
            _SUMMARY_END_MARKER, _content_text_for_contains,
        )

        analyzed = cls._analyze_summary_content(summary)
        if analyzed is not None and (
            allow_merged_carrier or analyzed[0] == "standalone"
        ):
            # Both standalone and merged carriers have already isolated the
            # summary-bearing region. In particular, authentic prior bytes are
            # never scanned globally for a delimiter or prefix.
            text = analyzed[2].strip()
        else:
            # Generated summaries and deterministic fallbacks are plain bodies
            # before persistence. Keep the historical best-effort normalizer
            # for that in-memory path without treating a mid-payload delimiter
            # as an owned transport boundary.
            text = _content_text_for_contains(summary).strip()
        for prefix in (
            SUMMARY_PREFIX,
            MICRO_SUMMARY_PREFIX,
            LEGACY_SUMMARY_PREFIX,
            *_HISTORICAL_SUMMARY_PREFIXES,
        ):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        # Strip the end marker too — a rehydrated handoff body that keeps it
        # would leak the boundary directive into the iterative-update
        # summarizer prompt (and the marker is re-appended on insertion anyway).
        # Forced user-leading merged summaries keep the live tail request after
        # this marker, so truncate at the marker even when it is not the final
        # content.
        marker_idx = (
            text.find(_SUMMARY_END_MARKER)
            if allow_merged_carrier
            else (
                len(text.rstrip()) - len(_SUMMARY_END_MARKER)
                if text.rstrip().endswith(_SUMMARY_END_MARKER)
                else -1
            )
        )
        if marker_idx >= 0:
            text = text[:marker_idx].rstrip()
        return text

    @classmethod
    def _with_summary_prefix(cls, summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        from agent.context_compressor import SUMMARY_PREFIX

        text = cls._strip_summary_prefix(
            summary,
            allow_merged_carrier=False,
        )
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _starts_with_summary_prefix(text: str) -> bool:
        """Return True if *text* begins with any known handoff prefix."""
        from agent.context_compressor import SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, _HISTORICAL_SUMMARY_PREFIXES

        if (
            text.startswith(SUMMARY_PREFIX)
            or text.startswith(MICRO_SUMMARY_PREFIX)
            or text.startswith(LEGACY_SUMMARY_PREFIX)
        ):
            return True
        return any(text.startswith(p) for p in _HISTORICAL_SUMMARY_PREFIXES)

    @classmethod
    def classify_summary_content(cls, content: Any) -> Optional[str]:
        """Classify how *content* relates to a compaction summary.

        Returns:
            ``"standalone"``: the entire message IS a compaction handoff
            (current, legacy, or historical prefix at the start). Frontends
            may restyle/collapse the whole message as a summary.

            ``"merged"``: a merge-into-tail message — real preserved turn
            content wrapped under ``_MERGED_PRIOR_CONTEXT_HEADER``, followed by
            ``_MERGED_SUMMARY_DELIMITER`` and the summary body. The message
            *contains* a summary but is not only a summary; collapsing the
            whole message would hide the preserved content.

            ``None``: no compaction summary detected.
        """
        analyzed = cls._analyze_summary_content(content)
        return analyzed[0] if analyzed is not None else None

    @classmethod
    def _strip_context_summary_handoff_message(
        cls,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Drop stale handoff data while preserving merged prior-tail content."""
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, COMPRESSED_SUMMARY_HAS_USER_TURN_KEY

        if not isinstance(message, dict):
            return message

        content = message.get("content")
        kind = cls.classify_summary_content(content)
        is_summary = kind is not None or cls._has_compressed_summary_metadata(message)
        if not is_summary:
            return message.copy()

        analyzed = cls._analyze_summary_content(content)
        if analyzed is not None:
            _kind, surviving_content, _summary_text = analyzed
            if surviving_content is not None:
                unwrapped = message.copy()
                unwrapped["content"] = surviving_content
                unwrapped.pop(COMPRESSED_SUMMARY_METADATA_KEY, None)
                unwrapped.pop(COMPRESSED_SUMMARY_HAS_USER_TURN_KEY, None)
                return unwrapped

        return None

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @classmethod
    def _find_context_summaries(
        cls,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> list[tuple[int, str]]:
        """Find handoff summaries inside a compression window."""
        from agent.context_compressor import _content_text_for_contains

        n = len(messages)
        # Defensive: clamp bounds so a caller passing an out-of-range end
        # (e.g. tail-cut returning len(messages)+1 when head_end >= n)
        # cannot trigger IndexError.  (#75588)
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        summaries: list[tuple[int, str]] = []
        for idx in range(start, end):
            message = messages[idx]
            content = message.get("content")
            kind = cls.classify_summary_content(content)
            has_metadata = cls._has_compressed_summary_metadata(message)
            if kind is not None:
                body = cls._strip_summary_prefix(content)
            elif has_metadata:
                # Compatibility for historical metadata-only rows. The flag
                # identifies the row as a summary, but it does not authorize a
                # first delimiter/prefix/end-marker cut through unknown payload
                # bytes. Preserve its complete textual view for rehydration.
                body = _content_text_for_contains(content).strip()
            else:
                continue
            summaries.append((idx, body))
        return summaries

    @classmethod
    def _handoff_only_content(cls, content: Any) -> Any:
        """Use the same owned boundary for history and live-user projections."""
        from agent.context_compressor import _SUMMARY_END_MARKER

        analyzed = cls._analyze_summary_content(content)
        if analyzed is None:
            # Historical metadata-only carriers have no owned transport cut.
            if isinstance(content, list):
                return [
                    item.copy() if isinstance(item, dict) else item
                    for item in content if cls._summary_text_block(item) is not None
                ]
            return content
        summary_text = analyzed[2]
        marker = summary_text.find(_SUMMARY_END_MARKER)
        handoff = summary_text[:marker + len(_SUMMARY_END_MARKER)] if marker >= 0 else summary_text
        if isinstance(content, str):
            return handoff
        # The analyzer's summary region belongs to one text block. Retain that
        # block's shape and metadata; images and live blocks are never historical.
        for item in reversed(content):
            text = cls._summary_text_block(item)
            if text is not None and summary_text in text:
                return [cls._replace_summary_text_block(item, handoff)]
        return []
