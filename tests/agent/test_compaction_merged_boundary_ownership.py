"""Behavioral regressions for compaction merged-boundary ownership.

Merged handoffs contain both authentic prior-tail content and model-only
summary scaffolding.  These tests pin which boundary Hermes owns without
depending on implementation text or a particular parser helper.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    MICRO_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    ContextCompressor,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
    _handoff_carries_live_user_content,
    is_compaction_summary_message,
)


def _summary(body: str = "REAL_SUMMARY_BODY", *, prefix: str = SUMMARY_PREFIX) -> str:
    return f"{prefix}\n{body}\n\n{_SUMMARY_END_MARKER}"


def _text(value: str) -> dict:
    return {"type": "text", "text": value}


def _image(label: str = "prior-image") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{label}"},
        "detail": "low",
    }


def _header_merged_string(prior: str, body: str = "REAL_SUMMARY_BODY") -> str:
    return (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{prior}\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n{_summary(body)}"
    )


def _legacy_merged_string(prior: str, body: str = "LEGACY_SUMMARY_BODY") -> str:
    return f"{prior}\n\n{_MERGED_SUMMARY_DELIMITER}\n\n{_summary(body)}"


def _header_merged_list(
    prior_blocks: list,
    body: str = "LIST_SUMMARY_BODY",
    *,
    trailing: list | None = None,
) -> list:
    return [
        _text(_MERGED_PRIOR_CONTEXT_HEADER + "\n"),
        *deepcopy(prior_blocks),
        _text(f"\n\n{_MERGED_SUMMARY_DELIMITER}\n\n{_summary(body)}"),
        *deepcopy(trailing or []),
    ]


def _message(content, *, metadata: bool = False, role: str = "user", **extra):
    message = {"role": role, "content": content, **extra}
    if metadata:
        message[COMPRESSED_SUMMARY_METADATA_KEY] = True
    return message


def test_standalone_prefix_owns_content_before_any_quoted_merged_delimiter():
    quoted_body = (
        "STANDALONE-BEFORE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n"
        f"{SUMMARY_PREFIX}\n"
        "STANDALONE-AFTER"
    )
    content = _summary(quoted_body)
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) == "standalone"
    assert ContextCompressor._strip_summary_prefix(content) == quoted_body
    assert is_compaction_summary_message(message) is True
    assert ContextCompressor._strip_context_summary_handoff_message(message) is None


def test_generated_body_does_not_promote_a_quoted_delimiter_to_transport():
    generated_body = (
        "GENERATED-BEFORE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n"
        f"{SUMMARY_PREFIX}\n"
        "GENERATED-AFTER"
    )

    prefixed = ContextCompressor._with_summary_prefix(generated_body)

    assert prefixed == f"{SUMMARY_PREFIX}\n{generated_body}"


def test_header_merged_string_selects_last_prefix_qualified_delimiter():
    prior = (
        "PRIOR-BEFORE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n"
        f"{SUMMARY_PREFIX}\n"
        "QUOTED-SUMMARY-PREFIX-IN-PRIOR\n"
        "PRIOR-AFTER"
    )
    content = _header_merged_string(prior, "OWNED-SUMMARY")
    message = _message(content)

    assert content.count(_MERGED_SUMMARY_DELIMITER) == 2
    assert ContextCompressor.classify_summary_content(content) == "merged"
    assert ContextCompressor._strip_summary_prefix(content) == "OWNED-SUMMARY"
    assert is_compaction_summary_message(message) is True

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    assert projected is not None
    assert projected["content"] == prior
    assert _MERGED_SUMMARY_DELIMITER in projected["content"]
    assert "PRIOR-AFTER" in projected["content"]


def test_headerless_string_accepts_exactly_one_prefix_qualified_candidate():
    prior = "LEGACY PRIOR CONTENT"
    content = _legacy_merged_string(prior)
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) == "merged"
    assert ContextCompressor._strip_summary_prefix(content) == "LEGACY_SUMMARY_BODY"
    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    assert projected is not None
    assert projected["content"] == prior


def test_headerless_string_rejects_multiple_prefix_qualified_candidates():
    ambiguous = (
        "LEGACY-PRIOR\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
        "FIRST-CANDIDATE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
        "SECOND-CANDIDATE\n"
        f"{_SUMMARY_END_MARKER}"
    )
    message = _message(ambiguous)

    assert ContextCompressor.classify_summary_content(ambiguous) is None
    assert is_compaction_summary_message(message) is False
    assert ContextCompressor._strip_context_summary_handoff_message(message) == message


def test_headerless_list_is_never_promoted_to_a_merged_handoff():
    content = [
        _text("HEADERLESS-LIST-PRIOR"),
        _text(f"{_MERGED_SUMMARY_DELIMITER}\n\n{_summary('LIST-CANDIDATE')}"),
    ]
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) is None
    assert is_compaction_summary_message(message) is False
    assert ContextCompressor._strip_context_summary_handoff_message(message) == message


@pytest.mark.parametrize(
    "trailing",
    [
        [_text(" \n\t")],
        ["\n  "],
        [_text(" \n"), "\t"],
    ],
)
def test_header_merged_list_preserves_prior_blocks_and_accepts_trailing_whitespace(
    trailing,
):
    prior_blocks = [
        _text(
            "PRIOR-BEFORE-LITERAL "
            f"{_MERGED_SUMMARY_DELIMITER} "
            "PRIOR-AFTER-LITERAL"
        ),
        _image(),
        _text("FINAL-PRIOR-REMAINDER"),
    ]
    content = _header_merged_list(prior_blocks, trailing=trailing)
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) == "merged"
    assert ContextCompressor._strip_summary_prefix(content) == "LIST_SUMMARY_BODY"
    assert is_compaction_summary_message(message) is True

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    assert projected is not None
    assert projected["content"] == prior_blocks
    assert projected["content"][1] == _image()
    assert "PRIOR-AFTER-LITERAL" in projected["content"][0]["text"]
    assert projected["content"][-1]["text"] == "FINAL-PRIOR-REMAINDER"


def test_header_merged_list_preserves_remainder_inside_boundary_block():
    content = [
        _text(_MERGED_PRIOR_CONTEXT_HEADER + "\n"),
        _image(),
        _text(
            "BOUNDARY-BLOCK-PRIOR\n\n"
            f"{_MERGED_SUMMARY_DELIMITER}\n\n"
            f"{_summary('BOUNDARY-BLOCK-SUMMARY')}"
        ),
    ]
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) == "merged"
    assert ContextCompressor._strip_summary_prefix(content) == (
        "BOUNDARY-BLOCK-SUMMARY"
    )
    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    assert projected is not None
    assert projected["content"] == [_image(), _text("BOUNDARY-BLOCK-PRIOR")]


@pytest.mark.parametrize(
    "trailing",
    [
        [_text("UNEXPECTED-NONEMPTY-TRAILING-TEXT")],
        [_image("trailing-image")],
    ],
)
def test_header_merged_list_rejects_nonempty_or_nontext_trailing_blocks(trailing):
    content = _header_merged_list([_text("AUTHENTIC-PRIOR")], trailing=trailing)
    message = _message(content)

    assert ContextCompressor.classify_summary_content(content) is None
    assert is_compaction_summary_message(message) is False
    assert ContextCompressor._strip_context_summary_handoff_message(message) == message


def test_force_leading_list_stays_standalone_and_preserves_live_blocks():
    live_blocks = [
        _text("LIVE-TEXT-BLOCK"),
        _image("live-image"),
    ]
    content = [
        _text(_summary("FORCE-LEADING-SUMMARY") + "\n\n"),
        *deepcopy(live_blocks),
    ]
    message = _message(content, metadata=True)

    assert ContextCompressor.classify_summary_content(content) == "standalone"
    assert ContextCompressor._strip_summary_prefix(content) == "FORCE-LEADING-SUMMARY"

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    assert projected is not None
    assert projected["content"] == live_blocks
    assert COMPRESSED_SUMMARY_METADATA_KEY not in projected


def test_find_context_summaries_without_metadata_accepts_only_valid_carriers():
    header_prior = (
        "HEADER-PRIOR-BEFORE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
        "QUOTED-CANDIDATE\n"
        "HEADER-PRIOR-AFTER"
    )
    ambiguous_headerless = (
        "AMBIGUOUS\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\nFIRST\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\nSECOND"
    )
    valid_list = _header_merged_list([_text("LIST-PRIOR")], "LIST-BODY")
    headerless_list = [
        _text("HEADERLESS-LIST"),
        _text(f"{_MERGED_SUMMARY_DELIMITER}\n{_summary('NOT-VALID-LIST')}"),
    ]
    messages = [
        _message("ordinary user content"),
        _message(_summary("STANDALONE-BODY"), role="assistant"),
        _message(_header_merged_string(header_prior, "HEADER-BODY")),
        _message(_legacy_merged_string("LEGACY-PRIOR", "LEGACY-BODY")),
        _message(ambiguous_headerless),
        _message(headerless_list),
        _message(valid_list),
    ]

    assert ContextCompressor._find_context_summaries(
        messages,
        0,
        len(messages),
    ) == [
        (1, "STANDALONE-BODY"),
        (2, "HEADER-BODY"),
        (3, "LEGACY-BODY"),
        (6, "LIST-BODY"),
    ]


def test_find_context_summaries_flagged_invalid_content_never_first_splits_payload():
    invalid_content = (
        "FLAGGED-BUT-UNSTRUCTURED-BEFORE\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n"
        "FLAGGED-MID-PAYLOAD-WITHOUT-A-SUMMARY-PREFIX"
    )
    message = _message(invalid_content, metadata=True)

    assert ContextCompressor._find_context_summaries([message], 0, 1) == [
        (0, invalid_content)
    ]


def test_handoff_only_blank_text_is_not_live_but_image_and_tool_calls_are():
    blank_text_handoff = _message(
        [
            _text(_summary("BLANK-HANDOFF") + "\n\n"),
            _text(" \n\t"),
        ],
        metadata=True,
    )
    image_handoff = _message(
        [
            _text(_summary("IMAGE-HANDOFF") + "\n\n"),
            _image("live-image-after-handoff"),
        ],
        metadata=True,
    )
    tool_handoff = _message(
        _summary("TOOL-HANDOFF"),
        metadata=True,
        role="assistant",
        tool_calls=[
            {
                "id": "call-live",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    )

    assert _handoff_carries_live_user_content(blank_text_handoff) is False
    assert _handoff_carries_live_user_content(image_handoff) is True
    assert _handoff_carries_live_user_content(tool_handoff) is True
