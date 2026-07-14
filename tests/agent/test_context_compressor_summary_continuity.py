"""Regression tests for iterative context-summary continuity."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _FALLBACK_SUMMARY_MAX_CHARS,
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _messages_with_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_existing_previous_summary_is_not_serialized_again_as_new_turn():
    """Same-process iterative compression should not feed the old handoff twice."""
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"
    compressor._previous_summary = old_summary

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_resume_rehydrates_previous_summary_from_handoff_message():
    """After restart/resume, the persisted handoff should regain summary identity."""
    compressor = _compressor()
    old_summary = "RESUMED-SUMMARY-BODY durable continuity facts"
    assert compressor._previous_summary is None

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert "TURNS TO SUMMARIZE:" not in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_handoff_in_protected_head_populates_previous_summary_before_update():
    """A resumed protected-head handoff should restore iterative-summary state."""
    compressor = _compressor()
    old_summary = "PROTECTED-HEAD-SUMMARY durable facts from before restart"
    seen_turns = []

    def fake_generate_summary(turns_to_summarize, focus_topic=None):
        seen_turns.extend(turns_to_summarize)
        return "new summary from resumed turns"

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == old_summary
    assert seen_turns
    assert all(old_summary not in str(msg.get("content", "")) for msg in seen_turns)


def test_summary_prompt_requires_full_span_continuity_note():
    compressor = _compressor()
    fake_secret = "FAKESECRET0123456789ABCDEF"
    turns = [
        {"role": "user", "content": "first-span fact: server uses port 8000"},
        {
            "role": "assistant",
            "content": (
                "middle-span decision: preserve the existing patch; "
                f"Authorization: Bearer {fake_secret}"
            ),
        },
        {
            "role": "user",
            "content": "last-span constraint: never invent unknowns",
        },
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("checkpoint"),
    ) as mock_call:
        summary = compressor._generate_summary(turns)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert summary is not None
    assert "## Continuity Note" in summary
    assert "ENTIRE span" in prompt
    assert "## Continuity Note" in prompt
    assert "Unknown — NEVER invent missing details" in prompt
    assert "Do not sanitize or omit explicit" in prompt
    assert "first-span fact: server uses port 8000" in prompt
    assert "middle-span decision: preserve the existing patch" in prompt
    assert "last-span constraint: never invent unknowns" in prompt
    assert fake_secret not in prompt
    middle_line = next(line for line in prompt.splitlines() if "middle-span" in line)
    assert "Authorization: Bearer " in middle_line
    assert "..." in middle_line


def test_empty_continuity_note_is_repaired_before_store():
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "first durable fact"},
        {"role": "assistant", "content": "middle durable decision"},
        {"role": "user", "content": "last durable constraint"},
    ]
    malformed = (
        "## Goal\nPreserve context.\n\n"
        "## Continuity Note\n\n"
        "## Active State\nUnknown."
    )

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(malformed),
    ):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    assert summary.count("## Continuity Note") == 1
    continuity = summary.split("## Continuity Note\n", 1)[1].split(
        "\n## Active State", 1
    )[0]
    assert "first durable fact" in continuity
    assert "middle durable decision" in continuity
    assert "last durable constraint" in continuity
    assert compressor._previous_summary is not None
    assert "## Continuity Note" in compressor._previous_summary


def test_compaction_forces_redaction_when_global_opt_out_is_active():
    compressor = _compressor()
    fake_secret = "FAKESECRET0123456789ABCDEF"
    compressor._previous_summary = (
        f"prior context Authorization: Bearer {fake_secret}"
    )
    turns = [
        {
            "role": "user",
            "content": f"new context Authorization: Bearer {fake_secret}",
        }
    ]

    with patch("agent.redact._REDACT_ENABLED", False), patch(
        "agent.context_compressor.call_llm",
        return_value=_response(
            f"checkpoint Authorization: Bearer {fake_secret}"
        ),
    ) as mock_call:
        summary = compressor._generate_summary(turns)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert summary is not None
    assert fake_secret not in prompt
    assert fake_secret not in summary
    assert fake_secret not in (compressor._previous_summary or "")


def test_static_fallback_has_full_span_anchors_within_cap():
    compressor = _compressor()
    fake_secret = "FAKESECRET0123456789ABCDEF"
    compressor._previous_summary = (
        f"PRIOR-SPAN-FACT Authorization: Bearer {fake_secret}"
    )
    turns = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index:02d} continuity fact " + ("x" * 900),
        }
        for index in range(21)
    ]

    with patch("agent.redact._REDACT_ENABLED", False):
        summary = compressor._build_static_fallback_summary(
            turns, reason="malformed summary"
        )

    assert len(summary) <= _FALLBACK_SUMMARY_MAX_CHARS
    assert summary.endswith("...[fallback summary truncated]")
    continuity = summary.split("## Continuity Note\n", 1)[1].split(
        "\n## Constraints & Preferences", 1
    )[0]
    assert "turn-00 continuity fact" in continuity
    assert "turn-10 continuity fact" in continuity
    assert "turn-20 continuity fact" in continuity
    assert "PRIOR-SPAN-FACT" in continuity
    assert fake_secret not in summary
    assert "Unknown. Do not invent" in continuity


def test_iterative_fallback_becomes_next_previous_summary():
    compressor = _compressor()
    old_summary = "OLD-SPAN-FACT durable prior context"
    compressor._previous_summary = old_summary

    with patch.object(
        compressor, "_find_tail_cut_by_tokens", return_value=5
    ), patch.object(
        compressor, "_generate_summary", return_value=None
    ):
        result = compressor.compress(
            _messages_with_handoff(old_summary), force=True
        )

    assert any(
        message.get(COMPRESSED_SUMMARY_METADATA_KEY)
        for message in result
    )
    assert compressor._previous_summary != old_summary
    assert "OLD-SPAN-FACT" in compressor._previous_summary
    assert "## Continuity Note" in compressor._previous_summary
    assert "new user turn after resume" in compressor._previous_summary


def test_media_directives_are_neutralized_at_all_compaction_boundaries():
    media_directive = "MEDIA:/tmp/compaction-secret.png"
    turns = [
        {
            "role": "assistant",
            "content": "tool call",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": (
                            '{"path":"' + media_directive + '"}'
                        ),
                    },
                }
            ],
        }
    ]
    compressor = _compressor()
    model_summary = (
        "## Continuity Note\n- model emitted " + media_directive
    )

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(model_summary),
    ) as mock_call:
        summary = compressor._generate_summary(
            turns, focus_topic=media_directive
        )

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    fallback = compressor._build_static_fallback_summary(
        turns, reason="failed near " + media_directive
    )
    auto_focus = ContextCompressor._derive_auto_focus_topic(
        [{"role": "user", "content": "latest " + media_directive}]
    )

    assert summary is not None
    assert auto_focus is not None
    for boundary_value in (
        prompt,
        fallback,
        summary,
        compressor._previous_summary or "",
        auto_focus,
    ):
        assert media_directive not in boundary_value
        assert "[media attachment]" in boundary_value


def test_empty_continuity_note_at_eof_gets_newline_before_repair():
    turns = [{"role": "user", "content": "first durable fact"}]

    for malformed in ("## Continuity Note", "## Continuity Note   "):
        compressor = _compressor()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(malformed),
        ):
            summary = compressor._generate_summary(turns)

        assert summary is not None
        body = compressor._strip_summary_prefix(summary)
        assert body.startswith(
            "## Continuity Note\n- Prior compaction context:"
        )
        assert "first durable fact" in body
        assert body.count("## Continuity Note") == 1
