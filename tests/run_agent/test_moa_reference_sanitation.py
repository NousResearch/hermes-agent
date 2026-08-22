"""Degenerate advisor outputs are excluded from aggregator guidance.

Observed degradation modes (moa trace, 5-turn sample): advisors fabricating
"[called tool: ...]"/"[tool result: ...]" transcripts, echoing context
scaffolding verbatim as their whole answer, and returning "(empty response)"
after a thinking model burned the full output cap on reasoning. None of that
is advice; injected into the aggregator prompt it reads as a fake action log
the aggregator must waste tokens dismissing. _is_failed_reference is the
single chokepoint — both the persistent facade and the one-shot /moa path
filter through it.
"""

from agent.moa_loop import (
    _ADVISORY_INSTRUCTION,
    _INTERRUPT_SCAFFOLD_MARKER,
    _degraded_notice,
    _failed_reference_labels,
    _is_failed_reference,
    _successful_references,
)


def test_internal_sentinels_still_detected():
    assert _is_failed_reference("[failed: HTTP 400]")
    assert _is_failed_reference("  [skipped: MoA presets cannot recursively reference MoA]")


def test_blank_and_empty_response_detected():
    assert _is_failed_reference("")
    assert _is_failed_reference("   \n  ")
    assert _is_failed_reference("(empty response)")


def test_fabricated_tool_transcripts_detected():
    fabricated = (
        "On it. Two things:\n"
        '[called tool: terminal({"command":"git clone https://example.com/x"})]\n'
        '[tool result: {"output": "Cloning...", "exit_code": 0}]Done.'
    )
    assert _is_failed_reference(fabricated)
    # Either bracket alone is enough — an advisor has no tools.
    assert _is_failed_reference('[tool result: {"output": "ok"}]')
    assert _is_failed_reference("[called tool: read_file({})]")


def test_quoted_bracket_markers_are_not_fabrication():
    """The digest hands advisors substance artifacts in fences and detail
    lines in `> ` quotes — an advisor QUOTING task material that contains
    bracket markers (reviewing a log/transcript) is real advice, not a
    fabricated action log (upstream review point on #85229)."""
    fenced = (
        "The transcript you're reviewing shows the degradation pattern:\n"
        "~~~\n"
        "[called tool: terminal({})]\n"
        "[tool result: {'exit': 0}]\n"
        "~~~\n"
        "Recommend stripping these markers before the advisor sees them."
    )
    assert not _is_failed_reference(fenced)

    backtick_fenced = fenced.replace("~~~", "```")
    assert not _is_failed_reference(backtick_fenced)

    quoted = (
        "The log line in question:\n"
        "> [tool result: {'output': 'Cloning...'}]\n"
        "This is the imitable syntax the digest view removes."
    )
    assert not _is_failed_reference(quoted)

    # Mid-sentence mention is commentary, not an action log.
    assert not _is_failed_reference(
        "advice: strip any [tool result: ...] markers from the digest"
    )

    # But an unquoted action-log line after a closed fence still fails.
    reopened = fenced + "\n[called tool: terminal({})]"
    assert _is_failed_reference(reopened)


def test_scaffold_echoes_detected():
    assert _is_failed_reference(_ADVISORY_INSTRUCTION)
    assert _is_failed_reference(_INTERRUPT_SCAFFOLD_MARKER)


def test_real_advice_passes():
    assert not _is_failed_reference(
        "The build failure is a missing JDK 21. The acting agent should run "
        "the install first, then retry the gradle build; check the mappings "
        "version if compilation still fails."
    )
    # Mentioning tools in prose is fine — only transcript syntax is degenerate.
    assert not _is_failed_reference("I suggest calling the terminal tool to run the tests.")


def test_degenerate_outputs_flow_into_degraded_policy():
    outputs = [
        ("ref-a", "solid advice here", None),
        ("ref-b", "(empty response)", None),
        ("ref-c", "[called tool: terminal({})]", None),
    ]

    assert [o[0] for o in _successful_references(outputs)] == ["ref-a"]
    failed = _failed_reference_labels(outputs)
    assert failed == ["ref-b", "ref-c"]
    assert "ref-b" in _degraded_notice(failed, "loud")
    assert _degraded_notice(failed, "silent") == ""


def test_truncated_thinking_becomes_actionable_failure(monkeypatch):
    """Empty text + output_tokens == cap -> actionable [failed:] note.

    Observed failure mode: a thinking advisor with reference_max_tokens=800
    burns the entire budget on reasoning and emits no visible text. The old
    "(empty response)" placeholder gave the user nothing to tune.
    """
    from unittest.mock import MagicMock, patch

    from agent.moa_loop import _run_reference

    def fake_call_llm(**kwargs):
        from types import SimpleNamespace

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=""))]
        mock_resp.usage = SimpleNamespace(completion_tokens=800, prompt_tokens=100)
        return mock_resp

    slot = {"provider": "openrouter", "model": "some/thinking-model"}

    with patch(
        "agent.moa_loop._slot_runtime",
        return_value={"provider": "openrouter", "model": "some/thinking-model"},
    ), patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), patch(
        "agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt, **kw: msgs
    ):
        _label, text, _acct = _run_reference(
            slot, [{"role": "user", "content": "hi"}], max_tokens=800
        )

    assert text.startswith("[failed: advisor hit reference_max_tokens=800")
    assert "reasoning" in text
    assert _is_failed_reference(text)


def test_empty_without_cap_stays_placeholder(monkeypatch):
    """Uncapped empty output keeps the plain placeholder (still filtered)."""
    from unittest.mock import MagicMock, patch

    from agent.moa_loop import _run_reference

    def fake_call_llm(**kwargs):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=""))]
        mock_resp.usage = None
        return mock_resp

    slot = {"provider": "openrouter", "model": "some/model"}

    with patch(
        "agent.moa_loop._slot_runtime",
        return_value={"provider": "openrouter", "model": "some/model"},
    ), patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), patch(
        "agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt, **kw: msgs
    ):
        _label, text, _acct = _run_reference(
            slot, [{"role": "user", "content": "hi"}], max_tokens=None
        )

    assert text == "(empty response)"
    assert _is_failed_reference(text)
