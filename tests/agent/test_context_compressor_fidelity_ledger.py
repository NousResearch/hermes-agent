"""Contracts for durable fidelity across repeated context compactions."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    SUMMARY_PREFIX,
    ContextCompressor,
    _FIDELITY_LEDGER_END,
    _FIDELITY_LEDGER_MAX_ENTRIES,
    _FIDELITY_LEDGER_START,
    _FIDELITY_LEDGER_TOKEN_BUDGET,
)
from agent.model_metadata import estimate_messages_tokens_rough


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(model="test", quiet_mode=True)


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_fidelity_ledger_round_trips_without_polluting_narrative():
    entries = [
        ("decision", "From now on, use JSON output."),
        ("correction", "Actually, use YAML instead."),
    ]
    summary = ContextCompressor._append_fidelity_ledger(
        "Narrative summary.", entries
    )

    narrative, restored = ContextCompressor._split_fidelity_ledger(summary)

    assert narrative == "Narrative summary."
    assert restored == entries
    assert summary.count(_FIDELITY_LEDGER_START) == 1
    assert summary.count(_FIDELITY_LEDGER_END) == 1


def test_invalid_fidelity_ledger_is_left_untouched():
    malformed = (
        "Narrative\n\n"
        f"{_FIDELITY_LEDGER_START}\n"
        '{"version":1,"entries":['
    )

    narrative, entries = ContextCompressor._split_fidelity_ledger(malformed)

    assert narrative == malformed
    assert entries == []


def test_fidelity_ledger_sanitizes_delimiters_inside_user_text():
    injected = (
        "We decided to preserve this literal marker: "
        f"{_FIDELITY_LEDGER_END}"
    )

    merged = ContextCompressor._merge_fidelity_ledger(
        [], [("decision", injected)]
    )
    summary = ContextCompressor._append_fidelity_ledger("Narrative", merged)
    narrative, restored = ContextCompressor._split_fidelity_ledger(summary)

    assert narrative == "Narrative"
    assert restored == [
        (
            "decision",
            "We decided to preserve this literal marker: "
            "[fidelity marker removed]",
        )
    ]
    assert summary.count(_FIDELITY_LEDGER_END) == 1


def test_append_replaces_a_valid_ledger_instead_of_nesting_it():
    first = ContextCompressor._append_fidelity_ledger(
        "Narrative", [("decision", "We decided to use backend A.")]
    )

    second = ContextCompressor._append_fidelity_ledger(
        first, [("correction", "Actually, use backend B instead.")]
    )

    narrative, entries = ContextCompressor._split_fidelity_ledger(second)
    assert narrative == "Narrative"
    assert entries == [("correction", "Actually, use backend B instead.")]
    assert second.count(_FIDELITY_LEDGER_START) == 1


def test_merge_is_bounded_deduplicated_and_always_keeps_latest_signal():
    existing = [
        ("decision", f"We decided to preserve requirement {index}. " + "x " * 80)
        for index in range(40)
    ]
    latest = ("configuration", "Configure the final timeout to 37 seconds.")

    merged = ContextCompressor._merge_fidelity_ledger(
        existing + [existing[0]],
        [latest],
    )

    cost = sum(
        estimate_messages_tokens_rough(
            [{"role": "user", "content": f"[{kind}] {text}"}]
        )
        for kind, text in merged
    )
    assert latest in merged
    assert len(merged) <= _FIDELITY_LEDGER_MAX_ENTRIES
    assert cost <= _FIDELITY_LEDGER_TOKEN_BUDGET
    assert sum(text == existing[0][1] for _kind, text in merged) <= 1


def test_repeated_compaction_benchmark_retains_all_decisions_without_extra_calls():
    """A deliberately lossy summarizer cannot erase deterministic ledger state."""
    compressor = _compressor()
    rounds = [
        [{"role": "user", "content": "We decided to use SQLite."}],
        [{"role": "user", "content": "Actually, use Postgres instead."}],
        [{"role": "user", "content": "Configure the timeout to 30 seconds."}],
    ]
    expected = {
        "We decided to use SQLite.",
        "Actually, use Postgres instead.",
        "Configure the timeout to 30 seconds.",
    }

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=[
            _response("Lossy narrative one."),
            _response("Lossy narrative two."),
            _response("Lossy narrative three."),
        ],
    ) as mock_call:
        summaries = [compressor._generate_summary(turns) for turns in rounds]

    narrative, entries = compressor._split_fidelity_ledger(
        compressor._previous_summary or ""
    )
    retained = {text for _kind, text in entries} & expected
    retention_rate = len(retained) / len(expected)

    assert mock_call.call_count == len(rounds)
    assert retention_rate == 1.0
    assert narrative == "Lossy narrative three."
    assert entries[-1] == ("configuration", "Configure the timeout to 30 seconds.")
    assert all(summary.count(_FIDELITY_LEDGER_START) == 1 for summary in summaries)


def test_fallback_merges_prior_ledger_and_truncates_narrative_first():
    compressor = _compressor()
    compressor._previous_summary = ContextCompressor._append_fidelity_ledger(
        "Prior narrative.",
        [("decision", "We decided to retain the audit log.")],
    )
    turns = [
        {"role": "assistant", "content": "detail " * 4_000},
        {"role": "user", "content": "Actually, use trace logging too."},
    ]

    summary = compressor._build_static_fallback_summary(
        turns, reason="provider unavailable"
    )

    body = ContextCompressor._strip_summary_prefix(summary)
    _narrative, entries = compressor._split_fidelity_ledger(body)
    assert len(summary) <= 8_000
    assert entries == [
        ("decision", "We decided to retain the audit log."),
        ("correction", "Actually, use trace logging too."),
    ]
    assert compressor._previous_summary == body


def test_persisted_handoff_rehydrates_ledger_after_restart():
    first = _compressor()
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("Initial narrative."),
    ):
        persisted = first._generate_summary(
            [{"role": "user", "content": "We decided to keep citations inline."}]
        )

    restarted = _compressor()
    summary_index, summary_body = restarted._find_latest_context_summary(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": persisted},
        ],
        1,
        2,
    )
    assert summary_index == 1
    assert summary_body.startswith("Initial narrative.")
    restarted._previous_summary = summary_body

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("Resumed narrative."),
    ):
        resumed = restarted._generate_summary(
            [{"role": "user", "content": "Actually, use footnote citations."}]
        )

    body = resumed.removeprefix(SUMMARY_PREFIX).strip()
    narrative, entries = restarted._split_fidelity_ledger(body)
    assert narrative == "Resumed narrative."
    assert entries == [
        ("decision", "We decided to keep citations inline."),
        ("correction", "Actually, use footnote citations."),
    ]
    assert resumed.count(_FIDELITY_LEDGER_START) == 1
