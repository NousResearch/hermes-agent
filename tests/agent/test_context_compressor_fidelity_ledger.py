"""Contracts for durable fidelity across repeated context compactions."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    ContextCompressor,
    _FIDELITY_LEDGER_END,
    _FIDELITY_LEDGER_MAX_ENTRIES,
    _FIDELITY_LEDGER_START,
    _FIDELITY_LEDGER_TOKEN_BUDGET,
    _HISTORICAL_SUMMARY_PREFIXES,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
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


@pytest.mark.parametrize(
    "marker",
    [
        _FIDELITY_LEDGER_START,
        _FIDELITY_LEDGER_END,
        SUMMARY_PREFIX,
        LEGACY_SUMMARY_PREFIX,
        *_HISTORICAL_SUMMARY_PREFIXES,
        _SUMMARY_END_MARKER,
        _MERGED_PRIOR_CONTEXT_HEADER,
        _MERGED_SUMMARY_DELIMITER,
    ],
)
def test_fidelity_ledger_sanitizes_compaction_markers_inside_user_text(marker: str):
    injected = (
        "We decided to preserve this literal marker: "
        f"{marker}"
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
            "[compaction marker removed]",
        )
    ]
    assert marker not in restored[0][1]
    assert summary.count(_FIDELITY_LEDGER_END) == 1

    final_handoff = (
        ContextCompressor._with_summary_prefix(summary)
        + "\n\n"
        + _SUMMARY_END_MARKER
    )
    assert final_handoff.endswith(_SUMMARY_END_MARKER)
    assert final_handoff.count(_SUMMARY_END_MARKER) == 1


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

    serialized = ContextCompressor._serialize_fidelity_ledger(merged)
    cost = estimate_messages_tokens_rough(
        [{"role": "user", "content": serialized}]
    )
    assert latest in merged
    assert len(merged) <= _FIDELITY_LEDGER_MAX_ENTRIES
    assert cost <= _FIDELITY_LEDGER_TOKEN_BUDGET
    assert sum(text == existing[0][1] for _kind, text in merged) <= 1


def test_merge_budget_counts_serialized_envelope_overhead():
    entry = ("decision", "We decided to preserve this bounded requirement.")
    entry_only_cost = estimate_messages_tokens_rough(
        [{"role": "user", "content": f"[{entry[0]}] {entry[1]}"}]
    )
    serialized_cost = ContextCompressor._fidelity_ledger_token_cost([entry])

    assert serialized_cost > entry_only_cost
    assert ContextCompressor._merge_fidelity_ledger(
        [],
        [entry],
        token_budget=entry_only_cost,
    ) == []


def test_repeated_compaction_benchmark_retains_all_decisions_without_extra_calls():
    """The public compression path survives a deliberately lossy summarizer."""
    compressor = _compressor()
    compressor.protect_first_n = 0
    compressor.protect_last_n = 2
    compressor.tail_token_budget = 120
    signals = [
        "We decided to use SQLite.",
        "Actually, use Postgres instead.",
        "Configure the timeout to 30 seconds.",
    ]
    messages = []

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=[
            _response("Lossy narrative one."),
            _response("Lossy narrative two."),
            _response("Lossy narrative three."),
        ],
    ) as mock_call:
        for round_index, signal in enumerate(signals):
            messages.extend(
                [
                    {"role": "user", "content": signal},
                    {"role": "assistant", "content": "Decision acknowledged."},
                ]
            )
            messages.extend(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": (
                        f"round {round_index} context {index} " + "detail " * 80
                    ),
                }
                for index in range(8)
            )
            messages = compressor.compress(
                messages,
                current_tokens=90_000,
                force=True,
            )

    summary_messages = [
        message
        for message in messages
        if SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    assert len(summary_messages) == 1
    summary_body = compressor._strip_summary_prefix(
        str(summary_messages[0]["content"])
    )
    narrative, entries = compressor._split_fidelity_ledger(summary_body)
    retained = {text for _kind, text in entries} & set(signals)
    retention_rate = len(retained) / len(signals)

    assert mock_call.call_count == len(signals)
    assert retention_rate == 1.0
    assert narrative == "Lossy narrative three."
    assert entries[-1] == ("configuration", "Configure the timeout to 30 seconds.")
    assert summary_body.count(_FIDELITY_LEDGER_START) == 1
    final_handoff = str(summary_messages[0]["content"])
    assert final_handoff.endswith(_SUMMARY_END_MARKER)
    assert final_handoff.count(_SUMMARY_END_MARKER) == 1


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
    restarted.protect_first_n = 1
    restarted.protect_last_n = 2
    restarted.tail_token_budget = 120
    resumed_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": persisted},
        {"role": "assistant", "content": "Handoff acknowledged."},
        {"role": "user", "content": "Actually, use footnote citations."},
        {"role": "assistant", "content": "Correction acknowledged."},
    ]
    resumed_messages.extend(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"resumed context {index} " + "detail " * 80,
        }
        for index in range(8)
    )

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("Resumed narrative."),
    ) as mock_call:
        resumed_messages = restarted.compress(
            resumed_messages,
            current_tokens=90_000,
            force=True,
        )

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert prompt.count("We decided to keep citations inline.") == 1
    assert "Actually, use footnote citations." in prompt
    summaries = [
        str(message.get("content", ""))
        for message in resumed_messages
        if SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    # The existing protected-head policy keeps the persisted handoff alongside
    # its updated successor. The later handoff is the authoritative one.
    body = next(
        restarted._strip_summary_prefix(summary)
        for summary in summaries
        if "Resumed narrative." in summary
    )
    narrative, entries = restarted._split_fidelity_ledger(body)
    assert narrative == "Resumed narrative."
    assert entries == [
        ("decision", "We decided to keep citations inline."),
        ("correction", "Actually, use footnote citations."),
    ]
    assert body.count(_FIDELITY_LEDGER_START) == 1
