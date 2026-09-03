"""Regression tests for #57682 — holographic auto_extract harvested
context-compaction handoff summaries into fact_store, and ran even when
configured off.

Two compounding defects:

1. Gate: the plugin's config schema declares ``auto_extract`` as a string enum
   (``"false"``/``"true"``), and the ``on_session_end`` gate used plain
   truthiness — ``not "false"`` is ``False`` — so extraction ran despite being
   explicitly configured off.

2. Eligibility: ``_auto_extract_facts`` scanned every ``role == "user"``
   message. Compaction handoff summaries can be inserted as ``role="user"``
   messages, and their prose reliably matches the decision patterns
   (``we decided/agreed/chose``, ``the project uses/needs/requires``), so the
   compactor's own output was stored as a durable ``project`` fact on every
   session rollover that followed a compaction.
"""

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
    _append_text_to_content,
    is_compaction_summary_message,
)
from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _fact_contents(provider):
    return [f["content"] for f in provider._store.list_facts(limit=100)]


def _user(content, **extra):
    msg = {"role": "user", "content": content}
    msg.update(extra)
    return msg


DECISION_MSG = "we decided to use PostgreSQL for the persistence layer"
SUMMARY_MSG = (
    f"{SUMMARY_PREFIX}\n## Historical Task Snapshot\n"
    "The project uses a kanban board for all dispatch. "
    "We agreed to route reviews through the fan-in consumer."
)


# ---------------------------------------------------------------------------
# Defect 1 — string-boolean gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("off_value", [False, "false", "False", "no", "off", "0", None, ""])
def test_auto_extract_off_values_disable_extraction(tmp_path, off_value):
    provider = _make_provider(tmp_path, auto_extract=off_value)
    provider.on_session_end([_user(DECISION_MSG)])
    assert _fact_contents(provider) == []
    provider.shutdown()


# ---------------------------------------------------------------------------
# Defect 2 — compaction summaries harvested as facts
# ---------------------------------------------------------------------------


def test_compaction_summary_not_harvested(tmp_path):
    """The exact failure mode from #57682: summary prose matches the decision
    patterns but must not become a fact."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(SUMMARY_MSG)])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_metadata_marked_summary_not_harvested(tmp_path):
    """In-process summaries carry COMPRESSED_SUMMARY_METADATA_KEY even if a
    future prefix rewrite changes the content sentinel."""
    provider = _make_provider(tmp_path, auto_extract=True)
    marked = _user(DECISION_MSG, **{COMPRESSED_SUMMARY_METADATA_KEY: True})
    provider.on_session_end([marked])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_merged_into_tail_summary_suffix_not_harvested_prefix_content_ignored(tmp_path):
    """Merge-into-tail summaries embed the handoff prefix after the delimiter,
    not at the start of the message. The wrapped pre-delimiter segment here has
    no fact-pattern match, so nothing is harvested from either side."""
    provider = _make_provider(tmp_path, auto_extract=True)
    merged = _user(
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\nplease fix the login bug\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}"
    )
    provider.on_session_end([merged])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_delimiter_merged_list_harvests_only_authentic_prior_text(tmp_path):
    """Multimodal delimiter carriers keep the genuine blocks before the
    summary, but the generated summary itself must never become a fact."""
    provider = _make_provider(tmp_path, auto_extract=True)
    authentic_prior = "I prefer PostgreSQL settings from the genuine user turn"
    prior_blocks = [
        {"type": "text", "text": authentic_prior},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        },
    ]
    suffix = (
        f"\n\n{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_MSG}\n\n{_SUMMARY_END_MARKER}"
    )
    merged = _append_text_to_content(
        _append_text_to_content(prior_blocks, suffix, prepend=False),
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n",
        prepend=True,
    )

    provider.on_session_end([_user(merged)])

    assert _fact_contents(provider) == [authentic_prior]
    provider.shutdown()


def test_force_user_leading_list_harvests_only_authentic_tail_text(tmp_path):
    """A force-user-leading list puts the summary before the real multimodal
    request; extraction must retain only the request after the end marker."""
    provider = _make_provider(tmp_path, auto_extract=True)
    authentic_tail = "I prefer tabs from the genuine user request"
    real_blocks = [
        {"type": "text", "text": authentic_tail},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,BBBB"},
        },
    ]
    prefix = f"{SUMMARY_MSG}\n\n{_SUMMARY_END_MARKER}\n\n"
    merged = _append_text_to_content(real_blocks, prefix, prepend=True)

    provider.on_session_end([_user(merged)])

    assert _fact_contents(provider) == [authentic_tail]
    provider.shutdown()


def test_real_user_messages_still_extracted_alongside_summary(tmp_path):
    """The guard must skip only the summary, not suppress extraction for the
    genuine user turns around it."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end(
        [
            _user(SUMMARY_MSG),
            _user("I prefer tabs over spaces for indentation"),
        ]
    )
    facts = _fact_contents(provider)
    assert len(facts) == 1
    assert "tabs over spaces" in facts[0]
    provider.shutdown()


# ---------------------------------------------------------------------------
# is_compaction_summary_message — public helper contract
# ---------------------------------------------------------------------------


def test_helper_detects_prefix_metadata_and_merged_forms():
    assert is_compaction_summary_message(_user(SUMMARY_MSG))
    assert is_compaction_summary_message(
        _user("anything", **{COMPRESSED_SUMMARY_METADATA_KEY: True})
    )
    assert is_compaction_summary_message(
        _user(f"prior tail\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}")
    )
    assert not is_compaction_summary_message(_user(DECISION_MSG))
    assert not is_compaction_summary_message(_user(""))
