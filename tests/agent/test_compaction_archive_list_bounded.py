"""Regression: the archive list appended to a compaction summary stays bounded.

``_assemble_with_archive_list`` appends every archive the replaced rows named to the committed summary. The
previous summary is part of the transcript and carries the list of the round before, so counting it as "still
named" made every earlier list permanent and each compaction re-committed all of them: the summary grew
monotonically (a real session reached 79 KB, of which 57 KB was the list). The list is now rendered from a
bounded newest-N window and the inherited block is replaced, not stacked.
"""

import os
from types import SimpleNamespace

import pytest

from agent.context_compressor_window import CompactionWindowMixin, _strip_archive_list
from tools.tool_result_storage import (
    ARCHIVE_LIST_HEADING,
    ARCHIVE_LIST_MAX_ENTRIES,
    archive_list_for_summary,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


def _refs(count, tag):
    return [f"cache/pruned/sess_{tag}_{i:04d}.json" for i in range(count)]


def _rows(refs):
    return [{"role": "tool", "content": f"see ~/.hermes/cache/{ref}"} for ref in refs]


class _Assembler(CompactionWindowMixin):
    """Captures the summary the mixin hands to ``_assemble_compressed`` (no real compressor needed)."""

    def __init__(self):
        self.summaries = []

    def _assemble_compressed(self, messages, start, end, scan, summary):
        self.summaries.append(summary)
        return [{"role": "user", "content": summary}]


_SCAN = SimpleNamespace(tail_start=10 ** 9)


def test_archive_list_is_capped_and_points_at_session_search(home):
    named = set(_refs(ARCHIVE_LIST_MAX_ENTRIES + 60, "r0"))

    block = archive_list_for_summary(named, kept=[])

    assert block.startswith("\n\n" + ARCHIVE_LIST_HEADING)
    listed = block.count("/sess_")
    assert listed == ARCHIVE_LIST_MAX_ENTRIES, f"cap not applied: {listed} entries"
    assert f"{60} earlier archive(s)" in block, "overflow count missing"
    assert "session_search" in block, "overflow must say where the rest are"


def test_strip_drops_every_stacked_block_from_the_first_heading():
    body = "## Goal\nship the feature."
    stacked = body
    for r in range(4):
        stacked += "\n\n" + ARCHIVE_LIST_HEADING + "".join(f"\n- <home>/{x}" for x in _refs(5, f"r{r}"))

    assert _strip_archive_list(stacked) == body


def test_strip_is_a_no_op_without_a_list():
    assert _strip_archive_list("## Goal\nno list here") == "## Goal\nno list here"


def test_repeated_compaction_keeps_the_summary_size_constant(home):
    """The regression: N compactions used to grow the summary by one more list block each round."""
    assembler = _Assembler()
    summary = "## Goal\nship the feature."
    sizes = []
    for r in range(6):
        out = assembler._assemble_with_archive_list(_rows(_refs(120, f"r{r}")), 0, 1, _SCAN, summary)
        summary = out[0]["content"]
        sizes.append(len(summary))

    assert len(set(sizes)) == 1, f"summary size drifted across compactions: {sizes}"
    assert summary.count(ARCHIVE_LIST_HEADING) == 1, "archive list blocks stacked"
    assert summary.count("\n- ") <= ARCHIVE_LIST_MAX_ENTRIES + 1, "list exceeded the cap"


def test_generated_body_is_kept_alongside_the_fresh_list(home):
    """The list is appended to the summary body; stripping must not eat the model's own text."""
    assembler = _Assembler()
    first = assembler._assemble_with_archive_list(_rows(_refs(3, "r0")), 0, 1, _SCAN, "## Goal\nkeep me")
    second = assembler._assemble_with_archive_list(
        _rows(_refs(3, "r1")), 0, 1, _SCAN, first[0]["content"])

    assert second[0]["content"].startswith("## Goal\nkeep me")
    assert second[0]["content"].count(ARCHIVE_LIST_HEADING) == 1