"""T3 — skew-distribution read-only reader (AC-6).

Parses COMPACTION_SKEW telemetry into a per-model ratio distribution, filtered to
task=main. Tolerant of malformed lines; never writes.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Load the script module by path (scripts/ isn't a package).
_SPEC = importlib.util.spec_from_file_location(
    "skew_distribution",
    Path(__file__).resolve().parents[2] / "scripts" / "skew-distribution.py",
)
skew_distribution = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(skew_distribution)  # type: ignore[union-attr]


def _write(p: Path, lines):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_collect_parses_and_groups_by_model(tmp_path):
    log = tmp_path / "skew-samples.log"
    _write(log, [
        "2026-06-27T10:00:00 COMPACTION_SKEW rough=1000 real=800 ratio=0.800 task=main model=claude-app/claude-opus-4-8 ctx=1000000",
        "2026-06-27T10:01:00 COMPACTION_SKEW rough=1000 real=900 ratio=0.900 task=main model=claude-app/claude-opus-4-8 ctx=1000000",
        "2026-06-27T10:02:00 COMPACTION_SKEW rough=2000 real=1000 ratio=0.500 task=main model=claude-app/claude-opus-4-8 ctx=1000000",
    ])
    by_model, total = skew_distribution.collect([str(log)], task_filter="main")
    assert total == 3
    assert sorted(by_model["claude-app/claude-opus-4-8"]) == [0.5, 0.8, 0.9]


def test_task_filter_excludes_aux(tmp_path):
    log = tmp_path / "skew-samples.log"
    _write(log, [
        "x COMPACTION_SKEW rough=1000 real=800 ratio=0.800 task=main model=m ctx=1",
        "x COMPACTION_SKEW rough=1000 real=600 ratio=0.600 task=compression model=m ctx=1",
    ])
    by_model, total = skew_distribution.collect([str(log)], task_filter="main")
    assert total == 1
    assert by_model["m"] == [0.8]
    # no filter → both
    _by, _total = skew_distribution.collect([str(log)], task_filter=None)
    assert _total == 2


def test_malformed_lines_skipped_not_raised(tmp_path):
    log = tmp_path / "skew-samples.log"
    _write(log, [
        "garbage line with no marker",
        "COMPACTION_SKEW rough=NOTANUM real=800 ratio=oops task=main model=m ctx=1",
        "COMPACTION_SKEW rough=1000 real=800 ratio=0.800 task=main model=m ctx=1",
    ])
    by_model, total = skew_distribution.collect([str(log)], task_filter="main")
    assert total == 1  # only the well-formed line
    assert by_model["m"] == [0.8]


def test_empty_source_zero_samples(tmp_path):
    log = tmp_path / "skew-samples.log"
    log.write_text("", encoding="utf-8")
    by_model, total = skew_distribution.collect([str(log)], task_filter="main")
    assert total == 0
    assert by_model == {}


def test_percentile_nearest_rank():
    vals = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert skew_distribution._percentile(vals, 50) == 0.7
    assert skew_distribution._percentile(vals, 1) == 0.5
    assert skew_distribution._percentile(vals, 100) == 0.9


def test_reader_is_read_only(tmp_path):
    """INV-5: the reader must not modify the log it reads."""
    log = tmp_path / "skew-samples.log"
    _write(log, ["x COMPACTION_SKEW rough=1000 real=800 ratio=0.800 task=main model=m ctx=1"])
    before = os.path.getmtime(log)
    before_content = log.read_text()
    skew_distribution.collect([str(log)], task_filter="main")
    assert os.path.getmtime(log) == before
    assert log.read_text() == before_content
