"""Startup surfacing for staged memory writes (#117894).

An unattended background review stages replace/remove proposals instead of applying them
(#105921). The staging message names ``/memory pending``, but it scrolls past with the turn that
produced it, so the queue was never reported again — one install accumulated 53 records over 10
days without the user knowing. These tests pin the read side.
"""

import pytest

from cli import HermesCLI
from tools import write_approval as wa


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return tmp_path


class _Recorder:
    """Minimal stand-in: only the two attributes the notice touches."""

    def __init__(self):
        self.printed = []

    def _console_print(self, text):
        self.printed.append(text)


def _notice(recorder):
    HermesCLI._show_pending_memory_notice(recorder)
    return "\n".join(recorder.printed)


def _stage(count):
    for i in range(count):
        wa.stage_write(
            wa.MEMORY,
            {"op": "replace", "old_text": f"old-{i}", "new_text": f"new-{i}"},
            summary=f"replace old-{i}",
            origin="background_review",
        )


def test_no_notice_when_the_queue_is_empty(isolated_home):
    assert _notice(_Recorder()) == ""


def test_notice_names_the_count_and_the_review_command(isolated_home):
    _stage(3)
    out = _notice(_Recorder())
    assert "3 memory writes" in out
    assert "/memory pending" in out


def test_single_record_reads_as_singular(isolated_home):
    _stage(1)
    assert "1 memory write staged" in _notice(_Recorder())
