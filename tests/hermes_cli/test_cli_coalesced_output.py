"""``_coalesced_output``: the erase/print/redraw sequence of one ``_cprint`` must reach the
terminal as a single write.

Split across writes, the erase leaves the bottom chrome blank for the frame in between —
invisible locally, a visible flicker over ssh (#1624). The context manager buffers every
flush and replays them as one, so the terminal renders only the final state.
"""
from hermes_cli.cli_render import _coalesced_output


class _RecordingOutput:
    """Minimal stand-in for ``Vt100_Output``: keeps its own buffer, records each flush."""

    def __init__(self):
        self._buffer = []
        self.writes = []

    def write_raw(self, text):
        self._buffer.append(text)

    def flush(self):
        if not self._buffer:
            return
        self.writes.append("".join(self._buffer))
        self._buffer = []


def test_sequence_of_flushes_becomes_one_write():
    out = _RecordingOutput()

    with _coalesced_output(out):
        out.write_raw("ERASE\x1b[J")
        out.flush()  # would show a blank chrome on its own
        out.write_raw("TEXT")
        out.flush()
        out.write_raw("CHROME")
        out.flush()

    assert len(out.writes) == 1
    assert out.writes[0] == "ERASE\x1b[JTEXTCHROME"


def test_without_coalescing_the_flushes_stay_split():
    out = _RecordingOutput()
    out.write_raw("A")
    out.flush()
    out.write_raw("B")
    out.flush()

    assert out.writes == ["A", "B"]


def test_flush_restored_after_the_block():
    out = _RecordingOutput()
    original = out.flush

    with _coalesced_output(out):
        pass

    assert out.flush == original

    # A later write flushes normally again.
    out.write_raw("later")
    out.flush()
    assert out.writes == ["later"]


def test_empty_block_writes_nothing():
    out = _RecordingOutput()

    with _coalesced_output(out):
        pass

    assert out.writes == []
