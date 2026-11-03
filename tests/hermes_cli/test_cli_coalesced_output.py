"""``_coalesced_output``: one ``_cprint`` paint sequence must reach the terminal as one write.

Split across writes, the erase ships a frame with the chrome already blanked — invisible on a
local terminal, a visible flicker over ssh or tmux on every streamed line (#24137). The
context manager defers every flush inside the block and runs the real one once, on the way out.
"""
import io

from prompt_toolkit.output.plain_text import PlainTextOutput

from hermes_cli.cli_render import _coalesced_output


class _CountingStdout(io.StringIO):
    """``flush_stdout`` writes through here; record what each write carried."""

    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return super().write(data)


def _output():
    return PlainTextOutput(_CountingStdout())


def test_sequence_of_flushes_becomes_one_write():
    out = _output()

    with _coalesced_output(out):
        out.write_raw("ERASE\x1b[J")
        out.flush()  # on its own this would show a blanked chrome
        out.write_raw("TEXT")
        out.flush()
        out.write_raw("CHROME")
        out.flush()

    assert out.stdout.writes == ["ERASE\x1b[JTEXTCHROME"]


def test_without_coalescing_the_flushes_stay_split():
    out = _output()
    out.write_raw("A")
    out.flush()
    out.write_raw("B")
    out.flush()

    assert out.stdout.writes == ["A", "B"]


def test_flush_restored_after_the_block():
    out = _output()

    with _coalesced_output(out):
        pass

    assert out.flush.__func__ is PlainTextOutput.flush

    # A later write flushes normally again.
    out.write_raw("later")
    out.flush()
    assert out.stdout.writes == ["later"]


def test_empty_block_writes_nothing():
    out = _output()

    with _coalesced_output(out):
        pass

    assert out.stdout.writes == []


def test_output_written_before_a_raise_still_ships():
    out = _output()

    try:
        with _coalesced_output(out):
            out.write_raw("PARTIAL")
            out.flush()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert out.stdout.writes == ["PARTIAL"]
    assert out.flush.__func__ is PlainTextOutput.flush


class _WriteThroughOutput:
    """Stands in for an Output that writes immediately and keeps no buffer."""

    def __init__(self):
        self.writes = []
        self.flushes = 0

    def write_raw(self, data):
        self.writes.append(data)

    def flush(self):
        self.flushes += 1


def test_write_through_output_still_gets_its_flush():
    out = _WriteThroughOutput()

    with _coalesced_output(out):
        out.write_raw("A")
        out.flush()

    # No buffer to merge, but the deferred call must still reach the output's own flush.
    assert out.flushes == 1
