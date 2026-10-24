"""Collapsed paste files must be unique per write.

``_tui_collapse_paste`` names the file it writes
``paste_<counter>_<HHMMSS>.txt``. The counter is per process (it is reset
when the TUI starts) and the pastes directory is shared by every Hermes
process on the host, so counter plus clock second is not unique across
processes: two processes that collapse their Nth paste in the same second
pick the same path, the second write truncates the file, and both sessions
then read whatever survived.

The test stands two sessions side by side in one process: each has its own
counter and both use the same pastes directory, which collides the same way.
Its inputs are captured, not generated. The two payloads are the bytes the
collapse path wrote for two real pastes (lines of this repository's README.md
and LICENSE) into two real CLI sessions sharing one Hermes home, and the
counter and clock second are those of the collision those sessions produced.
See fixtures/tui_paste_collision/PROVENANCE.md.
"""

import datetime as real_datetime
import sys
import types
from pathlib import Path

from hermes_cli.cli_tui_mixin import CLITuiMixin

FIXTURES = Path(__file__).parent / "fixtures" / "tui_paste_collision"


def _captured(name):
    # Line endings normalised as the paste handler does, so a CRLF checkout reads the same text.
    return (FIXTURES / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


# README.md lines 23-31 and LICENSE lines 5-21, as the collapse path wrote them.
FIRST_PASTE = _captured("readme_table.txt")
SECOND_PASTE = _captured("license_notice.txt")

# The captured collision: both sessions collapsed their first paste in this clock
# second, and both placeholders named this one file.
CAPTURED_SECOND = real_datetime.datetime(2026, 10, 22, 9, 54, 25)
CAPTURED_NAME = "paste_1_095425.txt"


class _FrozenClock:
    """Every ``now()`` in the test returns the captured second."""

    @staticmethod
    def now():
        return CAPTURED_SECOND


class _RecordingLogger:
    def __init__(self):
        self.lines = []

    def info(self, template, *args):
        self.lines.append(template % args)


class _Session(CLITuiMixin):
    """Minimal stand-in for a HermesCLI instance: the paste state is all the method touches."""

    def __init__(self, counter=0):
        self._tui_paste_counter = counter


def _install_cli_stub(monkeypatch, home, logger):
    stub = types.ModuleType("cli")
    stub._hermes_home = home
    stub.datetime = _FrozenClock
    stub.logger = logger
    monkeypatch.setitem(sys.modules, "cli", stub)


def _path_from_placeholder(placeholder):
    return placeholder.rsplit("→ ", 1)[1].rstrip("]")


def test_two_sessions_same_counter_same_second_write_different_files(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    _install_cli_stub(monkeypatch, tmp_path, logger)

    # The captured sessions logged "9 lines, 1892 chars" and "17 lines, 1022 chars".
    assert [FIRST_PASTE.count("\n"), len(FIRST_PASTE)] == [8, 1892]
    assert [SECOND_PASTE.count("\n"), len(SECOND_PASTE)] == [16, 1022]

    first, second = _Session(), _Session()
    placeholder_a = first._tui_collapse_paste(FIRST_PASTE, FIRST_PASTE.count("\n"), fallback=False)
    placeholder_b = second._tui_collapse_paste(SECOND_PASTE, SECOND_PASTE.count("\n"), fallback=False)

    path_a = _path_from_placeholder(placeholder_a)
    path_b = _path_from_placeholder(placeholder_b)
    assert path_a != path_b, "the same counter in the same second must not name one shared file"

    with open(path_a, encoding="utf-8") as handle:
        assert handle.read() == FIRST_PASTE
    with open(path_b, encoding="utf-8") as handle:
        assert handle.read() == SECOND_PASTE

    # Each session logged its own distinct target, so the log can be read per session.
    assert len(logger.lines) == 2
    assert path_a in logger.lines[0] and path_b in logger.lines[1]
    assert len({line.rsplit("-> ", 1)[1] for line in logger.lines}) == 2


def test_collapse_never_reuses_an_existing_paste_file(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    _install_cli_stub(monkeypatch, tmp_path, logger)

    # The file the captured collision left behind: its name, holding the paste that won.
    pastes = tmp_path / "pastes"
    pastes.mkdir()
    legacy = pastes / CAPTURED_NAME
    legacy.write_text(SECOND_PASTE, encoding="utf-8")

    placeholder = _Session()._tui_collapse_paste(FIRST_PASTE, FIRST_PASTE.count("\n"), fallback=False)
    path = _path_from_placeholder(placeholder)

    assert path != str(legacy)
    assert legacy.read_text(encoding="utf-8") == SECOND_PASTE
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == FIRST_PASTE


def test_placeholder_and_counter_still_work(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    _install_cli_stub(monkeypatch, tmp_path, logger)

    session = _Session()
    placeholder = session._tui_collapse_paste(FIRST_PASTE, FIRST_PASTE.count("\n"), fallback=True)

    # The captured session showed "[Pasted text #1: 9 lines → ...]" and logged
    # "Collapsed paste #1: 9 lines, 1892 chars -> ..."; the fallback path adds its marker.
    assert session._tui_paste_counter == 1
    assert session._tui_paste_just_collapsed is True
    assert placeholder.startswith("[Pasted text #1: 9 lines → ")
    assert logger.lines[0].startswith("Collapsed paste #1: 9 lines, 1892 chars -> ")
    assert logger.lines[0].endswith(" (fallback)")
