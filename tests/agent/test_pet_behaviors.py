"""pet.json ``stateRows`` / ``behaviors``: extra rows + state remaps, opt-in sleep."""

import json
import os
import signal

import pytest

from agent.pet import behaviors, render
from agent.pet.constants import CODEX_STATE_ROWS, FRAME_H, FRAME_W

PIL = pytest.importorskip("PIL")
from PIL import Image

EXTRA = ["sleeping", "smiling", "crying"]
ROWS_12 = [*CODEX_STATE_ROWS, *EXTRA]  # rows 9-11 = EXTRA


def _sheet(tmp_path, n_rows, meta=None, frames=None):
    """Sheet with ``n_rows`` rows; row *r* has ``frames[r]`` opaque frames (default 6)."""
    img = Image.new("RGBA", (FRAME_W * 8, FRAME_H * n_rows), (0, 0, 0, 0))
    for r in range(n_rows):
        for c in range((frames or {}).get(r, 6)):
            img.paste((r * 20 % 255, 100, 100, 255), (c * FRAME_W, r * FRAME_H, (c + 1) * FRAME_W, (r + 1) * FRAME_H))
    path = tmp_path / "spritesheet.png"
    img.save(path)
    if meta is not None:
        (tmp_path / "pet.json").write_text(json.dumps(meta))
    return path


def test_no_pet_json_is_unchanged(tmp_path):
    spec = behaviors.pet_spec(_sheet(tmp_path, 9))
    assert spec.rows == tuple(CODEX_STATE_ROWS)
    assert spec.overrides == {} and spec.sleep_after == 0 and spec.payload() == {} and spec.warnings == ()
    assert spec.row_index("failed") == 5 and spec.row_index("wave") == 3
    assert spec.row_index("sleep") == 0  # opt-in state renders idle


def test_declared_rows_and_overrides(tmp_path):
    meta = {
        "stateRows": ROWS_12,
        "behaviors": {
            "failed": {"row": "crying"},
            "jump": {"row": "smiling"},
            "sleep": {"row": "sleeping", "afterSeconds": 300},
            "review": {"row": "nope"},  # unknown row → ignored (more cases: test_cleaning_reports_...)
        },
    }
    sheet = _sheet(tmp_path, 12, meta, frames={11: 3})
    spec = behaviors.pet_spec(sheet)
    assert spec.rows[-3:] == tuple(EXTRA)
    assert spec.row_index("failed") == 11 and spec.row_index("jump") == 10 and spec.row_index("sleep") == 9
    assert spec.row_index("review") == 8  # standard row kept
    assert spec.payload() == {"rows": {"failed": ["crying"], "jump": ["smiling"], "sleep": ["sleeping"]}, "sleepAfterSeconds": 300}
    counts = render.state_frame_counts(sheet)
    assert counts["failed"] == 3  # frames come from the crying row


def test_bad_state_rows_fall_back_to_inferred(tmp_path):
    spec = behaviors.pet_spec(_sheet(tmp_path, 9, {"stateRows": ["a"] * 9, "behaviors": "x"}))
    assert spec.rows == tuple(CODEX_STATE_ROWS) and spec.overrides == {}


@pytest.mark.parametrize(("n_rows", "sleep", "expected"), [
    (10, {"row": "sleeping", "afterSeconds": 1}, 10),  # clamped up
    (10, {"row": "sleeping"}, 0),  # no delay → no sleep
    (9, {"row": "sleeping", "afterSeconds": 60}, 0),  # row not on the sheet
])
def test_sleep_delay_is_clamped_and_needs_a_row(tmp_path, n_rows, sleep, expected):
    meta = {"stateRows": ROWS_12[:n_rows], "behaviors": {"sleep": sleep}}
    assert behaviors.pet_spec(_sheet(tmp_path, n_rows, meta)).sleep_after == expected


def test_hold_seconds_is_per_state_and_clamped(tmp_path):
    spec = behaviors.pet_spec(_sheet(tmp_path, 12, {"stateRows": ROWS_12, "behaviors": {
        "run": {"rows": ["smiling", "crying"], "holdSeconds": 8},
        "review": {"row": "smiling", "holdSeconds": 99999},
        "failed": {"row": "crying"},
    }}))
    assert spec.payload()["holdSeconds"] == {"run": 8, "review": 3600}
    assert "review holdSeconds 99999 clamped to 3600" in spec.warnings[0]


def test_pet_json_edit_is_picked_up(tmp_path):
    """A pet.json edit changes the spec and the rendered rows (the gateway revision: test_pet_payload.py)."""
    sheet = _sheet(tmp_path, 12, {"stateRows": ROWS_12}, frames={11: 3})
    assert behaviors.pet_spec(sheet).row_index("failed") == 5 and render.state_frame_counts(sheet)["failed"] == 6
    (tmp_path / "pet.json").write_text(json.dumps({"stateRows": ROWS_12, "behaviors": {"failed": {"row": "crying"}}}))
    assert behaviors.pet_spec(sheet).row_index("failed") == 11 and render.state_frame_counts(sheet)["failed"] == 3


def test_row_lists_and_handling_states(tmp_path):
    """Row lists rotate on desktop (first row in terminal); handling states (held/fall/land) map like any other."""
    rows = [*CODEX_STATE_ROWS, "thinking", "dancing", "juggling"]
    meta = {"stateRows": rows, "behaviors": {
        "run": {"rows": ["dancing", "nope", "juggling", "dancing"]},  # unknown dropped, dupes collapsed
        "review": {"rows": ["thinking"]},
        "waiting": {"rows": ["nope"]},  # nothing left → ignored
        "held": {"row": "juggling"},
        "fall": {"row": "jumping"},  # a standard row works too
        "land": {"row": "nope"},  # unmapped → idle row, never played on desktop
    }}
    spec = behaviors.pet_spec(_sheet(tmp_path, 12, meta))
    assert spec.payload() == {"rows": {"run": ["dancing", "juggling"], "review": ["thinking"], "held": ["juggling"], "fall": ["jumping"]}}
    assert spec.row_index("run") == 10 and spec.row_index("review") == 9 and spec.row_index("waiting") == 6
    assert spec.row_index("held") == 11 and spec.row_index("fall") == 4 and spec.row_index("land") == 0


def test_cleaning_reports_what_it_dropped_or_clamped(tmp_path):
    """Pet authors hear about ignored/clamped pet.json entries (``hermes pets doctor``) instead of silence."""
    rows = [*CODEX_STATE_ROWS, "sleeping"]
    meta = {"stateRows": rows, "behaviors": {
        "failed": {"row": "failed"},
        "run": {"rows": ["running", "nope"]},  # kept, minus the unknown row
        "review": {"row": "nope"},
        "Sleep": {"row": "sleeping"},
        "sleep": {"row": "sleeping", "afterSeconds": 3},
        "jump": {"rows": ["jumping"] * 17},
    }}
    spec = behaviors.pet_spec(_sheet(tmp_path, 10, meta))
    assert spec.warnings == (
        "behaviors: 4 ok, 2 ignored (unknown row 'nope'; unknown state 'Sleep'; "
        "sleep afterSeconds 3 clamped to 10; 'jump' lists 17 rows; first 16 used)",
    )

    short = behaviors.pet_spec(_sheet(tmp_path, 9, {"stateRows": [f"r{i}" for i in range(12)]}))
    assert short.rows == tuple(CODEX_STATE_ROWS)
    assert short.warnings == ("stateRows: names 12 rows but the sheet has 9; rows inferred from the sheet",)


def test_doctor_prints_pet_json_warnings(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from agent.pet import store
    from hermes_cli import pets

    sheet = _sheet(tmp_path, 9, {"behaviors": {"poke": {"row": "giggle"}}})
    pet = SimpleNamespace(slug="demo", spritesheet=sheet)
    monkeypatch.setattr(pets, "_pet_config", lambda: {"enabled": True, "slug": "demo"})
    monkeypatch.setattr(store, "installed_pets", lambda: [pet])
    monkeypatch.setattr(store, "resolve_active_pet", lambda _slug: pet)
    pets._cmd_doctor(None)
    assert "⚠ pet.json behaviors: 0 ok, 1 ignored (unknown row 'giggle')" in capsys.readouterr().out


def test_hostile_pet_json_never_raises(tmp_path):
    """Third-party pet.json: huge integers and deep nesting fall back, never crash a render."""
    rows = [*CODEX_STATE_ROWS, "sleeping"]
    huge = {"stateRows": rows, "behaviors": {"sleep": {"row": "sleeping", "afterSeconds": 10**400}}}
    spec = behaviors.pet_spec(_sheet(tmp_path, 10, huge))
    assert spec.sleep_after == 0 and "sleep" not in spec.overrides

    deep = tmp_path / "deep"
    deep.mkdir()
    sheet = _sheet(deep, 10, {})
    for text, error in (("[" * 100_000 + "]" * 100_000, "RecursionError"), ("{", "JSONDecodeError")):
        (deep / "pet.json").write_text(text, encoding="utf-8")
        spec = behaviors.pet_spec(sheet)
        assert spec.overrides == {} and spec.sleep_after == 0
        assert spec.warnings[0].startswith(f"unreadable ({error}")  # surfaced by `hermes pets doctor`
        assert behaviors.pet_spec(sheet) is spec  # cached, not re-read every tick


def test_hostile_spritesheet_header_never_raises(tmp_path):
    """A tiny PNG claiming ~192x1,000,000 px trips Pillow's decompression-bomb guard; fall back."""
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 192, 1_000_000, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b"")
    sheet = tmp_path / "spritesheet.png"
    sheet.write_bytes(png)
    spec = behaviors.pet_spec(sheet)
    assert spec.overrides == {} and spec.sleep_after == 0


@pytest.mark.platforms("posix")  # FIFOs + SIGALRM
def test_special_file_pet_json_is_not_read(tmp_path):
    """A FIFO pet.json (or a symlink to a device) must not block or be slurped: regular files only."""
    sheet = _sheet(tmp_path, 10)
    os.mkfifo(tmp_path / "pet.json")
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("pet_spec blocked on a FIFO pet.json"))
    signal.alarm(5)
    try:
        spec = behaviors.pet_spec(sheet)
    finally:
        signal.alarm(0)
    assert spec.overrides == {}
