"""pet.json ``stateRows`` / ``behaviors``: extra rows + state remaps, opt-in sleep."""

import json
import os
import signal
import time

import pytest

from agent.pet import behaviors, render
from agent.pet.constants import CODEX_STATE_ROWS, FRAME_H, FRAME_W, PetState
from agent.pet.state import derive_pet_state

PIL = pytest.importorskip("PIL")
from PIL import Image

EXTRA = ["sleeping", "smiling", "crying"]


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
    render._raw_frames.cache_clear()
    return path


def test_no_pet_json_is_unchanged(tmp_path):
    spec = behaviors.pet_spec(_sheet(tmp_path, 9))
    assert spec.rows == tuple(CODEX_STATE_ROWS)
    assert spec.overrides == {} and spec.sleep_after == 0 and spec.payload() == {}
    assert spec.row_index("failed") == 5 and spec.row_index("wave") == 3
    assert spec.row_index("sleep") == 0  # opt-in state renders idle


def test_declared_rows_and_overrides(tmp_path):
    meta = {
        "stateRows": [*CODEX_STATE_ROWS, *EXTRA],
        "behaviors": {
            "failed": {"row": "crying"},
            "jump": {"row": "smiling"},
            "sleep": {"row": "sleeping", "afterSeconds": 300},
            "review": {"row": "nope"},  # unknown row → ignored
            "bogus": {"row": "idle"},  # unknown state → ignored
        },
    }
    sheet = _sheet(tmp_path, 12, meta, frames={11: 3})
    spec = behaviors.pet_spec(sheet)
    assert spec.rows[-3:] == tuple(EXTRA)
    assert spec.row_index("failed") == 11 and spec.row_index("jump") == 10 and spec.row_index("sleep") == 9
    assert spec.row_index("review") == 8  # standard row kept
    assert spec.payload() == {"rows": {"failed": "crying", "jump": "smiling", "sleep": "sleeping"}, "sleepAfterSeconds": 300}
    counts = render.state_frame_counts(sheet)
    assert counts["failed"] == 3  # frames come from the crying row


def test_bad_state_rows_fall_back_to_inferred(tmp_path):
    spec = behaviors.pet_spec(_sheet(tmp_path, 9, {"stateRows": ["a"] * 9, "behaviors": "x"}))
    assert spec.rows == tuple(CODEX_STATE_ROWS) and spec.overrides == {}
    too_many = behaviors.pet_spec(_sheet(tmp_path, 9, {"stateRows": [f"r{i}" for i in range(12)]}))
    assert too_many.rows == tuple(CODEX_STATE_ROWS)


def test_sleep_delay_is_clamped_and_needs_a_row(tmp_path):
    rows = [*CODEX_STATE_ROWS, "sleeping"]
    assert behaviors.pet_spec(_sheet(tmp_path, 10, {"stateRows": rows, "behaviors": {"sleep": {"row": "sleeping", "afterSeconds": 1}}})).sleep_after == 10
    assert behaviors.pet_spec(_sheet(tmp_path, 10, {"stateRows": rows, "behaviors": {"sleep": {"row": "sleeping"}}})).sleep_after == 0
    assert behaviors.pet_spec(_sheet(tmp_path, 9, {"behaviors": {"sleep": {"row": "sleeping", "afterSeconds": 60}}})).sleep_after == 0


def test_pet_json_edit_is_picked_up(tmp_path):
    sheet = _sheet(tmp_path, 12, {"stateRows": [*CODEX_STATE_ROWS, *EXTRA]})
    assert behaviors.pet_spec(sheet).row_index("failed") == 5
    (tmp_path / "pet.json").write_text(json.dumps({"stateRows": [*CODEX_STATE_ROWS, *EXTRA], "behaviors": {"failed": {"row": "crying"}}}))
    os.utime(tmp_path / "pet.json", ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    assert behaviors.pet_spec(sheet).row_index("failed") == 11


def test_sleepy_is_lowest_priority():
    assert derive_pet_state(sleepy=True) is PetState.SLEEP
    assert derive_pet_state(sleepy=True, busy=True) is PetState.RUN
    assert derive_pet_state(sleepy=True, awaiting_input=True) is PetState.WAITING
    assert derive_pet_state() is PetState.IDLE


def test_row_lists_rotate_on_desktop_first_row_in_terminal(tmp_path):
    rows = [*CODEX_STATE_ROWS, "thinking", "dancing", "juggling"]
    meta = {"stateRows": rows, "behaviors": {
        "run": {"rows": ["dancing", "nope", "juggling", "dancing"]},  # unknown dropped, dupes collapsed
        "review": {"rows": ["thinking"]},
        "waiting": {"rows": ["nope"]},  # nothing left → ignored
    }}
    spec = behaviors.pet_spec(_sheet(tmp_path, 12, meta))
    assert spec.overrides == {"run": ("dancing", "juggling"), "review": ("thinking",)}
    assert spec.row_index("run") == 10 and spec.row_index("review") == 9 and spec.row_index("waiting") == 6
    assert spec.payload() == {"rows": {"run": ["dancing", "juggling"], "review": "thinking"}}


def test_hostile_pet_json_never_raises(tmp_path):
    """Third-party pet.json: huge integers and deep nesting fall back, never crash a render."""
    rows = [*CODEX_STATE_ROWS, "sleeping"]
    huge = {"stateRows": rows, "behaviors": {"sleep": {"row": "sleeping", "afterSeconds": 10**400}}}
    spec = behaviors.pet_spec(_sheet(tmp_path, 10, huge))
    assert spec.sleep_after == 0 and "sleep" not in spec.overrides

    deep = tmp_path / "deep"
    deep.mkdir()
    sheet = _sheet(deep, 10, {})
    (deep / "pet.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    spec = behaviors.pet_spec(sheet)
    assert spec.overrides == {} and spec.sleep_after == 0


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
