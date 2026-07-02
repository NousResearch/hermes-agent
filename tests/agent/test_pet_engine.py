"""Tests for the petdex pet engine (agent/pet/*).

Behavior/invariant focused — no network, no live manifest. A tiny synthetic
spritesheet is generated with Pillow so render paths exercise real decode
without depending on a downloaded pet.
"""

from __future__ import annotations

import io

import pytest

from agent.pet import constants, render, state, store
from agent.pet.constants import FRAME_H, FRAME_W, PetState


# ─────────────────────────────────────────────────────────────────────────
# state mapping — priority invariants
# ─────────────────────────────────────────────────────────────────────────

def test_derive_idle_default():
    assert state.derive_pet_state() is PetState.IDLE
    # awaiting input uses the dedicated waiting row when available.
    assert state.derive_pet_state(awaiting_input=True) is PetState.WAITING


def test_derive_priority_order():
    # error beats everything
    assert state.derive_pet_state(error=True, celebrate=True, busy=True) is PetState.FAILED
    # celebrate beats completion/tool
    assert state.derive_pet_state(celebrate=True, just_completed=True, tool_running=True) is PetState.JUMP
    # completion beats waiting/tool
    assert state.derive_pet_state(just_completed=True, awaiting_input=True) is PetState.WAVE
    # waiting (blocked on the user) outranks the in-flight signals — a clarify
    # mid-turn pauses on you even though a tool is technically still open.
    assert state.derive_pet_state(awaiting_input=True, tool_running=True, busy=True) is PetState.WAITING
    # tool beats reasoning
    assert state.derive_pet_state(tool_running=True, reasoning=True) is PetState.RUN
    # reasoning beats bare-busy
    assert state.derive_pet_state(reasoning=True, busy=True) is PetState.REVIEW
    # bare busy runs
    assert state.derive_pet_state(busy=True) is PetState.RUN


def test_todos_all_done():
    # empty / falsy → not done (no plan to celebrate)
    assert state.todos_all_done(None) is False
    assert state.todos_all_done([]) is False
    # any open item → not done
    assert state.todos_all_done([{"status": "completed"}, {"status": "pending"}]) is False
    assert state.todos_all_done([{"status": "in_progress"}]) is False
    # every item terminal → done (completed and/or cancelled)
    assert state.todos_all_done([{"status": "completed"}, {"status": "cancelled"}]) is True

    # objects with a .status attr work too (mirrors dict + attr access)
    class _T:
        def __init__(self, status):
            self.status = status

    assert state.todos_all_done([_T("completed")]) is True
    assert state.todos_all_done([_T("completed"), _T("pending")]) is False


def test_state_row_index_maps_to_supported_atlas_taxonomies():
    # Current Petdex sheets are 8 columns x 9 rows.
    assert constants.state_row_index(PetState.IDLE, 9) == 0
    assert constants.state_row_index(PetState.WAVE, 9) == 3
    assert constants.state_row_index(PetState.JUMP, 9) == 4
    assert constants.state_row_index(PetState.FAILED, 9) == 5
    assert constants.state_row_index(PetState.WAITING, 9) == 6
    assert constants.state_row_index(PetState.RUN, 9) == 7
    assert constants.state_row_index(PetState.REVIEW, 9) == 8

    # Legacy Hermes/petdex sheets were 8 rows with Hermes state names packed in
    # order. Keep those readable instead of forcing old installs through the
    # newer Codex taxonomy.
    assert constants.state_row_index(PetState.WAVE, 8) == 1
    assert constants.state_row_index(PetState.RUN, 8) == 2
    assert constants.state_row_index(PetState.FAILED, 8) == 3
    assert constants.state_row_index(PetState.REVIEW, 8) == 4
    assert constants.state_row_index(PetState.JUMP, 8) == 5
    assert constants.state_row_index(PetState.WAITING, 8) == 0

    # Alias rows resolve as expected.
    assert constants.state_row_index("wave", 9) == constants.state_row_index("waving", 9) == 3
    assert constants.state_row_index("jump", 9) == constants.state_row_index("jumping", 9) == 4
    assert constants.state_row_index("run", 9) == constants.state_row_index("running", 9) == 7

    # unknown row names clamp to idle (row 0), never raise
    assert constants.state_row_index("nonsense") == 0


def test_cols_for_scale_is_monotonic_and_floored():
    # scale is the master size knob: smaller scale never yields more columns,
    # and half-blocks clamp to a legibility floor rather than devolving to mush.
    sizes = [constants.cols_for_scale(s) for s in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5)]
    assert sizes == sorted(sizes)
    assert all(c >= constants.UNICODE_MIN_COLS for c in sizes)
    # tiny scales pin to the floor; large scales grow past it.
    assert constants.cols_for_scale(0.05) == constants.UNICODE_MIN_COLS
    assert constants.cols_for_scale(0.33) == constants.UNICODE_MIN_COLS
    assert constants.cols_for_scale(2.0) > constants.UNICODE_MIN_COLS


def test_resolve_cols_override_else_scale():
    # 0 / falsy → derive from scale; a positive int hard-overrides scale.
    assert constants.resolve_cols(0.7, 0) == constants.cols_for_scale(0.7)
    assert constants.resolve_cols(0.7, None) == constants.cols_for_scale(0.7)
    assert constants.resolve_cols(2.0, 12) == 12
    assert constants.resolve_cols(0.1, -5) == constants.cols_for_scale(0.1)


# ─────────────────────────────────────────────────────────────────────────
# synthetic spritesheet fixture
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def boba_like(tmp_path, monkeypatch):
    """Install a synthetic 8-col × 9-row pet into a temp HERMES_HOME."""
    from PIL import Image

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    cols, rows = 8, 9
    sheet = Image.new("RGBA", (FRAME_W * cols, FRAME_H * rows), (0, 0, 0, 0))
    # paint each row a distinct opaque color so frames are non-empty
    for r in range(rows):
        color = (20 + r * 25, 60, 120, 255)
        for c in range(cols):
            block = Image.new("RGBA", (FRAME_W, FRAME_H), color)
            sheet.paste(block, (c * FRAME_W, r * FRAME_H))

    pet_dir = store.pets_dir() / "boba"
    pet_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(pet_dir / "spritesheet.webp")
    (pet_dir / "pet.json").write_text(
        '{"id":"boba","displayName":"Boba","description":"d","spritesheetPath":"spritesheet.webp"}'
    )
    return pet_dir


def test_store_install_resolution(boba_like):
    pets = store.installed_pets()
    assert [p.slug for p in pets] == ["boba"]
    assert store.installed_pets()[0].exists

    # configured slug wins when installed
    assert store.resolve_active_pet("boba").slug == "boba"
    # bogus slug falls back to first installed
    assert store.resolve_active_pet("does-not-exist").slug == "boba"
    # display metadata flows from pet.json
    assert store.load_pet("boba").display_name == "Boba"


def test_store_remove(boba_like):
    assert store.remove_pet("boba") is True
    assert store.installed_pets() == []
    assert store.remove_pet("boba") is False  # idempotent


# ─────────────────────────────────────────────────────────────────────────
# render — decode + every encoder produces output
# ─────────────────────────────────────────────────────────────────────────

def test_renderer_decodes_frames(boba_like):
    sprite = store.load_pet("boba").spritesheet
    r = render.PetRenderer(str(sprite), mode="unicode", scale=0.5, unicode_cols=12)
    assert r.available
    # standard sheet yields FRAMES_PER_STATE frames per state
    assert r.frame_count("idle") == constants.FRAMES_PER_STATE
    assert r.frame_count(PetState.RUN) == constants.FRAMES_PER_STATE


def test_trims_trailing_blank_frames(tmp_path):
    """Ragged state rows (real frames + transparent padding) trim to real count.

    petdex sheets are left-packed: a state with fewer than FRAMES_PER_STATE real
    frames pads the trailing columns transparent. Stepping into one flashes the
    pet blank, so the engine must stop the row at the first gap.
    """
    from PIL import Image

    cols, rows = 8, 9
    sheet = Image.new("RGBA", (FRAME_W * cols, FRAME_H * rows), (0, 0, 0, 0))
    # row index -> number of real (opaque) frames; the rest stay transparent.
    # Codex row taxonomy: idle, running-right, running-left, wave, jump, failed,
    # waiting, run, review.
    real = {0: 6, 3: 4, 4: 5, 5: 8, 7: 6, 8: 5}
    for r, k in real.items():
        for c in range(k):
            block = Image.new("RGBA", (FRAME_W, FRAME_H), (200, 80, 80, 255))
            sheet.paste(block, (c * FRAME_W, r * FRAME_H))
    sprite = tmp_path / "ragged.webp"
    sheet.save(sprite)

    r = render.PetRenderer(str(sprite), mode="unicode", scale=0.5)
    # Full rows cap at FRAMES_PER_STATE; ragged rows trim to their real count.
    assert r.frame_count("idle") == constants.FRAMES_PER_STATE
    assert r.frame_count("run") == constants.FRAMES_PER_STATE
    assert r.frame_count("wave") == 4
    assert r.frame_count("jump") == 5
    assert r.frame_count("failed") == constants.FRAMES_PER_STATE
    assert r.frame_count("review") == 5

    # Every stepped frame is non-empty — no blank flash for the trimmed states.
    for state in ("wave", "jump", "review"):
        for i in range(r.frame_count(state)):
            assert r.frame(state, i), f"{state}[{i}] rendered blank"

    counts = render.state_frame_counts(str(sprite))
    assert counts == {
        "idle": 6,
        "wave": 4,
        "run": 6,
        "failed": 6,
        "review": 5,
        "jump": 5,
        "waiting": 0,
    }


@pytest.mark.parametrize("mode", ["unicode", "kitty", "iterm", "sixel"])
def test_every_encoder_emits(boba_like, mode):
    sprite = store.load_pet("boba").spritesheet
    r = render.PetRenderer(str(sprite), mode=mode, scale=0.4)
    frame = r.frame("run", 1)
    assert isinstance(frame, str) and frame, f"{mode} produced no frame"
    if mode == "unicode":
        assert "\x1b[" in frame  # has color escapes
    elif mode == "kitty":
        assert frame.startswith("\x1b_G")
    elif mode == "iterm":
        assert frame.startswith("\x1b]1337;File=")
    elif mode == "sixel":
        assert frame.startswith("\x1bP")


def test_frame_index_wraps(boba_like):
    sprite = store.load_pet("boba").spritesheet
    r = render.PetRenderer(str(sprite), mode="unicode", scale=0.4)
    # index beyond count wraps rather than indexing out of range
    assert r.frame("idle", 999) == r.frame("idle", 999 % r.frame_count("idle"))


def test_cells_grid_shape(boba_like):
    sprite = store.load_pet("boba").spritesheet
    r = render.PetRenderer(str(sprite), mode="unicode", scale=0.4, unicode_cols=14)
    grid = r.cells("run", 0, cols=14)
    assert grid, "no cells produced"
    # every row is the requested width; every cell is (top, bottom) RGBA pairs
    assert all(len(row) == 14 for row in grid)
    (top, bottom) = grid[0][0]
    assert len(top) == 4 and len(bottom) == 4
    # missing-sheet renderer yields no cells, never raises
    assert render.PetRenderer(str(sprite.parent / "missing.webp"), mode="unicode").cells("idle", 0) == []


# ─────────────────────────────────────────────────────────────────────────
# render — kitty Unicode placeholders (TUI graphics path)
# ─────────────────────────────────────────────────────────────────────────

def test_kitty_image_id_stable_bounded_nonzero():
    # Deterministic per slug so re-renders reuse the same terminal-side image,
    # and always a valid 24-bit-encodable, non-zero id.
    a = render.kitty_image_id("boba")
    assert a == render.kitty_image_id("boba")
    assert 1 <= a <= 0x7FFF


def test_kitty_color_hex_decodes_to_id():
    # The placeholder's foreground color IS the image id (24-bit). The terminal
    # reconstructs id = (r<<16)|(g<<8)|b, so the hex must round-trip.
    for slug in ("boba", "clawd", "pixel-fox"):
        image_id = render.kitty_image_id(slug)
        h = render.kitty_color_hex(image_id)
        assert h.startswith("#") and len(h) == 7
        assert int(h[1:], 16) == image_id


def test_kitty_placeholder_rows_grid_contract():
    cols, rows = 18, 10
    grid = render.kitty_placeholder_rows(cols, rows)
    assert len(grid) == rows
    placeholder = "\U0010eeee"
    for r, row in enumerate(grid):
        # Each line is exactly `cols` placeholder cells (combining diacritics
        # are zero-width, so this is the rendered width Ink must measure).
        assert row.count(placeholder) == cols
        # First cell carries this row's diacritic; the rest inherit row + col.
        assert row.startswith(placeholder + chr(render._ROWCOL_DIACRITICS[r]))


def test_kitty_payload_structure(boba_like):
    sprite = store.load_pet("boba").spritesheet
    image_id = render.kitty_image_id("boba")
    scale = 0.4
    r = render.PetRenderer(str(sprite), mode="kitty", scale=scale, unicode_cols=18)
    payload = r.kitty_payload("run", image_id=image_id)
    assert payload is not None
    # Geometry is driven by the scaled/cropped sprite, not unicode_cols.
    assert payload["cols"] >= 1 and payload["rows"] >= 1
    assert payload["cols"] < 18  # 0.4 scale is much smaller than a pinned 18-col box
    # placeholder grid matches the requested geometry
    assert len(payload["placeholder"]) == payload["rows"]
    # one transmit escape per animation frame, each a kitty virtual placement
    assert len(payload["frames"]) == r.frame_count("run")
    for esc in payload["frames"]:
        assert esc.startswith("\x1b_G")
        assert esc.endswith("\x1b\\")
        assert f"i={image_id}" in esc
        assert "a=T" in esc and "U=1" in esc
        assert f"c={payload['cols']}" in esc and f"r={payload['rows']}" in esc


def test_kitty_payload_snaps_to_whole_cells(boba_like):
    # The transmitted frame must be an exact multiple of the cell box so kitty
    # doesn't round up + clip the bottom row / letterbox a blank row (the
    # "clipped feet" bug). cols/rows are derived as pixels // cell, so a snapped
    # frame round-trips exactly. Regression for ratatui-image #57.
    sprite = store.load_pet("boba").spritesheet
    r = render.PetRenderer(str(sprite), mode="kitty", scale=0.6, unicode_cols=18)
    frames = render._snap_frames_to_cell_grid(
        render._crop_frames_to_alpha_union(r._frames("run"))
    )
    for f in frames:
        assert f.width % render._CELL_W == 0
        assert f.height % render._CELL_H == 0


def test_kitty_payload_none_when_no_frames(tmp_path):
    r = render.PetRenderer(str(tmp_path / "missing.webp"), mode="kitty")
    assert r.kitty_payload("idle", image_id=1) is None


def test_off_mode_and_missing_sheet_degrade(tmp_path):
    # off mode never emits
    r_off = render.PetRenderer(str(tmp_path / "nope.webp"), mode="off")
    assert r_off.frame("idle", 0) == ""
    # missing sheet → not available, empty frames, no raise
    r_missing = render.PetRenderer(str(tmp_path / "nope.webp"), mode="unicode")
    assert not r_missing.available
    assert r_missing.frame("idle", 0) == ""


def test_resolve_mode_non_tty_is_off():
    # a non-tty stream forces 'off' regardless of configured mode
    assert render.resolve_mode("kitty", stream=io.StringIO()) == "off"
    assert render.resolve_mode("auto", stream=io.StringIO()) == "off"


def test_detect_terminal_graphics_env(monkeypatch):
    for key in ("KITTY_WINDOW_ID", "TERM_PROGRAM", "ITERM_SESSION_ID", "WEZTERM_PANE", "TERM"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    assert render.detect_terminal_graphics() == "kitty"
    monkeypatch.delenv("KITTY_WINDOW_ID")

    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert render.detect_terminal_graphics() == "iterm"
    monkeypatch.delenv("TERM_PROGRAM")

    monkeypatch.setenv("TERM", "xterm-256color")
    assert render.detect_terminal_graphics() == "unicode"


def test_vscode_terminal_ignores_leaked_graphics_env(monkeypatch):
    # The VS Code / Cursor integrated terminal can't show inline images by
    # default, yet inherits ITERM_SESSION_ID/KITTY_WINDOW_ID when launched from
    # those terminals. TERM_PROGRAM=vscode must win → unicode, never a protocol
    # whose escapes the embedded terminal would silently drop.
    for key in ("KITTY_WINDOW_ID", "TERM_PROGRAM", "ITERM_SESSION_ID", "WEZTERM_PANE", "TERM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert render.detect_terminal_graphics() == "unicode"
    for leaked in ("ITERM_SESSION_ID", "KITTY_WINDOW_ID", "WEZTERM_PANE"):
        monkeypatch.setenv(leaked, "1")
        assert render.detect_terminal_graphics() == "unicode"
        monkeypatch.delenv(leaked)
