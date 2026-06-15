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
    # awaiting input rests, doesn't run
    assert state.derive_pet_state(awaiting_input=True) is PetState.IDLE


def test_derive_priority_order():
    # error beats everything
    assert state.derive_pet_state(error=True, celebrate=True, busy=True) is PetState.FAILED
    # celebrate beats completion/tool
    assert state.derive_pet_state(celebrate=True, just_completed=True, tool_running=True) is PetState.JUMP
    # completion beats tool/reasoning
    assert state.derive_pet_state(just_completed=True, tool_running=True) is PetState.WAVE
    # tool beats reasoning
    assert state.derive_pet_state(tool_running=True, reasoning=True) is PetState.RUN
    # reasoning beats bare-busy
    assert state.derive_pet_state(reasoning=True, busy=True) is PetState.REVIEW
    # bare busy runs
    assert state.derive_pet_state(busy=True) is PetState.RUN


def test_state_row_index_maps_to_taxonomy():
    # row index must equal position in STATE_ROWS for every driveable state
    for st in PetState:
        assert constants.STATE_ROWS[constants.state_row_index(st)] == st.value
    # unknown row names clamp to idle (row 0), never raise
    assert constants.state_row_index("nonsense") == 0


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
    r = render.PetRenderer(str(sprite), mode="kitty", scale=0.4, unicode_cols=18)
    payload = r.kitty_payload("run", cols=18, image_id=image_id)
    assert payload is not None
    assert payload["cols"] == 18
    assert payload["rows"] == r.kitty_cell_rows(18) >= 1
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


def test_kitty_payload_none_when_no_frames(tmp_path):
    r = render.PetRenderer(str(tmp_path / "missing.webp"), mode="kitty")
    assert r.kitty_payload("idle", cols=18, image_id=1) is None


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
