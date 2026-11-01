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






# ─────────────────────────────────────────────────────────────────────────
# render — decode + every encoder produces output
# ─────────────────────────────────────────────────────────────────────────



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












def _clear_graphics_env(monkeypatch):
    for key in ("KITTY_WINDOW_ID", "TERM_PROGRAM", "ITERM_SESSION_ID", "WEZTERM_PANE", "TERM"):
        monkeypatch.delenv(key, raising=False)


def test_vscode_terminal_ignores_leaked_graphics_env(monkeypatch):
    # The VS Code / Cursor integrated terminal can't show inline images by
    # default, yet inherits ITERM_SESSION_ID/KITTY_WINDOW_ID when launched from
    # those terminals. TERM_PROGRAM=vscode must win → unicode, never a protocol
    # whose escapes the embedded terminal would silently drop.
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert render.detect_terminal_graphics() == "unicode"
    for leaked in ("ITERM_SESSION_ID", "KITTY_WINDOW_ID", "WEZTERM_PANE"):
        monkeypatch.setenv(leaked, "1")
        assert render.detect_terminal_graphics() == "unicode"
        monkeypatch.delenv(leaked)
        assert render.supports_kitty_placeholders() is False


def test_kitty_placeholders_on_kitty_and_ghostty(monkeypatch):
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    assert render.supports_kitty_placeholders() is True

    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-ghostty")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert render.supports_kitty_placeholders() is True


def test_wezterm_is_not_placeholder_capable(monkeypatch):
    # WezTerm speaks kitty APC but not U+10EEEE. Placeholders would tofu.
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    assert render.detect_terminal_graphics() == "kitty"
    assert render.supports_kitty_placeholders() is False

    monkeypatch.setenv("WEZTERM_PANE", "1")
    assert render.supports_kitty_placeholders() is False
