"""The base-CLI petdex pane: reactive half-block sprite above the prompt.

Mirrors the TUI's PetPane. The methods are tested in isolation via __new__ so
we don't pay the full HermesCLI.__init__ cost; a synthetic spritesheet exercises
the real engine decode + half-block fragment building.
"""

from __future__ import annotations

import threading

import pytest

from agent.pet import store
from agent.pet.constants import FRAME_H, FRAME_W
from agent.pet.render import PetRenderer
from cli import HermesCLI


@pytest.fixture
def boba_like(tmp_path, monkeypatch):
    """Install a synthetic pet into a temp HERMES_HOME and return its slug."""
    from PIL import Image

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    cols, rows = 8, 9
    sheet = Image.new("RGBA", (FRAME_W * cols, FRAME_H * rows), (0, 0, 0, 0))
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
    return "boba"


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._app = None
    cli_obj._pet_lock = threading.Lock()
    cli_obj._pet_enabled = False
    cli_obj._pet_renderer = None
    cli_obj._pet_slug = ""
    cli_obj._pet_cols = 18
    cli_obj._pet_scale = 0.7
    cli_obj._pet_frames_cache = {}
    cli_obj._pet_frame_idx = 0
    cli_obj._agent_running = False
    # Transient-beat + reasoning state (set by HermesCLI.__init__ in production).
    cli_obj._pet_event = ""
    cli_obj._pet_event_until = 0.0
    cli_obj._pet_reasoning = False
    # Blocking-modal state — a live one maps the pet to `waiting`.
    cli_obj._approval_state = None
    cli_obj._clarify_state = None
    cli_obj._sudo_state = None
    cli_obj._secret_state = None
    cli_obj._slash_confirm_state = None
    return cli_obj


def test_pet_state_tracks_agent_running():
    cli_obj = _make_cli()
    assert cli_obj._derive_pet_state() == "idle"
    cli_obj._agent_running = True
    assert cli_obj._derive_pet_state() == "run"


def test_pet_state_waits_on_a_blocking_modal():
    # A live clarify/approval pauses the agent on the user → `waiting`, even
    # while the turn is technically still running.
    cli_obj = _make_cli()
    cli_obj._agent_running = True
    cli_obj._clarify_state = {"question": "?"}
    assert cli_obj._derive_pet_state() == "waiting"


def test_pet_pane_collapsed_when_disabled():
    # No renderer resolved → the window reports zero height and no fragments,
    # so it's invisible for users without a pet.
    cli_obj = _make_cli()
    assert cli_obj._pet_widget_height() == 0
    assert cli_obj._pet_fragments() == []


def test_pet_fragments_render_half_blocks(boba_like):
    cli_obj = _make_cli()
    cli_obj._pet_renderer = PetRenderer(
        str(store.load_pet("boba").spritesheet), mode="unicode", scale=0.4, unicode_cols=14
    )
    cli_obj._pet_cols = 14
    cli_obj._pet_enabled = True

    height = cli_obj._pet_widget_height()
    assert height > 0

    frags = cli_obj._pet_fragments()
    assert frags, "expected fragments for an enabled pet"
    # Each fragment is a (style, text) pair; glyphs are half-blocks or blanks.
    glyphs = {text for _, text in frags}
    assert glyphs <= {"▀", "▄", " ", "\n"}
    # Opaque cells carry a truecolor foreground style.
    assert any(text == "▀" and "fg:#" in style for style, text in frags)
    # Row count in the fragment stream matches the reported window height.
    assert sum(1 for _, text in frags if text == "\n") == height - 1


def test_pet_resolve_config_enables_and_disables(boba_like):
    from hermes_cli.config import load_config, save_config

    cli_obj = _make_cli()

    cfg = load_config()
    cfg.setdefault("display", {}).setdefault("pet", {})
    cfg["display"]["pet"].update({"enabled": True, "slug": "boba"})
    save_config(cfg)

    cli_obj._pet_resolve_config()
    assert cli_obj._pet_enabled is True
    assert cli_obj._pet_renderer is not None
    assert cli_obj._pet_slug == "boba"

    cfg["display"]["pet"]["enabled"] = False
    save_config(cfg)
    cli_obj._pet_resolve_config()
    assert cli_obj._pet_enabled is False
    assert cli_obj._pet_renderer is None
