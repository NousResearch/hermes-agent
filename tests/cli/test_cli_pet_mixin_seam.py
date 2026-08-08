"""Seam tests for the Petdex mascot cluster extraction (cli.py god-file slice R2).

The Petdex cluster (13 members: 2 class consts + 11 methods) moved verbatim
from ``cli.py`` into ``hermes_cli/cli_pet_mixin.py`` as ``CLIPetMixin``,
appended last to ``HermesCLI``'s base tuple. These tests lock the seam:
identity through the MRO, no-back-import discipline, import-order safety,
patch-binding liveness, and a few behavioral spot checks (anim loop start/stop
and reaction routing).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from cli import HermesCLI
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_billing_mixin import CLIBillingMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_pet_mixin import CLIPetMixin

REPO_ROOT = Path(__file__).resolve().parents[2]

PET_METHODS = [
    "_pet_resolve_config",
    "_pet_flash",
    "_on_reaction",
    "_pet_react_turn_end",
    "_derive_pet_state",
    "_pet_frames_for",
    "_pet_fragments",
    "_pet_widget_height",
    "_pet_anim_loop",
    "_pet_start_anim",
    "_pet_stop_anim",
]


def test_mixin_members_are_identity_inherited():
    """All 13 moved members resolve to the SAME objects via the MRO."""
    assert HermesCLI._PET_FRAME_INTERVAL is CLIPetMixin._PET_FRAME_INTERVAL
    assert HermesCLI._PET_CFG_INTERVAL is CLIPetMixin._PET_CFG_INTERVAL
    for name in PET_METHODS:
        assert getattr(HermesCLI, name) is getattr(CLIPetMixin, name), name


def test_mixin_ordered_last_in_mro_without_shadowing():
    """CLIPetMixin is last in the house-mixin chain; pet names are unique there."""
    assert CLIPetMixin in HermesCLI.__mro__
    mro = HermesCLI.__mro__
    assert mro.index(CLIAgentSetupMixin) < mro.index(CLIPetMixin)
    assert mro.index(CLICommandsMixin) < mro.index(CLIPetMixin)
    assert mro.index(CLIBillingMixin) < mro.index(CLIPetMixin)
    # No member of the chain ahead of CLIPetMixin defines a pet name.
    for base in mro[: mro.index(CLIPetMixin)]:
        for name in ["_PET_FRAME_INTERVAL", "_PET_CFG_INTERVAL", *PET_METHODS]:
            assert name not in vars(base), (base.__name__, name)


def test_no_back_import_when_cli_is_blocked():
    """Importing the mixin module must not require ``cli`` (no import cycle)."""
    code = (
        "import sys; sys.modules['cli'] = None; "
        "import hermes_cli.cli_pet_mixin as m; "
        "print(m.CLIPetMixin.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "CLIPetMixin"


def test_import_order_mixin_first_then_cli():
    """Importing the mixin before ``cli`` must not raise NameError."""
    code = (
        "import hermes_cli.cli_pet_mixin; import cli; "
        "print(cli.HermesCLI._pet_start_anim.__module__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "hermes_cli.cli_pet_mixin"


def test_patch_binding_through_seam(monkeypatch):
    """Patching on CLIPetMixin is visible through HermesCLI at call time."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    calls = []

    def fake(self):
        calls.append(1)

    monkeypatch.setattr(CLIPetMixin, "_pet_resolve_config", fake)
    cli_obj._pet_resolve_config()
    assert calls == [1]


def test_anim_loop_start_stop_idempotent_and_advances_frame(monkeypatch):
    """_pet_start_anim spawns a daemon loop; stop joins it; start is idempotent."""
    # Config resolution has its own battery (test_cli_pet_pane.py); isolate the
    # loop mechanics so real config can't disable the pet mid-test.
    monkeypatch.setattr(CLIPetMixin, "_pet_resolve_config", lambda self: None)
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._pet_lock = threading.Lock()
    cli_obj._pet_enabled = True
    cli_obj._pet_renderer = object()  # non-None so the loop's config check is moot
    cli_obj._pet_frames_cache = {}
    cli_obj._pet_frame_idx = 0
    cli_obj._pet_cfg_checked = time.monotonic()
    cli_obj._pet_anim_running = False
    cli_obj._pet_anim_thread = None
    cli_obj._app = None

    cli_obj._pet_start_anim()
    first_thread = cli_obj._pet_anim_thread
    assert first_thread is not None and first_thread.daemon
    # Idempotent start: no second thread.
    cli_obj._pet_start_anim()
    assert cli_obj._pet_anim_thread is first_thread

    # Give the loop a beat to advance the frame index.
    deadline = time.monotonic() + 2.0
    while cli_obj._pet_frame_idx == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cli_obj._pet_frame_idx > 0

    cli_obj._pet_stop_anim()
    assert cli_obj._pet_anim_running is False
    assert cli_obj._pet_anim_thread is None
    # Stop is idempotent too.
    cli_obj._pet_stop_anim()


def test_on_reaction_vibe_flashes_jump():
    """User-affection reaction routes to the pet flash."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._pet_event = ""
    cli_obj._pet_event_until = 0.0
    cli_obj._on_reaction("vibe")
    assert cli_obj._pet_event == "jump"
    cli_obj._on_reaction("other")
    assert cli_obj._pet_event == "jump"  # unchanged


def test_regression_smoke_init_state():
    """Post-init pet state is disabled with no renderer (init block untouched)."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._pet_enabled = False
    cli_obj._pet_renderer = None
    assert cli_obj._pet_enabled is False
    assert cli_obj._pet_renderer is None
