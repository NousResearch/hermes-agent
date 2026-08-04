"""`hermes skin set` — deterministic single-color tweak of the active skin.

The whole point is that changing one token never disturbs the rest of the look
(background especially), which hand-authoring kept getting wrong.
"""

import os

import pytest
import yaml

from hermes_cli import skin_cmd
from hermes_constants import get_hermes_home


def _skins():
    d = get_hermes_home() / "skins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _activate(name: str) -> None:
    (get_hermes_home() / "config.yaml").write_text(f"display:\n  skin: {name}\n", encoding="utf-8")


def test_set_edits_active_user_skin_in_place_preserving_everything_else():
    (_skins() / "oasis.yaml").write_text(
        'name: oasis\ncolors:\n  background: "#08201f"\n  banner_title: "#f2dfb3"\n', encoding="utf-8"
    )
    _activate("oasis")

    assert skin_cmd._skin_set("ui_tool", "#00FFFF", None) == 0

    data = yaml.safe_load((_skins() / "oasis.yaml").read_text())
    assert data["colors"]["ui_tool"] == "#00FFFF"
    assert data["colors"]["background"] == "#08201f"  # untouched — the whole point
    assert data["colors"]["banner_title"] == "#f2dfb3"
    assert data["name"] == "oasis"  # no rename, no fork
    assert not (_skins() / "oasis-custom.yaml").exists()


def test_set_forks_a_builtin_without_inventing_a_background():
    _activate("default")  # a built-in — no file

    assert skin_cmd._skin_set("ui_tool", "#00FFFF", None) == 0

    fork = _skins() / "default-custom.yaml"
    assert fork.exists()
    data = yaml.safe_load(fork.read_text())
    assert data["colors"]["ui_tool"] == "#00FFFF"
    # default has no background, so the fork must not invent one (terminal stays put).
    assert "background" not in data["colors"]
    # full palette carried over, and it became active.
    assert data["colors"].get("banner_title")
    assert (get_hermes_home() / "config.yaml").read_text().find("default-custom") != -1


def test_set_rejects_non_hex():
    _activate("default")
    assert skin_cmd._skin_set("ui_tool", "teal", None) == 1


def test_set_persists_the_skin_durably():
    """The palette must reach disk before ``skin set`` returns.

    ``write_text`` truncates and writes without ever calling ``fsync``, so a
    power loss or container kill right after the command "succeeds" can leave
    ``<skin>.yaml`` zero-length.  That is not transient: ``_skin_set`` is a
    read-modify-write whose read half falls back to ``{}``, so the next tweak
    rewrites the file from that empty state and the rest of the palette is
    gone for good — and the gateway's skin watcher repaints every live
    surface from this file within ~1s either way.
    """
    path = _skins() / "oasis.yaml"
    path.write_text(
        'name: oasis\ncolors:\n  background: "#08201f"\n  banner_title: "#f2dfb3"\n',
        encoding="utf-8",
    )
    _activate("oasis")

    synced = []
    real_fsync = os.fsync

    def _tracking_fsync(fd):
        synced.append(fd)
        return real_fsync(fd)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "fsync", _tracking_fsync)
        assert skin_cmd._skin_set("ui_tool", "#00FFFF", None) == 0

    assert synced, "skin file written without fsync — a crash can leave it empty"

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["colors"]["ui_tool"] == "#00FFFF"
    assert data["colors"]["background"] == "#08201f"  # untouched — the whole point
    assert data["colors"]["banner_title"] == "#f2dfb3"
    assert [p.name for p in _skins().iterdir() if p.name.endswith(".tmp")] == []


def test_set_preserves_a_symlinked_skin_file():
    """Guard on the conversion, not a behavior change.

    ``write_text`` wrote through a symlink; a naive ``os.replace`` would
    detach it.  ``atomic_replace`` resolves the link first (GitHub #16743),
    so a skin file symlinked out to a dotfiles repo must stay a symlink.
    """
    real_dir = _skins().parent / "dotfiles"
    real_dir.mkdir(parents=True, exist_ok=True)
    real = real_dir / "oasis.yaml"
    real.write_text(
        'name: oasis\ncolors:\n  background: "#08201f"\n', encoding="utf-8"
    )
    link = _skins() / "oasis.yaml"
    link.symlink_to(real)
    _activate("oasis")

    assert skin_cmd._skin_set("ui_tool", "#00FFFF", None) == 0

    assert link.is_symlink()
    data = yaml.safe_load(real.read_text(encoding="utf-8"))
    assert data["colors"]["ui_tool"] == "#00FFFF"
    assert data["colors"]["background"] == "#08201f"
