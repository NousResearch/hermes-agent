"""`hermes skin set` — deterministic single-color tweak of the active skin.

The whole point is that changing one token never disturbs the rest of the look
(background especially), which hand-authoring kept getting wrong.
"""

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
