"""Skin files that cannot be parsed degrade to a usable skin and recover once repaired.

Covers whole-file failures (invalid YAML, undecodable bytes, a non-mapping document), which
``_load_skin_from_yaml`` deliberately swallows. Field-level and section-type damage inside a
valid mapping is covered in ``test_skin_engine.py``.
"""

from pathlib import Path

import pytest
from prompt_toolkit.styles import Style

from hermes_cli import skin_engine
from hermes_cli.skin_engine import (
    get_active_prompt_symbol,
    get_active_skin,
    get_prompt_toolkit_style_overrides,
    init_skin_from_config,
    list_skins,
    load_skin,
)

SIBLING = "name: sibling\ndescription: healthy\ncolors:\n  input_rule: '#123456'\n"
REPAIRED = (
    "name: {name}\n"
    "colors:\n  input_rule: '#0a1b2c'\n"
    "branding:\n  prompt_symbol: '>>'\n"
)

UNUSABLE = {
    "invalid_yaml": b"name: broken\ncolors: [unclosed\n  - : :\n",
    "undecodable_utf8": b"name: broken\n\xff\xfe\x80 not utf-8\n",
    "non_mapping": b"- just\n- a\n- list\n",
}


@pytest.fixture
def skins_dir(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skins").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skin_engine, "_active_skin", None)
    monkeypatch.setattr(skin_engine, "_active_skin_name", "default")
    monkeypatch.setattr(skin_engine, "_active_skin_by_home", {})
    return home / "skins"


def _activate(name):
    init_skin_from_config({"display": {"skin": name}})
    return get_active_skin()


@pytest.mark.parametrize("payload", UNUSABLE.values(), ids=UNUSABLE.keys())
def test_unusable_custom_skin_falls_back_then_recovers_when_repaired(skins_dir, payload):
    broken = skins_dir / "broken.yaml"
    sibling = skins_dir / "sibling.yaml"
    broken.write_bytes(payload)
    sibling.write_text(SIBLING, encoding="utf-8")
    sibling_bytes = sibling.read_bytes()
    default = load_skin("default")

    skin = _activate("broken")

    assert skin.colors == default.colors
    assert skin.branding == default.branding
    Style.from_dict(get_prompt_toolkit_style_overrides())
    names = {s["name"] for s in list_skins()}
    assert "sibling" in names
    assert "broken" not in names
    assert load_skin("sibling").colors["input_rule"] == "#123456"
    # The loader only reads: neither the broken file nor its sibling is rewritten or removed.
    assert broken.read_bytes() == payload
    assert sibling.read_bytes() == sibling_bytes

    broken.write_text(REPAIRED.format(name="broken"), encoding="utf-8")
    init_skin_from_config({"display": {"skin": "broken"}})

    assert get_active_skin().name == "broken"
    assert get_active_prompt_symbol() == ">> "
    styles = get_prompt_toolkit_style_overrides()
    Style.from_dict(styles)
    assert styles["input-rule"] == "#0a1b2c"


def test_unusable_override_of_builtin_keeps_builtin_look(skins_dir):
    baseline = load_skin("ares")
    baseline_symbol = baseline.branding["prompt_symbol"]
    (skins_dir / "sibling.yaml").write_text(SIBLING, encoding="utf-8")
    override = skins_dir / "ares.yaml"
    override.write_bytes(UNUSABLE["invalid_yaml"].replace(b"broken", b"ares"))

    skin = _activate("ares")

    assert skin.name == "ares"
    assert skin.colors == baseline.colors
    assert skin.branding == baseline.branding
    assert get_active_prompt_symbol() == f"{baseline_symbol.strip()} "
    Style.from_dict(get_prompt_toolkit_style_overrides())
    assert load_skin("sibling").colors["input_rule"] == "#123456"
    assert "sibling" in {s["name"] for s in list_skins()}
