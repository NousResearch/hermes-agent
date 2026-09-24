"""Tests for pet slash-command config helpers."""

from __future__ import annotations

import pytest



def test_pets_cli_quoted_false_disables_and_toggle_enables(tmp_path, monkeypatch):
    """Quoted `display.pet.enabled: "false"` must read as disabled.

    bool('false') is True — before the is_truthy_value fix, _has_active_pet
    reported an active pet and /pet toggle DISABLED instead of enabling.
    """
    import yaml

    from hermes_cli.pets import _has_active_pet, toggle_pet_display

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {"display": {"pet": {"enabled": "false", "slug": "", "scale": 0.33}}}
        ),
        encoding="utf-8",
    )

    assert _has_active_pet() is False
    # Toggle must take the ENABLE branch (reaching the "no pets installed"
    # error), not the disable branch (which would return err=None).
    enabled, name, err = toggle_pet_display()
    assert err is not None and "no pets installed" in err
    assert enabled is False
    assert name is None


@pytest.fixture
def empty_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_set_pet_scale_writes_clamped_value(empty_home):
    from agent.pet.constants import MAX_SCALE, MIN_SCALE
    from hermes_cli.pets import _pet_config, set_pet_scale

    applied, err = set_pet_scale("0.5")
    assert err is None
    assert applied == 0.5
    assert _pet_config()["scale"] == 0.5

    # Out-of-range values clamp to the bounds rather than erroring.
    assert set_pet_scale(99) == (MAX_SCALE, None)
    assert set_pet_scale(0) == (MIN_SCALE, None)


