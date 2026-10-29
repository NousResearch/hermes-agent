"""Tests for companion slash-command config helpers."""

from __future__ import annotations

import pytest



def test_companions_cli_quoted_false_disables_and_toggle_enables(tmp_path, monkeypatch):
    """Quoted `display.companion.enabled: "false"` must read as disabled.

    bool('false') is True — before the is_truthy_value fix, _has_active_companion
    reported an active companion and /companion toggle DISABLED instead of enabling.
    """
    import hermes_yaml as yaml

    from hermes_cli.companions import _has_active_companion, toggle_companion_display

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {"display": {"companion": {"enabled": "false", "slug": "", "scale": 0.33}}}
        ),
        encoding="utf-8",
    )

    assert _has_active_companion() is False
    # Toggle must take the ENABLE branch (reaching the "no companions installed"
    # error), not the disable branch (which would return err=None).
    enabled, name, err = toggle_companion_display()
    assert err is not None and "no companions installed" in err
    assert enabled is False
    assert name is None


@pytest.fixture
def empty_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_set_companion_scale_writes_clamped_value(empty_home):
    from agent.companion.constants import MAX_SCALE, MIN_SCALE
    from hermes_cli.companions import _companion_config, set_companion_scale

    applied, err = set_companion_scale("0.5")
    assert err is None
    assert applied == 0.5
    assert _companion_config()["scale"] == 0.5

    # Out-of-range values clamp to the bounds rather than erroring.
    assert set_companion_scale(99) == (MAX_SCALE, None)
    assert set_companion_scale(0) == (MIN_SCALE, None)


