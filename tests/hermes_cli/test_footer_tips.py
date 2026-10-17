"""The rotating tip line under the composer: it fills the live-work footer only while that footer has
nothing to list, rotates on a timer without repeating itself, prefers tips that fit the width, and
``display.cli_footer_tips: false`` turns it off."""
import random
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth

from agent.i18n import t
from hermes_cli import cli_footer_tips
from hermes_cli.cli_footer_tips import FEATURED_TIP_KEYS, ROTATE_SECONDS, FooterTips
from hermes_cli.tips import tip_keys

_AGENT = dict(subagent_id="sa-1", key="sa-1", goal="Worker", elapsed=1, status="running")


def _tip_lines():
    return {"  " + t("cli.tip_line", tip=t(key)): key for key in tip_keys()}


def _dock(tips_enabled=True):
    from hermes_cli.cli_subagent_monitor import install_dock
    cli = SimpleNamespace(agent=None, _footer_tips_enabled=tips_enabled)
    install_dock(cli)
    return cli, cli._subagent_monitor


def test_idle_footer_is_one_catalog_tip_and_live_work_replaces_it():
    cli, monitor = _dock()
    lines = monitor.dock_text(columns=100, rows=30).splitlines()
    assert len(lines) == 1 and lines[0] in _tip_lines()

    monitor.entries = [_AGENT]  # a running subagent: the footer is about it, not a tip
    text = monitor.dock_text(columns=100, rows=30)
    assert text and not any(line in _tip_lines() for line in text.splitlines())

    monitor.entries = []
    cli._approval_state = {"pending": True}  # a prompt owns the space under the composer
    assert monitor.dock_text(columns=100, rows=30) == ""
    cli._approval_state = None
    assert monitor.dock_text(columns=100, rows=14) == ""  # too short to spare the row
    assert monitor.dock_text(columns=100, rows=30) in _tip_lines()


def test_tips_off_leaves_the_idle_footer_empty():
    _, monitor = _dock(tips_enabled=False)
    assert monitor.dock_text(columns=100, rows=30) == ""


def test_tip_rotates_on_its_timer_and_repaints_only_when_it_changes(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cli, monitor = _dock()
    paints = []
    cli._invalidate = lambda *a, **k: paints.append(now[0])
    first = monitor.dock_text(columns=100, rows=30)
    monitor._last_poll = 0
    monitor.tick()
    paints.clear()
    for _ in range(int(ROTATE_SECONDS) - 2):  # the once-a-second poll before the tip is due
        now[0] += 1
        monitor.tick()
    assert paints == [] and monitor.dock_text(columns=100, rows=30) == first
    now[0] += 3
    monitor.tick()
    assert len(paints) == 1 and monitor.dock_text(columns=100, rows=30) != first


@pytest.mark.parametrize("columns", [60, 100])
def test_rotation_never_repeats_recent_tips_and_prefers_ones_that_fit(columns):
    tips = FooterTips(rng=random.Random(7))
    lines = _tip_lines()
    widths = {key: get_cwidth(line) for line, key in lines.items()}
    fits_some = any(w <= columns - 1 for w in widths.values())
    shown = []
    for _ in range(200):
        tips.line(columns)
        shown.append(tips.key)
        tips.rotate()
    assert fits_some
    assert all(widths[key] <= columns - 1 for key in shown)
    if len({k for k, w in widths.items() if w <= columns - 1}) > 8:
        assert all(key not in shown[i - 8:i] for i, key in enumerate(shown) if i >= 8)


def test_the_mouse_tip_comes_up_often_enough_to_be_seen():
    """``/mouse on`` and the footer controls are featured: in ~an hour of 30 s rotations each one
    is shown several times, far more than a uniform draw over the whole catalog would give."""
    mouse_key = next(key for key in FEATURED_TIP_KEYS if "/mouse on makes" in t(key))
    tips = FooterTips(rng=random.Random(3))
    seen = []
    for _ in range(120):
        tips.line(120)
        seen.append(tips.key)
        tips.rotate()
    assert seen.count(mouse_key) >= 3
    assert set(FEATURED_TIP_KEYS) <= set(tip_keys())


def test_footer_tips_setting_reaches_the_cli_through_the_loader(monkeypatch):
    """display.cli_footer_tips in config.yaml decides whether a fresh CLI's footer shows tips."""
    import cli as cli_mod
    import hermes_yaml as yaml
    from cli import load_cli_config
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    monkeypatch.setattr(cli_mod, "_hermes_home", home)
    assert cli_mod.HermesCLI(model="test-model", toolsets=[], provider="auto")._footer_tips_enabled is True
    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"cli_footer_tips": False}}), encoding="utf-8")
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", load_cli_config())
    fresh = cli_mod.HermesCLI(model="test-model", toolsets=[], provider="auto")
    assert fresh._footer_tips_enabled is False
    from hermes_cli.cli_subagent_monitor import install_dock
    install_dock(fresh)
    assert not cli_footer_tips.showing(fresh, rows=30)


def test_a_narrower_terminal_swaps_a_tip_that_no_longer_fits():
    tips = FooterTips(rng=random.Random(5))
    wide = next(key for key in tip_keys() if 70 < get_cwidth(_line(key)) < 120)
    tips.line(140)
    tips.key = wide
    assert tips.line(140) == tips.line(140) and tips.key == wide  # a steady width keeps it
    tips.line(60)
    assert tips.key != wide and get_cwidth(_line(tips.key)) <= 59


def _line(key):
    return "  " + t("cli.tip_line", tip=t(key))
