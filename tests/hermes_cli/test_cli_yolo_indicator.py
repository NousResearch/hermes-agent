"""The classic CLI badge reflects effective approval bypass, not just /yolo."""

from datetime import datetime

import pytest

import cli as cli_module
from agent.i18n import t
from cli import HermesCLI
from hermes_cli.config import atomic_config_write
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import approval, approval_context


@pytest.fixture
def indicator_cli(monkeypatch):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "_session_yolo", set())
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(cli_module, "CLI_CONFIG", {
        "display": {"status_bar": {"fields": ["model", "yolo", "title"]}},
    })
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "indicator-session"
    cli.model = "test-model"
    cli.session_start = datetime.now()
    cli.agent = None
    cli._status_bar_visible = True
    cli._pending_title = "indicator"
    return cli


@pytest.mark.parametrize("off_value", ["off", False], ids=["string-off", "yaml-bool-off"])
@pytest.mark.parametrize("width", [40, 60, 120], ids=["narrow", "medium", "wide"])
@pytest.mark.parametrize("styled", [False, True], ids=["plain", "styled"])
def test_badge_tracks_live_profile_approval_config(
    indicator_cli, tmp_path, monkeypatch, off_value, width, styled,
):
    """Read real profile YAML, revisit A after B, and revoke bypass without restart."""
    cli = indicator_cli
    monkeypatch.setattr(cli, "_get_tui_terminal_width", lambda: width)
    home_a = tmp_path / ".hermes" / "profiles" / "indicator-a"
    home_b = tmp_path / ".hermes" / "profiles" / "indicator-b"
    for home in (home_a, home_b):
        home.mkdir(parents=True)

    for home, mode, bypass in (
        (home_a, off_value, True),
        (home_b, "manual", False),
        (home_a, None, True),
        (home_a, "manual", False),
        (home_a, "smart", False),
        (home_a, off_value, True),
        (home_a, "manual", False),
    ):
        token = set_hermes_home_override(home)
        try:
            if mode is not None:
                atomic_config_write(home / "config.yaml", {"approvals": {"mode": mode}})
            assert approval.is_approval_bypass_active_for_session(cli.session_id) is bypass
            if styled:
                fragments = cli._get_status_bar_fragments()
                text = "".join(part for _, part in fragments)
                assert any(style == "class:status-bar-yolo" for style, _ in fragments) is bypass
            else:
                text = cli._build_status_bar_text(width=width)
            assert (t("cli.status_bar.yolo_badge") in text) is bypass, (home.name, mode, text)
            assert "test-model" in text
            assert text.endswith(" indicator ")
            assert cli._status_bar_display_width(text) == width
            assert not approval.is_session_yolo_enabled(cli.session_id)
        finally:
            reset_hermes_home_override(token)


@pytest.mark.parametrize("mode", ["manual", "smart"])
@pytest.mark.parametrize("session_id", ["indicator-session", "", None, "missing"])
def test_indicator_preserves_session_and_frozen_process_controls(
    indicator_cli, tmp_path, monkeypatch, mode, session_id,
):
    """Explicit CLI identity wins over another turn; bootstrap uses the default key."""
    cli = indicator_cli
    if session_id == "missing":
        del cli.session_id
    else:
        cli.session_id = session_id
    key = getattr(cli, "session_id", None) or "default"
    home_token = set_hermes_home_override(tmp_path)
    context_token = approval_context.set_current_session_key("another-session")
    try:
        atomic_config_write(tmp_path / "config.yaml", {"approvals": {"mode": mode}})
        assert cli._is_session_yolo_active() is False
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        assert cli._is_session_yolo_active() is False
        approval.enable_session_yolo("another-session")
        assert cli._is_session_yolo_active() is False
        approval.enable_session_yolo(key)
        assert cli._is_session_yolo_active() is True
        approval.disable_session_yolo(key)
        assert cli._is_session_yolo_active() is False
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
        assert cli._is_session_yolo_active() is True
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
        assert cli._is_session_yolo_active() is False
    finally:
        approval.clear_session(key)
        approval.clear_session("another-session")
        approval_context.reset_current_session_key(context_token)
        reset_hermes_home_override(home_token)
