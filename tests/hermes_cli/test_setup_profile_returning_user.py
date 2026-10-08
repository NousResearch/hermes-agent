"""A user who has already used Hermes must not get the first-run guide when the guest flag turns on.

``onboarding.state`` reports ``intro: unseen`` for any install without a setup profile, so a returning
user looked brand new. ``settle_returning_user`` latches ``seen`` on first launch from real history.
"""

from pathlib import Path

import pytest

from hermes_cli import setup_profile
from hermes_cli.config import read_user_config_raw
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _add_session(home: Path) -> None:
    db = SessionDB(home / "state.db")
    try:
        db.create_session("old-session", "desktop")
    finally:
        db.close()


def _settled_intro() -> str:
    setup_profile.settle_returning_user()
    return setup_profile.read_state()["intro"]


def test_fresh_install_stays_unseen(home):
    assert _settled_intro() == "unseen"
    assert "onboarding" not in read_user_config_raw(home / "config.yaml")


def test_empty_state_db_from_a_boot_stays_unseen(home):
    SessionDB(home / "state.db").close()
    (home / "SOUL.md").write_text("default soul")

    assert _settled_intro() == "unseen"


def test_install_with_a_session_is_latched_seen(home):
    _add_session(home)

    assert _settled_intro() == "seen"
    assert read_user_config_raw(home / "config.yaml")["onboarding"]["seen"][setup_profile.RETURNING_USER_FLAG] is True


def test_latch_survives_the_history_going_away(home):
    _add_session(home)
    assert _settled_intro() == "seen"

    (home / "state.db").unlink()

    assert _settled_intro() == "seen"


def test_install_with_a_user_profile_is_latched_seen(home):
    from hermes_cli import profiles

    profiles.create_profile("work", no_alias=True)

    assert _settled_intro() == "seen"


def test_a_user_mid_guide_is_left_to_the_setup_profile_marker(home):
    setup_profile.ensure_setup_profile()
    _add_session(home)

    assert _settled_intro() == "unseen"
    setup_profile.mark_intro_seen()
    assert setup_profile.read_state()["intro"] == "seen"
