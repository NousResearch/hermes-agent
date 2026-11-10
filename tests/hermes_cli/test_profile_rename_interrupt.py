"""An interrupted rename leaves the old profile listed and selectable."""
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_constants import named_profile_is_deleted


def test_interrupt_during_owner_wait_restores_the_old_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_notify_multiplexer",
                 "_stop_gateway_process", "_stop_bot_desktop", "_maybe_register_gateway_service"):
        monkeypatch.setattr(profiles, name, lambda *_a, **_k: None)
    old_dir = profiles.create_profile("interrupted")
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: True)

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(profiles, "wait_profile_unowned", interrupt)
    with pytest.raises(KeyboardInterrupt):
        profiles.rename_profile("interrupted", "renamed")
    assert not named_profile_is_deleted(old_dir)
    assert profiles.profile_exists("interrupted")
    assert not profiles.profile_exists("renamed")
