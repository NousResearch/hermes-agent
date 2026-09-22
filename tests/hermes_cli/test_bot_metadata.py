from __future__ import annotations

import threading
import time

import yaml


def test_concurrent_bot_metadata_transactions_preserve_both_updates(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "profile.yaml").write_text("setup_state: needs_setup\nroutines: []\n", encoding="utf-8")

    from hermes_cli.bot_metadata import mutate_bot_metadata

    entered = threading.Event()
    release = threading.Event()

    def setup_update(metadata):
        entered.set()
        assert release.wait(timeout=5)
        metadata["setup_state"] = "ready"

    def routine_update(metadata):
        metadata["routines"] = [{"id": "daily", "state": "active"}]

    first = threading.Thread(target=lambda: mutate_bot_metadata(setup_update))
    second = threading.Thread(target=lambda: mutate_bot_metadata(routine_update))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()

    stored = yaml.safe_load((home / "profile.yaml").read_text(encoding="utf-8"))
    assert stored["setup_state"] == "ready"
    assert stored["routines"] == [{"id": "daily", "state": "active"}]


def test_metadata_read_does_not_recreate_a_deleted_profile(tmp_path, monkeypatch):
    import pytest
    from hermes_cli.bot_metadata import read_bot_metadata

    home = tmp_path / "deleted-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(FileNotFoundError):
        read_bot_metadata()
    assert not home.exists()
