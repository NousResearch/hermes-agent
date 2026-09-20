"""Installation discovery is read-only, bounded, and independent of launch profile."""
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_tags import list_installation_session_tags


def test_catalogue_never_reads_outside_supplied_installation(tmp_path, monkeypatch):
    root = tmp_path / "isolated"
    a, b = root / "profiles" / "alpha", root / "profiles" / "beta"
    outside = tmp_path / "host" / ".hermes"
    for home, tag in [(a, "Shared"), (outside, "Private")]:
        home.mkdir(parents=True)
        with SessionDB(home / "state.db") as db:
            db.create_session("same", source="desktop")
            db.set_session_tag("same", tag, True)
    b.mkdir()
    (b / "config.yaml").write_text("{}")
    (root / "profiles" / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: outside.parent)
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("HERMES_BASE_HOME", str(root))
    for home in (a, b, root):
        assert list_installation_session_tags(home) == ["Shared"]
    assert list_installation_session_tags(b, "alpha") == ["Shared"]
    for profile in ("escape", "absent"):
        with pytest.raises(FileNotFoundError):
            list_installation_session_tags(b, profile)
    with pytest.raises(ValueError):
        list_installation_session_tags(b, "../host")
    assert not (b / "state.db").exists()
    assert not (root / "state.db").exists()
    with pytest.raises(ValueError, match="outside"):
        list_installation_session_tags(outside)
    monkeypatch.delenv("HERMES_BASE_HOME")
    assert list_installation_session_tags(b) == ["Shared"]
