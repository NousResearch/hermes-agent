"""Slot ``down`` markers and waits in the hermes-owned scandir never follow planted symlinks."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import gateway_multiplex_s6 as s6
from hermes_cli import service_manager as sm

pytestmark = pytest.mark.platforms("posix")


def test_down_marker_create_and_remove(tmp_path):
    slot = tmp_path / "gateway-a"
    slot.mkdir()
    s6._set_down_marker(slot, down=True)
    assert (slot / "down").is_file()
    s6._set_down_marker(slot, down=True)  # idempotent
    s6._set_down_marker(slot, down=False)
    assert not (slot / "down").exists()
    s6._set_down_marker(slot, down=False)  # idempotent


def test_down_marker_does_not_follow_a_down_symlink(tmp_path):
    slot, victim_dir = tmp_path / "gateway-a", tmp_path / "victim"
    slot.mkdir()
    victim_dir.mkdir()
    victim = victim_dir / "file"
    victim.write_text("ORIGINAL", encoding="utf-8")
    os.utime(victim, (1, 1))
    (slot / "down").symlink_to(victim)
    (slot / "dangling").symlink_to(victim_dir / "created")
    s6._set_down_marker(slot, down=True)
    assert victim.stat().st_mtime == 1 and victim.read_text(encoding="utf-8") == "ORIGINAL"
    (slot / "down").unlink()
    (slot / "dangling").rename(slot / "down")
    s6._set_down_marker(slot, down=True)
    assert list(victim_dir.iterdir()) == [victim]
    s6._set_down_marker(slot, down=False)  # removes the link, not its target
    assert not os.path.lexists(slot / "down") and victim.exists()


def test_down_marker_refuses_a_symlinked_slot(tmp_path):
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    (tmp_path / "gateway-a").symlink_to(victim_dir)
    with pytest.raises(OSError):
        s6._set_down_marker(tmp_path / "gateway-a", down=True)
    assert list(victim_dir.iterdir()) == []


def test_root_down_marker_belongs_to_hermes(tmp_path, monkeypatch):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    fchown = Mock()
    monkeypatch.setattr(s6.os, "fchown", fchown)
    slot = tmp_path / "gateway-a"
    slot.mkdir()
    s6._set_down_marker(slot, down=True)
    assert fchown.call_count == 1 and fchown.call_args.args[1:] == (1000, 1000)


@pytest.mark.parametrize("root", [True, False])
def test_wait_down_drops_to_hermes_when_root(tmp_path, monkeypatch, root):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0 if root else 1000)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    calls = []
    monkeypatch.setattr(sm, "_s6_run", lambda *a, **kw: calls.append(a))
    s6._wait_down(tmp_path / "gateway-a", timeout_ms=10)
    expected = ("s6-setuidgid", "hermes", f"{sm._S6_BIN_DIR}/s6-svwait") if root else ("s6-svwait",)
    assert calls and calls[0][:len(expected)] == expected
