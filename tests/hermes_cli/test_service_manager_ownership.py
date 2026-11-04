"""Root registration follows the runtime hermes account, including UID remaps."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import service_manager as sm

pytestmark = pytest.mark.platforms("posix")


@pytest.mark.parametrize("root", [True, False])
def test_registration_uses_current_hermes_owner(tmp_path, monkeypatch, root):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0 if root else 1000)
    lookup = Mock(return_value=SimpleNamespace(pw_uid=1000, pw_gid=1000))
    monkeypatch.setattr(pwd, "getpwnam", lookup)
    chown = Mock()
    monkeypatch.setattr(sm.os, "chown", chown)
    monkeypatch.setattr(sm, "_s6_run", lambda *a, **kw: SimpleNamespace(returncode=0))
    # Even a root caller's restrictive umask must not hide scripts or slot directories.
    old_umask = sm.os.umask(0o077)
    try:
        sm.S6ServiceManager(tmp_path).register_profile_gateway("remapped", start_now=False)
    finally:
        sm.os.umask(old_umask)
    slot = tmp_path / "gateway-remapped"
    if root:
        owned = {Path(call.args[0]).relative_to(tmp_path) for call in chown.call_args_list}
        expected = {Path(".gateway-remapped.tmp") / p.relative_to(slot)
                    for p in [slot, *slot.rglob("*")]}
        assert expected <= owned
        assert {call.args[1:] for call in chown.call_args_list} == {(1000, 1000)}
        lookup.assert_called_with("hermes")
    else:
        chown.assert_not_called()
        lookup.assert_not_called()
    for name in ("run", "finish", "log/run"):
        assert (slot / name).stat().st_mode & 0o555 == 0o555
    for name in (".", "log"):
        assert (slot / name).stat().st_mode & 0o555 == 0o555


def test_seed_uses_remapped_account(tmp_path, monkeypatch):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    chown = Mock()
    monkeypatch.setattr(sm.os, "chown", chown)
    (tmp_path / "log").mkdir()
    sm._seed_supervise_skeleton(tmp_path)
    assert {call.args[1:] for call in chown.call_args_list} == {(1000, 1000)}
    assert {Path(call.args[0]).relative_to(tmp_path) for call in chown.call_args_list} == {
        Path(prefix) / name for prefix in (".", "log")
        for name in ("event", "supervise", "supervise/event", "supervise/control")
    }


def test_root_seed_without_container_account_retains_owner(tmp_path, monkeypatch):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0)
    monkeypatch.setattr(pwd, "getpwnam", Mock(side_effect=KeyError("hermes")))
    chown = Mock()
    monkeypatch.setattr(sm.os, "chown", chown)
    sm._seed_supervise_skeleton(tmp_path)
    chown.assert_not_called()
    assert (tmp_path / "supervise/control").is_fifo()
