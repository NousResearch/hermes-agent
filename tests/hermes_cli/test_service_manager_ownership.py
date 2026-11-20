"""Root registration follows the runtime hermes account, including UID remaps."""
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import service_manager as sm

pytestmark = pytest.mark.platforms("posix")


def _as_root(monkeypatch, *, uid=1000, gid=1000):
    import pwd

    monkeypatch.setattr(sm.os, "geteuid", lambda: 0)
    lookup = Mock(return_value=SimpleNamespace(pw_uid=uid, pw_gid=gid))
    monkeypatch.setattr(pwd, "getpwnam", lookup)
    return lookup


@pytest.mark.parametrize("root", [True, False])
def test_registration_uses_current_hermes_owner(tmp_path, monkeypatch, root):
    import pwd

    scandir, staging = tmp_path / "scan", tmp_path / "staging"
    scandir.mkdir()
    staging.mkdir()
    monkeypatch.setattr(sm.os, "geteuid", lambda: 0 if root else 1000)
    lookup = Mock(return_value=SimpleNamespace(pw_uid=1000, pw_gid=1000))
    monkeypatch.setattr(pwd, "getpwnam", lookup)
    staging_for = Mock(return_value=staging)
    monkeypatch.setattr(sm, "_root_staging_dir", staging_for)
    chown = Mock()
    monkeypatch.setattr(sm.os, "chown", chown)
    monkeypatch.setattr(sm, "_s6_run", lambda *a, **kw: SimpleNamespace(returncode=0))
    # Even a root caller's restrictive umask must not hide scripts or slot directories.
    old_umask = sm.os.umask(0o077)
    try:
        sm.S6ServiceManager(scandir).register_profile_gateway("remapped", start_now=False)
    finally:
        sm.os.umask(old_umask)
    slot = scandir / "gateway-remapped"
    if root:
        # Built in the root-private staging dir, then published by rename.
        staging_for.assert_called_once_with(scandir)
        owned = {Path(*Path(call.args[0]).relative_to(staging).parts[1:])
                 for call in chown.call_args_list}
        assert {p.relative_to(slot) for p in [slot, *slot.rglob("*")]} | {Path()} <= owned
        assert {call.args[1:] for call in chown.call_args_list} == {(1000, 1000)}
        assert {call.kwargs.get("follow_symlinks") for call in chown.call_args_list} == {False}
        assert list(staging.iterdir()) == []
        lookup.assert_called_with("hermes")
    else:
        staging_for.assert_not_called()
        chown.assert_not_called()
        lookup.assert_not_called()
    assert not slot.is_symlink()
    for name in ("run", "finish", "log/run"):
        assert (slot / name).stat().st_mode & 0o555 == 0o555
    for name in (".", "log"):
        assert (slot / name).stat().st_mode & 0o555 == 0o555


def test_root_registration_ignores_staging_swap_in_scandir(tmp_path, monkeypatch):
    """hermes owns the scandir, so it can swap any dot dir root builds there for a symlink."""
    scandir, staging, victim = tmp_path / "scan", tmp_path / "staging", tmp_path / "victim"
    for path in (scandir, staging, victim):
        path.mkdir()
    victim.chmod(0o700)
    _as_root(monkeypatch)
    monkeypatch.setattr(sm, "_root_staging_dir", lambda _scandir: staging, raising=False)
    chown = Mock()
    monkeypatch.setattr(sm.os, "chown", chown)
    monkeypatch.setattr(sm, "_s6_run", lambda *a, **kw: SimpleNamespace(returncode=0))
    seed = sm._seed_supervise_skeleton

    def seed_then_swap(svc_dir):
        seed(svc_dir)
        # The concurrent hermes process: rename the half-built slot, plant a symlink.
        for entry in list(scandir.iterdir()):
            if entry.name.startswith("."):
                entry.rename(scandir / f"moved{entry.name}")
                entry.symlink_to(victim)

    monkeypatch.setattr(sm, "_seed_supervise_skeleton", seed_then_swap)
    sm.S6ServiceManager(scandir).register_profile_gateway("swap", start_now=False)

    assert victim.stat().st_mode & 0o777 == 0o700
    assert list(victim.iterdir()) == []
    touched = {Path(call.args[0]).resolve() for call in chown.call_args_list}
    assert not {p for p in touched if p == victim or victim in p.parents}
    slot = scandir / "gateway-swap"
    assert slot.is_dir() and not slot.is_symlink()
    assert (slot / "run").is_file()


@pytest.mark.parametrize("st_mode, st_uid, private, ok", [
    (stat.S_IFDIR | 0o755, 0, False, True),
    (stat.S_IFDIR | 0o1777, 0, False, True),       # sticky: others cannot rename root's entries
    (stat.S_IFDIR | 0o700, 0, True, True),
    (stat.S_IFDIR | 0o755, 0, True, False),        # staging must be private
    (stat.S_IFDIR | 0o775, 0, False, False),       # group-writable parent
    (stat.S_IFDIR | 0o757, 0, False, False),       # world-writable parent
    (stat.S_IFDIR | 0o755, 10000, False, False),   # hermes-owned
    (stat.S_IFLNK | 0o777, 0, False, False),       # symlink
])
def test_require_root_controlled(monkeypatch, st_mode, st_uid, private, ok):
    fake = SimpleNamespace(st_mode=st_mode, st_uid=st_uid)
    monkeypatch.setattr(sm.os, "lstat", lambda path: fake)
    if ok:
        assert sm._require_root_controlled(Path("/x"), private=private) is fake
    else:
        with pytest.raises(RuntimeError):
            sm._require_root_controlled(Path("/x"), private=private)


def test_root_staging_refuses_unprivileged_ancestor(tmp_path, monkeypatch):
    """An ancestor owned by a non-root account (here: the scandir's parent) is refused."""
    monkeypatch.setattr(sm.os, "geteuid", lambda: 0)
    scandir = tmp_path / "scan"
    scandir.mkdir()
    real_lstat = sm.os.lstat

    def lstat(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if Path(path) != tmp_path:
            return SimpleNamespace(st_mode=st.st_mode & ~0o022, st_uid=0)
        return SimpleNamespace(st_mode=st.st_mode, st_uid=10000)

    monkeypatch.setattr(sm.os, "lstat", lstat)
    with pytest.raises(RuntimeError, match="root-owned"):
        sm._root_staging_dir(scandir)
    assert not (tmp_path / ".scan.staging").exists()


@pytest.mark.parametrize("root", [True, False])
def test_s6_tools_drop_to_hermes_when_root(tmp_path, monkeypatch, root):
    """As root, s6 tools would follow hermes-planted symlinks for control FIFOs/fifodirs."""
    if root:
        _as_root(monkeypatch)
    else:
        monkeypatch.setattr(sm.os, "geteuid", lambda: 1000)
    (tmp_path / "gateway-p").mkdir()
    calls = []
    monkeypatch.setattr(sm, "_s6_run", lambda *a, **kw: calls.append(a) or SimpleNamespace(
        returncode=0, stdout="true\n", stderr=""))
    monkeypatch.setattr(sm, "_write_gateway_desired_state", lambda *a: None)
    mgr = sm.S6ServiceManager(tmp_path)
    mgr.restart("gateway-p")
    mgr.is_running("gateway-p")
    mgr.unregister_profile_gateway("p")
    assert calls
    for call in calls:
        if root:
            assert call[:2] == ("s6-setuidgid", "hermes")
            assert call[2].startswith(f"{sm._S6_BIN_DIR}/s6-")
        else:
            assert call[0].startswith("s6-") and call[0] != "s6-setuidgid"


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
