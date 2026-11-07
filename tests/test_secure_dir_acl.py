"""apply_secure_dir_policy keeps the POSIX ACL mask so named-user grants survive its chmod."""

import os
import shutil
import stat
import struct
import subprocess

import pytest

from hermes_constants import apply_secure_dir_policy

pytestmark = pytest.mark.platforms("posix")  # POSIX file modes


def _acl_blob(*entries):
    """Linux ``system.posix_acl_access`` xattr bytes for (tag, perm) entries."""
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", t, p, 0) for t, p in entries)


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for var in ("HERMES_HOME_MODE", "HERMES_MANAGED", "HERMES_CONTAINER",
                "HERMES_SKIP_CHMOD", "HERMES_UID", "HERMES_GID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_constants._detect_container", lambda: False)
    d = tmp_path / "shared"
    d.mkdir()
    os.chmod(d, 0o755)
    return d


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_named_user_acl_mask_is_not_narrowed(target):
    if not (shutil.which("setfacl") and shutil.which("getfacl")):
        pytest.skip("setfacl/getfacl not installed")
    grant = subprocess.run(["setfacl", "-m", f"u:{os.getuid()}:r-x,m::r-x", str(target)],
                           capture_output=True, stdin=subprocess.DEVNULL)
    if grant.returncode:
        pytest.skip("filesystem has no POSIX ACL support")
    apply_secure_dir_policy(target, home=target.parent / "home")
    acl = subprocess.run(["getfacl", "-cp", str(target)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=True).stdout
    assert "mask::r-x" in acl.splitlines()
    assert _mode(target) & 0o007 == 0  # other bits still follow the policy


def test_plain_dir_is_still_owner_only(target):
    apply_secure_dir_policy(target, home=target.parent / "home")
    assert _mode(target) == 0o700


def test_mask_from_xattr_widens_only_group_bits(target, monkeypatch):
    blob = _acl_blob((0x01, 7), (0x02, 5), (0x04, 0), (0x10, 5), (0x20, 0))
    calls = []
    monkeypatch.setattr(os, "getxattr", lambda *_a: blob, raising=False)
    monkeypatch.setattr(os, "chmod", lambda p, m: calls.append(m))
    apply_secure_dir_policy(target, home=target.parent / "home")
    assert calls == [0o750]


@pytest.mark.parametrize("getxattr", [
    lambda *_a: _acl_blob((0x01, 7), (0x04, 5), (0x10, 5), (0x20, 0)),  # no named user
    lambda *_a: b"\x02\x00",  # malformed
])
def test_acl_without_named_users_or_malformed_keeps_policy(target, monkeypatch, getxattr):
    monkeypatch.setattr(os, "getxattr", getxattr, raising=False)
    apply_secure_dir_policy(target, home=target.parent / "home")
    assert _mode(target) == 0o700


def test_platform_without_getxattr_keeps_policy(target, monkeypatch):
    monkeypatch.delattr(os, "getxattr", raising=False)
    apply_secure_dir_policy(target, home=target.parent / "home")
    assert _mode(target) == 0o700
