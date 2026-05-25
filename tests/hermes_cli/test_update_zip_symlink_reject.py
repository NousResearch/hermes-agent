"""Regression: _update_via_zip must reject ZIP members with symlink mode.

A symlink member in a downloaded update ZIP would let an attacker who can
serve / MITM the update mirror plant a symlink that extractall() then
follows, writing arbitrary file content outside the staging directory.
The Linux mode bits live in the upper 16 bits of ``ZipInfo.external_attr``;
we explicitly reject any member whose type bits are S_IFLNK.
"""

import io
import os
import stat
import tempfile
import zipfile
from unittest.mock import patch

import pytest


def _build_zip_with_symlink_member(zip_path: str, link_name: str, target: str) -> None:
    """Write a ZIP containing a single member with S_IFLNK mode bits set."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo(link_name)
        # Upper 16 bits = Unix mode; mark as symlink (0o120000) + 0o777 perms.
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        # The "data" of a symlink ZIP member is the link target string.
        zf.writestr(info, target)


def _build_normal_zip(zip_path: str) -> None:
    """Write a regular ZIP with a normal file member (no symlink)."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hermes-agent-main/README.md", "ok\n")


def test_update_via_zip_rejects_symlink_member(tmp_path, monkeypatch):
    """A symlink member in the update ZIP must raise before extractall."""
    zip_path = tmp_path / "evil.zip"
    _build_zip_with_symlink_member(
        str(zip_path),
        link_name="hermes-agent-main/evil-link",
        target="/etc/passwd",
    )

    from hermes_cli.main import _update_via_zip

    args = type("Args", (), {})()

    # Patch urlretrieve to "download" our pre-built malicious ZIP into the
    # _update_via_zip tempdir. Capture the tempdir so we can prove no
    # extraction happened.
    captured = {}
    original_mkdtemp = tempfile.mkdtemp

    def capturing_mkdtemp(*args, **kwargs):
        d = original_mkdtemp(*args, **kwargs)
        captured["tmp_dir"] = d
        return d

    def fake_urlretrieve(url, dest):
        # Copy our malicious zip into the destination dest path.
        with open(zip_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        return dest, None

    with patch("tempfile.mkdtemp", side_effect=capturing_mkdtemp), \
         patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        # _update_via_zip catches ValueError, prints the message, and exits 1.
        # That's the contract: a malicious ZIP must fail the update, not
        # silently materialize a symlink.
        with pytest.raises(SystemExit) as exc_info:
            _update_via_zip(args)
        assert exc_info.value.code == 1

    # Belt: confirm extractall never produced the link.
    tmp_dir = captured.get("tmp_dir")
    if tmp_dir:
        evil_path = os.path.join(tmp_dir, "hermes-agent-main", "evil-link")
        assert not os.path.lexists(evil_path), (
            "symlink member should never be materialized"
        )


def test_update_via_zip_accepts_normal_member(tmp_path, monkeypatch, capsys):
    """A ZIP with only regular file members must extract without raising.

    Sanity check that the symlink reject didn't break the happy path.
    Wraps just enough of _update_via_zip's I/O to verify extraction
    proceeds past the symlink check. We let the rest of the function
    fail naturally (no real git checkout to update); the assertion is
    just that we got past the ZIP validation.
    """
    zip_path = tmp_path / "normal.zip"
    _build_normal_zip(str(zip_path))

    from hermes_cli.main import _update_via_zip

    args = type("Args", (), {})()

    def fake_urlretrieve(url, dest):
        with open(zip_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        return dest, None

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        # The function will fail later (no real install dir to update into),
        # but it must get past the ZIP validation without raising
        # ValueError("symlink member").
        try:
            _update_via_zip(args)
        except SystemExit:
            pass

    captured = capsys.readouterr()
    assert "symlink member" not in captured.out
    assert "symlink member" not in captured.err
