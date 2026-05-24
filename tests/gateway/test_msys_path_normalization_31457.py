"""Regression tests for #31457 — Windows MSYS path normalization.

The agent, when run under Git Bash on Windows, emits paths in the
shape ``/c/Users/...`` (and very-old Cygwin shells produce
``/cygdrive/c/...``).  Native Windows Python's :class:`pathlib.Path`
treats the leading slash as the current drive's root, so
``Path('/c/Users/foo').resolve(strict=True)`` raises
``FileNotFoundError`` and ``read_bytes()`` reports
``[Errno 2] No such file or directory`` — even when the file exists
at ``C:\\Users\\foo``.

The helpers ``normalize_msys_path`` and ``strip_file_url_prefix``
rewrite those shapes before they reach :class:`pathlib.Path`.  These
tests pin the helpers' contract and the call-site wiring inside
``validate_media_delivery_path`` and the Weixin adapter.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.platforms import base as base_module


# ---------------------------------------------------------------------------
# normalize_msys_path
# ---------------------------------------------------------------------------


class TestNormalizeMsysPath:
    """Helper contract: rewrite MSYS drive paths only on Windows."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("/c/Users/foo/bar.jpg", "C:/Users/foo/bar.jpg"),
            ("/d/data/file.bin", "D:/data/file.bin"),
            # Single-letter drive must be at position 1; the trailing
            # slash and rest are required (so we don't rewrite ``/c``
            # alone, which on a real POSIX disk could be a directory).
            ("/cygdrive/c/Users/foo.png", "C:/Users/foo.png"),
            # Drive letter is uppercased to match Windows convention.
            ("/x/games/save.dat", "X:/games/save.dat"),
        ],
    )
    def test_windows_rewrite(self, raw, expected):
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "win32"
            assert base_module.normalize_msys_path(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            # Already native: unchanged.
            r"C:\Users\foo\bar.jpg",
            "C:/Users/foo/bar.jpg",
            # Drive-relative: unchanged (Windows resolves against drive cwd).
            r"foo\bar.jpg",
            # Two-character first segment is not a drive — leave alone.
            "/cd/foo.jpg",
            # Bare drive root with no rest must NOT be rewritten,
            # otherwise ``/c`` (a hypothetical literal directory)
            # gets corrupted to ``C:`` which is a totally different
            # location.
            "/c",
            "/cygdrive",
            "/cygdrive/",
            # Empty string is a no-op.
            "",
        ],
    )
    def test_windows_unchanged(self, raw):
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "win32"
            assert base_module.normalize_msys_path(raw) == raw

    @pytest.mark.parametrize("plat", ["linux", "darwin", "freebsd"])
    def test_non_windows_is_passthrough(self, plat):
        # On real POSIX systems ``/c`` is a legitimate filesystem path.
        # We must never rewrite it.
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = plat
            for raw in ("/c/Users/foo", "/cygdrive/c/foo", "/usr/bin/python"):
                assert base_module.normalize_msys_path(raw) == raw


# ---------------------------------------------------------------------------
# strip_file_url_prefix
# ---------------------------------------------------------------------------


class TestStripFileUrlPrefix:
    def test_canonical_windows_file_url(self):
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "win32"
            # ``file:///C:/Users/foo`` is the canonical Windows file URL.
            assert (
                base_module.strip_file_url_prefix("file:///C:/Users/foo.jpg")
                == "C:/Users/foo.jpg"
            )

    def test_msys_file_url_on_windows(self):
        # The exact bug shape from #31457: agent emits
        # ``file:///c/Users/.../file.jpg`` (lowercase drive, no colon).
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "win32"
            assert (
                base_module.strip_file_url_prefix(
                    "file:///c/Users/qhttttt/cache/file.jpg"
                )
                == "C:/Users/qhttttt/cache/file.jpg"
            )

    def test_no_prefix_is_passthrough(self):
        assert base_module.strip_file_url_prefix("/already/bare") == "/already/bare"
        assert base_module.strip_file_url_prefix("") == ""

    def test_posix_file_url_keeps_leading_slash(self):
        # POSIX-style ``file:///srv/...`` is the common shape on Linux
        # and macOS — the leading ``/`` after the prefix is the actual
        # path root and must survive.
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "linux"
            assert (
                base_module.strip_file_url_prefix("file:///srv/data/x.png")
                == "/srv/data/x.png"
            )

    def test_strips_synthetic_slash_before_windows_drive(self):
        # ``file:///C:/foo`` is the canonical Windows file URL; the
        # leading slash before ``C:`` is a URL artefact, not a path
        # component.  Trim it so callers get a clean ``C:/foo``.
        with patch.object(base_module, "sys") as fake_sys:
            fake_sys.platform = "win32"
            assert (
                base_module.strip_file_url_prefix("file:///D:/games/save.dat")
                == "D:/games/save.dat"
            )


# ---------------------------------------------------------------------------
# validate_media_delivery_path call-site wiring
# ---------------------------------------------------------------------------


class TestValidateMediaDeliveryPathCallsNormalize:
    """Source-level guard — fast and platform-independent."""

    def test_validator_invokes_normalize_msys_path(self):
        src = inspect.getsource(base_module.validate_media_delivery_path)
        assert "normalize_msys_path" in src, (
            "validate_media_delivery_path must normalize MSYS paths before "
            "Path() consumes them; otherwise Windows Git-Bash MEDIA: "
            "directives silently fail validation (#31457)."
        )

    def test_extract_local_files_invokes_normalize_msys_path(self):
        src = inspect.getsource(base_module.BasePlatformAdapter.extract_local_files)
        assert "normalize_msys_path" in src, (
            "extract_local_files must rewrite MSYS drive paths before "
            "os.path.isfile, otherwise bare local paths emitted by an "
            "agent under Git Bash on Windows are silently dropped (#31457)."
        )


# ---------------------------------------------------------------------------
# Behavioural integration via validate_media_delivery_path
# ---------------------------------------------------------------------------


class TestValidateMediaDeliveryPathBehaviour:
    """Run on POSIX hosts via simulated MSYS prefix that points at a real file."""

    def test_real_posix_path_under_allowlist_still_validates(self, tmp_path):
        # The non-Windows branch must still accept ordinary absolute
        # paths that live under an allowlisted root — i.e. our shim
        # is not interfering with the existing happy path.
        target = tmp_path / "cache.png"
        target.write_bytes(b"x")
        with patch.object(
            base_module,
            "_media_delivery_allowed_roots",
            return_value=[tmp_path],
        ):
            resolved = base_module.validate_media_delivery_path(str(target))
        assert resolved is not None
        assert Path(resolved).resolve() == target.resolve()


# ---------------------------------------------------------------------------
# Weixin adapter wiring
# ---------------------------------------------------------------------------


class TestWeixinSendImageWiring:
    """The naive ``replace('file://', '')`` was the visible bug; assert
    the adapter now routes file URLs through ``strip_file_url_prefix``."""

    def test_send_image_no_longer_uses_naive_replace(self):
        from gateway.platforms import weixin as weixin_module

        src = inspect.getsource(weixin_module.WeixinAdapter.send_image)
        # The fragile pattern that broke on Windows MSYS.
        assert 'replace("file://"' not in src, (
            "send_image must use strip_file_url_prefix; the naive "
            "replace() loses MSYS drive normalization (#31457)."
        )
        assert "strip_file_url_prefix" in src

    def test_send_file_invokes_normalize_msys_path(self):
        from gateway.platforms import weixin as weixin_module

        src = inspect.getsource(weixin_module.WeixinAdapter._send_file)
        assert "normalize_msys_path" in src, (
            "_send_file must defensively normalize MSYS drive paths "
            "before Path(path).read_bytes() to survive any caller that "
            "bypasses the validator (#31457)."
        )
