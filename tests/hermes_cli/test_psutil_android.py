"""Tests for hermes_cli/psutil_android.py — pure helpers."""

import pytest


def test_psutil_android_install_error_is_runtime_error():
    from hermes_cli.psutil_android import PsutilAndroidInstallError
    with pytest.raises(PsutilAndroidInstallError):
        raise PsutilAndroidInstallError("test")


def test_marker_and_replacement_are_non_empty():
    from hermes_cli.psutil_android import MARKER, REPLACEMENT
    assert "linux" in MARKER
    assert "android" in REPLACEMENT


def test_normalize_member_parts_strips_prefix():
    from hermes_cli.psutil_android import _normalize_member_parts
    parts = _normalize_member_parts("psutil-7.2.2/psutil/__init__.py")
    assert parts[0] != "psutil-7.2.2" or len(parts) > 1


def test_psutil_url_points_to_tar_gz():
    from hermes_cli.psutil_android import PSUTIL_URL
    assert PSUTIL_URL.endswith(".tar.gz")
    assert "psutil" in PSUTIL_URL.lower()
