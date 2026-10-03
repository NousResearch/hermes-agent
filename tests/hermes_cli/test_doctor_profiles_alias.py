"""Doctor profile-alias reporting must follow the platform wrapper layout.

Windows wrappers are ``<alias>.bat`` (``_wrapper_path``), extensionless
elsewhere; the profile summary used to probe the bare name and reported
"no alias" for every profile on Windows (#131753).
"""

import io
import sys
import contextlib

import pytest

from hermes_cli.doctor_state import _check_profiles
from hermes_cli.profiles import _wrapper_path


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "dev").mkdir(parents=True)
    (home / "profiles" / "dev" / "config.yaml").write_text("model:\n  provider: test\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.profiles._get_wrapper_dir", lambda: tmp_path / "bin")
    return tmp_path / "bin"


def _profile_summary() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _check_profiles(False)
    return buf.getvalue()


def test_alias_present_when_platform_wrapper_exists(profile_env):
    wrapper = _wrapper_path("dev")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        wrapper.write_text(f"@echo off\r\nhermes -p dev %*\r\n", encoding="utf-8")
    else:
        wrapper.write_text("#!/bin/sh\nexec hermes -p dev \"$@\"\n", encoding="utf-8")
    out = _profile_summary()
    assert "dev:" in out
    assert "no alias" not in out


def test_alias_missing_reported_when_no_wrapper_at_all(profile_env):
    out = _profile_summary()
    assert "no alias" in out
