"""Verify HERMES_PYTHON is exported when hermes applies a profile."""
import os
import sys
from unittest.mock import patch

# Import at module level so the module-level _apply_profile_override() call
# happens with normal argv (not our test argv), avoiding side effects.
import hermes_cli.main as _main_mod


def test_hermes_python_set_to_sys_executable(monkeypatch, tmp_path):
    """HERMES_PYTHON must be set to sys.executable so skills use the agent's interpreter."""
    hermes_home = str(tmp_path / ".hermes")

    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "--profile", "myprofile"])

    with patch("hermes_cli.profiles.resolve_profile_env", return_value=hermes_home):
        _main_mod._apply_profile_override()

    assert os.environ.get("HERMES_PYTHON") == sys.executable, (
        f"Expected HERMES_PYTHON={sys.executable!r}, got {os.environ.get('HERMES_PYTHON')!r}"
    )


def test_hermes_python_not_overwritten_if_already_set(monkeypatch, tmp_path):
    """A user-set HERMES_PYTHON must not be overwritten by the agent."""
    hermes_home = str(tmp_path / ".hermes")

    monkeypatch.setenv("HERMES_PYTHON", "/custom/python3")
    monkeypatch.setattr(sys, "argv", ["hermes", "--profile", "myprofile"])

    with patch("hermes_cli.profiles.resolve_profile_env", return_value=hermes_home):
        _main_mod._apply_profile_override()

    assert os.environ.get("HERMES_PYTHON") == "/custom/python3", (
        "HERMES_PYTHON should not be overwritten when already set by user"
    )
