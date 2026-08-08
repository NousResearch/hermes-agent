"""The Google Workspace setup script must not install packages itself.

The Google SDKs ship in the [google] extra, which [all] contains, so each
supported install path already has them. A stripped environment is a broken
install, and the repair is `hermes update`, not a pip run from inside a
skill script that writes to whatever interpreter it happens to be under.

These tests hold that line: the script reports the repair, and it exits
rather than continuing into an API call that would fail with an import
error further down.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _break_google_imports(monkeypatch, setup_module) -> None:
    """Make the three Google SDK imports fail, and leave the rest alone."""
    real_import = builtins.__import__
    broken = {"googleapiclient", "google.auth", "google_auth_oauthlib"}

    def fake_import(name, *args, **kwargs):
        if name in broken:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_sdks_stop_the_script(setup_module, monkeypatch, capsys):
    """Exit rather than run on into an API call that cannot work."""
    _break_google_imports(monkeypatch, setup_module)
    with pytest.raises(SystemExit) as exc:
        setup_module._require_google_libs()
    assert exc.value.code == 1


def test_the_message_names_the_repair(setup_module, monkeypatch, capsys):
    """A user needs the command that fixes this, not the import error.

    The packages come from an extra, so the fix is to repair the install.
    A pip command here would write to whichever interpreter the script runs
    under, which is not always the one Hermes uses.
    """
    _break_google_imports(monkeypatch, setup_module)
    with pytest.raises(SystemExit):
        setup_module._require_google_libs()
    err = capsys.readouterr().err
    assert "hermes update" in err
    assert "uv sync --extra all" in err
    assert "[google]" in err


def test_present_sdks_are_not_an_error(setup_module):
    """The venv running the tests has the [google] extra, so this passes."""
    pytest.importorskip("googleapiclient")
    pytest.importorskip("google.auth")
    pytest.importorskip("google_auth_oauthlib")
    setup_module._require_google_libs()


def test_the_script_installs_nothing(setup_module):
    """No install path may reappear here.

    An earlier version pip-installed a hardcoded list of pins, which is a
    second copy of what the [google] extra already declares.
    """
    assert not hasattr(setup_module, "install_deps")
    assert not hasattr(setup_module, "_missing_required_packages")
