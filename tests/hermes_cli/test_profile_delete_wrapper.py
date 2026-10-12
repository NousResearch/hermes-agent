"""`hermes profile delete` removes the profile's wrapper on Windows (#131753).

Windows wrappers are ``<alias>.bat``; delete used to look only for the bare name, so the
``.bat`` survived as an orphan alias. A wrapper written before Windows wrappers became
``.bat`` (#34796) is bare-named and must still go.
"""

import sys
import types
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profiles import create_profile, create_wrapper_script, delete_profile


@pytest.mark.parametrize("legacy", [False, True], ids=["bat", "pre-bat-bare-name"])
def test_delete_removes_the_windows_wrapper(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # profiles.py only: a global sys.platform patch makes modules first imported by
    # delete_profile cache a Windows lock backend (msvcrt) for the rest of the run.
    monkeypatch.setattr(profiles, "sys", types.SimpleNamespace(**{**vars(sys), "platform": "win32"}))
    for stub in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_stop_profile_backends"):
        monkeypatch.setattr(profiles, stub, lambda *a, **k: None)
    profile_dir = create_profile("imported", no_alias=True)
    if legacy:
        wrapper = tmp_path / ".local" / "bin" / "imported"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text('#!/bin/sh\nexec hermes -p imported "$@"\n', encoding="utf-8")
    else:
        wrapper = create_wrapper_script("imported")
        assert wrapper is not None and wrapper.name == "imported.bat"

    delete_profile("imported", yes=True)

    assert not profile_dir.exists()
    assert not wrapper.exists()
