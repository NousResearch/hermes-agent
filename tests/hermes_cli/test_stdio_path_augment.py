"""Regression tests for ``_augment_path_with_known_tools`` HERMES_HOME support.

The first-launch Windows PATH prefill in ``hermes_cli.stdio`` must resolve its
Hermes-owned tool directories through ``get_hermes_home()`` — the single source
of truth ``scripts/install.ps1`` mirrors — so a custom ``HERMES_HOME`` install
(multi-profile / non-C-drive / redirected AppData) is honored rather than
silently ignored.  These tests run headless on any platform (Windows detection
and, where needed, the platform default are monkeypatched).
"""

from __future__ import annotations

import os
import sys

import hermes_cli.stdio as stdio


def test_augment_path_honors_custom_hermes_home(monkeypatch, tmp_path):
    """A custom HERMES_HOME's git\\cmd dir is prepended to PATH.

    Fails before the fix (prefill hardcodes %LOCALAPPDATA%\\hermes, which does
    not contain the tool dir); passes after (dirs resolve via get_hermes_home).
    """
    monkeypatch.setattr(stdio, "is_windows", lambda: True)

    hermes_home = tmp_path / "custom_home"
    git_cmd = hermes_home / "git" / "cmd"
    git_cmd.mkdir(parents=True)

    # LOCALAPPDATA points at a DIFFERENT dir with no hermes/git/cmd under it,
    # so the only way git_cmd gets onto PATH is via HERMES_HOME resolution.
    appdata = tmp_path / "appdata"
    appdata.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    baseline = tmp_path / "baseline"
    monkeypatch.setenv("PATH", str(baseline))

    stdio._augment_path_with_known_tools()

    entries = os.environ["PATH"].split(os.pathsep)
    assert str(git_cmd) in entries
    # baseline must be preserved (prepend, not replace)...
    assert str(baseline) in entries
    # ...and the Hermes tool dir must be PREPENDED — appear before the prior
    # PATH — so Hermes-managed tools win name collisions with the ambient PATH.
    assert entries.index(str(git_cmd)) < entries.index(str(baseline))


def test_augment_path_empty_existing_has_no_trailing_separator(monkeypatch, tmp_path):
    """An empty existing PATH must not produce an empty PATH element.

    ``os.pathsep.join([dir, ""])`` would leave a trailing separator; on Windows
    an empty PATH element resolves to the current working directory, which is
    unintended and unsafe.  This is reachable when LOCALAPPDATA is unset (custom
    HERMES_HOME), where PATH can legitimately be empty.  Fails before the fix
    (join leaves ``"...git\\cmd;"``); passes after (empty segments filtered).
    """
    monkeypatch.setattr(stdio, "is_windows", lambda: True)

    hermes_home = tmp_path / "custom_home"
    git_cmd = hermes_home / "git" / "cmd"
    git_cmd.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("PATH", "")

    stdio._augment_path_with_known_tools()

    entries = os.environ["PATH"].split(os.pathsep)
    assert "" not in entries, "empty PATH element (trailing separator) must not be emitted"
    assert str(git_cmd) in entries


def test_augment_path_default_profile_uses_localappdata(monkeypatch, tmp_path):
    """Default profile (HERMES_HOME unset) still prepends %LOCALAPPDATA%\\hermes.

    Guards the superset property: when HERMES_HOME is unset the resolved dirs
    must be byte-identical to the old %LOCALAPPDATA%\\hermes behavior.
    """
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    # Force the win32 platform default so get_hermes_home() falls back to
    # %LOCALAPPDATA%\hermes exactly as it does on a real default-profile
    # Windows install.
    monkeypatch.setattr(sys, "platform", "win32")

    appdata = tmp_path / "appdata"
    git_cmd = appdata / "hermes" / "git" / "cmd"
    git_cmd.mkdir(parents=True)

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    baseline = tmp_path / "baseline"
    monkeypatch.setenv("PATH", str(baseline))

    stdio._augment_path_with_known_tools()

    entries = os.environ["PATH"].split(os.pathsep)
    assert str(git_cmd) in entries
    # The prior PATH must be preserved and the tool dir prepended before it.
    assert str(baseline) in entries
    assert entries.index(str(git_cmd)) < entries.index(str(baseline))
