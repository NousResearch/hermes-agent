"""Routed-temp contract: parent-side staging temp follows the active (context-routed) home.

A multiplexed gateway/Desktop backend boots with one profile's scratch exported into the
process-global ``TMPDIR``/``TMP``/``TEMP`` (tagged by ``HERMES_SCRATCH_DIR``), then serves
every profile through per-turn context overrides. ``routed_tmpdir()`` is what parent-side
staging (code kernel dirs, RPC sockets) must consult so profile A's turn never stages into
the startup profile's scratch (#138501).
"""

import tempfile

from hermes_constants import (
    SCRATCH_DIR_MARKER_ENV,
    SCRATCH_TMP_ENV_VARS,
    get_hermes_home_override,
    reset_hermes_home_override,
    routed_tmpdir,
    set_hermes_home_override,
)
import pytest


@pytest.fixture
def home_override():
    tokens = []

    def _set(path):
        tokens.append(set_hermes_home_override(path))
        return str(path)

    yield _set
    for token in reversed(tokens):
        reset_hermes_home_override(token)


def _export_hermes_owned_tmp(monkeypatch, scratch: str) -> None:
    """Make the process env look like a backend booted under that scratch dir."""
    monkeypatch.setenv(SCRATCH_DIR_MARKER_ENV, scratch)
    for key in SCRATCH_TMP_ENV_VARS:
        monkeypatch.setenv(key, scratch)


def test_routed_profile_scratch_wins_over_boot_profile_scratch(
    monkeypatch, tmp_path, home_override
):
    """The reported bug: backend booted under profile B, turn routed to profile A — staging
    must use A's scratch, not the ambient temp left over from B."""
    home_b = tmp_path / "profiles" / "beta"
    scratch_b = str(home_b / "cache" / "scratch")
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _export_hermes_owned_tmp(monkeypatch, scratch_b)
    home_override(tmp_path / "profiles" / "alpha")

    assert routed_tmpdir() == str(tmp_path / "profiles" / "alpha" / "cache" / "scratch")


def test_without_override_ambient_boot_scratch_stands(monkeypatch, tmp_path):
    """A dedicated per-profile process (no context override): the boot scratch is already
    this profile's, so the ambient value is returned unchanged."""
    assert get_hermes_home_override() is None
    home_b = tmp_path / "profiles" / "beta"
    scratch_b = str(home_b / "cache" / "scratch")
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _export_hermes_owned_tmp(monkeypatch, scratch_b)

    assert routed_tmpdir() == scratch_b


def test_user_set_temp_var_keeps_ambient_default(monkeypatch, tmp_path, home_override):
    """A user/OS temp var (macOS ``/var/folders``, Windows ``%TEMP%``) is never overridden:
    the ambient ``tempfile`` default is returned even while a profile is routed."""
    home_override(tmp_path / "profiles" / "alpha")
    monkeypatch.setenv("TMPDIR", "/var/folders/user-set-tmp")
    monkeypatch.setattr(tempfile, "tempdir", "/ambient/system/tmp", raising=False)

    assert routed_tmpdir() == "/ambient/system/tmp"


def test_partially_set_user_triple_counts_as_override(
    monkeypatch, tmp_path, home_override
):
    """One user-set var among TMPDIR/TMP/TEMP (marker unequal) is respected — mirrors
    apply_scratch_tmp_env, which leaves a partially user-set triple exactly as found."""
    home_b = tmp_path / "profiles" / "beta"
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _export_hermes_owned_tmp(monkeypatch, str(home_b / "cache" / "scratch"))
    monkeypatch.setenv("TMP", "/user/chosen/tmp")
    monkeypatch.setattr(tempfile, "tempdir", "/ambient/system/tmp", raising=False)
    home_override(tmp_path / "profiles" / "alpha")

    assert routed_tmpdir() == "/ambient/system/tmp"
