"""A transient read failure on auth.json must not wipe the credential store.

``_load_auth`` treated every failure as "start from empty" and returned ``{}``.
Every writer in the module does read-modify-write into ``_save_auth``, which
publishes the *whole* file via ``tmp.replace(path)`` — and ``replace`` only
needs a writable parent directory, so a present-but-unreadable ``auth.json``
does not stop the overwrite. One ``OSError`` (EACCES after a root-owned write,
EIO, a stalled mount) followed by any photon write therefore replaced the
shared store with a photon-only file, destroying every other provider's
credentials and OAuth refresh tokens.

This is the same defect ``hermes_cli.auth._load_auth_store`` was fixed for in
#75206 / #75258; the plugin carried an unpatched copy of the pre-fix shape.

The two arms are now separated the way the core module separates them:

* ``OSError`` — the contents are not known to be bad, so re-raise and leave the
  file on disk untouched.
* unparseable JSON / non-UTF-8 bytes — genuine corruption, degrade to an empty
  store, but only *after* preserving a copy at ``auth.json.corrupt``, because
  the next ``_save_auth`` publishes the whole file and a truncated store
  usually still holds the other providers' tokens verbatim.

A BOM-prefixed store is read with ``utf-8-sig`` and is *not* corruption — the
core module has always read it that way, and classifying it as corrupt was a
wipe vector for a perfectly healthy file.
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
from pathlib import Path

import pytest

from plugins.platforms.photon import auth as photon_auth

# Mirrors the scrubbing in the sibling ``tmp_hermes_home`` fixture
# (tests/plugins/platforms/photon/test_auth.py): ``save_env_value`` mutates
# ``os.environ`` directly, so these must be cleared before and after or state
# leaks between tests. Kept in sync deliberately; consolidating both fixtures
# into a package-level conftest is a separate cleanup.
_PHOTON_ENV = (
    "PHOTON_PROJECT_ID",
    "PHOTON_PROJECT_SECRET",
    "PHOTON_DASHBOARD_PROJECT_ID",
    "PHOTON_SPECTRUM_HOST",
    "PHOTON_ALLOWED_USERS",
    "PHOTON_HOME_CHANNEL",
)

_OTHER_PROVIDER_TOKENS = {"refresh_token": "single-use-refresh"}


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in _PHOTON_ENV:
        monkeypatch.delenv(key, raising=False)
    yield home
    for key in _PHOTON_ENV:
        os.environ.pop(key, None)


def _seed_shared_store(home: Path, encoding: str = "utf-8") -> Path:
    """Write an auth.json holding another provider's credentials."""
    path = home / "auth.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"some-provider": {"tokens": _OTHER_PROVIDER_TOKENS}},
                "credential_pool": {"anthropic": [{"access_token": "anthropic-token"}]},
            },
            indent=2,
        ),
        encoding=encoding,
    )
    return path


def _assert_other_providers_survived(path: Path) -> None:
    surviving = json.loads(path.read_text(encoding="utf-8-sig"))
    assert surviving["providers"]["some-provider"]["tokens"] == _OTHER_PROVIDER_TOKENS
    assert surviving["credential_pool"]["anthropic"] == [
        {"access_token": "anthropic-token"}
    ]


def test_unreadable_store_raises_instead_of_degrading(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _seed_shared_store(hermes_home)
    before = path.read_bytes()

    def _boom(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied")

    # A context, not monkeypatch.undo(): the fixture and the test share one
    # function-scoped MonkeyPatch, so undo() would also revert the HERMES_HOME
    # redirection and point every later call at the real ~/.hermes/auth.json.
    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", _boom)
        with pytest.raises(OSError):
            photon_auth._load_auth()

    assert path.read_bytes() == before


def test_write_does_not_erase_other_providers_when_store_unreadable(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression itself: a read failure must not cost the whole store.

    ``suppress`` rather than ``pytest.raises`` deliberately — the assertion that
    must fail against the unpatched source is the *survival* of the other
    provider's credentials, not the presence of the raise. With
    ``pytest.raises`` the unpatched run stops at "DID NOT RAISE" and never
    reaches the assertions this test is named for.
    """
    path = _seed_shared_store(hermes_home)
    before = path.read_bytes()

    real_open = Path.open

    def _boom_on_auth_json(self, *args, **kwargs):
        if self.name == "auth.json":
            raise OSError(errno.EIO, "Input/output error")
        return real_open(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", _boom_on_auth_json)
        with contextlib.suppress(OSError):
            photon_auth.store_photon_token("synthetic-token")

    _assert_other_providers_survived(path)
    assert path.read_bytes() == before


def test_bom_prefixed_store_is_not_corruption(hermes_home: Path) -> None:
    """A BOM'd store is valid to the core module and must survive a photon write.

    ``json.load`` over an ``encoding="utf-8"`` handle raises "Unexpected UTF-8
    BOM" on a file the core module reads without complaint, so reading as plain
    utf-8 turned a healthy store into a corrupt one — and the degrade arm then
    erased it on the next write.
    """
    path = _seed_shared_store(hermes_home, encoding="utf-8-sig")

    assert photon_auth._load_auth()["providers"]["some-provider"]["tokens"] == (
        _OTHER_PROVIDER_TOKENS
    )

    photon_auth.store_photon_token("synthetic-token")

    _assert_other_providers_survived(path)
    assert not path.with_suffix(".json.corrupt").exists()


def test_corrupt_json_degrades_to_empty_but_preserves_a_copy(
    hermes_home: Path,
) -> None:
    """Unparseable JSON still degrades — but the original must survive on disk.

    A truncated store is the realistic corruption case and it still holds the
    other providers' tokens verbatim, so the degrade arm has to preserve a copy
    before the next full-file ``_save_auth`` overwrites the only one.
    """
    path = hermes_home / "auth.json"
    truncated = json.dumps(
        {"providers": {"some-provider": {"tokens": _OTHER_PROVIDER_TOKENS}}}
    )[:-3]
    path.write_text(truncated, encoding="utf-8")

    assert photon_auth._load_auth() == {}

    corrupt_copy = path.with_suffix(".json.corrupt")
    assert corrupt_copy.exists()
    assert corrupt_copy.read_text(encoding="utf-8") == truncated
    assert "single-use-refresh" in corrupt_copy.read_text(encoding="utf-8")

    # The degrade is still a degrade: a photon write publishes a photon-only
    # store. That is the established behaviour — recoverability comes from the
    # preserved copy, which must outlive the write.
    photon_auth.store_photon_token("synthetic-token")
    assert corrupt_copy.read_text(encoding="utf-8") == truncated


def test_missing_store_still_returns_empty(hermes_home: Path) -> None:
    assert not (hermes_home / "auth.json").exists()
    assert photon_auth._load_auth() == {}
