"""The Bitwarden and 1Password secret caches and the Photon sidecar record are covered by every guard
over credential stores (profile export, file tools, gateway media delivery, the dashboard Files API,
backups), at their current location and at the one earlier releases used.

The current paths come from the writers' own path functions, so moving a store without updating the
guards, or dropping it from one of them, fails here. The stores sit directly in the profile home: a
sandbox that hides files by path can hide them safely only there (a writable folder above a hidden file
can be renamed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Where earlier releases kept the same stores; Hermes deletes a copy there unread.
PREVIOUS_LOCATIONS = (
    "cache/bws_cache.json", "cache/bws_cache.enc.json", "cache/op_cache.json", "runtime/photon-sidecar.json",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _current(home: Path) -> tuple[str, ...]:
    from agent.secret_sources import bitwarden as bw
    from agent.secret_sources import onepassword as op
    from plugins.platforms.photon import adapter as photon

    paths = (bw._disk_cache_path(home), bw._encrypted_disk_cache_path(home), op._STORE.disk.path(home),
             photon._runtime_record_path())
    return tuple(p.relative_to(home).as_posix() for p in paths)


def _stores(home: Path) -> tuple[str, ...]:
    return (*_current(home), *PREVIOUS_LOCATIONS)


def _touch(base: Path, rel: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def test_the_stores_sit_at_the_top_of_the_profile_home(home):
    assert _current(home) == ("bws_cache.json", "bws_cache.enc.json", "op_cache.json", "photon-sidecar.json")


def test_the_shared_list_names_both_locations(home):
    from hermes_cli.home_data_layout import LEGACY_SECRET_CACHE_PATHS, SECRET_CACHE_FILES, SECRET_CACHE_STORES

    assert SECRET_CACHE_FILES == _current(home)
    assert LEGACY_SECRET_CACHE_PATHS == PREVIOUS_LOCATIONS
    assert SECRET_CACHE_STORES == (*SECRET_CACHE_FILES, *LEGACY_SECRET_CACHE_PATHS)


def test_profile_credential_paths_list_both_locations(home):
    from hermes_cli.profiles import PROFILE_CREDENTIAL_PATHS, profile_path_is_private

    for rel in _stores(home):
        assert rel in PROFILE_CREDENTIAL_PATHS, rel
        assert profile_path_is_private(tuple(rel.split("/"))), rel


def test_file_tools_refuse_to_read_or_write_them(home, monkeypatch):
    import agent.file_safety as fs

    root = home.parent / "root"
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    for base in (home, root):
        for rel in _stores(home):
            path = _touch(base, rel)
            assert fs.get_read_block_error(str(path)), f"read allowed: {path}"
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_gateway_media_delivery_refuses_them(home, monkeypatch):
    from gateway.platforms import base

    monkeypatch.setattr(base, "_HERMES_HOME", home)
    monkeypatch.setattr(base, "_HERMES_ROOT", home)
    denied = base._media_delivery_denied_paths()
    for rel in _stores(home):
        path = _touch(home, rel)
        assert path in denied, rel
        assert base.validate_media_delivery_path(str(path)) is None, rel


def test_dashboard_files_api_hides_them(home):
    from hermes_cli.web_routers.files import _is_sensitive_path

    for rel in _stores(home):
        assert _is_sensitive_path(home / rel), rel


def test_backups_leave_them_out(home):
    from hermes_cli.backup import _should_exclude

    for rel in _current(home):
        assert _should_exclude(Path(rel)), rel
        assert _should_exclude(Path("profiles", "work", rel)), rel
    for rel in (r for r in PREVIOUS_LOCATIONS if r.startswith("cache/")):  # cache/ is left out as a whole
        assert _should_exclude(Path(rel)), rel
    # Same names deeper in the tree are user data.
    assert not _should_exclude(Path("skills", "demo", "op_cache.json"))
