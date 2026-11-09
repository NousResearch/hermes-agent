"""The blob storage port: resolution rules and the local backend's contract.

Mirrors the session storage port's tests (``test_storage_port.py``): a named provider is
resolved by explicit name and never falls back to the local store; the local backend is
today's files at today's paths, serves its own profile only, and refuses a write at a level
it does not store.
"""

import textwrap

import pytest

from hermes_blob_port import (
    LEVELS, BlobKeyRefused, BlobLevelRefused, BlobNotFound, BlobPinRefused, BlobStore,
    BlobStoreProvider, resolve_blob_store,
)
from hermes_blob_port_local import LocalBlobStoreProvider

_PLUGIN = '''
from pathlib import Path

from hermes_blob_port import BlobStoreProvider

Path(__file__).with_name("imported").write_text("loaded")


class Probe(BlobStoreProvider):
    @property
    def name(self):
        return "probe"

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("this backend's role bypasses row security")


def register(ctx):
    ctx.register_blob_storage(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_blob_store(name)
        assert isinstance(provider, LocalBlobStoreProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(BlobStoreProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, home=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_blob_store("probe")
    assert isinstance(provider, BlobStoreProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):",
        "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_blob_store("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="bucket"):
        resolve_blob_store("bucket")


def test_the_local_backend_serves_todays_paths(tmp_path, monkeypatch):
    """Each family's key names today's file under the profile home, and a populated legacy
    cache directory is honoured exactly as today's callers honour it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = LocalBlobStoreProvider().open()
    assert isinstance(store, BlobStore)
    store.put("media/image/x.png", b"photo", level="agent")
    store.put("pets/ghost/sheet.webp", b"sheet", level="agent")
    store.put("skills/greet/SKILL.md", b"skill", level="agent")
    store.put("attachments/room.bin", b"attach", level="agent")
    assert (tmp_path / "cache" / "images" / "x.png").read_bytes() == b"photo"
    assert (tmp_path / "pets" / "ghost" / "sheet.webp").read_bytes() == b"sheet"
    assert (tmp_path / "skills" / "greet" / "SKILL.md").read_bytes() == b"skill"
    assert (tmp_path / "hosted-room-attachments" / "room.bin").read_bytes() == b"attach"
    legacy = tmp_path / "video_cache"
    legacy.mkdir()
    (legacy / "clip.mp4").write_bytes(b"clip")
    store.put("media/video/clip2.mp4", b"clip2", level="agent")
    assert (legacy / "clip2.mp4").read_bytes() == b"clip2"
    assert not (tmp_path / "cache" / "videos").exists()


def test_the_local_backend_lists_across_families(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = LocalBlobStoreProvider().open()
    for key in ("media/image/a.png", "media/audio/b.ogg", "pets/g/c.webp", "skills/s/SKILL.md"):
        store.put(key, b"x", level="agent")
    assert store.list("") == ["media/audio/b.ogg", "media/image/a.png", "pets/g/c.webp", "skills/s/SKILL.md"]
    assert store.list("pets/") == ["pets/g/c.webp"]
    assert store.list("media/im") == ["media/image/a.png"]


def test_the_local_backend_refuses_any_pin_but_its_own_profile(tmp_path, monkeypatch):
    home = tmp_path / "mine"
    home.mkdir()
    store = LocalBlobStoreProvider().open(home)
    other = tmp_path / "theirs"
    other.mkdir()
    with pytest.raises(BlobPinRefused):
        store.get("pets/g/c.webp")
    monkeypatch.setenv("HERMES_HOME", str(other))
    with pytest.raises(BlobPinRefused):
        store.get("pets/g/c.webp")
        # ^ no scope bound: the ambient profile is not the store's own


def test_a_write_at_a_level_the_local_backend_does_not_store_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = LocalBlobStoreProvider().open()
    with pytest.raises(BlobLevelRefused, match="participant"):
        store.put("media/image/x.png", b"x", level="participant")
    with pytest.raises(BlobLevelRefused, match="nonsense"):
        store.put("media/image/x.png", b"x", level="nonsense")
    assert not store.exists("media/image/x.png")
    assert "participant" in LEVELS  # the level exists; this backend just does not store it


def test_a_missing_blob_and_a_key_that_names_nothing_look_the_same(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = LocalBlobStoreProvider().open()
    with pytest.raises(BlobNotFound):
        store.get("pets/ghost/absent.webp")
    assert store.delete("pets/ghost/absent.webp", level="agent") is False
    assert not store.exists("pets/ghost/absent.webp")


def test_a_key_of_a_family_the_local_backend_does_not_serve_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = LocalBlobStoreProvider().open()
    for key in ("bogus/x.bin", "media/bogus/x.bin", "media"):
        with pytest.raises(BlobKeyRefused):
            store.put(key, b"x", level="agent")
        with pytest.raises(BlobKeyRefused):
            store.get(key)
