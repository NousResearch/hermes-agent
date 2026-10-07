"""Package cache selection leaves other installations' downloads and state untouched."""

import pytest

import hermes_cache
import hermes_constants
from hermes_cli.local_runtime import binaries, bootstrap
from pm import paths
from tools import transcription_whisper_cpp


@pytest.fixture
def cache_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    home = hermes_constants._get_platform_default_hermes_home()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cache = tmp_path / "package" / "LocalCache"
    monkeypatch.setattr(hermes_cache, "local_cache_folder", lambda: cache)
    sealed = tmp_path / "payload"
    sealed.mkdir()
    (sealed / "manifest.json").write_text("{}")
    monkeypatch.setattr(paths, "store_root", lambda: sealed / "tools")
    return home, cache


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("layout", ["default", "profile", "custom", "unpackaged"])
def test_download_paths_leave_existing_installations_untouched(cache_layout, monkeypatch, layout):
    home, cache = cache_layout
    if layout == "custom":
        home = home.parent / "user-selected"
    active = home / "profiles" / "work" if layout == "profile" else home
    monkeypatch.setenv("HERMES_HOME", str(active))
    if layout == "unpackaged":
        monkeypatch.setattr(hermes_cache, "local_cache_folder", lambda: None)
    active.mkdir(parents=True)
    state = active / "config.yaml"
    state.write_bytes(b"user state")
    callers = {
        "models": bootstrap.models_dir,
        "runtimes/llamacpp": binaries.runtimes_root,
        "cache/partials": paths.partials_root,
        "tools": paths.writable_store_root,
    }
    callers[str(active.relative_to(home) / "cache/whisper.cpp")] = transcription_whisper_cpp._model_dir
    packaged = layout in ("default", "profile")
    for relative, resolve in callers.items():
        original = home / relative
        original.mkdir(parents=True)
        weight = original / "download.bin"
        weight.write_bytes(b"existing install")
        identity = weight.stat().st_ino
        expected = cache / "hermes-inference" / relative if packaged else original
        assert resolve() == expected
        assert weight.read_bytes() == b"existing install"
        assert weight.stat().st_ino == identity
        if packaged:
            assert not expected.exists()  # Path lookup creates/moves/copies nothing.
            expected.mkdir(parents=True)
            (expected / weight.name).write_bytes(b"package download")
            assert resolve() == expected
            assert weight.read_bytes() == b"existing install"
    assert state.read_bytes() == b"user state"
    assert not (home / ".cache-migration.lock").exists()
    if not packaged:
        assert not cache.exists()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("linked", [False, True])
def test_packaged_inference_does_not_adopt_old_profile_models(cache_layout, tmp_path, linked):
    import _winapi

    home, cache = cache_layout
    profile = home / "profiles/work"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}")
    existing = tmp_path / "user-models" if linked else profile / "models"
    existing.mkdir()
    if linked:
        _winapi.CreateJunction(str(existing), str(profile / "models"))
        _winapi.CreateJunction(str(existing), str(home / "models"))
    (existing / "keep.gguf").write_bytes(b"other installation's model")
    assert bootstrap.models_dir() == cache / "hermes-inference/models"
    assert bootstrap.adopt_legacy_models() == []
    assert bootstrap.staged_models() == []
    assert (existing / "keep.gguf").read_bytes() == b"other installation's model"
    assert not cache.exists()
