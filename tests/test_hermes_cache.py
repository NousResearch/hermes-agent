"""Store-owned inference downloads move without adopting custom storage or user state."""

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
def test_download_owners_move_together_without_moving_user_state(cache_layout, monkeypatch, layout):
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
    speech_relative = str(active.relative_to(home) / "cache/whisper.cpp")
    callers[speech_relative] = transcription_whisper_cpp._model_dir
    should_move = layout in ("default", "profile")
    for relative, resolve in callers.items():
        source = home / relative
        source.mkdir(parents=True)
        weight = source / "download.bin"
        weight.write_bytes(b"downloaded bytes")
        identity = weight.stat().st_ino
        expected = cache / "hermes-inference" / relative if should_move else source
        assert resolve() == expected
        assert (expected / weight.name).read_bytes() == b"downloaded bytes"
        assert (expected / weight.name).stat().st_ino == identity  # Rename, no second model copy.
        assert source.exists() is not should_move
        assert resolve() == expected  # Restart/repeated status lookup is idempotent.
    assert state.read_bytes() == b"user state"
    if not should_move:
        assert not cache.exists()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("obstacle", ["collision", "locked", "link", "nested-link"])
def test_migration_preserves_existing_bytes_and_can_resume(cache_layout, tmp_path, obstacle):
    import ctypes

    home, cache = cache_layout
    source = home / "models"
    source.mkdir(parents=True)
    target = cache / "hermes-inference" / "models"
    weight = source / "model.gguf"
    weight.write_bytes(b"original model")
    if obstacle in ("link", "nested-link"):
        linked = tmp_path / "user-models"
        linked.mkdir()
        (linked / "keep.gguf").write_bytes(b"user model")
        link = source / "external" if obstacle == "nested-link" else home / "linked-models"
        import _winapi

        _winapi.CreateJunction(str(linked), str(link))
        relative = "models" if obstacle == "nested-link" else "linked-models"
        assert hermes_cache.managed_cache_dir(relative) == home / relative
        assert (linked / "keep.gguf").read_bytes() == b"user model"
        assert weight.read_bytes() == b"original model"
        assert not cache.exists()
        return

    if obstacle == "collision":
        target.mkdir(parents=True)
        (target / weight.name).write_bytes(b"different model")
        with pytest.raises(RuntimeError, match="existing files were preserved"):
            bootstrap.models_dir()
        assert (target / weight.name).read_bytes() == b"different model"
        (target / weight.name).unlink()
    else:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.CreateFileW(str(weight), 0x80000000, 0, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value
        try:
            with pytest.raises(RuntimeError, match="existing files were preserved"):
                bootstrap.models_dir()
        finally:
            kernel.CloseHandle(handle)
    assert weight.read_bytes() == b"original model"
    assert bootstrap.models_dir() == target
    assert (target / weight.name).read_bytes() == b"original model"
    assert not source.exists()
