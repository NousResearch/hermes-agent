"""--clone / --clone-all must not carry the memory provider's live data (#133308).

``clone_memory_provider_config`` copies ``<provider>/`` whole so the clone keeps the
provider *configured* (#120115) — but for providers whose directory is also the live data
store (mnemosyne: ``config.yaml`` + ``data/mnemosyne.db``), that byte-copies the SOURCE
profile's episodic memory into the clone: an identity leak, same class as the
state.db/sessions exclusion in ``_CLONE_ALL_HISTORY_EXCLUDE_ROOT``. These tests lock the
boundary: config travels, the source's DB never does.
"""

from pathlib import Path

import pytest

from hermes_cli.profiles import create_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _mnemosyne_source(default_home: Path) -> None:
    """A source profile running mnemosyne with non-trivial private memory."""
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n", encoding="utf-8"
    )
    (default_home / "mnemosyne").mkdir()
    (default_home / "mnemosyne" / "config.yaml").write_text(
        "embeddings: local\n", encoding="utf-8"
    )
    db_dir = default_home / "mnemosyne" / "data"
    db_dir.mkdir()
    (db_dir / "mnemosyne.db").write_text("SOURCE-EPISODIC-HISTORY", encoding="utf-8")
    (default_home / "mnemosyne" / "banks").mkdir()
    (default_home / "mnemosyne" / "banks" / "notes.md").write_text(
        "bank", encoding="utf-8"
    )


def test_clone_leaves_builtin_provider_live_data_behind(profile_env):
    default_home = profile_env / ".hermes"
    _mnemosyne_source(default_home)

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    cloned = profile_dir / "mnemosyne"
    # The clone keeps the provider configured (#120115 goal) ...
    assert (cloned / "config.yaml").read_text(
        encoding="utf-8-sig"
    ) == "embeddings: local\n"
    assert (cloned / "banks" / "notes.md").exists()
    # ... but never the source profile's private memory DB: the data dir is empty, and no
    # byte of the source's episodic history landed anywhere under the clone.
    assert (cloned / "data").is_dir()
    assert not any((cloned / "data").iterdir())
    assert "SOURCE-EPISODIC-HISTORY" not in "\n".join(
        p.read_text(errors="replace") for p in profile_dir.rglob("*") if p.is_file()
    )


def test_clone_all_drops_provider_live_data(profile_env):
    default_home = profile_env / ".hermes"
    _mnemosyne_source(default_home)

    profile_dir = create_profile("coder", clone_all=True, no_alias=True)

    cloned = profile_dir / "mnemosyne"
    assert (cloned / "config.yaml").exists()
    assert (cloned / "data").is_dir()
    assert not any((cloned / "data").iterdir())
    assert not (cloned / "data" / "mnemosyne.db").exists()


def test_clone_manifest_extends_excludes_and_travels(profile_env):
    """A provider opts its own live-data subpaths in via CLONE-EXCLUDE — a file convention,
    not a plugin hook, because the provider may live in the catalog out of tree. The manifest
    is copied too, so clones of clones stay protected."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: acme\n", encoding="utf-8"
    )
    (default_home / "acme").mkdir()
    (default_home / "acme" / "settings.json").write_text("{}", encoding="utf-8")
    (default_home / "acme" / "CLONE-EXCLUDE").write_text(
        "# live data\nstore\n", encoding="utf-8"
    )
    (default_home / "acme" / "store").mkdir()
    (default_home / "acme" / "store" / "acme.db").write_text("SECRET", encoding="utf-8")

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    cloned = profile_dir / "acme"
    assert (cloned / "settings.json").exists()
    assert (cloned / "CLONE-EXCLUDE").exists()
    assert (cloned / "store").is_dir()
    assert not any((cloned / "store").iterdir())


@pytest.mark.parametrize("bad", ["../outside", "/abs", "..", "a/../.."])
def test_clone_manifest_ignores_unsafe_entries(profile_env, bad):
    """A malformed manifest line can only ever keep data flowing — it must never widen the
    copy outside the provider directory."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: acme\n", encoding="utf-8"
    )
    (default_home / "acme").mkdir()
    (default_home / "acme" / "CLONE-EXCLUDE").write_text(f"{bad}\n", encoding="utf-8")
    (default_home / "acme" / "store").mkdir()
    (default_home / "acme" / "store" / "acme.db").write_text("k", encoding="utf-8")
    (profile_env / "outside").mkdir()

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    # The unsafe line is ignored: the subpath copies as before, nothing escapes the provider dir.
    assert (profile_dir / "acme" / "store" / "acme.db").exists()
    assert not (profile_dir.parent / "outside" / "store").exists()
    assert not (profile_dir / "abs").exists()


def test_flat_json_provider_is_unaffected(profile_env):
    """Providers storing config in a flat ``<provider>.json`` never had a directory to leak —
    their copy must stay byte-identical (default stays copy-whole)."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: mem0\n", encoding="utf-8"
    )
    (default_home / "mem0.json").write_text('{"agent_id": "hermes"}', encoding="utf-8")

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    assert (profile_dir / "mem0.json").read_text(
        encoding="utf-8-sig"
    ) == '{"agent_id": "hermes"}'


def test_unlisted_directory_provider_still_copies_whole(profile_env):
    """hindsight keeps everything-but-credentials in its dir; with no builtin entry and no
    manifest, --clone must copy it whole exactly as before (#120115 behavior unchanged)."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: hindsight\n", encoding="utf-8"
    )
    (default_home / "hindsight").mkdir()
    (default_home / "hindsight" / "config.json").write_text(
        '{"bank_id": "hermes"}', encoding="utf-8"
    )
    (default_home / "hindsight" / "data").mkdir()
    (default_home / "hindsight" / "data" / "cache.bin").write_bytes(b"\x00\x01")

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    assert (profile_dir / "hindsight" / "config.json").exists()
    assert (
        profile_dir / "hindsight" / "data" / "cache.bin"
    ).read_bytes() == b"\x00\x01"
