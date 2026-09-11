"""The managed runtime is machine-scoped, not profile-scoped.

Engine binaries, models, presets, and server state are machine assets: a
second profile must reuse them, never re-download 20 GB of GGUFs or fight
the running server for its port. Profile-scoped decisions (default model,
enabled flag) stay in each profile's config.yaml."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """A NAMED-profile HERMES_HOME under <root>/profiles/<name>."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    # hermes_constants memoizes root resolution per (native, env) pair;
    # reload to make the new env authoritative for this test.
    import hermes_constants

    importlib.reload(hermes_constants)
    yield root, profile
    importlib.reload(hermes_constants)


def test_models_and_runtimes_resolve_to_the_shared_root(profile_home):
    root, profile = profile_home
    import hermes_cli.local_runtime.binaries as binaries
    import hermes_cli.local_runtime.bootstrap as bootstrap

    models = bootstrap.models_dir()
    runtimes = binaries.runtimes_root()

    assert models == root / "models", (
        f"models dir leaked into the profile: {models}")
    assert runtimes == root / "runtimes" / "llamacpp", (
        f"runtimes dir leaked into the profile: {runtimes}")
    assert "profiles" not in models.parts
    assert "profiles" not in runtimes.parts


def test_all_runtime_state_follows_runtimes_root(profile_home):
    """Presets, window overrides, server state, and the api key all live
    under runtimes_root() — one resolver, so profile-scoping bugs cannot
    come back one file at a time."""
    root, profile = profile_home
    from hermes_cli.local_runtime.growth import window_overrides_path
    from hermes_cli.local_runtime.presets import read_preset_decisions
    import hermes_cli.local_runtime.binaries as binaries

    shared = root / "runtimes" / "llamacpp"
    assert window_overrides_path() == shared / "window_overrides.json"
    # read_preset_decisions' default path must be the shared INI: write a
    # section there and read it back through the default-path branch.
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "presets.ini").write_text("[m1]\nctx-size = 65536\n",
                                        encoding="utf-8")
    assert "m1" in read_preset_decisions()


def test_default_profile_paths_unchanged(tmp_path, monkeypatch):
    """HERMES_HOME at the root itself (default profile) resolves exactly
    as before the scoping change — no migration for existing installs."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants

    importlib.reload(hermes_constants)
    try:
        import hermes_cli.local_runtime.binaries as binaries
        import hermes_cli.local_runtime.bootstrap as bootstrap

        assert bootstrap.models_dir() == root / "models"
        assert binaries.runtimes_root() == root / "runtimes" / "llamacpp"
    finally:
        importlib.reload(hermes_constants)
