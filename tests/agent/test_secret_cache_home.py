"""resolve_cache_home() must match load_hermes_dotenv()'s home resolution.

The disk cache's fallback (``home_path=None``) resolves the *environment* home
— ``HERMES_HOME`` with the ``HT_HOME`` alias, else the platform default — and
deliberately NOT ``get_hermes_home()``: the per-task profile contextvar
override would silently relocate the cache between the threaded path (callers
passing the home ``load_hermes_dotenv`` resolved) and the fallback path,
causing cache misses and duplicate cache files within one process.
"""

from pathlib import Path

import hermes_constants
from agent.secret_sources._cache import resolve_cache_home


def test_threaded_home_passes_through(tmp_path):
    assert resolve_cache_home(tmp_path) == tmp_path


def test_fallback_reads_env_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HT_HOME", raising=False)
    assert resolve_cache_home() == tmp_path


def test_fallback_honors_ht_home_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HT_HOME", str(tmp_path))
    assert resolve_cache_home() == tmp_path


def test_fallback_ignores_profile_contextvar(monkeypatch, tmp_path):
    # An active profile context must NOT relocate the cache away from the
    # env-resolved home the threaded path uses.
    env_home = tmp_path / "env-home"
    profile_home = tmp_path / "profiles" / "coder"
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    monkeypatch.delenv("HT_HOME", raising=False)
    token = hermes_constants.set_hermes_home_override(profile_home)
    try:
        assert resolve_cache_home() == env_home
        # Sanity: get_hermes_home() WOULD have returned the profile home —
        # the divergence this test pins down.
        assert hermes_constants.get_hermes_home() == Path(profile_home)
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_fallback_defaults_to_platform_home(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HT_HOME", raising=False)
    assert resolve_cache_home() == hermes_constants._get_platform_default_hermes_home()
