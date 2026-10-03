"""Regression for #128632: ``load_config_readonly()`` must not materialise the home tree.

The read-only entry point's contract says callers must not see side effects. The
implementation went through ``_load_config_impl``, which called ``ensure_hermes_home()``
on every cache miss — so a plugin enable/disable list probe or a dashboard auth check
for a fresh ``HERMES_HOME`` would create the directory and seed ``SOUL.md``. Read-only
must skip the ensure on cache miss; the write path (``load_config``, ``save_config_value``,
``_write_user_config``) keeps the original behaviour.
"""

from __future__ import annotations

import pytest

from hermes_cli import config as cfg


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Each test gets a fresh cache so the readonly path actually exercises a miss."""
    cache = getattr(cfg, "_LOAD_CONFIG_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    yield
    if isinstance(cache, dict):
        cache.clear()


def test_load_config_readonly_does_not_create_home_on_cache_miss(tmp_path, monkeypatch):
    missing = tmp_path / "readonly-target"
    assert not missing.exists()
    monkeypatch.setenv("HERMES_HOME", str(missing))

    snapshot = cfg.load_config_readonly()
    assert isinstance(snapshot, dict)

    assert not missing.exists(), (
        "load_config_readonly() created the home tree on a cache miss: "
        f"{sorted(p.name for p in missing.iterdir())}"
    )


def test_load_config_still_creates_home(tmp_path, monkeypatch):
    """The fix is scoped to the readonly path; the write path must keep ensuring the home."""
    missing = tmp_path / "full-target"
    monkeypatch.setenv("HERMES_HOME", str(missing))

    cfg.load_config()
    assert missing.is_dir(), "load_config() must still ensure_hermes_home() on cache miss"


def test_load_config_readonly_returns_defaults_when_home_missing(tmp_path, monkeypatch):
    """A read against an empty home must still return the merged default config."""
    missing = tmp_path / "readonly-defaults"
    monkeypatch.setenv("HERMES_HOME", str(missing))

    snapshot = cfg.load_config_readonly()
    assert "agent" in snapshot, "expected merged DEFAULT_CONFIG keys in the readonly snapshot"
    assert not missing.exists()