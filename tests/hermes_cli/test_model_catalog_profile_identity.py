"""A warm manifest must not supply another profile's catalog or silent default."""

import json
import os
import time

import pytest

from hermes_cli import model_catalog, models
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_catalogs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    model_catalog.reset_cache()
    catalogs = []
    timestamp = time.time()
    for name in ("first", "second"):
        home = tmp_path / "profiles" / name
        path = home / "cache" / "model_catalog.json"
        path.parent.mkdir(parents=True)
        manifest = {"version": 1, "providers": {"nous": {"models": [
            {"id": f"{name}-model", "default": True}
        ]}}}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        # Copied/seeded caches can share timestamps; identity must include the path.
        os.utime(path, (timestamp, timestamp))
        catalogs.append((home, manifest))
    yield catalogs
    model_catalog.reset_cache()


def test_silent_default_uses_current_profile_after_another_profile_warms_cache(profile_catalogs):
    first, second = profile_catalogs
    token = set_hermes_home_override(first[0])
    try:
        assert model_catalog.get_catalog() == first[1]
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(second[0])
    try:
        assert models.pick_silent_default_model(
            ["first-model", "second-model"], provider="nous"
        ) == "second-model"
    finally:
        reset_hermes_home_override(token)


def test_equal_timestamps_do_not_alias_profile_catalogs(profile_catalogs):
    for home, manifest in [*profile_catalogs, profile_catalogs[0]]:
        token = set_hermes_home_override(home)
        try:
            assert model_catalog.get_catalog() == manifest
        finally:
            reset_hermes_home_override(token)
