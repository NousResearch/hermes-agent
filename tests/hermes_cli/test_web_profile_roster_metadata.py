"""Regression tests for metadata exposed by the Web profile roster."""

import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def clear_roster_cache():
    from hermes_cli import web_server_profiles

    with web_server_profiles._PROFILE_ROSTER_FIELDS_CACHE_LOCK:
        web_server_profiles._PROFILE_ROSTER_FIELDS_CACHE.clear()
    yield
    with web_server_profiles._PROFILE_ROSTER_FIELDS_CACHE_LOCK:
        web_server_profiles._PROFILE_ROSTER_FIELDS_CACHE.clear()


@pytest.fixture()
def client():
    from hermes_cli import web_server

    with TestClient(web_server.app) as test_client:
        test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        yield test_client


def test_profiles_list_includes_cross_source_identity_metadata(client):
    from hermes_constants import get_hermes_home

    root = get_hermes_home()
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yaml").write_text(
        yaml.safe_dump({
            "display_name": "Asus",
            "ui_meta": {
                "hermes-bots": {"title": "Asus Bot", "color": "#abcdef"},
                "private-plugin": {"token": "must-not-roam"},
            },
        }),
        encoding="utf-8",
    )
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "avatar.png").write_bytes(b"avatar")

    response = client.get("/api/profiles")

    assert response.status_code == 200
    default = next(row for row in response.json()["profiles"] if row["name"] == "default")
    assert default["display_name"] == "Asus"
    assert default["ui_meta"] == {
        "hermes-bots": {"title": "Asus Bot", "color": "#abcdef"}
    }
    assert default["has_avatar"] is True


def test_profile_to_dict_has_stable_empty_metadata_shape():
    from hermes_cli.web_routers.profiles import _profile_to_dict

    row = _profile_to_dict(SimpleNamespace(path=""))

    assert row["ui_meta"] == {}
    assert row["has_avatar"] is False


def test_profile_roster_fields_cache_invalidates_on_metadata_change(
    monkeypatch, tmp_path
):
    from hermes_cli import web_server_profiles

    meta_path = tmp_path / "profile.yaml"
    meta_path.write_text(
        yaml.safe_dump({"ui_meta": {"hermes-bots": {"title": "One"}}}),
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    avatar_path = assets / "avatar.png"
    avatar_path.write_bytes(b"avatar")
    calls = 0
    original_safe_load = web_server_profiles.yaml.safe_load

    def counted_safe_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_safe_load(*args, **kwargs)

    monkeypatch.setattr(web_server_profiles.yaml, "safe_load", counted_safe_load)

    first = web_server_profiles._profile_roster_fields(tmp_path)
    cached = web_server_profiles._profile_roster_fields(tmp_path)
    assert first == cached
    assert calls == 1

    avatar_mtime = avatar_path.stat().st_mtime_ns
    os.utime(avatar_path, ns=(avatar_mtime + 1_000_000, avatar_mtime + 1_000_000))
    avatar_touched = web_server_profiles._profile_roster_fields(tmp_path)
    assert avatar_touched == cached
    assert calls == 1

    meta_path.write_text(
        yaml.safe_dump({"ui_meta": {"hermes-bots": {"title": "Updated"}}}),
        encoding="utf-8",
    )
    current_mtime = meta_path.stat().st_mtime_ns
    os.utime(meta_path, ns=(current_mtime + 1_000_000, current_mtime + 1_000_000))

    refreshed = web_server_profiles._profile_roster_fields(tmp_path)
    assert refreshed["ui_meta"]["hermes-bots"]["title"] == "Updated"
    assert calls == 2


def test_profile_roster_fields_cache_is_deep_copied(tmp_path):
    from hermes_cli import web_server_profiles

    (tmp_path / "profile.yaml").write_text(
        yaml.safe_dump({"ui_meta": {"hermes-bots": {"title": "Original"}}}),
        encoding="utf-8",
    )

    first = web_server_profiles._profile_roster_fields(tmp_path)
    first["ui_meta"]["hermes-bots"]["title"] = "Mutated"

    cached = web_server_profiles._profile_roster_fields(tmp_path)
    assert cached["ui_meta"]["hermes-bots"]["title"] == "Original"


def test_profile_roster_fields_cache_evicts_least_recently_used(
    monkeypatch, tmp_path
):
    from hermes_cli import web_server_profiles

    monkeypatch.setattr(web_server_profiles, "_PROFILE_ROSTER_FIELDS_CACHE_MAX", 2)
    paths = [tmp_path / name for name in ("one", "two", "three")]
    for path in paths:
        path.mkdir()
        (path / "profile.yaml").write_text(
            yaml.safe_dump({"ui_meta": {"hermes-bots": {"title": path.name}}}),
            encoding="utf-8",
        )

    web_server_profiles._profile_roster_fields(paths[0])
    web_server_profiles._profile_roster_fields(paths[1])
    web_server_profiles._profile_roster_fields(paths[0])
    web_server_profiles._profile_roster_fields(paths[2])

    assert list(web_server_profiles._PROFILE_ROSTER_FIELDS_CACHE) == [
        str(paths[0]),
        str(paths[2]),
    ]


def test_profile_roster_fields_has_stable_empty_metadata_shape(tmp_path):
    from hermes_cli import web_server_profiles

    assert web_server_profiles._profile_roster_fields(tmp_path) == {
        "ui_meta": {},
        "has_avatar": False,
    }


def test_profile_roster_fields_warns_once_when_ui_meta_is_dropped(
    monkeypatch, tmp_path, caplog
):
    from hermes_cli import web_server_profiles

    monkeypatch.setattr(web_server_profiles, "_PROFILE_ROSTER_UI_META_MAX_BYTES", 8)
    (tmp_path / "profile.yaml").write_text(
        yaml.safe_dump({"ui_meta": {"hermes-bots": {"title": "Too long"}}}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        first = web_server_profiles._profile_roster_fields(tmp_path)
        second = web_server_profiles._profile_roster_fields(tmp_path)

    assert first["ui_meta"] == {}
    assert first == second
    assert caplog.text.count("Ignoring oversized hermes-bots ui_meta") == 1


def test_profile_roster_fields_warns_once_for_non_json_ui_meta(tmp_path, caplog):
    from hermes_cli import web_server_profiles

    (tmp_path / "profile.yaml").write_text(
        yaml.safe_dump({"ui_meta": {"hermes-bots": {"weight": float("nan")}}}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        first = web_server_profiles._profile_roster_fields(tmp_path)
        second = web_server_profiles._profile_roster_fields(tmp_path)

    assert first["ui_meta"] == {}
    assert first == second
    assert caplog.text.count("Ignoring non-JSON hermes-bots ui_meta") == 1


def test_fallback_profiles_read_metadata_once_per_profile(tmp_path):
    from hermes_cli import web_server_profiles

    default_home = tmp_path / "default"
    profiles_root = tmp_path / "profiles"
    worker_home = profiles_root / "worker"
    default_home.mkdir()
    worker_home.mkdir(parents=True)
    reads = []

    class ProfilesStub:
        _PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

        @staticmethod
        def _get_default_hermes_home():
            return default_home

        @staticmethod
        def _get_profiles_root():
            return profiles_root

        @staticmethod
        def _read_config_model(_path):
            return None, None

        @staticmethod
        def _count_skills(_path):
            return 0

        @staticmethod
        def _check_gateway_running(_path):
            return False

        @staticmethod
        def _served_by_running_multiplexer(_name):
            return False

        @staticmethod
        def read_profile_meta(path):
            reads.append(Path(path))
            return {
                "description": f"{Path(path).name} description",
                "description_auto": True,
                "display_name": Path(path).name.title(),
            }

    rows = web_server_profiles._fallback_profile_dicts(ProfilesStub)

    assert reads == [default_home, worker_home]
    assert [row["display_name"] for row in rows] == ["Default", "Worker"]
    assert all(row["ui_meta"] == {} for row in rows)
    assert all(row["has_avatar"] is False for row in rows)
