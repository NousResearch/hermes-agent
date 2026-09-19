"""``/api/profiles`` carries each profile's declared roster group.

Grouping already has a home: a profile's ``ui_meta`` block in ``profile.yaml``, where the
Desktop roster keeps the user-made section a bot is filed under
(``ui_meta['hermes-bots'].sectionId`` — membership lives on the bot, so it rides the
profile; see ``apps/desktop/src/plugins/hermes-bots/user-sections.ts``). The gateway's
``profiles.list`` row already ships that blob (``tui_gateway/methods_profiles.py``), but
``/api/profiles`` — what the Desktop sidebar and its profile pickers read — did not, so no
client could group. Additive and default-off: no declaration, no key.
"""

import pytest

# The payload a client sees today, unchanged for a profile that declares no group.
_HISTORICAL_KEYS = {
    "name", "path", "is_default", "model", "provider", "has_env", "skill_count",
    "gateway_running", "description", "description_auto", "display_name",
    "distribution_name", "distribution_version", "distribution_source", "has_alias",
}


@pytest.fixture
def profiles_on_disk(monkeypatch):
    """An isolated default home plus named profiles, each with a config.yaml."""
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    default_home.mkdir(parents=True, exist_ok=True)
    (default_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    def add(name: str, profile_yaml: str = "") -> None:
        home = profiles_root / name
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        if profile_yaml:
            (home / "profile.yaml").write_text(profile_yaml, encoding="utf-8")

    add("analyst", "ui_meta:\n  hermes-bots:\n    sectionId: crypto-desk\n")
    add("audit")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return add


@pytest.fixture
def client(profiles_on_disk):
    try:
        from fastapi import FastAPI
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_routers import profiles as profiles_routes

    app = FastAPI()
    app.include_router(profiles_routes.router)
    return TestClient(app)


def _by_name(client) -> dict:
    response = client.get("/api/profiles")
    assert response.status_code == 200
    return {entry["name"]: entry for entry in response.json()["profiles"]}


def test_declared_group_rides_the_profile_list(client):
    assert _by_name(client)["analyst"]["ui_meta"] == {"hermes-bots": {"sectionId": "crypto-desk"}}


def test_undeclared_profile_payload_is_byte_for_byte_what_it_was(client):
    undeclared = _by_name(client)["audit"]
    assert "ui_meta" not in undeclared
    assert set(undeclared) == _HISTORICAL_KEYS


def test_group_ids_that_collide_with_profile_names_still_list_everyone(client, profiles_on_disk):
    """A group id is not a profile name and need not look like one: ids may repeat a
    profile's name, or carry a slash the profile-name regex forbids. Listing stays whole."""
    profiles_on_disk("derivatives", "ui_meta:\n  hermes-bots:\n    sectionId: crypto-desk/derivatives\n")
    profiles_on_disk("coder", "ui_meta:\n  hermes-bots:\n    sectionId: analyst\n")

    listed = _by_name(client)
    assert sorted(listed) == ["analyst", "audit", "coder", "default", "derivatives"]
    assert listed["coder"]["ui_meta"]["hermes-bots"]["sectionId"] == "analyst"
    assert listed["derivatives"]["ui_meta"]["hermes-bots"]["sectionId"] == "crypto-desk/derivatives"
