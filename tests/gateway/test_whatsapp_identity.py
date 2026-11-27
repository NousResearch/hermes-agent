"""Tests for gateway.whatsapp_identity alias resolution path."""

import json

from gateway.whatsapp_identity import expand_whatsapp_aliases


def test_aliases_resolve_on_modern_platforms_layout(tmp_path, monkeypatch):
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    assert expand_whatsapp_aliases("999999999999999@lid") == {
        "999999999999999",
        "15551234567",
    }


def test_aliases_resolve_from_launch_home_under_profile_scope(tmp_path, monkeypatch):
    # Multiplexed host (#135139): the bridge writes lid-mapping files under the gateway's
    # launch home while the resolver runs inside a routed profile's scope whose own
    # platforms/whatsapp/session dir does not exist.
    launch_home = tmp_path / "owner-home"
    mapping_dir = launch_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    try:
        assert expand_whatsapp_aliases("999999999999999@lid") == {
            "999999999999999",
            "15551234567",
        }
    finally:
        reset_hermes_home_override(token)


def test_aliases_still_resolve_when_only_the_scoped_dir_has_mappings(tmp_path, monkeypatch):
    # Both candidates are probed, not just the last one: a mapping living only in the scoped
    # dir must still resolve while the launch home has an empty session dir of its own.
    profile_home = tmp_path / "profile-home"
    mapping_dir = profile_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    launch_home = tmp_path / "owner-home"
    (launch_home / "platforms" / "whatsapp" / "session").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(profile_home))
    try:
        assert expand_whatsapp_aliases("999999999999999@lid") == {
            "999999999999999",
            "15551234567",
        }
    finally:
        reset_hermes_home_override(token)
