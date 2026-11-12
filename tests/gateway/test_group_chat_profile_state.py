# ABOUTME: Verify the canonical roster reads group history from separate profile runtime files.
# ABOUTME: Cover the owning profile and cached sibling discovery without starting inference.

import asyncio
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB


def test_canonical_roster_reads_separate_group_state_for_owner_and_sibling(tmp_path, monkeypatch):
    home = tmp_path / "state"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    key = "hermes-bots-groups"
    for name, target in (("default", home), ("helper", home / "profiles" / "helper")):
        (target / "state").mkdir(parents=True)
        (target / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        (target / "profile.yaml").write_text(f"display_name: {name}\n", encoding="utf-8")
        (target / "state" / "group-chat.yaml").write_text(yaml.safe_dump({
            "revision": 4, "snapshot": {"rooms": {name: {"log": [{"text": name}]}}},
        }), encoding="utf-8")

    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(profile_id=str(home), instance_id="owner", db=db, events={}, sessions={})
        connection = AuthorityConnection(authority, object(), {"user_id": "reader"})

        def rows():
            result = asyncio.run(connection.dispatch({"id": "read", "method": "profiles.list",
                                                      "params": {"include_sessions": False}}))
            return {p["name"]: p for p in result["result"]["profiles"]}

        first = rows()
        assert first["default"]["ui_meta"][key]["rooms"]["default"]["log"] == [{"text": "default"}]
        assert first["default"]["ui_meta_revisions"][key] == 4
        assert first["helper"]["ui_meta"][key]["rooms"]["helper"]["log"] == [{"text": "helper"}]
        target = home / "profiles" / "helper" / "state" / "group-chat.yaml"
        target.write_text(yaml.safe_dump({"revision": 5, "snapshot": {"rooms": {}}}), encoding="utf-8")
        assert rows()["helper"]["ui_meta"][key] == {"rooms": {}}

        for damaged, state_path in (("helper", target), ("default", home / "state" / "group-chat.yaml")):
            state_path.write_text("[invalid", encoding="utf-8")
            roster = rows()
            assert {"default", "helper"} <= roster.keys()
            assert roster[damaged]["display_name"] == damaged
            assert key not in roster[damaged].get("ui_meta", {})
            assert state_path.read_text() == "[invalid"


def test_canonical_roster_projects_pending_mixed_ui_update_after_profile_write_failure(tmp_path, monkeypatch):
    """The canonical roster must honor a committed journal even before metadata materialization."""
    import tui_gateway.server as srv
    import utils

    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    group_key = "hermes-bots-groups"
    original = {
        "display_name": "default",
        "ui_meta": {group_key: {"rooms": {"old": {"log": [{"text": "old"}]}}},
                    "room-theme": {"color": "blue"}},
        "_ui_meta_revisions": {group_key: 3, "room-theme": 1},
    }
    profile = home / "profile.yaml"
    profile.write_text(yaml.safe_dump(original), encoding="utf-8")
    unchanged = profile.read_bytes()
    new_group = {"rooms": {"new": {"log": [{"text": "new"}]}}}
    real_write = utils.atomic_yaml_write

    def fail_profile_write(path, data, **kwargs):
        if Path(path) == profile:
            raise OSError("synthetic read-only profile.yaml")
        return real_write(path, data, **kwargs)

    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(profile_id=str(home), instance_id="owner", db=db, events={}, sessions={})
        connection = AuthorityConnection(authority, object(), {"user_id": "reader"})
        with monkeypatch.context() as fault:
            fault.setattr(utils, "atomic_yaml_write", fail_profile_write)
            result = srv._methods["profiles.configure"]("write", {
                "name": "default",
                "ui_meta": {group_key: new_group, "room-theme": {"color": "green"}},
                "ui_meta_expected_revisions": {group_key: 3, "room-theme": 1},
            })
            assert result["result"]["applied"]["ui_meta"] is True
            state = yaml.safe_load((home / "state" / "group-chat.yaml").read_text())
            assert "pending_ui_meta" in state
            assert profile.read_bytes() == unchanged

            response = asyncio.run(connection.dispatch({
                "id": "read", "method": "profiles.list", "params": {"include_sessions": False},
            }))
            assert "result" in response, response
            row = next(p for p in response["result"]["profiles"] if p["name"] == "default")
            assert row["ui_meta"][group_key] == new_group
            assert row["ui_meta"]["room-theme"] == {"color": "green"}
            assert row["ui_meta_revisions"][group_key] == 4
            assert row["ui_meta_revisions"]["room-theme"] == 2
            # A read must neither write profile.yaml nor retire the pending journal.
            assert profile.read_bytes() == unchanged
            assert "pending_ui_meta" in yaml.safe_load((home / "state" / "group-chat.yaml").read_text())
