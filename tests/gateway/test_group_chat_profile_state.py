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
