"""Both parent branches used config v46 for different transformations."""
import pytest
import yaml

from hermes_cli.config import DEFAULT_CONFIG, check_config_version, migrate_config


@pytest.mark.parametrize("parent", ["before_either", "main_v46", "runtime_v46", "runtime_v47"])
def test_join_migrates_both_parent_histories(tmp_path, monkeypatch, parent):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cron = {"max_parallel_jobs": 2}
    server = {"command": "fixture-only-not-executed", "args": ["--example"]}
    if parent not in ("runtime_v46", "runtime_v47"):
        cron["bot_chat_delivery_timeout_seconds"] = 900
    if parent == "main_v46":
        server["enabled"] = False
    else:
        server.update(enabled=True, disabled=True)
    raw = {
        "_config_version": {"before_either": 45, "runtime_v47": 47}.get(parent, 46),
        "cron": cron,
        "mcp_servers": {"legacy": server, "active": {"command": "also-inert", "enabled": True}},
    }
    path = tmp_path / "config.yaml"
    path.write_text("# preserve user comment\n" + yaml.safe_dump(raw), encoding="utf-8")

    results = migrate_config(interactive=False, quiet=True)

    migrated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "bot_chat_delivery_timeout_seconds" not in migrated["cron"]
    assert migrated["cron"]["max_parallel_jobs"] == 2
    assert migrated["mcp_servers"]["legacy"] == {
        "command": "fixture-only-not-executed", "args": ["--example"], "enabled": False,
    }
    assert migrated["mcp_servers"]["active"] == {"command": "also-inert", "enabled": True}
    assert "# preserve user comment" in path.read_text(encoding="utf-8")
    assert migrated["_config_version"] == DEFAULT_CONFIG["_config_version"] > 46
    assert check_config_version() == (DEFAULT_CONFIG["_config_version"],) * 2
    assert not results["warnings"]
    assert any("bot_chat_delivery_timeout_seconds" in note for note in results["config_added"]) == (parent not in ("runtime_v46", "runtime_v47"))
    assert any("disabled → enabled: false" in note for note in results["config_added"]) == (parent != "main_v46")

    before = path.read_bytes()
    repeated = migrate_config(interactive=False, quiet=True)
    assert path.read_bytes() == before
    assert not any("bot_chat_delivery_timeout_seconds" in note or "disabled → enabled: false" in note
                   for note in repeated["config_added"])
    assert repeated["warnings"] == []
