import json

from hermes_cli.profile_identity import migrate_profile_durable_references


def test_temp_profile_rename_migrates_durable_identity_and_path_consumers(tmp_path):
    root = tmp_path / ".hermes"
    old_dir = root / "profiles" / "old-bot"
    new_dir = root / "profiles" / "new-bot"
    new_dir.mkdir(parents=True)
    old_path = str(old_dir)
    (new_dir / "cron").mkdir()
    (new_dir / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
        "profile": "old-bot", "transport_profile": "old-bot",
        "deliver": "bot-chat:old-bot", "session_key": "agent:old-bot:telegram:dm:1",
        "home": old_path,
    }]}))
    (new_dir / "plugins").mkdir()
    (new_dir / "plugins" / "state.json").write_text(json.dumps({
        "target_profile": "old-bot", "profile_home": old_path,
    }))
    (root / "config.yaml").write_text(
        "routes:\n  - profile: old-bot\n    transport_profile: old-bot\n")

    assert migrate_profile_durable_references(old_dir, new_dir)
    assert migrate_profile_durable_references(old_dir, new_dir)  # retry-safe

    job = json.loads((new_dir / "cron" / "jobs.json").read_text())["jobs"][0]
    assert job == {
        "profile": "new-bot", "transport_profile": "new-bot",
        "deliver": "bot-chat:new-bot", "session_key": "agent:new-bot:telegram:dm:1",
        "home": str(new_dir),
    }
    plugin = json.loads((new_dir / "plugins" / "state.json").read_text())
    assert plugin == {"target_profile": "new-bot", "profile_home": str(new_dir)}
    assert "profile: new-bot" in (root / "config.yaml").read_text()
    assert "transport_profile: new-bot" in (root / "config.yaml").read_text()


def _layout(tmp_path, old="bot", new="chat"):
    root = tmp_path / ".hermes"
    old_dir, new_dir = root / "profiles" / old, root / "profiles" / new
    (new_dir / "cron").mkdir(parents=True)
    return root, old_dir, new_dir


def test_rename_does_not_retarget_profiles_sharing_a_name_prefix(tmp_path):
    root, old_dir, new_dir = _layout(tmp_path)
    (root / "profiles" / "bot2" / "cron").mkdir(parents=True)
    sibling_jobs = root / "profiles" / "bot2" / "cron" / "jobs.json"
    sibling_jobs.write_text(json.dumps({"jobs": [
        {"profile": "bot2", "deliver": "bot-chat:bot2"},
        {"profile": "bot2", "deliver": "telegram,bot-chat:bot,bot-chat:bot-dev"},
    ]}))
    (new_dir / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
        "profile": "bot", "deliver": "bot-chat:bot", "home": str(old_dir) + "2/notes",
    }]}))

    assert migrate_profile_durable_references(old_dir, new_dir)

    sibling = json.loads(sibling_jobs.read_text())["jobs"]
    assert sibling[0] == {"profile": "bot2", "deliver": "bot-chat:bot2"}
    assert sibling[1]["deliver"] == "telegram,bot-chat:chat,bot-chat:bot-dev"
    own = json.loads((new_dir / "cron" / "jobs.json").read_text())["jobs"][0]
    assert own == {"profile": "chat", "deliver": "bot-chat:chat", "home": str(old_dir) + "2/notes"}


def test_rename_leaves_free_text_that_merely_contains_the_name(tmp_path):
    root, old_dir, new_dir = _layout(tmp_path)
    (new_dir / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
        "profile": "bot", "command": "python ~/scripts/bot.py --send", "prompt": "ask bot to report",
    }]}))
    config = (
        "# routes for bot\n"
        "terminal:\n  cwd: /srv/bot\n"
        "routes:\n  - profile: bot  # main\n    transport_profile: \"bot\"\n  - profile: bot2\n"
    )
    (root / "config.yaml").write_text(config)

    assert migrate_profile_durable_references(old_dir, new_dir)

    job = json.loads((new_dir / "cron" / "jobs.json").read_text())["jobs"][0]
    assert job == {"profile": "chat", "command": "python ~/scripts/bot.py --send", "prompt": "ask bot to report"}
    assert (root / "config.yaml").read_text() == (
        "# routes for bot\n"
        "terminal:\n  cwd: /srv/bot\n"
        "routes:\n  - profile: chat  # main\n    transport_profile: \"chat\"\n  - profile: bot2\n"
    )


def test_rename_rewrites_root_cron_jobs_targeting_the_profile(tmp_path):
    root, old_dir, new_dir = _layout(tmp_path)
    (root / "cron").mkdir(parents=True)
    root_jobs = root / "cron" / "jobs.json"
    root_jobs.write_text(json.dumps({"jobs": [{"profile": "default", "deliver": "bot-chat:bot"}]}))

    assert migrate_profile_durable_references(old_dir, new_dir)
    assert migrate_profile_durable_references(old_dir, new_dir)  # retry-safe

    assert json.loads(root_jobs.read_text())["jobs"] == [{"profile": "default", "deliver": "bot-chat:chat"}]
