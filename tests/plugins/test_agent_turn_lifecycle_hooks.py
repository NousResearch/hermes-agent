"""Exercise agent turn hooks through the real user-plugin discovery path."""

import json
from types import SimpleNamespace

from hermes_cli.plugins import get_plugin_manager


def test_agent_turn_dispatch_reaches_user_plugin_loaded_from_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "turn-lifecycle-probe"
    plugin.mkdir(parents=True)
    events_file = tmp_path / "events.jsonl"
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [turn-lifecycle-probe]\n",
        encoding="utf-8",
    )
    (plugin / "plugin.yaml").write_text(
        "name: turn-lifecycle-probe\nversion: 1.0.0\nkind: standalone\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"_events = Path({str(events_file)!r})\n"
        "def _record(name, payload):\n"
        "    with _events.open('a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps({'name': name, 'payload_keys': sorted(payload),\n"
        "                                 'turn_id': payload.get('turn_id')}) + '\\n')\n"
        "def _before(**payload):\n"
        "    _record('pre', payload)\n"
        "    return {'action': 'allow'}\n"
        "def _after(**payload):\n"
        "    _record('post', payload)\n"
        "def register(ctx):\n"
        "    ctx.register_hook('pre_agent_turn', _before)\n"
        "    ctx.register_hook('post_agent_turn', _after)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "bundled-empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)

    manager = get_plugin_manager()
    manager.unload()
    try:
        manager.discover_and_load(force=True)
        from agent.turn_lifecycle_hooks import begin_agent_turn_hooks, finish_agent_turn_hooks

        agent = SimpleNamespace(platform="desktop", model="test-model")
        started, message = begin_agent_turn_hooks(
            agent, session_id="session-1", turn_id="turn-1", task_id="task-1"
        )
        assert started is True
        assert message is None
        finish_agent_turn_hooks(
            agent,
            enabled=started,
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            outcome="success",
        )

        records = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
        assert [record["name"] for record in records] == ["pre", "post"]
        assert all(record["turn_id"] == "turn-1" for record in records)
        assert all("prompt" not in record["payload_keys"] for record in records)
    finally:
        manager.unload()
