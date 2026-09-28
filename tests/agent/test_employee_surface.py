"""Employee exclusions apply before native skill loading or background work."""
from types import SimpleNamespace
import asyncio
import json


def test_old_skills_cannot_reenter_through_native_loaders(tmp_path, monkeypatch):
    from agent.secret_scope import set_multiplex_active, is_multiplex_active
    from gateway.run import _profile_runtime_scope
    from gateway.run_turn import GatewayTurnMixin
    from agent import skill_commands as skills, skill_bundles as bundles
    from tools.skills_tool import skill_view
    from hermes_cli.commands import resolve_command, COMMAND_REGISTRY
    from agent.employee_policy import EXCLUDED_COMMANDS
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'launch'))
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for name in ('a', 'b', 'a'):
            home = tmp_path / name
            folder = home / 'skills' / 'old'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'SKILL.md').write_text('---\nname: old\ndescription: Old instructions.\n---\nSKILL PAYLOAD')
            with _profile_runtime_scope(home):
                assert skills.scan_skill_commands() == skills.get_skill_commands() == {}
                assert skills.build_skill_invocation_message('/old') is None
                assert not skills.build_preloaded_skills_prompt(['old'])[0]
                assert bundles.build_bundle_invocation_message('/old') is None
                assert not json.loads(skill_view('old')).get('success')
                event = SimpleNamespace(text='User request')
                GatewayTurnMixin._hmwa_auto_load_skills(None, event, 'old', 'turn', 'session')
                assert event.text == 'User request'
        assert not {command.name for command in COMMAND_REGISTRY} & EXCLUDED_COMMANDS
        for command in EXCLUDED_COMMANDS:
            assert resolve_command(command) is None
        assert resolve_command('model') and resolve_command('stop')
    finally:
        set_multiplex_active(previous)


def test_retired_background_services_and_personality_stay_inactive(tmp_path, monkeypatch):
    from agent.curator import maybe_run_curator
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    from hermes_cli.personality import resolve_ephemeral_system_prompt
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('kanban:\n  dispatch_in_gateway: true\n  notify_in_gateway: true\ncurator:\n  enabled: true\n')
    assert maybe_run_curator(idle_for_seconds=float('inf')) is None
    watcher = GatewayKanbanWatchersMixin()
    assert watcher._kanban_dispatcher_boot() is None
    asyncio.run(watcher._kanban_notifier_watcher())
    assert resolve_ephemeral_system_prompt({'display': {'personality': 'pirate'}, 'agent': {'system_prompt': 'old persona'}}) == ''


def test_retired_commands_and_aliases_rejected_at_dispatch():
    from cli import HermesCLI
    from gateway.run_inbound import GatewayInboundMixin
    from hermes_cli.commands import EMPLOYEE_EXCLUDED_COMMAND_NAMES

    messages = []
    cli = SimpleNamespace(_console_print=messages.append)
    gateway = GatewayInboundMixin()
    for name in EMPLOYEE_EXCLUDED_COMMAND_NAMES:
        # No initialized handler/config or admin identity: rejection must precede
        # handler lookup, hooks, quick commands and authorization fallthrough.
        assert HermesCLI.process_command(cli, f'/{name} add anything') is True
        assert 'unavailable' in messages[-1]
        event = SimpleNamespace(get_command=lambda: name)
        handled, result, *_ = asyncio.run(gateway._hm_resolve_command(event, None, ''))
        assert handled and 'unavailable' in result
        # Alias/hook rewriting reaches the canonical sink independently.
        handled, result = asyncio.run(gateway._hm_dispatch_canonical_command(event, None, '', name))
        assert handled and 'unavailable' in result
