"""Employee exclusions apply before native skill loading or background work."""
from types import SimpleNamespace
import json


def test_old_skills_cannot_reenter_through_native_loaders(tmp_path, monkeypatch):
    from agent.secret_scope import set_multiplex_active, is_multiplex_active
    from gateway.run import _profile_runtime_scope
    from gateway.run_turn import GatewayTurnMixin
    from agent import skill_commands as skills, skill_bundles as bundles
    from tools.skills_tool import skill_view
    from hermes_cli.commands import resolve_command
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
        assert resolve_command('model') and resolve_command('stop')
    finally:
        set_multiplex_active(previous)


def test_skill_curator_stays_inactive(tmp_path, monkeypatch):
    from agent.curator import maybe_run_curator
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('curator:\n  enabled: true\n')
    assert maybe_run_curator(idle_for_seconds=float('inf')) is None
