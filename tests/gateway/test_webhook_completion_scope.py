"""Completion scripts retain the admitted profile and dispatch-time environment expansion."""
import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from gateway.platforms.webhook_filters import WebhookRouteProcessor
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from hermes_state_runtime import claim_session_input, settle_session_input


@pytest.mark.asyncio
async def test_bare_route_completion_uses_admitted_profile(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    secondary = home / 'profiles' / 'sec'
    secondary.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda *args: {})
    (secondary / 'config.yaml').write_text('{}', encoding='utf-8')
    (home / '.env').write_text('COMPLETION_DEST=launch\nLAUNCH_ONLY_COMPLETION=private\n', encoding='utf-8')
    (secondary / '.env').write_text('COMPLETION_DEST=routed\n', encoding='utf-8')
    monkeypatch.setenv('COMPLETION_DEST', 'launch')
    monkeypatch.setenv('LAUNCH_ONLY_COMPLETION', 'private')
    for profile_home, label in ((home, 'launch'), (secondary, 'routed')):
        scripts = profile_home / 'scripts'
        scripts.mkdir()
        (scripts / 'complete.py').write_text(
            "import os, pathlib, sys\nsys.stdin.read()\n"
            f"pathlib.Path('ran').write_text('{label}:' + os.environ.get('COMPLETION_DEST', 'missing')"
            " + ':' + str('LAUNCH_ONLY_COMPLETION' in os.environ))\n",
            encoding='utf-8')
    config = cast(Any, GatewayConfig(sessions_dir=home / 'sessions'))
    config.multiplex_profiles = True
    config._runtime_profile_homes = (('default', home), ('sec', secondary))
    config.profile_routes = [ProfileRoute(name='alerts-to-sec', platform='webhook',
                                         profile='sec', user_id='webhook:alerts')]
    runner = cast(Any, GatewayRunner(config))
    route = {'secret': 'completion-secret', 'completion_script': 'complete.py'}
    adapter = cast(Any, WebhookAdapter(PlatformConfig(enabled=True, extra={'routes': {'alerts': route}})))
    adapter.gateway_runner = runner
    process_ownership.reserve([home, secondary])
    try:
        await initialize_gateway_runtime(runner)
        runner.adapters[Platform.WEBHOOK] = adapter
        runner._wire_adapter_handlers(adapter)
        authorities = runner.session_authorities
        for authority in authorities:
            # Exercise real routing/admission, but settle without starting model inference.
            authority._schedule = lambda ref: None
        outcome = await adapter._admit_agent_run(
            {'text': 'hello'}, 'hello', 'routed-completion', 100.0,
            route_config=route, route_name='alerts',
            profile=None, event_type='push')
        assert outcome == 'accepted'
        authority = authorities.for_home(secondary)
        row = authority.db._read_one('SELECT * FROM session_admissions')
        assert row is not None
        claim = claim_session_input(authority.db, epoch=authority.epoch,
                                    session_id=row['target_session_id'])
        assert isinstance(claim, dict)
        settle_session_input(authority.db, epoch=authority.epoch,
                             admission_id=row['admission_id'], generation=claim['generation'],
                             outcome='completed')
        await asyncio.gather(*adapter._background_tasks)
        assert (secondary / 'scripts' / 'ran').read_text(encoding='utf-8') == 'routed:routed:False'
        assert not (home / 'scripts' / 'ran').exists()
        assert authority.db.get_session(row['target_session_id'])['end_reason'] == 'webhook_complete'
    finally:
        for task in list(adapter._background_tasks):
            task.cancel()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        for authority in getattr(runner, 'session_authorities', ()):
            authority.db.close()
        for profile_home in (home, secondary):
            process_ownership.release(profile_home)


@pytest.mark.parametrize('setting', ['${LATE_SCRIPT}', '~/.hermes/scripts/${LATE_SCRIPT}', '${SCRIPT_NAME}'])
def test_completion_script_does_not_expand_snapshot_again(tmp_path, monkeypatch, setting):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('LATE_SCRIPT', raising=False)
    monkeypatch.setenv('SCRIPT_NAME', '${LATE_SCRIPT}')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for filename, label in (('${LATE_SCRIPT}', 'snapshot'), ('second.py', 'late')):
        (scripts / filename).write_text(
            f"import pathlib, sys\nsys.stdin.read()\npathlib.Path('ran').write_text('{label}')\n",
            encoding='utf-8')
    processor = WebhookRouteProcessor()
    snapshot = processor.snapshot_script_value(setting)
    monkeypatch.setenv('LATE_SCRIPT', 'second.py')

    assert processor.run_completion_script(snapshot, {'version': 1})
    assert (scripts / 'ran').read_text(encoding='utf-8') == 'snapshot'
