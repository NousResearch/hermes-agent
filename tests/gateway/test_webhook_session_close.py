"""A late webhook finalizer cannot end a replacement route or generation."""
import json

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.webhook import WebhookAdapter
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input


@pytest.mark.asyncio
async def test_webhook_finalization_is_exact_session_and_generation(tmp_path):
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='close-test')
    runner._session_db = authority.db
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-close-secret'}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='webhook:close:one', user_id='webhook:close', chat_type='webhook')
    ref = authority.register(source)

    def settled(identity):
        row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='fixture',
            session_id=ref.session_id, request_id=identity, payload={'text': 'fixture'})
        claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
        return authority._receipt(settle_session_input(authority.db, epoch=authority.epoch,
            admission_id=row['admission_id'], generation=claimed['generation'], outcome='completed'))

    receipt = settled('first')
    event = MessageEvent(text='fixture', source=source, message_id='first')
    event._webhook_receipt = (authority, receipt)
    successor = runner.session_store.get_or_create_session(source, force_new=True)
    assert successor.session_id != ref.session_id
    # Simulate a delayed callback after route publication. The old authority
    # receipt, not the current route, is the only admissible finalization target.
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.db.get_session(successor.session_id)['ended_at'] is None

    authority.db.reopen_session(ref.session_id)
    newer = settled('second')
    assert newer.execution_generation > receipt.execution_generation
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.db.get_session(ref.session_id)['ended_at'] is None
    event._webhook_receipt = (authority, newer)
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.db.get_session(ref.session_id)['end_reason'] == 'webhook_complete'


async def _settled_delivery(tmp_path, monkeypatch, outcome, script_body):
    """A real authority with one settled webhook admission and a profile-local completion script."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / 'complete.py').write_text(script_body, encoding='utf-8')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='completion-test')
    runner._session_db = authority.db
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-completion-secret'}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id=f'webhook:alerts:{outcome}', user_id='webhook:alerts', chat_type='webhook')
    ref = authority.register(source)
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='fixture',
        session_id=ref.session_id, request_id=outcome, payload={'text': 'fixture'})
    claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
    assert isinstance(claimed, dict)
    receipt = authority._receipt(settle_session_input(authority.db, epoch=authority.epoch,
        admission_id=row['admission_id'], generation=claimed['generation'], outcome=outcome))
    event = MessageEvent(text='fixture', source=source, message_id=f'delivery-{outcome}',
                         raw_message={'message': 'payload; touch should-not-run'})
    return adapter, authority, receipt, event, ref


_RECORD_ENVELOPE = (
    "import json, os, pathlib, sys\n"
    "payload = json.load(sys.stdin)\n"
    "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text(json.dumps(payload))\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(('admission_outcome', 'expected_outcome'),
                         [('completed', 'success'), ('failed', 'failure')])
async def test_completion_script_receives_terminal_outcome(tmp_path, monkeypatch, admission_outcome,
                                                           expected_outcome):
    output_path = tmp_path / 'completion.json'
    monkeypatch.setenv('COMPLETION_OUTPUT', str(output_path))
    adapter, authority, receipt, event, ref = await _settled_delivery(
        tmp_path, monkeypatch, admission_outcome, _RECORD_ENVELOPE)

    await adapter._finalize_delivery(event, authority, receipt, route_name='alerts',
                                     profile=None, completion_script='complete.py')

    assert json.loads(output_path.read_text(encoding='utf-8')) == {
        'version': 1, 'route': 'alerts', 'outcome': expected_outcome,
        'delivery_id': f'delivery-{admission_outcome}',
        'payload': {'message': 'payload; touch should-not-run'}}
    assert not (tmp_path / 'should-not-run').exists()
    assert authority.db.get_session(ref.session_id)['end_reason'] == 'webhook_complete'


@pytest.mark.asyncio
async def test_interrupted_webhook_does_not_run_completion_script(tmp_path, monkeypatch):
    output_path = tmp_path / 'completion.json'
    monkeypatch.setenv('COMPLETION_OUTPUT', str(output_path))
    adapter, authority, receipt, event, _ref = await _settled_delivery(
        tmp_path, monkeypatch, 'interrupted', _RECORD_ENVELOPE)

    await adapter._finalize_delivery(event, authority, receipt, route_name='alerts',
                                     profile=None, completion_script='complete.py')

    assert not output_path.exists()


@pytest.mark.asyncio
async def test_failing_completion_script_does_not_block_session_cleanup(tmp_path, monkeypatch):
    marker = tmp_path / 'ran'
    monkeypatch.setenv('COMPLETION_OUTPUT', str(marker))
    adapter, authority, receipt, event, ref = await _settled_delivery(
        tmp_path, monkeypatch, 'completed',
        "import os, pathlib, sys\nsys.stdin.read()\n"
        "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text('ran')\nsys.exit(3)\n")

    await adapter._finalize_delivery(event, authority, receipt, route_name='alerts',
                                     profile=None, completion_script='complete.py')

    assert marker.read_text(encoding='utf-8') == 'ran'
    assert authority.db.get_session(ref.session_id)['end_reason'] == 'webhook_complete'


@pytest.mark.asyncio
async def test_completion_script_runs_in_the_routed_profile(tmp_path, monkeypatch):
    """A ``/p/<profile>/`` route's finalizer resolves the script under THAT profile's home, not the
    launch profile's, even though it runs after the request's profile scope has closed."""
    output_path = tmp_path / 'completion.json'
    monkeypatch.setenv('COMPLETION_OUTPUT', str(output_path))
    adapter, authority, receipt, event, _ref = await _settled_delivery(
        tmp_path, monkeypatch, 'completed',
        "import os, pathlib, sys\nsys.stdin.read()\n"
        "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text('launch')\n")
    routed_scripts = tmp_path / 'profiles' / 'sec' / 'scripts'
    routed_scripts.mkdir(parents=True)
    (routed_scripts / 'complete.py').write_text(
        "import os, pathlib, sys\nsys.stdin.read()\n"
        "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text('routed')\n", encoding='utf-8')
    monkeypatch.setattr('hermes_cli.profiles._get_default_hermes_home', lambda: tmp_path)
    monkeypatch.setattr('hermes_cli.profiles._get_profiles_root', lambda: tmp_path / 'profiles')

    await adapter._finalize_delivery(event, authority, receipt, route_name='alerts',
                                     profile='sec', completion_script='complete.py')

    assert output_path.read_text(encoding='utf-8') == 'routed'
