"""A real profile gateway authority that admits cron fires without running an agent turn.

Shared by the cron run-history suites (tests/hermes_state, tests/hermes_cli): fires go through
the real producer (``gateway/session_cron.py``) over SQLite, so run history reads exactly what a
scheduler fire writes.
"""
from types import SimpleNamespace

from cron import executions
from gateway import run, session_cron
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from tests.fixtures.storage_seed import seed_messages
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def cron_owner(home, monkeypatch):
    """The owner of the profile at *home*, over ``home/state.db``."""
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('model: test-model\n', encoding='utf-8')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    persistence = storage_handle(home)
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(home / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(home), instance_id='test', persistence=persistence,
                                 epoch=persistence.runtime.begin_epoch(instance_id='test'))
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    runner.session_authority = authority
    return authority


async def admit_fire(authority, job_id, *, running=False):
    """One scheduler fire: its ledger row (marked running when asked), then the owner's submit and
    the run's first user message. Returns ``(session_id, execution_id)``."""
    execution_id = executions.create_execution(job_id, source='test')['id']
    if running:
        executions.mark_execution_running(execution_id)
    receipt = await session_cron.operation(
        authority, 'submit', {'job_id': job_id, 'request_id': execution_id, 'extra_prompt': None})
    seed_messages(authority.persistence, receipt['session_id'], {'role': 'user', 'content': f'run {job_id}'})
    return receipt['session_id'], execution_id
