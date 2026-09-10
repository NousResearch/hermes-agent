"""Cron's owner-side constructor retains provider-specific wire options."""
from unittest.mock import MagicMock

from cron.scheduler import _construct_cron_agent, _CronAgentSetup


def test_cron_constructor_forwards_runtime_request_overrides():
    overrides = {'extra_headers': {'User-Agent': 'cron-provider-client'}}
    factory = MagicMock()
    setup = _CronAgentSetup(model='model', runtime={
        'api_key': 'test-key', 'base_url': 'https://example.invalid/v1',
        'provider': 'custom', 'api_mode': 'codex_responses', 'request_overrides': overrides})
    _construct_cron_agent(factory, {'id': 'job'}, {}, setup,
                          workdir=None, session_id='session', session_db=MagicMock())
    assert factory.call_args.kwargs['request_overrides'] == overrides
