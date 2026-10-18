"""A pre-authority Bot Chat served by an external-process provider opens from the Bots roster.

Live repro (2026-10-18): the Desktop roster click asks ``session.list title="Bot Chat"`` — no
``base_url`` on the wire — and the title lookup adopts the hidden pre-authority ``desktop`` row
(``resolve_titled_session`` -> ``resolve_local_target`` -> ``adopt_legacy_session``). That row's only
route record is the billing stamp the provider's runtime wrote: ``billing_provider=<process provider>``,
``billing_base_url=process://<process provider>`` (the provider's own registered endpoint). The
profile still selects that provider, but its ``model.base_url`` is a leftover https URL the
external-process runtime never reads, so ``same_route`` saw two endpoints, the stamp went into the
launch overrides, and ``_apply_launch_overrides`` refused it: every click toasted
"Could not check <bot>'s Bot Chat registry (invalid_params)". An endpoint that is not the saved
provider's own registered one must still be refused.
"""
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

SID = '20261017_144305_80d3ce'


@pytest.fixture(autouse=True)
def _unregister_proc_provider():
    """``register_provider`` writes into the process-wide registry (``providers._REGISTRY``), so a
    plugin load from one test's fixture ``home`` would otherwise leak into every test run after it
    in the same pytest process (e.g. the per-file isolation ``scripts/run_tests.sh`` relies on hides
    this; running files together does not)."""
    yield
    import providers
    providers._REGISTRY.pop('proc-provider', None)
    providers._SOURCES.pop('proc-provider', None)

_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="proc-provider", auth_type="external_process",
                                      base_url="process://proc-provider", api_mode="chat_completions"))
    """
)


async def _list_bot_chat(home, monkeypatch, *, billing_provider, billing_base_url, profile_provider='proc-provider'):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setattr('gateway.run._hermes_home', home)
    plugin = home / 'plugins' / 'proc-provider'
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / 'plugin.yaml').write_text('name: proc-provider\nkind: model-provider\n', encoding='utf-8')
    (plugin / '__init__.py').write_text(_PLUGIN, encoding='utf-8')
    # The live profile: provider moved to the process provider, the old HTTP endpoint and key stayed.
    (home / 'config.yaml').write_text(
        f'model:\n  default: sonnet\n  provider: {profile_provider}\n'
        '  base_url: https://cloud-api.near.ai/v1\n  api_key: stale-key\n  context_length: 1000000\n'
        'platform_toolsets:\n  gui: []\n', encoding='utf-8')
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    db.create_session(SID, source='desktop', model='sonnet', cwd=str(home),
                      model_config={'follow_profile_config': True})
    db.update_token_counts(SID, input_tokens=1, model='sonnet', billing_provider=billing_provider,
                           billing_base_url=billing_base_url)
    db.set_session_title(SID, 'Bot Chat')
    db.set_session_hidden(SID, True)
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
                                                       'profile_id': str(home), 'instance_id': 'fixture'})
    # Exactly the params canonical-chat.ts findExistingCanonicalChat sends (``profile`` is our own home).
    reply = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {
        'title': 'Bot Chat', 'limit': 200, 'include_hidden': True}})
    return reply, db, store


@pytest.mark.asyncio
async def test_bot_chat_stamped_with_process_providers_own_endpoint_opens(monkeypatch):
    import json
    from hermes_state_local import local_receipt
    home = Path(os.environ['HERMES_HOME'])
    reply, db, store = await _list_bot_chat(home, monkeypatch, billing_provider='proc-provider',
                                            billing_base_url='process://proc-provider')
    try:
        assert 'error' not in reply, reply
        assert [row['session_id'] for row in reply['result']['sessions']] == [SID]
        # Same provider as the profile: the profile's route stands (its own pins kept), and no
        # pseudo-URL is frozen as a launch override.
        model = json.loads(local_receipt(db, SID)['policy']['config_json'])['model']
        assert model['provider'] == 'proc-provider'
        assert model.get('base_url') != 'process://proc-provider'
        assert model.get('context_length') == 1000000
    finally:
        store._db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider, stamp', [('proc-provider', 'process://proc-provider'),
                                             ('copilot-acp', 'acp://copilot')])
async def test_process_route_survives_a_profile_that_moved_to_another_provider(monkeypatch, provider, stamp):
    """The profile now selects an HTTP provider: the row resumes on its saved process provider."""
    import json
    from hermes_state_local import local_receipt
    home = Path(os.environ['HERMES_HOME'])
    reply, db, store = await _list_bot_chat(home, monkeypatch, billing_provider=provider,
                                            billing_base_url=stamp, profile_provider='openrouter')
    try:
        assert 'error' not in reply, reply
        model = json.loads(local_receipt(db, SID)['policy']['config_json'])['model']
        assert model['provider'] == provider
        # The replaced profile's endpoint and key are never paired with the saved provider.
        assert model.get('base_url') != 'https://cloud-api.near.ai/v1' and 'api_key' not in model, model
    finally:
        store._db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stamp', ['ftp://evil.example/v1', 'process://unregistered-provider',
                                   'process://proc-provider?x=1', 'acp://copilot'])
async def test_endpoint_that_is_not_the_providers_own_is_still_refused(monkeypatch, stamp):
    home = Path(os.environ['HERMES_HOME'])
    reply, db, store = await _list_bot_chat(home, monkeypatch, billing_provider='proc-provider',
                                            billing_base_url=stamp)
    try:
        assert reply.get('error', {}).get('data', {}).get('field') == 'base_url', reply
        assert db.get_session(SID)['chat_id'] in (None, ''), 'refused adoption must not write'
    finally:
        store._db.close()
