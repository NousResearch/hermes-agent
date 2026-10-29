"""Adopting a pre-authority chat that ran on an external-process provider.

Such a provider runs a local CLI. The runtime takes its address from the provider's own profile
(``_external_process_spec``), never from ``model.base_url``, and the old surfaces still persisted
that address (``process://<provider>``) as the row's route. Adoption replayed it as an explicit
``base_url`` launch override, which only accepts http(s) endpoints, so the chat could never be opened
and every bot-to-bot delivery into a Bot Chat holding it was refused with ``invalid_params``.

Real: SessionStore/SessionAuthority, ``resolve_local_target`` -> ``adopt_legacy_session`` ->
``build_policy``, a real config.yaml. Seam: the provider registry lookup gains one external-process
profile. No process is spawned.
"""
import json

import pytest

from tests.gateway.test_session_follow_profile_config import _owner

PROVIDER = 'proc-provider'
ADDRESS = 'process://proc-provider'


@pytest.fixture
def home(monkeypatch):
    import providers
    from gateway import run
    from hermes_constants import get_hermes_home
    from providers.base import ProviderProfile
    profile = ProviderProfile(name=PROVIDER, auth_type='external_process', base_url=ADDRESS,
                              api_mode='chat_completions')
    real = providers.get_provider_profile
    monkeypatch.setattr(providers, 'get_provider_profile', lambda name: profile if name == PROVIDER else real(name))
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _profile(home, **model):
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': PROVIDER, 'default': 'profile-model', **model}}))


def _legacy(db, home, sid, title=None, **model_config):
    db.create_session(sid, cwd=str(home), source='gui', model='picked-model', model_config={
        'provider': PROVIDER, 'base_url': ADDRESS, 'api_mode': 'chat_completions', **model_config})
    if title:
        db.set_session_title(sid, title)
    db.end_session(sid, 'tui_close')


def _model(runner, live):
    from gateway.session_policy import policy_for_source
    return json.loads(policy_for_source(runner, live.source).config_json).get('model') or {}


@pytest.mark.asyncio
@pytest.mark.parametrize('profile_url', [None, 'https://unused.example/v1'], ids=['bare', 'stray-base-url'])
async def test_a_bot_chat_with_a_held_pick_on_a_process_provider_adopts(home, profile_url):
    """The work-bot shape: a Bot Chat whose composer pick still holds keeps its saved route, which
    carried the provider's ``process://`` address. A ``model.base_url`` the profile still lists for the
    provider is ignored by the runtime too, so it must not change the outcome."""
    from gateway.session_local_migration import resolve_local_target
    _profile(home, **({'base_url': profile_url} if profile_url else {}))
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-bot', title='Bot Chat',
            composer_override_profile={'model': 'profile-model', 'provider': PROVIDER})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        bot = authority.sessions[resolve_local_target(authority, owner.actor, 'old-bot').session_id]
        model = _model(runner, bot)
        assert model.get('provider') == PROVIDER
        assert not str(model.get('base_url') or '').startswith('process://')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_old_desktop_chat_on_a_process_provider_keeps_its_provider_and_opens(home):
    """Not a Bot Chat: the row keeps its saved route, the provider. The address is not part of it."""
    from gateway.session_local_migration import resolve_local_target
    _profile(home, provider='custom', base_url='http://127.0.0.1:1/v1', api_key='k1')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-desktop')
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        live = authority.sessions[resolve_local_target(authority, owner.actor, 'old-desktop').session_id]
        model = _model(runner, live)
        assert model.get('provider') == PROVIDER
        assert not model.get('base_url')
        assert model.get('api_key') != 'k1'  # the profile's custom key never follows the saved provider
    finally:
        await owner.close()
        store.close_all_db_handles()
