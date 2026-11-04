"""Side tasks (``auxiliary:``) of a gateway-owned local chat follow the profile's config.yaml.

A chat keeps its main model and route from creation. Its side tasks (compression, vision, titles, ...)
read ``auxiliary`` from the profile's ``config.yaml``, as messaging chats and the other side-task readers
already do, so a side task's model, address and key come from the same file. Real policy, real per-turn
scope, real auxiliary resolver; loopback fake providers.
"""
import json

from tests.gateway.test_config_policy_recovery import _bound_session


def _compress(policy, owner):
    """One side-task call as a turn makes it: the per-turn scope, then the real resolver."""
    from agent.auxiliary_client import get_text_auxiliary_client
    from gateway.session_policy import policy_scope
    with policy_scope(policy, authority=owner()):
        client, model = get_text_auxiliary_client('compression')
        client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': 'summarise'}])
        return model


def _write(path, cfg):
    path.write_text(json.dumps(cfg))


def test_a_changed_compression_model_and_key_reach_the_next_call(tmp_path, monkeypatch):
    from tests.fakes.fake_llm_provider import FakeLLMServer
    with FakeLLMServer() as main, FakeLLMServer() as aux:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'main-key'},
               'auxiliary': {'compression': {'provider': 'custom', 'model': 'retired-model',
                                             'base_url': aux.base_url, 'api_key': 'retired-key'}}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        assert _compress(policy, owner) == 'retired-model'
        # The provider retired the model; the user switches to the paid one with its own key.
        cfg['auxiliary']['compression'].update(model='replacement-model', api_key='replacement-key')
        _write(config_path, cfg)
        assert _compress(policy, owner) == 'replacement-model'
        assert [r['body']['model'] for r in aux.requests] == ['retired-model', 'replacement-model']
        assert [r['auth'] for r in aux.requests] == ['Bearer retired-key', 'Bearer replacement-key']
        assert main.requests == []  # the replacement serves; nothing falls back to the main model


def test_a_side_task_key_goes_only_to_the_address_it_is_configured_with(tmp_path, monkeypatch):
    """The block carries its own key and names a provider entry; the entry then moves. Address and key
    come from the same file, so the old key never reaches the new address."""
    from tests.fakes.fake_llm_provider import FakeLLMServer
    with FakeLLMServer() as main, FakeLLMServer() as old, FakeLLMServer() as new:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'main-key'},
               'providers': {'work-gateway': {'base_url': old.base_url}},
               'auxiliary': {'compression': {'provider': 'work-gateway', 'model': 'small',
                                             'api_key': 'key-for-old'}}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        _compress(policy, owner)
        cfg['providers']['work-gateway']['base_url'] = new.base_url
        cfg['auxiliary']['compression']['api_key'] = 'key-for-new'
        _write(config_path, cfg)
        _compress(policy, owner)
        assert [r['auth'] for r in old.requests] == ['Bearer key-for-old']
        assert [r['auth'] for r in new.requests] == ['Bearer key-for-new']


def test_removing_the_override_returns_compression_to_automatic_selection(tmp_path, monkeypatch):
    from tests.fakes.fake_llm_provider import FakeLLMServer
    from agent.auxiliary_client import reset_runtime_main, set_runtime_main
    with FakeLLMServer() as main, FakeLLMServer() as aux:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'main-key'},
               'auxiliary': {'compression': {'provider': 'custom', 'model': 'retired-model',
                                             'base_url': aux.base_url, 'api_key': 'retired-key'}}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        cfg['auxiliary'].pop('compression')
        _write(config_path, cfg)
        token = set_runtime_main('custom', 'primary-model', base_url=main.base_url, api_key='main-key')
        try:
            assert _compress(policy, owner) == 'primary-model'
        finally:
            reset_runtime_main(token)
        assert aux.requests == []
        assert [r['auth'] for r in main.requests] == ['Bearer main-key']


def test_the_chats_main_model_stays_where_it_was(tmp_path, monkeypatch):
    from tests.fakes.fake_llm_provider import FakeLLMServer
    with FakeLLMServer() as main, FakeLLMServer() as aux:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'main-key'},
               'auxiliary': {'compression': {'provider': 'custom', 'model': 'small', 'base_url': aux.base_url,
                                             'api_key': 'aux-key'}}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        cfg['model'].update(default='other-model', base_url=aux.base_url)
        cfg['auxiliary']['compression']['model'] = 'smaller'
        _write(config_path, cfg)
        effective = policy.config(owner())
        assert effective['model']['default'] == 'primary-model'
        assert effective['model']['base_url'] == main.base_url
        assert _compress(policy, owner) == 'smaller'


def test_an_ignore_user_config_chat_never_reads_the_profiles_side_tasks(tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.fakes.fake_llm_provider import FakeLLMServer
    from gateway.session_policy import policy_scope
    from agent.auxiliary_task_config import _get_auxiliary_task_config
    with FakeLLMServer() as main, FakeLLMServer() as aux:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'main-key'}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        bypass = replace(policy, ignore_user_config=True)
        cfg['auxiliary'] = {'compression': {'provider': 'custom', 'model': 'late', 'base_url': aux.base_url,
                                            'api_key': 'late-key'}}
        _write(config_path, cfg)
        with policy_scope(bypass, authority=owner()):
            assert _get_auxiliary_task_config('compression') == {}


def test_every_side_task_reader_sees_the_same_route(tmp_path, monkeypatch):
    """The shared resolver and a reader that loads config.yaml itself (background review, ``/btw``, vision's
    model) agree inside one chat's turn: there is no second copy for them to disagree about."""
    from agent.auxiliary_task_config import _get_auxiliary_task_config
    from gateway.session_policy import policy_scope
    from hermes_cli.config import load_config_readonly
    cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': 'http://127.0.0.1:9/v1',
                     'api_key': 'main-key'},
           'auxiliary': {'compression': {'provider': 'custom', 'model': 'first', 'base_url': 'http://127.0.0.1:9/a/v1',
                                         'api_key': 'aux-key'}}}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
    cfg['auxiliary']['compression']['model'] = 'second'
    _write(config_path, cfg)
    with policy_scope(policy, authority=owner()):
        assert _get_auxiliary_task_config('compression')['model'] == 'second'
        assert load_config_readonly()['auxiliary']['compression']['model'] == 'second'
