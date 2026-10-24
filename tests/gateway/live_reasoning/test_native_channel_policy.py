"""Resolve channel policies through canonical configuration and a real gateway session store."""
import os
from uuid import uuid4

import pytest


@pytest.fixture
def configured_gateway(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    parent, child, topic, ordinary = (uuid4().hex for _ in range(4))
    data = {
        'gateway': {'multiplex_profiles': False},
        'model': {'default': 'gpt-6.1-sol'}, 'agent': {'reasoning_effort': 'low'},
        'platforms': {
            'discord': {'channel_overrides': {parent: {'reasoning_effort': 'high'}, child: {'model': 'gpt-6.1-sol'}}},
            'telegram': {'channel_overrides': {parent: {'reasoning_effort': 'high'}, f'{parent}:{topic}': {'reasoning_effort': False}}},
        },
    }
    atomic_config_write(tmp_path / 'config.yaml', data)
    from gateway.config import load_gateway_config
    from gateway.run import GatewayRunner

    config = load_gateway_config()
    config.multiplex_profiles = False
    runner = GatewayRunner(config)
    assert runner._session_db_init_error is None
    try:
        yield runner, parent, child, topic, ordinary
    finally:
        runner.session_store.close_all_db_handles()
        runner.close_all_session_db_handles()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


@pytest.mark.parametrize('kind', ['discord-thread', 'telegram-topic', 'unscoped'])
def test_field_inheritance_disable_and_session_precedence(configured_gateway, kind):
    from gateway.config import Platform
    from gateway.session import SessionSource

    runner, parent, child, topic, ordinary = configured_gateway
    if kind == 'discord-thread':
        source = SessionSource(platform=Platform.DISCORD, chat_id=child, parent_chat_id=parent, chat_type='thread')
        expected, scoped = {'enabled': True, 'effort': 'high'}, True
    elif kind == 'telegram-topic':
        source = SessionSource(platform=Platform.TELEGRAM, chat_id=parent, thread_id=topic, chat_type='group')
        expected, scoped = {'enabled': False}, True
    else:
        source = SessionSource(platform=Platform.DISCORD, chat_id=ordinary, chat_type='channel')
        expected, scoped = {'enabled': True, 'effort': 'low'}, False
    entry = runner.session_store.get_or_create_session(source)
    assert runner._resolve_session_reasoning_config(source=source, model='gpt-6.1-sol') == expected
    assert runner._has_scoped_reasoning_override(source=source) is scoped
    session_policy = {'enabled': True, 'effort': 'max'}
    runner._set_session_reasoning_override(entry.session_key, session_policy)
    assert runner._resolve_session_reasoning_config(source=source, model='gpt-6.1-sol') == session_policy
    assert runner._has_scoped_reasoning_override(source=source) is True
    runner._set_session_reasoning_override(entry.session_key, None)
    assert runner._resolve_session_reasoning_config(source=source, model='gpt-6.1-sol') == expected
