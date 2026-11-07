"""The grant uses the canonical inbound runtime profile."""

def test_gateway_binds_canonical_profile_with_blank_legacy_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session_identity import RoutingIdentity
    from gateway import session_context
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id='fixture', chat_type='dm',
        chat_name='', thread_id='', user_id='fixture', user_id_alt='', user_name='', scope_id='',
        message_id='fixture', profile='')
    source._identity = RoutingIdentity('default', 'default', tmp_path, tmp_path, False)
    context = SimpleNamespace(source=source, session_key='fixture', session_id='fixture')
    try:
        GatewayRunner._set_session_env(SimpleNamespace(adapters={}), context)
        assert session_context.get_session_env('HERMES_SESSION_PROFILE') == 'default'
    finally:
        session_context.reset_session_vars()
