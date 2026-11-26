"""The transport stamp follows ingress identity without replacing runtime ownership."""
from gateway.config import Platform
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env
from gateway.session_identity import canonical_identity
from tests.gateway.test_kanban_routed_transport import setup_runner
from tools.kanban_provenance import conversation_origin


def test_origin_separates_transport_from_routed_runtime(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    source = SessionSource(platform=Platform.DISCORD, chat_id="parent", user_id="creator", guild_id="guild")
    identity = canonical_identity(source, runner=runner, adapter=primary, transport_profile="default")
    assert identity.runtime_profile == "yuki"
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_id="conversation")
    tokens = runner._set_session_env(context)
    try:
        assert get_session_env("HERMES_SESSION_TRANSPORT_PROFILE") == "default"
        assert get_session_env("HERMES_SESSION_PROFILE") == "yuki"
        origin = conversation_origin("conversation")
        assert origin["transport_profile"] == "default"
    finally:
        runner._clear_session_env(tokens)
    assert get_session_env("HERMES_SESSION_TRANSPORT_PROFILE") == ""
