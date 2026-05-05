from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, SessionStore, build_session_key, resolve_channel_route


def _make_source(
    *,
    chat_id: str = "1234567890",
    thread_id: str | None = None,
    chat_type: str = "group",
    user_id: str = "user-1",
):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        thread_id=thread_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="Alice",
    )


def _make_config(routes):
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                extra={"channel_routes": routes},
            )
        }
    )


def test_build_session_key_uses_custom_agent_id():
    key = build_session_key(_make_source(), agent_id="research")
    assert key == "agent:research:discord:group:1234567890:user-1"


def test_resolve_channel_route_matches_channel_id(tmp_path):
    workspace = tmp_path / "research-workspace"
    workspace.mkdir()
    config = _make_config(
        [
            {
                "id": "1234567890",
                "agent_id": "research",
                "cwd": str(workspace),
            }
        ]
    )

    route = resolve_channel_route(_make_source(), config)

    assert route is not None
    assert route["agent_id"] == "research"
    assert route["cwd"] == str(workspace.resolve())


def test_resolve_channel_route_matches_thread_id_and_workspace_alias(tmp_path):
    workspace = tmp_path / "support-workspace"
    workspace.mkdir()
    config = _make_config(
        [
            {
                "id": "thread-123",
                "agentId": "support",
                "workspace": str(workspace),
            }
        ]
    )

    route = resolve_channel_route(_make_source(thread_id="thread-123"), config)

    assert route is not None
    assert route["agent_id"] == "support"
    assert route["cwd"] == str(workspace.resolve())


def test_resolve_channel_route_ignores_malformed_routes():
    config = _make_config(
        [
            "not-a-route",
            {"id": ""},
            {"id": "1234567890", "agent_id": "bad:agent"},
            {"id": "different", "agent_id": "other"},
        ]
    )

    assert resolve_channel_route(_make_source(), config) is None


def test_resolve_channel_route_ignores_invalid_cwd_but_keeps_agent_id():
    config = _make_config(
        [
            {"id": "1234567890", "agent_id": "relative", "cwd": "relative/path"},
        ]
    )

    route = resolve_channel_route(_make_source(), config)

    assert route is not None
    assert route["agent_id"] == "relative"
    assert route["cwd"] is None


def test_session_store_generate_session_key_uses_routed_agent_id(tmp_path):
    workspace = tmp_path / "research-workspace"
    workspace.mkdir()
    config = _make_config(
        [
            {
                "id": "1234567890",
                "agent_id": "research",
                "cwd": str(workspace),
            }
        ]
    )
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_key = store._generate_session_key(_make_source())

    assert session_key == "agent:research:discord:group:1234567890:user-1"
