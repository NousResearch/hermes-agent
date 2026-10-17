from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import clear_session_vars, set_session_vars


def test_desktop_ui_session_id_uses_tab_id_not_durable_session_id(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return []

    tab_id = "29952e9a"
    durable_session_id = "20261003_102410_572750"
    tokens = set_session_vars(ui_session_id=tab_id)
    try:
        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        context = SessionContext(
            source=SessionSource(platform=Platform.LOCAL, chat_id="desktop"),
            connected_platforms=[],
            home_channels={},
            session_key="desktop-session-key",
            session_id=durable_session_id,
        )

        monkeypatch.setattr("gateway.session_context.set_session_vars", capture)
        GatewayRunner._set_session_env(runner, context)
    finally:
        clear_session_vars(tokens)

    assert captured["ui_session_id"] == tab_id
    assert captured["ui_session_id"] != durable_session_id
