from unittest.mock import MagicMock


def _make_stub():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    class Stub(CLICommandsMixin):
        def __init__(self):
            self.reasoning_config = {"enabled": True, "effort": "medium"}
            self.show_reasoning = False
            self.agent = MagicMock()

        def _current_reasoning_callback(self):
            return None

    return Stub()


def test_cli_reasoning_effort_is_session_scoped_by_default(monkeypatch):
    import cli

    saved = []
    monkeypatch.setattr(cli, "save_config_value", lambda key, value: saved.append((key, value)) or True)
    monkeypatch.setattr(cli, "_cprint", lambda *args, **kwargs: None)

    stub = _make_stub()
    stub._handle_reasoning_command("/reasoning high")

    assert stub.reasoning_config == {"enabled": True, "effort": "high"}
    assert stub.agent is None
    assert saved == []


def test_cli_reasoning_global_flag_persists(monkeypatch):
    import cli

    saved = []
    monkeypatch.setattr(cli, "save_config_value", lambda key, value: saved.append((key, value)) or True)
    monkeypatch.setattr(cli, "_cprint", lambda *args, **kwargs: None)

    stub = _make_stub()
    stub._handle_reasoning_command("/reasoning low --global")

    assert stub.reasoning_config == {"enabled": True, "effort": "low"}
    assert saved == [("agent.reasoning_effort", "low")]
