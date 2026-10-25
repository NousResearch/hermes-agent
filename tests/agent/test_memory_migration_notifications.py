"""A configured memory provider that cannot be recovered at agent start is shown to the user,
never only logged (#119769): the CLI prints it; other drivers get a warn notice, which a
messaging-gateway agent (callbacks wired per turn) replays on its first turn."""
from agent.agent_init import _init_memory
from agent.status_output import StatusOutputMixin
from hermes_cli import memory_provider_migration as mig


class StartupAgent(StatusOutputMixin):
    enabled_toolsets = disabled_toolsets = tools = []
    valid_tool_names = set()
    log_prefix = ""
    quiet_mode = False
    status_callback = notice_callback = tool_progress_callback = None

    def __init__(self, platform):
        self.platform = platform
        self.printed = []

    def _vprint(self, message, **kwargs):
        self.printed.append(message)


def _start(tmp_path, monkeypatch, platform):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mig, "_attempted", set())
    monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: False)
    agent = StartupAgent(platform)
    monkeypatch.setattr(agent, "_warning_presentation_enabled", lambda: True)
    _init_memory(agent, {"memory": {"provider": "scout_missing_provider"}}, False, platform)
    assert agent._memory_manager is None and not (tmp_path / "plugins").exists()
    return agent


def test_cli_start_prints_the_refusal(tmp_path, monkeypatch):
    agent = _start(tmp_path, monkeypatch, "cli")
    assert any("hermes plugins install scout_missing_provider" in line for line in agent.printed)


def test_gateway_start_replays_the_refusal_once_on_first_turn(tmp_path, monkeypatch):
    agent = _start(tmp_path, monkeypatch, "telegram")
    notices = []
    agent.notice_callback = notices.append  # the gateway wires this per turn, after __init__
    agent._replay_startup_warnings()
    agent._replay_startup_warnings()
    assert len(notices) == 1 and notices[0].level == "warn"
    assert "hermes plugins install scout_missing_provider" in notices[0].text
