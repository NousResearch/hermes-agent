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


def test_a_gateway_hosted_cli_session_prints_the_startup_warning_on_its_first_turn():
    """``hermes chat`` runs its agent in the gateway: built before the viewer's print sink is wired, a
    startup line printed during init reached only the gateway's stdout log. It prints on the first turn, once."""
    agent = StartupAgent("cli")
    agent._gateway_session_key = "agent:main:cli:dm:local"
    agent._warning_presentation_enabled = lambda: True
    agent._emit_startup_warning("✓ Something moved out of core — installed its plugin.")
    assert agent.printed == []
    agent._replay_startup_warnings()  # first turn: the viewer's sink is wired now
    agent._replay_startup_warnings()
    assert agent.printed == ["✓ Something moved out of core — installed its plugin."]
