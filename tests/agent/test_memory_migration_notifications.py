"""Startup migration outcomes reach the existing gated warning sink."""
import pytest

from agent.agent_init import _init_memory
from agent.status_output import StatusOutputMixin
from hermes_cli import memory_provider_migration as mig


class StartupAgent(StatusOutputMixin):
    enabled_toolsets = []
    disabled_toolsets = []
    platform = "cli"
    log_prefix = ""
    quiet_mode = False
    status_callback = None
    tools = []
    valid_tool_names = set()

    def _vprint(self, message, **kwargs):
        print(message)


@pytest.fixture
def startup_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mig, "_attempted", set())
    (tmp_path / "config.yaml").write_text(
        "security:\n  allow_lazy_installs: false\nmemory:\n  provider: scout_missing_provider\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("show", [True, False])
def test_startup_refusal_reaches_gated_cli_sink(startup_home, capsys, monkeypatch, show):
    agent = StartupAgent()
    monkeypatch.setattr(agent, "_warning_presentation_enabled", lambda: show)
    _init_memory(agent, {"memory": {"provider": "scout_missing_provider"}}, False, "cli")
    output = capsys.readouterr().out
    assert ("hermes plugins install scout_missing_provider" in output) is show
    assert agent._memory_manager is None
    assert not (startup_home / "plugins").exists()


@pytest.mark.parametrize("outcome", ["success", "unknown", "error", "present"])
def test_recovery_outcomes_keep_log_and_return_contract(startup_home, monkeypatch, caplog, outcome):
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: outcome == "present")
    monkeypatch.setattr(mig, "catalog_source", lambda name: None if outcome == "unknown" else name)
    calls = []

    def install(name):
        calls.append(name)
        if outcome == "error":
            raise RuntimeError("offline fixture")
        return {"ok": True}

    monkeypatch.setattr(mig, "_install_into", lambda home: install)
    said = []
    result = mig.recover_at_startup("scout_missing_provider", say=said.append)
    assert result is (outcome == "success")
    assert len(said) == (0 if outcome == "present" else 1)
    assert calls == (["scout_missing_provider"] if outcome in {"success", "error"} else [])
    if said:
        assert said[0] in caplog.text
    assert mig.recover_at_startup("scout_missing_provider", say=said.append) is False
    assert len(said) == (0 if outcome == "present" else 1)


def test_broken_presentation_does_not_undo_success(startup_home, monkeypatch, caplog):
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: False)
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig, "_install_into", lambda home: lambda name: {"ok": True})

    def broken(message):
        raise RuntimeError("presentation unavailable")

    assert mig.recover_at_startup("scout_missing_provider", say=broken) is True
    assert "installed its plugin" in caplog.text


def test_default_caller_retains_logging(startup_home, caplog):
    assert mig.recover_at_startup("scout_missing_provider") is False
    assert "security.allow_lazy_installs is off" in caplog.text
