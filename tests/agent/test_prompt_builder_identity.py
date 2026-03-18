from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, resolve_default_agent_identity


def test_default_agent_identity_without_override(monkeypatch):
    monkeypatch.delenv("MYNAH_AGENT_IDENTITY", raising=False)
    monkeypatch.delenv("HERMES_AGENT_IDENTITY", raising=False)

    assert resolve_default_agent_identity() == DEFAULT_AGENT_IDENTITY


def test_mynah_agent_identity_override_wins(monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_IDENTITY", "Hermes override")
    monkeypatch.setenv("MYNAH_AGENT_IDENTITY", "  MYNAH override  ")

    assert resolve_default_agent_identity() == "MYNAH override"
