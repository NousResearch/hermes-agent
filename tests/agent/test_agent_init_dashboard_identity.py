from agent import agent_init


def test_dashboard_env_identity_requires_dashboard_source(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "alice")
    monkeypatch.setenv("HERMES_SESSION_USER_NAME", "Alice Example")
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert agent_init._dashboard_env_identity() == {}


def test_dashboard_env_identity_reads_non_secret_user_fields(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "dashboard")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "alice")
    monkeypatch.setenv("HERMES_SESSION_USER_NAME", "Alice Example")

    assert agent_init._dashboard_env_identity() == {
        "user_id": "alice",
        "user_name": "Alice Example",
    }


def test_dashboard_env_identity_does_not_use_user_id_as_name(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "dashboard")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "alice")
    monkeypatch.setenv("HERMES_SESSION_USER_NAME", "alice")

    assert agent_init._dashboard_env_identity() == {"user_id": "alice"}
