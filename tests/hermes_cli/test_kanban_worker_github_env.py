from __future__ import annotations


FAKE_TOKEN = "fake-token-for-test"


def test_normal_terminal_env_still_strips_github_tokens(monkeypatch):
    from tools.environments.local import _make_run_env

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    run_env = _make_run_env({"GH_TOKEN": FAKE_TOKEN, "GITHUB_TOKEN": FAKE_TOKEN})

    assert "GH_TOKEN" not in run_env
    assert "GITHUB_TOKEN" not in run_env


def test_force_prefix_can_forward_github_token_to_subprocess(monkeypatch):
    from tools.environments.local import _make_run_env

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    run_env = _make_run_env({"_HERMES_FORCE_GITHUB_TOKEN": FAKE_TOKEN})

    assert run_env["GITHUB_TOKEN"] == FAKE_TOKEN
    assert "_HERMES_FORCE_GITHUB_TOKEN" not in run_env


def test_force_prefix_can_forward_gh_token_to_subprocess(monkeypatch):
    from tools.environments.local import _make_run_env

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    run_env = _make_run_env({"_HERMES_FORCE_GH_TOKEN": FAKE_TOKEN})

    assert run_env["GH_TOKEN"] == FAKE_TOKEN
    assert "_HERMES_FORCE_GH_TOKEN" not in run_env


def test_kanban_worker_forced_github_env_only_in_worker_context(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setenv("GH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert terminal_tool._kanban_worker_forced_github_env() == {}

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auth_probe")
    forced = terminal_tool._kanban_worker_forced_github_env()

    assert forced["_HERMES_FORCE_GH_TOKEN"] == FAKE_TOKEN
    assert forced["_HERMES_FORCE_GITHUB_TOKEN"] == FAKE_TOKEN
    assert "_HERMES_FORCE_COPILOT_GITHUB_TOKEN" not in forced
    assert "GH_TOKEN" not in forced
    assert "GITHUB_TOKEN" not in forced
    assert "COPILOT_GITHUB_TOKEN" not in forced


def test_kanban_worker_does_not_forward_copilot_provider_token(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auth_probe")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", FAKE_TOKEN)
    forced = terminal_tool._kanban_worker_forced_github_env()

    assert forced == {}


def test_kanban_worker_force_env_keeps_copilot_out_of_subprocess(monkeypatch):
    from tools import terminal_tool
    from tools.environments.local import _make_run_env

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auth_probe")
    monkeypatch.setenv("GH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", FAKE_TOKEN)

    run_env = _make_run_env(terminal_tool._kanban_worker_forced_github_env())

    assert run_env["GH_TOKEN"] == FAKE_TOKEN
    assert run_env["GITHUB_TOKEN"] == FAKE_TOKEN
    assert "COPILOT_GITHUB_TOKEN" not in run_env
    assert "_HERMES_FORCE_GH_TOKEN" not in run_env
    assert "_HERMES_FORCE_GITHUB_TOKEN" not in run_env
    assert "_HERMES_FORCE_COPILOT_GITHUB_TOKEN" not in run_env


def test_local_environment_receives_kanban_force_env(monkeypatch):
    from tools import terminal_tool

    captured = {}

    class FakeLocalEnvironment:
        def __init__(self, *, cwd, timeout, env):
            captured["cwd"] = cwd
            captured["timeout"] = timeout
            captured["env"] = env

    monkeypatch.setattr(terminal_tool, "_LocalEnvironment", FakeLocalEnvironment)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_auth_probe")
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)

    env = terminal_tool._create_environment(
        env_type="local",
        image="",
        cwd="/tmp",
        timeout=60,
        local_config={"force_env": terminal_tool._kanban_worker_forced_github_env()},
    )

    assert isinstance(env, FakeLocalEnvironment)
    assert captured["env"] == {"_HERMES_FORCE_GITHUB_TOKEN": FAKE_TOKEN}
