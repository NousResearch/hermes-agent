"""Regression tests for credential-safe tools_config subprocess spawns."""

import os


def test_pip_install_fallbacks_use_sanitized_env(monkeypatch):
    from hermes_cli import tools_config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider-must-strip")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "ant-provider-must-strip")
    monkeypatch.setenv("GH_TOKEN", "gh-must-strip")
    monkeypatch.setenv("GITHUB_TOKEN", "github-must-strip")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "discord-must-strip")
    monkeypatch.setattr(tools_config.shutil, "which", lambda _name: None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[-1:] == ["--version"]:
            return tools_config.subprocess.CompletedProcess(cmd, returncode=1)
        return tools_config.subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(tools_config.subprocess, "run", fake_run)

    result = tools_config._pip_install(["example-package"], timeout=123)

    assert result.returncode == 0
    assert len(calls) == 3
    assert calls[0][0][-1:] == ["--version"]
    assert calls[1][0][2] == "ensurepip"
    assert calls[2][0][-2:] == ["install", "example-package"]

    for _cmd, kwargs in calls:
        env = kwargs.get("env")
        assert env is not None
        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_TOKEN" not in env
        assert "GH_TOKEN" not in env
        assert "GITHUB_TOKEN" not in env
        assert "DISCORD_HOME_CHANNEL" not in env
        assert env["VIRTUAL_ENV"] == str(os.path.dirname(os.path.dirname(tools_config.sys.executable)))
