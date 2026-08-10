"""Cross-profile gateway restart environment isolation."""

from pathlib import Path

import hermes_cli.gateway as gateway


def test_cross_profile_restart_env_replaces_source_profile_secrets(
    monkeypatch, tmp_path: Path
):
    source_home = tmp_path / "default"
    target_home = tmp_path / "profiles" / "testing"
    source_home.mkdir()
    target_home.mkdir(parents=True)

    (source_home / ".env").write_text(
        "DISCORD_BOT_TOKEN=source-token\n"
        "DISCORD_ALLOWED_USERS=source-user\n"
        "CUSTOM_PLUGIN_SECRET=source-secret\n",
        encoding="utf-8",
    )
    (target_home / ".env").write_text(
        "OPENROUTER_API_KEY=target-provider-key\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway, "get_hermes_home", lambda: source_home)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _profile: target_home
    )
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "inherited-source-token")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "inherited-source-user")
    monkeypatch.setenv("CUSTOM_PLUGIN_SECRET", "inherited-source-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "inherited-source-provider-key")
    monkeypatch.setenv("PATH", "safe-path")

    restart_env = gateway._profile_gateway_restart_env("testing")

    assert "DISCORD_BOT_TOKEN" not in restart_env
    assert "DISCORD_ALLOWED_USERS" not in restart_env
    assert "CUSTOM_PLUGIN_SECRET" not in restart_env
    assert restart_env["OPENROUTER_API_KEY"] == "target-provider-key"
    assert restart_env["PATH"] == "safe-path"


def test_same_profile_restart_env_preserves_shell_credentials(monkeypatch, tmp_path):
    profile_home = tmp_path / "testing"
    profile_home.mkdir()
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _profile: profile_home
    )
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "shell-only-token")

    restart_env = gateway._profile_gateway_restart_env("testing")

    assert restart_env["DISCORD_BOT_TOKEN"] == "shell-only-token"


def test_profile_restart_passes_isolated_env_to_watcher(monkeypatch):
    isolated_env = {"PATH": "safe-path", "HERMES_HOME": "target-home"}
    popen_calls = []

    monkeypatch.setattr(gateway.sys, "platform", "linux")
    monkeypatch.setattr(
        gateway,
        "_gateway_run_args_for_profile",
        lambda _profile: ["python", "-m", "hermes_cli.main", "gateway", "run"],
    )
    monkeypatch.setattr(
        gateway,
        "_profile_gateway_restart_env",
        lambda _profile: isolated_env,
    )
    monkeypatch.setattr(
        gateway.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    assert gateway.launch_detached_profile_gateway_restart("testing", 1234) is True

    assert len(popen_calls) == 1
    assert popen_calls[0][1]["env"] is isolated_env


def test_captured_profile_cmdline_restart_uses_isolated_env(monkeypatch):
    isolated_env = {"PATH": "safe-path", "HERMES_HOME": "target-home"}
    spawn_calls = []
    run_argv = [
        "python",
        "-m",
        "hermes_cli.main",
        "--profile",
        "testing",
        "gateway",
        "run",
    ]

    monkeypatch.setattr(
        gateway,
        "_profile_gateway_restart_env",
        lambda _profile: isolated_env,
    )
    monkeypatch.setattr(
        gateway,
        "_spawn_gateway_restart_watcher",
        lambda *args, **kwargs: spawn_calls.append((args, kwargs)) or True,
    )

    assert gateway.launch_detached_gateway_restart_by_cmdline(1234, run_argv) is True

    assert spawn_calls == [((1234, run_argv), {"watcher_env": isolated_env})]
