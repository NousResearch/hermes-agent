"""``hermes update`` respawns a sibling profile's gateway with that profile's credentials, never the updater's.

The updater has its own profile's ``.env`` in ``os.environ``. The restart watcher hands the env it was
spawned with to the gateway it respawns, so a profile that declares no bot token of its own (an
API-only profile) used to start the ROOT bot and steal its polling lock (#135792).
"""

import os
import subprocess

import pytest

_ROOT_TOKEN = "root-bot-token-111"
_OWN_TOKEN = "api1-own-token-222"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    api1 = root / "profiles" / "api1"
    api1.mkdir(parents=True)
    for home in (root, api1):
        (home / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_ROOT_TOKEN}\n", encoding="utf-8")
    (api1 / ".env").write_text("API_SERVER_ENABLED=true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The updater runs on the default profile: its .env is loaded into the process env.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _ROOT_TOKEN)
    # Which names a dotenv load put into os.environ is process-wide state: isolate it per test.
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    return root, api1


def _watcher_env(monkeypatch, profile: str) -> dict:
    """Env of the restart watcher ``launch_detached_profile_gateway_restart`` spawns (recorded, not run)."""
    from hermes_cli import gateway

    spawned = []

    def _record(argv, **kwargs):
        if len(argv) > 2 and argv[1] == "-c" and "pid_exists_stdlib" in argv[2]:
            spawned.append(kwargs.get("env"))

    monkeypatch.setattr(subprocess, "Popen", _record)
    assert gateway.launch_detached_profile_gateway_restart(profile, 424242)
    assert len(spawned) == 1
    return spawned[0] if spawned[0] is not None else dict(os.environ)


@pytest.mark.platforms("posix")
def test_sibling_profile_without_a_token_never_gets_the_updaters(homes, monkeypatch):
    _root, api1 = homes
    env = _watcher_env(monkeypatch, "api1")
    assert env.get("HERMES_HOME") == str(api1)
    assert env.get("TELEGRAM_BOT_TOKEN") != _ROOT_TOKEN
    assert env.get("API_SERVER_ENABLED") == "true"

    # The profile's own token still reaches it; the host restart still carries the root's.
    (api1 / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_OWN_TOKEN}\n", encoding="utf-8")
    from hermes_cli.env_loader import reset_secret_source_cache
    reset_secret_source_cache()
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") == _OWN_TOKEN
    assert _watcher_env(monkeypatch, "default").get("TELEGRAM_BOT_TOKEN") == _ROOT_TOKEN


@pytest.mark.platforms("posix")
def test_updater_on_the_same_profile_keeps_its_exported_token(homes, monkeypatch):
    _root, api1 = homes
    # `hermes -p api1 update` with a token exported in the shell, api1's .env declaring none.
    monkeypatch.setenv("HERMES_HOME", str(api1))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "exported-api1-token-333")
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") == "exported-api1-token-333"


@pytest.mark.platforms("posix")
def test_shell_exported_keys_reach_a_sibling_but_the_root_bot_token_does_not(homes, monkeypatch):
    """Only the launch profile's own bot tokens are withheld. A model key or a platform token that
    lives only in the user's shell reached the sibling on its manual launch, so the update restart
    must keep it, or every turn fails until someone restarts the gateway by hand."""
    root, _api1 = homes
    from hermes_cli.env_loader import load_hermes_dotenv
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-or-key")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "shell-discord-token")
    load_hermes_dotenv(hermes_home=root)  # the updater loads its own profile's .env
    env = _watcher_env(monkeypatch, "api1")
    assert env.get("TELEGRAM_BOT_TOKEN") is None
    assert (env.get("OPENROUTER_API_KEY"), env.get("DISCORD_BOT_TOKEN")) == ("shell-or-key", "shell-discord-token")


@pytest.mark.platforms("posix")
def test_a_unit_injected_root_token_never_follows_a_sibling(homes, monkeypatch):
    """``/update`` sent to a systemd gateway: its token came from ``Environment=``, not any .env."""
    root, _api1 = homes
    (root / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "unit-injected-token")
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") is None
