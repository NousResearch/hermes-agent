"""The multiplexed host gateway's launch profile keeps its env-only credentials and ``TERMINAL_*``.

Once the host arms multiplexing, a scoped secret miss returns the default instead of reading
``os.environ``, and a bound terminal scope is the whole policy. ``_profile_runtime_scope`` rebuilt
every home from its files, the launch home included, so a value that reached the launch profile only
through the process env (systemd ``Environment=`` / ``EnvironmentFile=``, ``op run``, a shell export)
vanished from its own turns: provider keys and bot tokens read as unset and ``TERMINAL_ENV=docker``
ran on the host. The launch home must bind its files over the env frozen at activation (what
``hermes serve`` already does); every other served home stays files-only.
"""
import asyncio
import os

import pytest

from agent.secret_scope import get_secret
from gateway.config import Platform
from gateway.run import _async_profile_runtime_scope, _profile_runtime_scope, load_gateway_config_for_runner
from tools.terminal_scope import terminal_env

LAUNCH_ENV = {
    "OPENROUTER_API_KEY": "sk-from-systemd",
    "TELEGRAM_BOT_TOKEN": "123:from-systemd",
    "TERMINAL_ENV": "docker",
    "TERMINAL_LOCAL_MEMORY_MAX_MB": "64",
}


@pytest.fixture
def host(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    for home in (root, coder):
        (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    (root / ".env").write_text("ROOT_FILE_KEY=root\n", encoding="utf-8")
    (coder / ".env").write_text("CODER_FILE_KEY=coder\n", encoding="utf-8")
    for name, value in LAUNCH_ENV.items():
        monkeypatch.setenv(name, value)
    return root, coder


def _boot(monkeypatch, launch_home):
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    cfg = load_gateway_config_for_runner()
    assert cfg.multiplex_profiles
    # A secondary's context rewrites the process env after activation; the frozen snapshot must win.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-poisoned-after-activation")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return cfg


def _seen():
    return (get_secret("OPENROUTER_API_KEY"), get_secret("TELEGRAM_BOT_TOKEN"),
            terminal_env("TERMINAL_ENV"), terminal_env("TERMINAL_LOCAL_MEMORY_MAX_MB"))


async def _seen_async(home):
    async with _async_profile_runtime_scope(home):
        return _seen()


LAUNCH_VIEW = ("sk-from-systemd", "123:from-systemd", "docker", "64")
FILES_ONLY_VIEW = (None, None, "local", "")


def test_launch_home_keeps_env_only_values_across_a_secondary_turn(host, monkeypatch):
    root, coder = host
    cfg = _boot(monkeypatch, root)
    # The primary adapter map is loaded under the default root's scope: its env-only token survives.
    assert cfg.platforms[Platform.TELEGRAM].token == "123:from-systemd"

    with _profile_runtime_scope(root):
        assert _seen() == LAUNCH_VIEW
    assert asyncio.run(_seen_async(coder)) == FILES_ONLY_VIEW
    with _profile_runtime_scope(coder, hydrate_secrets=False):
        assert _seen() == FILES_ONLY_VIEW
    assert asyncio.run(_seen_async(root)) == LAUNCH_VIEW
    with _profile_runtime_scope(root, hydrate_secrets=False):
        assert _seen() == LAUNCH_VIEW
    assert os.environ["OPENROUTER_API_KEY"] == "sk-poisoned-after-activation"


def test_named_launcher_env_belongs_to_the_launcher_not_the_default_root(host, monkeypatch):
    root, coder = host
    cfg = _boot(monkeypatch, coder)
    # The default root is a served secondary here: the launcher's env never reaches its adapters.
    assert Platform.TELEGRAM not in cfg.platforms

    with _profile_runtime_scope(coder):
        assert _seen() == LAUNCH_VIEW
    assert asyncio.run(_seen_async(root)) == FILES_ONLY_VIEW
    with _profile_runtime_scope(root):
        assert _seen() == FILES_ONLY_VIEW
    assert asyncio.run(_seen_async(coder)) == LAUNCH_VIEW
