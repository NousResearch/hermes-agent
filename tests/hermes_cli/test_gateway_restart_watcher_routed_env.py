"""The detached restart watcher must not hand a sibling profile the updater's env (#135792).

``hermes update`` restarts sibling profile gateways through ``gateway._spawn_gateway_restart_watcher``.
The watcher copies its own environ into the respawned gateway, and the updater has already loaded
the ROOT profile's ``.env`` into ``os.environ`` — so an API-only profile whose own dotenv lacks
``TELEGRAM_BOT_TOKEN`` resolved the inherited root token (``get_env_value_prefer_dotenv``'s environ
fallback), claimed the default bot's gateway lock, and broke the default gateway's startup. The
watcher now runs on ``served_profile_child_env`` for a routed home: launch residue stripped,
credentials scrubbed, the target home's own secrets overlaid — the same env ``_spawn_detached``
and the POSIX replay (``update_cmd_posix_pause._replay_env``) already use.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import gateway
from hermes_cli.gateway_restart_env import routed_home_watcher_env

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="watcher respawn driven via the POSIX branch"
)


def _vanished_pid() -> int:
    """A PID beyond every supported platform's pid_max: the watcher's poll loop reads it as
    already-gone and respawns immediately, without a real process to wait on."""
    return 99_999_999


def _wait_for(condition, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


def _make_home(root: Path, env_lines: list) -> Path:
    home = root
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return home


class TestRoutedHomeWatcherEnv:
    def test_same_home_replay_keeps_the_launchers_exported_env(
        self, tmp_path, monkeypatch
    ):
        launch_home = _make_home(tmp_path / "launch", [])
        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        assert routed_home_watcher_env(str(launch_home)) is None

    def test_routed_home_scrubs_launch_credentials_and_overlays_the_targets_own(
        self, tmp_path, monkeypatch
    ):
        launch_home = _make_home(tmp_path / "launch", ["HERMES_LANGUAGE=en"])
        target_home = _make_home(
            tmp_path / "api1", ["TELEGRAM_BOT_TOKEN=target-profile-token"]
        )
        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        # The updater loaded the ROOT profile's .env into os.environ before the fleet restart.
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "root-profile-token")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "root-discord-token")

        env = routed_home_watcher_env(str(target_home))

        assert env is not None
        assert env.get("TELEGRAM_BOT_TOKEN") == "target-profile-token", (
            "the respawned sibling must resolve its OWN dotenv token, never the root one"
        )
        assert "DISCORD_BOT_TOKEN" not in env, (
            "a credential the target dotenv does not declare must not survive from the updater env"
        )
        assert env.get("HERMES_HOME") == str(target_home.resolve())
        assert "_HERMES_GATEWAY" not in env

    def test_env_helpers_unavailable_degrades_to_the_inherited_env(
        self, tmp_path, monkeypatch
    ):
        launch_home = _make_home(tmp_path / "launch", [])
        target_home = _make_home(
            tmp_path / "api1", ["TELEGRAM_BOT_TOKEN=target-profile-token"]
        )
        monkeypatch.setenv("HERMES_HOME", str(launch_home))

        import tools.environments.local as local_env

        def _unavailable(*_a, **_kw):
            raise RuntimeError("bare store python without the dependency environment")

        monkeypatch.setattr(local_env, "served_profile_child_env", _unavailable)
        assert routed_home_watcher_env(str(target_home)) is None


class TestWatcherRespawnOnRoutedHome:
    def test_respawned_child_env_carries_the_target_not_the_updater_credentials(
        self, tmp_path, monkeypatch
    ):
        launch_home = _make_home(tmp_path / "launch", [])
        target_home = _make_home(
            tmp_path / "api1", ["TELEGRAM_BOT_TOKEN=target-profile-token"]
        )
        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "root-profile-token")

        dump = tmp_path / "respawned-env.json"
        monkeypatch.setenv("PROBE_ENV_DUMP_TARGET", str(dump))
        probe_script = tmp_path / "probe_dump_env.py"
        probe_script.write_text(
            "import json, os, pathlib\n"
            'pathlib.Path(os.environ["PROBE_ENV_DUMP_TARGET"]).write_text(json.dumps(dict(os.environ)))\n',
            encoding="utf-8",
        )
        relaunch = [sys.executable, str(probe_script)]

        assert gateway._spawn_gateway_restart_watcher(
            _vanished_pid(), relaunch, host=False, home=str(target_home)
        )
        assert _wait_for(dump.exists), (
            "the restart watcher never relaunched the probe command"
        )

        respawned = json.loads(dump.read_text(encoding="utf-8"))
        assert respawned.get("TELEGRAM_BOT_TOKEN") == "target-profile-token", (
            "the respawned gateway resolved the root updater's token instead of its own profile's"
        )
        assert respawned.get("HERMES_HOME") == str(target_home.resolve())
