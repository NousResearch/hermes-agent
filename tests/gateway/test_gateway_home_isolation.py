import json
import os
import subprocess
import sys
import textwrap
from collections.abc import MutableMapping
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_home(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "sessions").mkdir(exist_ok=True)
    (path / "platforms" / "pairing").mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("", encoding="utf-8")
    return path


def _run_probe(script: str, host_home: Path, isolated_home: Path) -> dict:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(host_home)
    env["HOST_HOME"] = str(host_home)
    env["ISOLATED_HOME"] = str(isolated_home)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    for key in list(env):
        if key.endswith("_ALLOW_ALL_USERS") or key.endswith("_ALLOWED_USERS"):
            env.pop(key, None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_gateway_runner_reloads_env_from_patched_hermes_home(tmp_path):
    host_home = _make_home(tmp_path / "host_home")
    isolated_home = _make_home(tmp_path / "isolated_home")
    (host_home / ".env").write_text("GATEWAY_ALLOW_ALL_USERS=true\n", encoding="utf-8")

    script = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path

        import gateway.run as gateway_run
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionSource

        isolated_home = Path(os.environ["ISOLATED_HOME"])
        os.environ["HERMES_HOME"] = str(isolated_home)

        runner = gateway_run.GatewayRunner(GatewayConfig())
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-1",
            chat_type="dm",
            user_id="unknown-user",
        )
        print(json.dumps({"authorized": runner._is_user_authorized(source)}))
        """
    )

    result = _run_probe(script, host_home, isolated_home)
    assert result == {"authorized": False}


def test_gateway_voice_mode_path_uses_patched_hermes_home_after_preimport(tmp_path):
    host_home = _make_home(tmp_path / "host_home")
    isolated_home = _make_home(tmp_path / "isolated_home")
    (host_home / "gateway_voice_mode.json").write_text(
        json.dumps({"host-chat": "all"}),
        encoding="utf-8",
    )

    script = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path

        import gateway.run as gateway_run
        from gateway.config import GatewayConfig

        isolated_home = Path(os.environ["ISOLATED_HOME"])
        host_home = Path(os.environ["HOST_HOME"])
        os.environ["HERMES_HOME"] = str(isolated_home)

        runner = gateway_run.GatewayRunner(GatewayConfig())
        runner._voice_mode = {"isolated-chat": "voice_only"}
        runner._save_voice_modes()

        payload = {
            "isolated_exists": (isolated_home / "gateway_voice_mode.json").exists(),
            "isolated_data": json.loads((isolated_home / "gateway_voice_mode.json").read_text(encoding="utf-8")),
            "host_data": json.loads((host_home / "gateway_voice_mode.json").read_text(encoding="utf-8")),
        }
        print(json.dumps(payload))
        """
    )

    result = _run_probe(script, host_home, isolated_home)
    assert result["isolated_exists"] is True
    assert result["isolated_data"] == {"isolated-chat": "voice_only"}
    assert result["host_data"] == {"host-chat": "all"}


def test_gateway_rebootstrap_preserves_in_process_env_override_for_second_runner(tmp_path):
    host_home = _make_home(tmp_path / "host_home")
    isolated_home = _make_home(tmp_path / "isolated_home")
    (host_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=user1\n", encoding="utf-8")

    script = textwrap.dedent(
        """
        import json
        import os

        import gateway.run as gateway_run
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionSource

        os.environ["HERMES_HOME"] = os.environ["HOST_HOME"]
        gateway_run.GatewayRunner(GatewayConfig())

        os.environ["TELEGRAM_ALLOWED_USERS"] = "user2"
        os.environ["HERMES_HOME"] = os.environ["ISOLATED_HOME"]
        runner = gateway_run.GatewayRunner(GatewayConfig())

        source_user1 = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm", user_id="user1")
        source_user2 = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm", user_id="user2")
        print(json.dumps({
            "env": os.environ.get("TELEGRAM_ALLOWED_USERS"),
            "user1": runner._is_user_authorized(source_user1),
            "user2": runner._is_user_authorized(source_user2),
        }))
        """
    )

    result = _run_probe(script, host_home, isolated_home)
    assert result == {"env": "user2", "user1": False, "user2": True}


class _OverlayWithoutUnion(MutableMapping):
    def __init__(self):
        self._data = {"PATH": "/usr/bin", "BASE": "1"}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def test_local_run_env_accepts_environ_overlay_without_mapping_union(monkeypatch):
    from tools.environments import local

    monkeypatch.setattr(local.os, "environ", _OverlayWithoutUnion())
    run_env = local._make_run_env({"EXTRA": "2"})

    assert run_env["BASE"] == "1"
    assert run_env["EXTRA"] == "2"
