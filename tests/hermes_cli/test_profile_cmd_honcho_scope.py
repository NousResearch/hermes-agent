"""Integration regression for clone-based profile creation under shared Honcho policy."""

from argparse import Namespace
import json
from pathlib import Path

from hermes_cli import profile_cmd
from plugins.memory.honcho import cli as honcho_cli


def test_clone_command_from_named_home_only_syncs_default_shared_honcho(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    default_config = {
        "apiKey": "default-key",
        "hosts": {"hermes": {"workspace": "default-workspace", "aiPeer": "default"}},
    }
    default_config_path = default_home / "honcho.json"
    default_config_path.write_text(json.dumps(default_config), encoding="utf-8")
    (default_home / ".sync-profile-honcho").write_text("enabled\n", encoding="utf-8")

    launch_home = default_home / "profiles" / "launch"
    launch_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    launch_config = {
        "apiKey": "launch-key",
        "hosts": {"hermes": {"workspace": "launch-workspace", "aiPeer": "launch"}},
    }
    launch_config_path = launch_home / "honcho.json"
    launch_config_path.write_text(json.dumps(launch_config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    peer_operations = []

    def record_peer_operation(host_key=None, *, config_path=None):
        path = Path(config_path) if config_path is not None else honcho_cli._config_path()
        config = json.loads(path.read_text(encoding="utf-8"))
        block = config["hosts"][host_key]
        peer_operations.append((path, block["workspace"], block["aiPeer"]))
        return True

    monkeypatch.setattr(honcho_cli, "_ensure_peer_exists", record_peer_operation)

    profile_cmd.cmd_profile(Namespace(
        profile_action="create",
        profile_name="coder",
        clone=True,
        clone_all=False,
        clone_from=None,
        clone_channels=False,
        sync_imports=False,
        no_alias=True,
        no_skills=False,
        description=None,
    ))

    saved_default = json.loads(default_config_path.read_text(encoding="utf-8"))
    assert saved_default["hosts"]["hermes_coder"]["aiPeer"] == "coder"
    assert saved_default["hosts"]["hermes_coder"]["workspace"] == "default-workspace"
    assert json.loads(launch_config_path.read_text(encoding="utf-8")) == launch_config
    assert peer_operations == [(default_config_path, "default-workspace", "coder")]
