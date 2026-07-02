import argparse
import json

from hermes_cli.plugins import PluginManager, PluginManifest
from plugins.platforms.photon import adapter as photon_adapter


def test_deferred_photon_platform_exposes_cli_command():
    manifest = PluginManifest(
        name="photon-platform",
        version="0.3.0",
        description="Photon Spectrum gateway adapter",
        source="bundled",
        path="plugins/platforms/photon",
        kind="platform",
        key="platforms/photon",
    )
    manager = PluginManager()

    manager._register_deferred_platform_cli(manifest)

    cmd = manager._cli_commands["photon"]
    parser = argparse.ArgumentParser()
    cmd["setup_fn"](parser)
    args = parser.parse_args(["status"])
    assert args.photon_command == "status"
    assert callable(cmd["handler_fn"])


def test_photon_sidecar_token_state_roundtrip(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(photon_adapter.PhotonAdapter, "_pid_is_sidecar", staticmethod(lambda pid: pid == 1234))

    photon_adapter._write_sidecar_token_state(
        bind="127.0.0.1",
        port=8789,
        token="secret-token",
        pid=1234,
    )

    path = tmp_path / "photon" / "sidecar-token.json"
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert photon_adapter._read_sidecar_token_state(bind="127.0.0.1", port=8789) == "secret-token"
    assert photon_adapter._read_sidecar_token_state(bind="127.0.0.1", port=8790) is None

    data = json.loads(path.read_text())
    data["pid"] = 9999
    path.write_text(json.dumps(data))
    assert photon_adapter._read_sidecar_token_state(bind="127.0.0.1", port=8789) is None


def test_photon_sidecar_token_state_clear_only_matching_token(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    photon_adapter._write_sidecar_token_state(
        bind="127.0.0.1",
        port=8789,
        token="current-token",
        pid=0,
    )

    path = tmp_path / "photon" / "sidecar-token.json"
    photon_adapter._clear_sidecar_token_state("other-token")
    assert path.exists()

    photon_adapter._clear_sidecar_token_state("current-token")
    assert not path.exists()
