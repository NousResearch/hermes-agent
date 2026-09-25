"""G3 guard rail: `hermes update` verifies (config parity, gateway liveness,
MCP fingerprint parity) against a pre-update snapshot before reporting
success; violations roll back. Spec §5."""
import json
import socket

import pytest
import yaml

from hermes_cli import update_verification as uv


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    bobby = root / "profiles" / "bobby"
    bobby.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    seeds = (
        (root, {"github": {"command": "npx", "args": ["-y", "pkg"], "enabled": True}}),
        (bobby, {"lovable": {"url": "https://mcp.example", "enabled": False}}),
    )
    for home, servers in seeds:
        (home / "config.yaml").write_text(yaml.safe_dump({
            "mcp_servers": servers,
            "custom_providers": {"ark": {"base_url": "https://ark.example"}},
        }), encoding="utf-8")
    return {"root": root, "bobby": bobby}


def test_capture_roundtrip_and_parity(two_homes):
    state = uv.capture_pre_state(
        [("default", two_homes["root"]), ("bobby", two_homes["bobby"])], backup=False)
    assert state["profiles"]["bobby"]["root_keys"] == ["custom_providers", "mcp_servers"]
    assert state["profiles"]["bobby"]["mcp_fingerprint"] == {
        "lovable": {"enabled": False, "command": None, "url": "https://mcp.example"}}
    assert state["gateway_was_running"] is False  # no gateway_state.json in tmp home
    ok, _ = uv.check_config_parity(state)
    assert ok
    ok, _ = uv.check_mcp_fingerprint_parity(state)
    assert ok


def test_dropped_key_fails_config_parity(two_homes):
    state = uv.capture_pre_state([("default", two_homes["root"])], backup=False)
    (two_homes["root"] / "config.yaml").write_text(
        yaml.safe_dump({"custom_providers": {"ark": {"base_url": "https://ark.example"}}}),
        encoding="utf-8")
    ok, detail = uv.check_config_parity(state)
    assert not ok
    assert "mcp_servers" in detail


def test_fingerprint_drift_detected(two_homes):
    state = uv.capture_pre_state([("bobby", two_homes["bobby"])], backup=False)
    raw = yaml.safe_load((two_homes["bobby"] / "config.yaml").read_text(encoding="utf-8"))
    raw["mcp_servers"]["lovable"]["enabled"] = True
    (two_homes["bobby"] / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    ok, detail = uv.check_mcp_fingerprint_parity(state)
    assert not ok
    assert "changed" in detail


def test_fingerprint_never_contains_env_or_args_secrets(two_homes):
    state = uv.capture_pre_state([("default", two_homes["root"])], backup=False)
    blob = json.dumps(state)
    assert "args" not in state["profiles"]["default"]["mcp_fingerprint"]["github"]
    assert "env" not in blob  # only enabled/command/url per server


def test_liveness_passes_on_running_state_with_open_port(monkeypatch):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        monkeypatch.setattr(uv, "_read_gateway_state", lambda: {
            "gateway_state": "running",
            "platforms": {"webhook": {"listener_base": f"http://127.0.0.1:{port}"}},
        })
        ok, detail = uv.check_gateway_liveness(timeout=2.0)
        assert ok
        assert "accepting" in detail
    finally:
        listener.close()


def test_liveness_times_out_bounded(monkeypatch):
    monkeypatch.setattr(uv, "_read_gateway_state", lambda: {"gateway_state": "stopped"})
    import time
    started = time.monotonic()
    ok, _ = uv.check_gateway_liveness(timeout=1.0)
    assert not ok
    assert time.monotonic() - started < 10  # no infinite wait


def test_run_verification_skips_liveness_when_gateway_was_not_running(two_homes, monkeypatch):
    monkeypatch.setattr(uv, "check_gateway_liveness",
                        lambda *a, **k: pytest.fail("liveness must be skipped"))
    state = uv.capture_pre_state([("default", two_homes["root"])], backup=False)
    results = uv.run_verification(state)
    assert [c["name"] for c in results] == ["config_parity", "mcp_fingerprint_parity"]


def test_save_and_load_pre_state_roundtrip(two_homes):
    state = uv.capture_pre_state([("default", two_homes["root"])], backup=False)
    path = uv.save_pre_state(state)
    assert path.name == uv.PRE_STATE_FILENAME
    assert uv.load_pre_state(path) == state
