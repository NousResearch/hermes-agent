"""#152152: a `gateway run --no-supervise` container CMD is the one gateway, so boot must leave
the s6 root slot down even when the volume persists `desired_state: running`."""

import json

import pytest

from hermes_cli.container_boot import reconcile_profile_gateways

_WRAPPER = ["/init", "/opt/hermes/docker/main-wrapper.sh"]


@pytest.mark.parametrize(
    ("cmd", "env_opt_out", "slot_up"),
    [
        (["gateway", "run", "--no-supervise"], False, False),
        (["gateway", "run"], True, False),
        (["-p", "coder", "gateway", "run", "--no-supervise"], False, False),
        (["gateway", "run"], False, True),  # bare run redirects to s6: the slot IS the gateway
    ],
)
def test_foreground_gateway_cmd_leaves_s6_root_slot_down(tmp_path, monkeypatch, cmd, env_opt_out, slot_up):
    home, scandir = tmp_path / "data", tmp_path / "scan"
    home.mkdir()
    (home / "gateway_state.json").write_text(
        json.dumps({"gateway_state": "stopped", "desired_state": "running"}), encoding="utf-8")
    if env_opt_out:
        monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "1")
    else:
        monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    actions = reconcile_profile_gateways(hermes_home=home, scandir=scandir, container_argv=_WRAPPER + cmd)

    assert actions[0].profile == "default"
    assert actions[0].action == ("started" if slot_up else "registered")
    assert (scandir / "gateway-default" / "down").exists() is not slot_up
