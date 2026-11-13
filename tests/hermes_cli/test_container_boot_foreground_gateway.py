"""#152152: a foreground (`--no-supervise`) gateway container CMD is the one gateway, so boot must
leave the s6 root slot down even when the volume persists `desired_state: running`, unless that
foreground gateway serves only a `gateway.standalone` profile."""

import json

import pytest

from hermes_cli.container_boot import reconcile_profile_gateways

_WRAPPER = ["/init", "/opt/hermes/docker/main-wrapper.sh"]


@pytest.mark.parametrize(
    ("cmd", "env_opt_out", "slot_up"),
    [
        (["gateway", "run", "--no-supervise"], False, False),
        (["gateway", "run"], True, False),
        (["gateway"], True, False),  # bare `gateway` runs the gateway too
        (["-p", "coder", "gateway", "run", "--no-supervise"], False, False),
        (["-p", "gateway", "gateway", "run", "--no-supervise"], False, False),  # profile named gateway
        (["-p", "solo", "gateway", "run", "--no-supervise"], False, True),  # standalone: serves only solo
        (["gateway", "run"], False, True),  # bare run redirects to s6: the slot IS the gateway
        (["gateway", "status"], True, True),
    ],
)
def test_foreground_gateway_cmd_leaves_s6_root_slot_down(tmp_path, monkeypatch, cmd, env_opt_out, slot_up):
    home, scandir = tmp_path / "data", tmp_path / "scan"
    for name, standalone in (("coder", False), ("gateway", False), ("solo", True)):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "SOUL.md").write_text("x", encoding="utf-8")
        if standalone:
            (profile / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
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
