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
        (["gateway", "-p", "coder", "run", "--no-supervise"], False, False),  # -p after `gateway`
        (["gateway", "--accept-hooks", "run", "--no-supervise"], False, False),  # gateway-level flag first
        (["python", "-m", "hermes_cli.main", "gateway", "run", "--no-supervise"], False, False),  # wrapper exec path
        (["/opt/hermes/.venv/bin/python3", "-u", "/opt/hermes/hermes_cli/main.py", "gateway", "run", "--no-supervise"],
         False, False),  # script launcher
        (["gateway", "run", "--no-supervis"], False, False),  # argparse accepts the abbreviation
        (["gateway", "--no-supervise"], False, True),  # argparse rejects it: no foreground gateway
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
            (profile / "gateway_state.json").write_text(json.dumps({"desired_state": "running"}), encoding="utf-8")
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
    # solo's own slot boots from its saved intent, unless the CMD already runs solo in the foreground.
    solo_up = cmd[:2] != ["-p", "solo"]
    assert [a.action for a in actions if a.profile == "solo"] == ["started" if solo_up else "registered"]


def test_boot_notice_says_the_foreground_gateway_serves_folded_profiles(tmp_path, monkeypatch, caplog):
    home = tmp_path / "data"
    coder = home / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "SOUL.md").write_text("x", encoding="utf-8")
    (coder / "gateway_state.json").write_text(json.dumps({"desired_state": "running"}), encoding="utf-8")
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    caplog.set_level("WARNING")

    actions = reconcile_profile_gateways(hermes_home=home, scandir=tmp_path / "scan",
                                         container_argv=_WRAPPER + ["gateway", "run", "--no-supervise"])

    assert actions[0].action == "registered"
    assert "foreground gateway (the container CMD) serves them" in caplog.text
    assert "was started instead" not in caplog.text


def test_invalid_profile_in_foreground_cmd_does_not_exit_the_boot_reconciler(tmp_path, monkeypatch, capsys):
    """cont-init runs container_boot.py as argv[0]: an invalid `-p` in the CMD (which exits 2 by
    itself) must not abort the reconciler before the other profiles' slots are registered."""
    home = tmp_path / "data"
    coder = home / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "SOUL.md").write_text("x", encoding="utf-8")
    (coder / "gateway_state.json").write_text(json.dumps({"desired_state": "running"}), encoding="utf-8")
    (home / "gateway_state.json").write_text(
        json.dumps({"gateway_state": "stopped", "desired_state": "running"}), encoding="utf-8")
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    monkeypatch.setattr("sys.argv", ["/opt/hermes/hermes_cli/container_boot.py"])

    actions = reconcile_profile_gateways(
        hermes_home=home, scandir=tmp_path / "scan",
        container_argv=_WRAPPER + ["-p", "Work Bot", "gateway", "run", "--no-supervise"])

    assert {a.profile: a.action for a in actions} == {"default": "started", "coder": "registered"}
    assert capsys.readouterr().err == ""
