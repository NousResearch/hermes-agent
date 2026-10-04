from argparse import Namespace

from hermes_cli import fleet_estop as fe


def _units():
    return [
        fe.FleetGatewayUnit("hermes-gateway.service", "default", "active", "running", 100),
        fe.FleetGatewayUnit("hermes-gateway-chloe.service", "chloe", "active", "running", 200),
        fe.FleetGatewayUnit("hermes-gateway-atlantis.service", "atlantis", "inactive", "dead", 0),
    ]


def test_profile_unit_roundtrip_and_default_mapping():
    assert fe.profile_from_unit("hermes-gateway.service") == "default"
    assert fe.profile_from_unit("hermes-gateway-chloe.service") == "chloe"
    assert fe.unit_from_profile("default") == "hermes-gateway.service"
    assert fe.unit_from_profile("chloe") == "hermes-gateway-chloe.service"
    assert fe.unit_from_profile("tony") == "hermes-gateway-tony.service"
    assert fe.unit_from_profile("ops watch") == "hermes-gateway-opswatch.service"
    assert fe.unit_from_profile("Maverick Revenue Engine") == "hermes-gateway-MaverickRevenueEngine.service"
    assert fe.profile_from_unit("hermes-gateway-tony.service") == "tony"


def test_parse_systemctl_list_filters_gateway_units():
    text = """
      hermes-gateway.service loaded active running Hermes Gateway
      hermes-gateway-chloe.service loaded active running Hermes Gateway Chloe
      hermes-dashboard.service loaded active running Dashboard
    """
    assert fe.parse_systemctl_list(text) == [
        "hermes-gateway.service",
        "hermes-gateway-chloe.service",
    ]


def test_parse_show_extracts_state_and_pid():
    unit = fe.parse_show("hermes-gateway-chloe.service", "ActiveState=active\nSubState=running\nMainPID=123\n")
    assert unit.profile == "chloe"
    assert unit.active_state == "active"
    assert unit.sub_state == "running"
    assert unit.main_pid == 123
    assert unit.protected is False


def test_stop_plan_defaults_to_non_default_profiles_only():
    plan = fe.build_plan(action="stop", units=_units())
    assert [p["profile"] for p in plan] == ["chloe", "atlantis"]
    assert all(not p["blocked"] for p in plan)


def test_default_gateway_is_blocked_without_explicit_override():
    plan = fe.build_plan(action="stop", profiles=["default"], units=_units())
    assert len(plan) == 1
    assert plan[0]["unit"] == "hermes-gateway.service"
    assert plan[0]["protected"] is True
    assert plan[0]["blocked"] is True


def test_default_gateway_can_only_be_planned_with_override():
    plan = fe.build_plan(action="stop", profiles=["default"], include_default=True, units=_units())
    assert plan[0]["protected"] is True
    assert plan[0]["blocked"] is False


def test_execute_plan_dry_run_writes_audit_without_systemctl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    called = []
    monkeypatch.setattr(fe, "_run_systemctl", lambda args: called.append(args))
    plan = fe.build_plan(action="stop", profiles=["chloe"], units=_units())
    result = fe.execute_plan(plan, dry_run=True, reason="test")
    assert result["results"][0]["status"] == "dry-run"
    assert called == []
    assert (tmp_path / "workspace" / "fleet-estop" / "audit.jsonl").exists()


def test_execute_plan_blocks_protected_unit_even_when_not_dry_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    called = []
    monkeypatch.setattr(fe, "_run_systemctl", lambda args: called.append(args))
    plan = fe.build_plan(action="stop", profiles=["default"], units=_units())
    result = fe.execute_plan(plan, dry_run=False, reason="test")
    assert result["results"][0]["status"] == "blocked"
    assert called == []


def test_configured_profile_unit_override(monkeypatch):
    monkeypatch.setenv("HERMES_FLEET_ESTOP_UNIT_OVERRIDES", '{"tony":"hermes-tony-gateway.service"}')
    assert fe.unit_from_profile("tony") == "hermes-tony-gateway.service"
    assert fe.profile_from_unit("hermes-tony-gateway.service") == "tony"


def test_invalid_profile_unit_override_is_ignored(monkeypatch):
    monkeypatch.setenv("HERMES_FLEET_ESTOP_UNIT_OVERRIDES", '{"tony":"not-a-gateway.timer"}')
    assert fe.unit_from_profile("tony") == "hermes-gateway-tony.service"


def test_explicit_profile_is_added_when_systemctl_list_omits_failed_unit(monkeypatch):
    omitted_inventory = [
        fe.FleetGatewayUnit("hermes-gateway.service", "default", "active", "running", 100)
    ]

    def fake_run(args):
        class Result:
            stdout = "ActiveState=failed\nSubState=failed\nMainPID=0\n"
            stderr = ""
            returncode = 0

        assert args[0] == "show"
        assert args[1] == "hermes-gateway-research-hub.service"
        return Result()

    monkeypatch.setattr(fe, "list_gateway_units", lambda: omitted_inventory)
    monkeypatch.setattr(fe, "_run_systemctl", fake_run)
    plan = fe.build_plan(action="resume", profiles=["research-hub"])
    assert len(plan) == 1
    assert plan[0]["unit"] == "hermes-gateway-research-hub.service"
    assert plan[0]["active_state"] == "failed"
    assert plan[0]["blocked"] is False


def test_format_plan_marks_blocked_default():
    plan = fe.build_plan(action="stop", profiles=["default"], units=_units())
    text = fe.format_plan(plan)
    assert "default" in text
    assert "BLOCKED" in text



def test_fleet_estop_registered_in_top_level_cli_parser():
    from hermes_cli.main import _build_cli_parser
    from hermes_cli.fleet_estop import cmd_fleet_estop

    parser, subparsers = _build_cli_parser()
    assert "fleet-estop" in subparsers.choices

    args = parser.parse_args(["fleet-estop", "plan-stop", "--target-profile", "tony"])
    assert args.func is cmd_fleet_estop
    assert args.fleet_action == "plan-stop"
    assert args.target_profile == ["tony"]
