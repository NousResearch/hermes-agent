from types import SimpleNamespace

from gateway.run import GatewayRunner


def _source(platform: str = "telegram"):
    return SimpleNamespace(platform=SimpleNamespace(value=platform))


def test_company_os_routes_investment_research_language():
    runner = GatewayRunner.__new__(GatewayRunner)

    assert runner._should_company_os_route_message("研判一下拓维信息走势", _source())
    assert runner._should_company_os_route_message("不要自动交易，只做投资分析", _source())


def test_company_os_skips_casual_and_non_telegram_messages():
    runner = GatewayRunner.__new__(GatewayRunner)

    assert not runner._should_company_os_route_message("你好", _source())
    assert not runner._should_company_os_route_message("/status", _source())
    assert not runner._should_company_os_route_message("研判一下拓维信息走势", _source("discord"))


def test_company_os_parses_control_envelope_ids():
    parsed = GatewayRunner._parse_company_os_route_output(
        """
control_version: 1
run_id: run-20260624-test
mcp_profile_id: research
task:
  task_id: task-20260624-test
approval:
  approval_id: approval-20260624-test
route:
  domain_id: market.investing
  workflow_id: investment_research_loop
  review_gates:
  - investment_action
"""
    )

    assert parsed["domain_id"] == "market.investing"
    assert parsed["workflow_id"] == "investment_research_loop"
    assert parsed["run_id"] == "run-20260624-test"
    assert parsed["mcp_profile_id"] == "research"
    assert parsed["task_id"] == "task-20260624-test"
    assert parsed["approval_id"] == "approval-20260624-test"
    assert parsed["review_gates"] == ["investment_action"]
    assert parsed["hard_confirmation_required"] is False


def test_company_os_parses_hard_confirmation_required():
    parsed = GatewayRunner._parse_company_os_route_output(
        """
control_version: 2
route:
  domain_id: money.services
  workflow_id: service_delivery_loop
  review_gates:
  - external_send
  gate_plan:
    hard_confirmation_required: true
"""
    )

    assert parsed["hard_confirmation_required"] is True
