"""A run's terminal ``usage`` carries the cost Hermes already priced.

BLK-2 (HERMES-PAPERCLIP-COST-PROPAGATION-001): ``/v1/runs/{id}`` published tokens only, so the
official Paperclip hermes_gateway adapter (which reads ``usage.cost_usd`` in USD) booked every
real run as ``costCents=0`` / ``costStatus=unpriced``. These tests pin the wire contract:

* a priced run reports ``cost_usd`` from ``session_estimated_cost_usd`` (no second price table),
* an unpriced route reports no amount at all (never a fake zero),
* an included subscription reports a real $0,
* a retained agent reports this run's delta, not the session's lifetime spend,
* the existing token keys are unchanged.
"""
from gateway.platforms.api_server_runs import _run_usage
from gateway.session_usage import cost_fields


class _Agent:
    """The session-cost/token attributes ``run_conversation`` leaves on a real agent."""

    def __init__(self, *, prompt=0, completion=0, total=0, read=0, write=0,
                 cost=0.0, status="unknown", source="none"):
        self.session_prompt_tokens = prompt
        self.session_completion_tokens = completion
        self.session_total_tokens = total
        self.session_cache_read_tokens = read
        self.session_cache_write_tokens = write
        self.session_estimated_cost_usd = cost
        self.session_cost_status = status
        self.session_cost_source = source


def test_priced_run_reports_the_agents_own_cost():
    agent = _Agent(prompt=1_000, completion=500, total=1_500, cost=0.0075,
                   status="estimated", source="official_docs_snapshot")
    usage = _run_usage(agent)
    assert usage["input_tokens"] == 1_000 and usage["output_tokens"] == 500
    assert usage["total_tokens"] == 1_500
    assert usage["cost_usd"] == 0.0075
    assert usage["cost_status"] == "estimated"


def test_unpriced_route_omits_the_amount_instead_of_reporting_zero():
    usage = _run_usage(_Agent(prompt=12, completion=3, total=15))
    assert "cost_usd" not in usage, "unknown pricing must not read as a priced zero"
    assert usage["cost_status"] == "unknown"
    # Tokens still publish, so a client can price externally or bill on usage.
    assert usage["total_tokens"] == 15


def test_included_subscription_reports_a_real_zero():
    usage = _run_usage(_Agent(prompt=10, completion=2, total=12, status="included"))
    assert usage["cost_usd"] == 0.0 and usage["cost_status"] == "included"


def test_zero_token_run_keeps_its_zero_cost_and_status():
    usage = _run_usage(_Agent(status="estimated"))
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                     "cache_read_tokens": 0, "cache_write_tokens": 0,
                     "cost_usd": 0.0, "cost_status": "estimated"}


def test_retained_agent_reports_only_this_runs_delta():
    agent = _Agent(cost=0.0300, status="estimated", source="official_docs_snapshot")
    fields = cost_fields(agent, before=0.0125)
    assert round(fields["cost_usd"], 10) == 0.0175
    assert fields["cost_status"] == "estimated"


def test_a_negative_delta_cannot_report_negative_money():
    fields = cost_fields(_Agent(cost=0.01, status="estimated"), before=0.02)
    assert fields["cost_usd"] == 0.0


def test_cost_fields_survive_the_run_status_json():
    """The usage dict is persisted in the run status JSON and replayed on poll."""
    import json
    usage = _run_usage(_Agent(total=9, cost=0.0005, status="estimated"))
    assert json.loads(json.dumps(usage)) == usage
