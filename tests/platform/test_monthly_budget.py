"""Monthly budgets: an agent that has spent its budget stops taking work and calling tools.

No plugin can veto a model call in this runtime, so the budget is enforced where NOVA does
have a veto — a task refused before it starts, every tool call but the ones that close the
task refused. These run the installed policy plugin against real session stores and a real
board.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from nova.apply import apply_bundle
from nova.errors import SpecError
from nova.policy.decide import ALLOW, DENY, decide_budget
from nova.spec.agent import LimitsSpec
from nova.spec.deployment import DeploymentSpec
from nova._fields import Doc

from .test_policy_enforcement import load_installed_plugin

OPS, SUPPORT = "operations", "customer-support"


# -- declaring it --------------------------------------------------------------------


def test_an_agent_budget_is_an_amount():
    assert LimitsSpec.parse(Doc({"monthly_budget_usd": 25})).monthly_budget_usd == 25.0
    assert LimitsSpec.parse(Doc({"monthly_budget_usd": 12.5})).monthly_budget_usd == 12.5
    for bad in (-1, "50", True):
        with pytest.raises(SpecError, match="monthly_budget_usd"):
            LimitsSpec.parse(Doc({"monthly_budget_usd": bad}))


def test_a_tenant_budget_is_declared_in_the_deployment_and_moves_its_digest():
    with_budget = DeploymentSpec.parse({"budget": {"monthly_usd": 200}})
    assert with_budget.monthly_budget_usd == 200.0
    assert with_budget.to_dict()["budget"] == {"monthly_usd": 200.0}
    assert "budget" not in DeploymentSpec.parse({}).to_dict()
    with pytest.raises(SpecError):
        DeploymentSpec.parse({"budget": {"monthly_usd": 200, "daily_usd": 5}})


# -- the decision, pure --------------------------------------------------------------


@pytest.mark.parametrize("agent_limit, tenant_limit, agent_spend, tenant_spend, effect", [
    (0, 0, 999, 999, ALLOW),          # no budget, no stop
    (10, 0, 9.99, 0, ALLOW),
    (10, 0, 10, 0, DENY),             # reaching it is spending it
    (0, 50, 1, 50, DENY),             # the tenant's is spent by the others
    (10, 50, 5, 20, ALLOW),
])
def test_the_budget_decision(agent_limit, tenant_limit, agent_spend, tenant_spend, effect):
    policy = {"agent_id": OPS, "monthly_budget_usd": agent_limit, "tenant_monthly_budget_usd": tenant_limit}
    assert decide_budget(policy, agent_spend=agent_spend, tenant_spend=tenant_spend).effect == effect


def test_a_spent_budget_says_whose_and_how_much():
    reason = decide_budget({"agent_id": OPS, "monthly_budget_usd": 10},
                           agent_spend=10.4, tenant_spend=0).reason
    assert "operations's monthly budget of $10.00" in reason and "$10.40" in reason


# -- in the worker -------------------------------------------------------------------


def _usage(profile: Path, *rows: tuple[float, float]) -> None:
    """Write (cost_usd, seconds_ago) rows into a profile's session store, schema as the runtime's."""
    db = sqlite3.connect(profile / "state.db")
    db.execute(
        "CREATE TABLE IF NOT EXISTS session_model_usage (session_id TEXT, model TEXT, "
        "billing_provider TEXT DEFAULT '', estimated_cost_usd REAL DEFAULT 0, "
        "actual_cost_usd REAL DEFAULT 0, first_seen REAL, last_seen REAL)"
    )
    now = time.time()
    for index, (cost, ago) in enumerate(rows):
        db.execute("INSERT INTO session_model_usage VALUES (?, 'm', '', ?, 0, ?, ?)",
                   (f"s{index}-{now}", cost, now - ago, now - ago))
    db.commit()
    db.close()


@pytest.fixture
def applied(bundle, runtime, audit, home, monkeypatch, tmp_path):
    """The example bundle with operations on a $10 budget and the tenant on $30."""
    import shutil

    from nova.spec import load_bundle

    from .conftest import EXAMPLE_BUNDLE

    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    ops = root / "agents" / "operations.yaml"
    text = ops.read_text()
    ops.write_text(text.replace("limits:\n", "limits:\n  monthly_budget_usd: 10\n", 1)
                   if "limits:\n" in text else text + "\nlimits:\n  monthly_budget_usd: 10\n")
    deployment = root / "deployment.yaml"
    deployment.write_text(deployment.read_text() + "\nbudget:\n  monthly_usd: 30\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    apply_bundle(load_bundle(root), runtime, audit=audit)
    return home


def plugin_for(home, agent, monkeypatch, name="p"):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return load_installed_plugin(home, agent, f"nova_budget_{agent.replace('-', '_')}_{name}")


def refused(plugin, tool="web_search"):
    result = plugin.pre_tool_call(tool_name=tool, args={})
    return bool(result and result.get("action") == "block" and "budget" in result["message"])


def test_the_budgets_reach_the_compiled_policy(applied):
    import json

    document = json.loads((applied / "profiles" / OPS / "nova-policy.json").read_text())
    assert document["monthly_budget_usd"] == 10
    assert document["tenant_monthly_budget_usd"] == 30


def test_under_budget_nothing_is_refused(applied, monkeypatch):
    _usage(applied / "profiles" / OPS, (4.0, 60))
    assert not refused(plugin_for(applied, OPS, monkeypatch))


def test_a_spent_budget_stops_tools_but_not_closing_the_task(applied, monkeypatch):
    _usage(applied / "profiles" / OPS, (6.0, 3600), (4.5, 60))
    plugin = plugin_for(applied, OPS, monkeypatch)
    assert refused(plugin)
    assert plugin.pre_tool_call(tool_name="kanban_block", args={}) is None, "it must be able to stop"


def test_last_months_spend_does_not_count(applied, monkeypatch):
    _usage(applied / "profiles" / OPS, (50.0, 40 * 86400), (2.0, 60))
    assert not refused(plugin_for(applied, OPS, monkeypatch))


def test_the_tenant_budget_counts_every_agent(applied, monkeypatch):
    """Support has no budget of its own; the tenant's $30 is spent by the two of them."""
    _usage(applied / "profiles" / OPS, (9.0, 60))
    _usage(applied / "profiles" / SUPPORT, (22.0, 60))
    assert refused(plugin_for(applied, SUPPORT, monkeypatch))


def test_unreadable_spend_with_a_budget_set_refuses(applied, monkeypatch):
    (applied / "profiles" / OPS / "state.db").write_bytes(b"not a database")
    assert refused(plugin_for(applied, OPS, monkeypatch))


def test_a_task_over_budget_is_blocked_on_the_board_with_a_readable_reason(applied, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db")
    from hermes_cli import kanban_db_connect as kbc

    from nova.runtime.model_errors import is_nova_failure_block, summarize_task_error

    monkeypatch.setenv("HERMES_KANBAN_DB", str(applied / "kanban.db"))
    kbc.init_db()
    _usage(applied / "profiles" / OPS, (11.0, 60))
    with kbc.connect_closing() as c:
        task_id = kb.create_task(c, title="t", assignee=OPS, created_by="nova-supervisor", tenant="acme")
        claimed = kb.claim_task(c, task_id)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    load_installed_plugin(applied, OPS, "nova_budget_task").on_session_start()
    with kbc.connect_closing() as c:
        assert kb.get_task(c, task_id).status == "blocked"
        reason = c.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'",
                           (task_id,)).fetchone()[0]
    import json

    reason = json.loads(reason)["reason"]
    assert is_nova_failure_block(reason)
    assert summarize_task_error(reason)["headline"] == "Monthly budget reached"


def test_the_usage_screen_shows_this_month_as_the_stop_counts_it(applied, runtime, monkeypatch, tmp_path):
    from nova.control import ControlAPI
    from nova.spec import load_bundle

    _usage(applied / "profiles" / OPS, (7.5, 60), (100.0, 40 * 86400))
    _usage(applied / "profiles" / SUPPORT, (2.0, 60))
    body = ControlAPI(load_bundle(tmp_path / "bundle"), runtime).handle("/platform/v1/budget").body
    month = body["this_month"]
    rows = {row["agent_id"]: row for row in month["agents"]}
    assert rows[OPS]["spent_usd"] == pytest.approx(7.5) and rows[OPS]["budget_usd"] == 10
    assert month["tenant"] == {"spent_usd": pytest.approx(9.5), "budget_usd": 30.0}
    assert any(c["key"] == "monthly_budget_usd" and c["enforcement"] == "hard_boundary" for c in body["controls"])
