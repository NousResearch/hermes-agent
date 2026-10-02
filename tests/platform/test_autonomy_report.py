"""The autonomy report (earned autonomy, phase 4): measured numbers, labelled estimates.

The arithmetic is checked on records with known answers. The command is then run end to
end against an installed tenant whose decisions came through the plugin, and the files it
writes are read back.
"""

from __future__ import annotations

import json

import pytest

from nova.autonomy.report import build_report, to_markdown

from .test_autonomy_ledger import A, answered, ev, escalated, triaged
from .test_autonomy_triage import Scripted, autonomy_block, call, install, plugin


def home_log(home):
    from nova.audit import AuditLog

    path = AuditLog.for_home(home, tenant_id="acme").path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def triage_with(n, latency, *, usage=(500, 150), tenant="acme", failed=None, model="jev-1.13.0"):
    event = triaged(n, tenant=tenant, model=model)
    event["detail"].update(latency_ms=latency, usage={"input_tokens": usage[0], "output_tokens": usage[1]},
                           failed=failed or [])
    return event


def records():
    events = []
    for n in range(4):  # four escalations, all answered, three safe and approved, one rejected
        events += [escalated(n), triage_with(n, 800 + n * 100), answered(n, "approved" if n < 3 else "rejected")]
    for n in range(6):  # six calls that ran without a person
        events += [ev("policy.autonomous_action", {"action": A}, phase="intent", corr=f"a{n}"),
                   ev("policy.autonomous_action", {"status": "ok"}, phase="committed", corr=f"a{n}")]
    events.append(triage_with(9, 2100, failed=[{"id": "", "why": "provider_unavailable (HTTP 529)"}], model=""))
    events.append(ev("autonomy.demoted", {"action": A, "reason": "model changed"}, phase="committed"))
    return events


def test_the_numbers_add_up():
    report = build_report(records(), [], tenant_id="acme", minutes_per_approval=4)
    totals = report["totals"]
    assert (totals["approval_gated_calls"], totals["handled_autonomously"], totals["escalated_to_a_person"]) == (10, 6, 4)
    assert totals["handled_autonomously_share"] == pytest.approx(0.6)
    assert report["estimates"]["human_hours_saved"] == pytest.approx(6 * 4 / 60)
    [action] = report["actions"]
    assert action["false_safe"] == 1 and action["shadow_agreement"] == pytest.approx(3 / 4)
    tri = report["triage"]
    assert tri["calls"] == 5 and tri["provider_unavailable"] == 1
    assert tri["latency_ms_median"] == 950, "an unanswered call's wait is not provider latency"
    assert tri["tokens"] == {"input": 2500, "output": 750}
    assert [d["reason"] for d in report["demotions"]] == ["model changed"]


def test_cost_is_not_invented_without_a_price():
    unpriced = build_report(records(), [], tenant_id="acme")
    assert unpriced["triage"]["cost_usd"] is None and "not computed" in unpriced["triage"]["cost_note"]
    priced = build_report(records(), [], tenant_id="acme", price_input_per_mtok=1.0, price_output_per_mtok=4.0)
    assert priced["triage"]["cost_usd"] == pytest.approx((2500 * 1.0 + 750 * 4.0) / 1_000_000)


def test_estimates_are_labelled_and_the_unobservable_is_said():
    markdown = to_markdown(build_report(records(), [], tenant_id="acme", company="Acme Ltd"))
    assert "# Autonomy report — Acme Ltd" in markdown
    hours = next(line for line in markdown.splitlines() if line.startswith("| Human hours saved"))
    assert "**estimate**" in hours and "minutes of a person's time" in hours
    assert "not observable" in markdown and "Provider cost | not computed" in markdown


def test_another_tenants_records_are_not_reported():
    theirs = [ev("policy.autonomous_action", {"action": A}, phase="intent", corr="x", tenant="globex"),
              ev("policy.autonomous_action", {"status": "ok"}, phase="committed", corr="x", tenant="globex"),
              triage_with(1, 700, tenant="globex")]
    report = build_report(theirs, [], tenant_id="acme")
    assert report["totals"]["approval_gated_calls"] == 0 and report["triage"]["calls"] == 0
    assert "not measured yet" in to_markdown(report)


def test_the_command_writes_both_files_from_real_records(tmp_path, monkeypatch, capsys):
    from nova.cli import main

    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = lambda s, q, t: (Scripted()(s, q, t)[:3] + ({"input_tokens": 520, "output_tokens": 160},))
    for n in range(2):
        d = call(p, tool_call_id=f"r{n}")
        p.post_approval_response(pattern_key=f"plugin_rule:{d['rule_key']}", choice="once", tool_call_id=f"r{n}")
    # The CLI reads the home's conventional audit log; put the installed tenant's there.
    home_log(home).write_text(audit.path.read_text())
    out = tmp_path / "out"
    assert main(["--home", str(home), "autonomy", "report", str(tmp_path / "b"), "--out", str(out),
                 "--minutes-per-approval", "5"]) == 0
    report = json.loads((out / "autonomy-report-acme.json").read_text())
    assert report["triage"]["calls"] == 2 and report["triage"]["tokens"] == {"input": 1040, "output": 320}
    assert report["actions"][0]["shadow_agreement"] == 1.0
    assert (out / "autonomy-report-acme.md").read_text().startswith("# Autonomy report")
    assert "Written:" in capsys.readouterr().out


def test_review_runs_the_same_review_as_the_screen(tmp_path, monkeypatch, capsys):
    from nova.cli import main

    rules = "  promotion:\n    min_shadow_decisions: 2\n    window: 3\n"
    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow") + rules)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    for n in range(2):
        d = call(p, tool_call_id=f"v{n}")
        p.post_approval_response(pattern_key=f"plugin_rule:{d['rule_key']}", choice="once", tool_call_id=f"v{n}")
    home_log(home).write_text(audit.path.read_text())
    assert main(["--home", str(home), "autonomy", "review", str(tmp_path / "b")]) == 0
    assert "promotion proposed on jev-1.13.0" in capsys.readouterr().out
    proposals = [l for l in home_log(home).read_text().splitlines() if "autonomy.promotion_proposed" in l]
    assert len(proposals) == 1
