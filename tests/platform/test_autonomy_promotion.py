"""Promotion and demotion (earned autonomy, phase 3), through the Control API and the plugin.

A small promotion bar (3 reviewed safe verdicts, window 5) keeps the scenarios short; the
rules are the same ones the defaults (50, window 100) run. Decisions are produced by the
installed plugin and answered through its own approval hook, so promotion is earned the way
it would be in a chat — never by writing records by hand.
"""

from __future__ import annotations

import json

import pytest

from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.runtime.hermes import HermesRuntime
from nova.runtime.hermes.paths import HermesPaths
from nova.spec import load_bundle

from .test_autonomy_triage import SUPPORT, Scripted, autonomy_block, call, install, plugin

A = "send_external_email"
ADMIN = Principal(name="priya-ops", role="admin", via="test")
VIEWER = Principal(name="vic", role="viewer", via="test")
RULES = ("  promotion:\n    min_shadow_decisions: 3\n    min_agreement: 0.98\n"
         "    max_false_safe: 0\n    window: 5\n")


def enforce_block(**kw):
    return autonomy_block(provider="typesafe", mode="enforce", **kw) + RULES


@pytest.fixture
def tenant(tmp_path, monkeypatch):
    home, audit = install(tmp_path, enforce_block())
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundle = load_bundle(tmp_path / "b")
    api = ControlAPI(bundle, HermesRuntime(home=home, tenant_id=bundle.tenant_id), audit=audit)
    return home, audit, api


def decide_in_chat(home, monkeypatch, outcomes, *, model="jev-1.13.0", start=0):
    """Make one escalated call per outcome and answer it through the plugin's hook."""
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted(model=model)
    choice = {"approved": "once", "rejected": "deny"}
    for n, outcome in enumerate(outcomes, start=start):
        directive = call(p, tool_call_id=f"c{n}")
        assert directive["action"] == "approve"
        p.post_approval_response(pattern_key=f"plugin_rule:{directive['rule_key']}",
                                 choice=choice[outcome], tool_call_id=f"c{n}")


def screen(api):
    response = api.handle("/platform/v1/autonomy")
    assert response.status == 200, response.body
    return {row["action"]: row for row in response.body["actions"]}, response.body


def compiled_state(home):
    document = json.loads(HermesPaths(home=home).policy_path(SUPPORT).read_text())
    entry = document["autonomy"]["actions"][A]
    return entry["state"], entry["model_version"]


def kinds(audit, kind):
    return [json.loads(l) for l in audit.path.read_text().splitlines() if f'"{kind}"' in l]


# -- proposing ---------------------------------------------------------------------------------


def test_nothing_is_proposed_below_the_minimum(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved", "approved"])
    rows, _ = screen(api)
    assert rows[A]["state"] == "supervised" and rows[A]["proposal"] is None
    assert "2 of 3" in " ".join(rows[A]["assessment"]["waiting_on"])
    assert not kinds(audit, "autonomy.promotion_proposed")


def test_a_single_false_safe_in_the_window_blocks_a_proposal(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved", "rejected", "approved", "approved"])
    rows, _ = screen(api)
    assert rows[A]["proposal"] is None and "false-safe" in " ".join(rows[A]["assessment"]["waiting_on"])


def test_meeting_the_bar_records_one_proposal(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 3)
    rows, _ = screen(api)
    screen(api)  # looking again must not propose again
    assert rows[A]["state"] == "proposed" and rows[A]["proposal"]["model_version"] == "jev-1.13.0"
    assert len(kinds(audit, "autonomy.promotion_proposed")) == 1
    assert compiled_state(home) == ("supervised", ""), "proposed is not promoted"


def test_decisions_on_two_models_do_not_add_up(tenant, monkeypatch):
    home, _, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 2, model="jev-1.13.0")
    decide_in_chat(home, monkeypatch, ["approved"], model="jev-1.14.0", start=10)
    rows, _ = screen(api)
    assert rows[A]["proposal"] is None and "more than one model" in " ".join(rows[A]["assessment"]["waiting_on"])


def test_the_fake_provider_can_never_be_promoted(tmp_path, monkeypatch):
    home, audit = install(tmp_path, autonomy_block(provider="fake", mode="shadow") + RULES)
    bundle = load_bundle(tmp_path / "b")
    api = ControlAPI(bundle, HermesRuntime(home=home, tenant_id="acme"), audit=audit)
    p = plugin(home, monkeypatch)
    for n in range(4):
        d = call(p, tool_call_id=f"f{n}")
        p.post_approval_response(pattern_key=f"plugin_rule:{d['rule_key']}", choice="once", tool_call_id=f"f{n}")
    rows, body = screen(api)
    assert rows[A]["proposal"] is None and any(n["kind"] == "fake" for n in body["notices"])


# -- confirming ---------------------------------------------------------------------------------


def test_a_viewer_cannot_confirm_a_promotion(tenant, monkeypatch):
    home, _, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 3)
    screen(api)
    response = api.write(f"/platform/v1/autonomy/{A}/promote", VIEWER, {})
    assert response.status == 403
    assert compiled_state(home) == ("supervised", "")


def test_confirming_writes_intent_and_commit_and_recompiles_to_graduated(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 3)
    screen(api)
    response = api.write(f"/platform/v1/autonomy/{A}/promote", ADMIN, {})
    assert response.status == 200 and response.body["state"] == "graduated", response.body
    phases = [e["phase"] for e in kinds(audit, "autonomy.promoted")]
    assert phases == ["intent", "committed"]
    assert compiled_state(home) == ("graduated", "jev-1.13.0")
    # The next safe call runs without a person, with its own write-ahead pair.
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p, tool_call_id="auto-1") is None


def test_a_promotion_that_no_longer_holds_is_refused(tenant, monkeypatch):
    home, _, api = tenant
    response = api.write(f"/platform/v1/autonomy/{A}/promote", ADMIN, {})
    assert response.status == 409 and "does not qualify" in response.body["error"]["message"]


def test_a_crash_between_intent_and_commit_leaves_an_open_intent(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 3)
    screen(api)

    class Crash(BaseException):
        pass

    def crash(*a, **k):
        raise Crash()

    monkeypatch.setattr(api, "_apply_to_runtime", crash)
    with pytest.raises(Crash):
        api.write(f"/platform/v1/autonomy/{A}/promote", ADMIN, {})
    dangling = [e for e in AuditLog(audit.path, tenant_id="acme").open_intents() if e.kind == "autonomy.promoted"]
    assert len(dangling) == 1


# -- demoting ----------------------------------------------------------------------------------


def promoted(tenant, monkeypatch):
    home, audit, api = tenant
    decide_in_chat(home, monkeypatch, ["approved"] * 3)
    screen(api)
    assert api.write(f"/platform/v1/autonomy/{A}/promote", ADMIN, {}).status == 200
    return home, audit, api


def test_rejecting_an_autonomous_call_demotes_before_the_next_call(tenant, monkeypatch):
    home, audit, api = promoted(tenant, monkeypatch)
    # A long-lived process (the gateway) that loaded the graduated policy before the demotion.
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p, tool_call_id="auto-1") is None
    response = api.write(f"/platform/v1/autonomy/{A}/incident", ADMIN,
                         {"reason": "sent to the wrong customer", "autonomous": True, "ref": "auto-1"})
    assert response.status == 200 and response.body["state"] == "supervised"
    assert compiled_state(home) == ("supervised", "")
    assert call(p, tool_call_id="auto-2")["action"] == "approve", "the same process asks a person now"
    assert [e["phase"] for e in kinds(audit, "autonomy.demoted")] == ["intent", "committed"]


def test_a_provider_model_change_demotes_every_graduated_action(tenant, monkeypatch):
    home, _, api = promoted(tenant, monkeypatch)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted(model="jev-2.0.0")
    assert call(p, tool_call_id="m1")["action"] == "approve", "the hook already refuses the new model"
    rows, _ = screen(api)
    assert rows[A]["state"] == "supervised" and "jev-2.0.0" in rows[A]["last_demotion"]
    assert compiled_state(home) == ("supervised", "")


def test_an_admin_can_demote_now_and_a_viewer_cannot(tenant, monkeypatch):
    home, _, api = promoted(tenant, monkeypatch)
    assert api.write(f"/platform/v1/autonomy/{A}/demote", VIEWER, {}).status == 403
    assert api.write(f"/platform/v1/autonomy/{A}/demote", ADMIN, {"reason": "quarterly review"}).status == 200
    assert compiled_state(home) == ("supervised", "")


def test_the_screen_is_readable_by_a_viewer_and_says_what_it_cannot_see(tenant):
    _, _, api = tenant
    assert VIEWER.may("/autonomy") and not VIEWER.may_write("/autonomy/promote")
    _, body = screen(api)
    assert body["configured"] and body["mode"] == "enforce"
    assert any("not observable" in c for c in body["caveats"])
