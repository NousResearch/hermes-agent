"""Risk-gated approvals (earned autonomy, phase 2): triage inside the policy hook.

Each test installs the example bundle with an ``autonomy:`` block, loads the policy plugin
from the profile the way the runtime does, and scripts the provider by replacing its entry
in the plugin's provider table — so what is exercised is the shipped plugin, its copied
triage rule and its audit records, with no network.

The contract: triage can only ever let *one escalated call* run, and only when the tenant
chose ``enforce``, the action is ``graduated`` on the model that is answering, and every
question passed. Every failure is the escalation that would have happened anyway.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog
from nova.autonomy import triage
from nova.errors import SpecError
from nova.runtime.hermes import HermesRuntime
from nova.runtime.hermes.paths import HermesPaths
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE
from .test_policy_enforcement import load_installed_plugin

SUPPORT = "customer-support"
TOOL = "send_message"  # performs send_external_email, which customer-support must escalate
BODY = "Hi Sam, as agreed we will refund the full $450 to your card within 3 days."
ARGS = {"action": "send", "target": "telegram:123456789", "message": BODY}
MODEL = "jev-1.13.0"


def autonomy_block(*, provider="fake", mode="shadow", data="metadata_only", state="supervised",
                   model_version="", timeout=2.0):
    lines = ["autonomy:", f"  provider: {provider}", f"  mode: {mode}", f"  data: {data}",
             f"  timeout_seconds: {timeout}", "  actions:", "    send_external_email:",
             "      questions: default", f"      state: {state}"]
    if model_version:
        lines.append(f"      model_version: {model_version}")
    return "\n".join(lines) + "\n"


def install(tmp_path, block="", name="b"):
    root = tmp_path / name
    shutil.copytree(EXAMPLE_BUNDLE, root)
    if block:
        policy = root / "policy.yaml"
        policy.write_text(policy.read_text() + "\n" + block)
    home = tmp_path / f"{name}-home"
    home.mkdir()
    bundle = load_bundle(root)
    runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog(tmp_path / f"{name}-audit.jsonl", tenant_id=bundle.tenant_id, actor="test")
    apply_bundle(bundle, runtime, audit=audit)
    return home, audit


_LOADED = [0]


def plugin(home, monkeypatch, agent=SUPPORT):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _LOADED[0] += 1
    return load_installed_plugin(home, agent, f"nova_triage_{_LOADED[0]}")


class Scripted:
    """A provider entry for the plugin's table: records the state, returns scripted answers."""

    def __init__(self, overrides=None, *, model=MODEL, fail=None, delay=0.0):
        self.overrides, self.model, self.fail, self.delay, self.states = overrides or {}, model, fail, delay, []

    def __call__(self, state, questions, timeout):
        self.states.append(state)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        answers = triage.safe_answers(questions)
        answers.update(self.overrides)
        return answers, self.model, 5


def call(p, tool=TOOL, args=None, tool_call_id="call-1"):
    return p.pre_tool_call(tool_name=tool, args=dict(ARGS if args is None else args), tool_call_id=tool_call_id)


def events(audit, kind):
    return [e for e in (json.loads(line) for line in audit.path.read_text().splitlines() if line.strip())
            if e["kind"] == kind]


@pytest.fixture
def graduated(tmp_path):
    return install(tmp_path, autonomy_block(provider="typesafe", mode="enforce", state="graduated",
                                            model_version=MODEL))


# -- off means off ---------------------------------------------------------------------------


def test_without_autonomy_the_compiled_policy_and_every_answer_are_unchanged(tmp_path, monkeypatch):
    plain_home, _ = install(tmp_path, name="plain")
    none_home, _ = install(tmp_path, autonomy_block(provider="none"), name="none")
    plain = json.loads(HermesPaths(home=plain_home).policy_path(SUPPORT).read_text())
    off = json.loads(HermesPaths(home=none_home).policy_path(SUPPORT).read_text())
    assert "autonomy" not in plain and "autonomy" not in off
    strip = lambda d: {k: v for k, v in d.items() if k != "audit_log"}
    assert strip(plain) == strip(off)
    for tool in (TOOL, "crm_refund", "terminal", "kanban_complete", "crm_lookup", "made_up_tool"):
        assert call(plugin(plain_home, monkeypatch), tool) == call(plugin(none_home, monkeypatch), tool)


def test_deny_baseline_ceiling_and_allow_list_are_untouched(tmp_path, graduated, monkeypatch):
    plain_home, _ = install(tmp_path, name="plain")
    home, _ = graduated
    for tool in ("terminal", "kanban_complete", "crm_lookup", "made_up_tool", "crm_refund"):
        with_autonomy, without = plugin(home, monkeypatch), plugin(plain_home, monkeypatch)
        with_autonomy._PROVIDERS["typesafe"] = Scripted()
        assert call(with_autonomy, tool) == call(without, tool), tool
    # The per-run ceiling still refuses the triaged action outright.
    ceiling = plugin(home, monkeypatch)
    ceiling._PROVIDERS["typesafe"] = scripted = Scripted()
    policy = ceiling._load_policy()
    policy["max_tool_calls_per_run"] = 1
    ceiling._CALLS_USED = 1
    assert call(ceiling)["action"] == "block" and scripted.states == [], "a refused call is never triaged"


# -- shadow and supervised never let a call through ------------------------------------------


@pytest.mark.parametrize("state", ["supervised", "graduated"])
def test_shadow_escalates_even_when_every_answer_is_safe(tmp_path, monkeypatch, state):
    """Shadow on a real provider, a graduated action, the right model: still a person decides."""
    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow", state=state,
                                                   model_version=MODEL))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    directive = call(p)
    assert directive["action"] == "approve"
    assert "Triage (shadow) would have let this through" in directive["message"]
    [event] = events(audit, "policy.triage")
    assert event["detail"]["verdict"] == "auto_ok" and event["detail"]["proceed"] is False


def test_the_fake_provider_never_lets_a_call_through(tmp_path, monkeypatch):
    home, _ = install(tmp_path, autonomy_block(provider="fake", mode="shadow"))
    p = plugin(home, monkeypatch)
    policy = p._load_policy()
    policy["autonomy"].update(mode="enforce")  # as if the compiler's refusal were bypassed
    policy["autonomy"]["actions"]["send_external_email"].update(state="graduated", model_version="fake-1")
    assert call(p)["action"] == "approve"


def test_shadow_explains_what_would_have_stopped_it(tmp_path, monkeypatch):
    home, _ = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted({"financial_commitment": {"type": "noul", "p": 0.98}})
    message = call(p)["message"]
    assert "would have asked a person anyway" in message and "financial_commitment" in message


def test_enforce_on_a_supervised_action_escalates(tmp_path, monkeypatch):
    home, _ = install(tmp_path, autonomy_block(provider="typesafe", mode="enforce", state="supervised"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p)["action"] == "approve"


# -- enforce on a graduated action --------------------------------------------------------------


def test_all_safe_answers_let_the_call_run_with_an_intent_and_commit(graduated, monkeypatch):
    home, audit = graduated
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p, tool_call_id="call-7") is None
    p.post_tool_call(tool_name=TOOL, tool_call_id="call-7", status="ok")
    pair = events(audit, "policy.autonomous_action")
    assert [e["phase"] for e in pair] == ["intent", "committed"]
    assert pair[0]["correlation_id"] == pair[1]["correlation_id"]
    assert not [e for e in AuditLog(audit.path, tenant_id="acme").open_intents()
                if e.kind == "policy.autonomous_action"]
    assert not events(audit, "policy.decision"), "a call that ran without a person is not an escalation"


def test_a_tool_that_fails_after_running_unasked_is_recorded_as_failed(graduated, monkeypatch):
    home, audit = graduated
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    call(p, tool_call_id="call-8")
    p.post_tool_call(tool_name=TOOL, tool_call_id="call-8", status="error", error_type="Timeout")
    assert [e["phase"] for e in events(audit, "policy.autonomous_action")] == ["intent", "failed"]


def test_a_crash_between_intent_and_outcome_leaves_a_detectable_open_intent(graduated, monkeypatch):
    home, audit = graduated
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p, tool_call_id="call-9") is None  # ...and the worker dies before post_tool_call
    dangling = AuditLog(audit.path, tenant_id="acme").open_intents()
    assert [e.kind for e in dangling] == ["policy.autonomous_action"]


@pytest.mark.parametrize("overrides", [
    {"financial_commitment": {"type": "noul", "p": 0.10}},                       # at its limit
    {"sensitivity": {"type": "score", "probabilities": [1.0, 0.0, 0.0, 0.0], "confidence": 0.8999}},
    {"recipient": {"type": "choice", "choice": "new_contact", "confidence": 0.99, "probabilities": {
        "existing_customer": 0.0, "new_contact": 1.0, "internal": 0.0, "unknown": 0.0}}},
    {"recipient": {"type": "choice"}},                                            # malformed
], ids=["noul-at-threshold", "confidence-just-below", "choice-not-allowed", "malformed-answer"])
def test_one_failing_answer_escalates_with_its_reason(graduated, monkeypatch, overrides):
    home, _ = graduated
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted(overrides)
    directive = call(p)
    assert directive["action"] == "approve"
    assert next(iter(overrides)) in directive["message"]


def test_a_different_model_version_escalates(graduated, monkeypatch):
    home, _ = graduated
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted(model="jev-2.0.0")
    directive = call(p)
    assert directive["action"] == "approve" and "graduated on 'jev-1.13.0'" in directive["message"]


@pytest.mark.parametrize("provider", [
    Scripted(fail=RuntimeError("boom")),
    Scripted(delay=1.0),
    Scripted(model=""),
], ids=["exception", "timeout", "no-model-version"])
def test_a_provider_that_fails_escalates(tmp_path, monkeypatch, provider):
    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="enforce", state="graduated",
                                                   model_version=MODEL, timeout=0.3))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = provider
    started = time.monotonic()
    assert call(p)["action"] == "approve"
    assert time.monotonic() - started < 0.9
    assert events(audit, "policy.triage")[-1]["detail"]["proceed"] is False


def test_a_missing_key_escalates_without_a_request(graduated, monkeypatch):
    """The real client, unscripted: no key means no call and no autonomy."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    home, audit = graduated
    directive = call(plugin(home, monkeypatch))
    assert directive["action"] == "approve" and "provider_unavailable" in directive["message"]
    assert "no API key" in events(audit, "policy.triage")[-1]["detail"]["failed"][0]["why"]


# -- what leaves, and what is kept --------------------------------------------------------------


def test_metadata_only_sends_no_content(tmp_path, monkeypatch):
    home, _ = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = scripted = Scripted()
    call(p)
    [state] = scripted.states
    sent = json.dumps(state)
    assert "refund" not in sent and "$450" not in sent and "123456789" not in sent
    assert state["platform"] == "telegram" and state["data"] == "metadata_only"
    assert state["arguments"]["message"]["chars"] == len(BODY)


@pytest.mark.parametrize("data", ["metadata_only", "full_args"])
def test_secrets_never_leave_in_any_mode(tmp_path, monkeypatch, data):
    monkeypatch.setenv("CRM_API_TOKEN", "crm-live-7Hq2wPz9Kx")
    home, _ = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow", data=data))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = scripted = Scripted()
    leaky = {"action": "send", "target": "slack:C1", "api_key": "plain-secret-value",
             "message": "use token crm-live-7Hq2wPz9Kx or sk-abcdefghijklmnopqrstu or ghp_abcdefghijklmnopqrstuvwxyz12"}
    call(p, args=leaky)
    sent = json.dumps(scripted.states[0])
    for secret in ("crm-live-7Hq2wPz9Kx", "sk-abcdefghijklmnopqrstu", "ghp_abcdefghijklmnop", "plain-secret-value"):
        assert secret not in sent, secret
    if data == "full_args":
        assert "use token" in sent, "the rest of the content is what full_args is for"


def test_the_triage_record_keeps_fingerprints_not_content(tmp_path, monkeypatch):
    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow", data="full_args"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted({"sensitive_data": {"type": "noul", "p": 0.4}})
    call(p, tool_call_id="call-3")
    [event] = events(audit, "policy.triage")
    detail = event["detail"]
    assert BODY not in audit.path.read_text() and "123456789" not in audit.path.read_text()
    assert detail["state_digest"].startswith("sha256:") and detail["args_digest"].startswith("sha256:")
    assert detail["answers"]["sensitive_data"] == {"type": "noul", "p": 0.4}
    assert {"tool", "action", "mode", "action_state", "verdict", "failed", "provider", "model_version",
            "latency_ms", "data"} <= set(detail)
    assert detail["tool_call_id"] == "call-3"


# -- task work: the board request carries what triage found -------------------------------------


def test_in_task_work_the_held_call_shows_the_triage_note(tmp_path, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db")
    from hermes_cli import kanban_db_connect as kbc

    home, _ = install(tmp_path, autonomy_block(provider="fake", mode="shadow"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    kbc.init_db()
    with kbc.connect_closing() as c:
        task_id = kb.create_task(c, title="reply", assignee=SUPPORT, created_by="nova-supervisor", tenant="acme")
        claimed = kb.claim_task(c, task_id)
    p = plugin(home, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    assert "HELD" in call(p)["message"]
    view = HermesRuntime(home=home, tenant_id="acme").get_task(task_id)
    assert view.detail["approval"]["triage"].startswith("Triage (shadow) would have let this through")


# -- the policy refuses what could not be safe ----------------------------------------------------


@pytest.mark.parametrize("block, words", [
    (autonomy_block(provider="fake", mode="enforce"), "needs a real provider"),
    (autonomy_block(provider="none", mode="enforce"), "needs a real provider"),
    (autonomy_block(provider="typesafe", state="graduated"), "which provider model version"),
    (autonomy_block(provider="typesafe", timeout=30), "at most 10"),
    (autonomy_block(provider="typesafe").replace("send_external_email", "send_postcard"), "not a business action"),
    (autonomy_block(provider="typesafe").replace("data: metadata_only", "data: everything"), "must be one of"),
])
def test_an_unsafe_autonomy_block_is_refused_at_load(tmp_path, block, words):
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    (root / "policy.yaml").write_text((root / "policy.yaml").read_text() + "\n" + block)
    with pytest.raises(SpecError, match=words):
        load_bundle(root)


# -- what ships ------------------------------------------------------------------------------------


def test_the_installed_triage_files_are_the_shipped_ones(tmp_path):
    from nova.runtime.hermes import materialize

    home, _ = install(tmp_path, autonomy_block())
    plugin_dir = HermesPaths(home=home).policy_plugin_dir(SUPPORT)
    assert (plugin_dir / "_triage.py").read_bytes() == materialize.PLUGIN_TRIAGE.read_bytes()
    assert (plugin_dir / "_triage_typesafe.py").read_bytes() == materialize.PLUGIN_TRIAGE_WIRE.read_bytes()


def test_the_installed_plugin_triages_with_only_the_standard_library(tmp_path):
    """Run from the profile with site-packages and the repository off the import path."""
    home, _ = install(tmp_path, autonomy_block(provider="fake", mode="shadow"))
    plugin_dir = HermesPaths(home=home).policy_plugin_dir(SUPPORT)
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('probe_plugin', {str(plugin_dir / '__init__.py')!r},"
        f" submodule_search_locations=[{str(plugin_dir)!r}])\n"
        "module = importlib.util.module_from_spec(spec); sys.modules['probe_plugin'] = module\n"
        "spec.loader.exec_module(module)\n"
        f"out = module.pre_tool_call(tool_name={TOOL!r}, args={json.dumps(ARGS)}, tool_call_id='c')\n"
        "print(out['action'], 'shadow' in out['message'])\n"
        "print(sorted(m for m in sys.modules if m == 'nova' or m.startswith('nova.')))\n"
    )
    result = subprocess.run([sys.executable, "-I", "-S", str(script)], capture_output=True, text=True,
                            cwd=tmp_path, timeout=60, env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["approve True", "[]"]
