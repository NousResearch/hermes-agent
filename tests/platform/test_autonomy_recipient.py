"""Who a message goes to: looked up, not guessed.

The recipient question is answered by NOVA from the tenant's contact list and the gateway's
channel directory — records the agent cannot write — and never sent to the provider. These
check the lookup itself, the contact list and its editor, and the whole path through the
installed plugin, where a routine message to a listed customer finally passes every check.
"""

from __future__ import annotations

import json

import pytest

from nova.autonomy import triage
from nova.autonomy.contacts import Contacts, load_contacts
from nova.autonomy.providers.typesafe import build_request
from nova.autonomy.questions import QuestionSet, question_set_for
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.runtime.hermes import HermesRuntime
from nova.runtime.hermes.paths import HermesPaths
from nova.spec import load_bundle

from .test_autonomy_triage import (
    ARGS, MODEL, SUPPORT, Scripted, autonomy_block, call, install, plugin,
)

CONTACTS = {"customers": ["telegram:42", "matrix:!room:example.org"], "internal": ["slack:C0TEAM"]}
DIRECTORY = {"platforms": {
    "telegram": [{"id": "42", "name": "Lena Park", "type": "dm"},
                 {"id": "77", "name": "Sam", "type": "dm"},
                 {"id": "-100500", "name": "Acme fans", "type": "group"},
                 {"id": "88", "name": "Jo", "type": "dm"}, {"id": "89", "name": "Jo", "type": "dm"}],
    "slack": [{"id": "C0TEAM", "name": "#ops", "type": "channel"}],
}}


def who(target, contacts=CONTACTS, directories=(DIRECTORY,)):
    return triage.recipient_answer({"target": target}, contacts, directories)["choice"]


# -- the lookup --------------------------------------------------------------------------


@pytest.mark.parametrize("target, expected", [
    ("telegram:42", "existing_customer"),       # on the contact list
    ("TELEGRAM:42", "existing_customer"),       # platform is case-insensitive
    ("telegram:Lena Park", "existing_customer"),  # a friendly name, resolved through the directory
    ("slack:C0TEAM", "internal"),               # the list wins over the directory's "channel"
    ("slack:#ops", "internal"),
    ("telegram:77", "known_contact"),           # wrote to us, not listed
    ("telegram:-100500", "broadcast"),          # a group
    ("telegram:Acme fans", "broadcast"),
    ("telegram:12345", "unknown"),              # never seen
    ("telegram:Jo", "unknown"),                 # ambiguous name: not resolved at all
    ("matrix:!room:example.org", "existing_customer"),  # ids containing ':' still match
    ("telegram:42:7", "existing_customer"),     # a thread in a listed chat
])
def test_the_recipient_is_looked_up(target, expected):
    assert who(target) == expected


@pytest.mark.parametrize("args", [{}, {"target": ""}, {"target": "no-colon"}, {"target": 42}, None])
def test_no_recipient_in_the_call_is_unknown(args):
    assert triage.recipient_answer(args, CONTACTS, (DIRECTORY,))["choice"] == "unknown"


def test_a_missing_or_broken_directory_establishes_nothing():
    assert who("telegram:77", directories=()) == "unknown"
    assert who("telegram:77", directories=({"platforms": "nonsense"}, ["junk"])) == "unknown"
    assert who("telegram:42", contacts=None, directories=()) == "unknown"


def test_the_lookup_never_raises():
    class Hostile(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    assert triage.recipient_answer({"target": "telegram:42"}, Hostile(), (DIRECTORY,))["choice"] == "unknown"


def test_the_rule_passes_only_the_allowed_recipients():
    questions = question_set_for("send_external_email", "default").to_list()
    safe = triage.safe_answers(questions)
    for choice, verdict in (("existing_customer", "auto_ok"), ("internal", "auto_ok"),
                            ("known_contact", "escalate"), ("broadcast", "escalate"), ("unknown", "escalate")):
        result = triage.combine(questions, {**safe, "recipient": {"type": "recipient", "choice": choice}})
        assert result["verdict"] == verdict, choice
        if verdict == "escalate":
            assert result["failed"][0]["why"].startswith("the recipient is "), result


def test_a_tenant_can_allow_known_contacts_in_its_own_set():
    allowed = QuestionSet.parse([{"id": "recipient", "type": "recipient",
                                  "allowed": ["existing_customer", "known_contact"]}]).to_list()
    assert triage.combine(allowed, {"recipient": {"type": "recipient", "choice": "known_contact"}})["verdict"] == "auto_ok"
    with pytest.raises(SpecError, match="subset"):
        QuestionSet.parse([{"id": "r", "type": "recipient", "allowed": ["vip"]}])


def test_the_provider_is_never_asked_about_the_recipient():
    body = build_request({}, question_set_for("send_external_email", "default").to_list())
    assert "recipient" not in body["questions"] and len(body["questions"]) == 6


# -- the contact list ----------------------------------------------------------------------


def test_contacts_are_normalized_and_checked():
    contacts = Contacts.parse({"customers": ["Telegram:42", "telegram:42", "slack:U1"], "internal": []})
    assert contacts.customers == ("telegram:42", "slack:U1")
    for bad, words in (({"customers": ["42"]}, "send target"), ({"customers": ["telegram:4 2"]}, "send target"),
                       ({"vips": []}, "unknown keys"),
                       ({"customers": ["slack:U1"], "internal": ["slack:U1"]}, "both internal and customer")):
        with pytest.raises(SpecError, match=words):
            Contacts.parse(bad)


def test_a_malformed_contact_list_is_refused_when_the_bundle_loads(tmp_path):
    with pytest.raises(SpecError, match="send target"):
        install(tmp_path, autonomy_block(), contacts="customers: [not a target]\n")


def test_the_list_is_compiled_only_for_agents_that_ask_about_the_recipient(tmp_path):
    home, _ = install(tmp_path, autonomy_block(provider="typesafe"))
    support = json.loads(HermesPaths(home=home).policy_path(SUPPORT).read_text())
    ops = json.loads(HermesPaths(home=home).policy_path("operations").read_text())
    assert support["autonomy"]["contacts"]["customers"] == ["telegram:123456789"]
    assert "autonomy" not in ops, "an agent that never triages does not carry the list"


# -- through the installed plugin -----------------------------------------------------------


def graduated(tmp_path, **kw):
    return install(tmp_path, autonomy_block(provider="typesafe", mode="enforce", state="graduated",
                                            model_version=MODEL), **kw)


def test_a_routine_message_to_a_listed_customer_passes_every_check(tmp_path, monkeypatch):
    home, audit = graduated(tmp_path)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = scripted = Scripted()
    assert call(p, tool_call_id="ok-1") is None
    [event] = [json.loads(l) for l in audit.path.read_text().splitlines() if '"policy.triage"' in l]
    assert event["detail"]["answers"]["recipient"] == {"type": "recipient", "choice": "existing_customer",
                                                       "source": "contact list"}
    assert "123456789" not in audit.path.read_text(), "the audit records what was found, not the chat id"
    assert "recipient" not in json.dumps(scripted.states[0].get("arguments", {})), "nothing about it was sent"


@pytest.mark.parametrize("target, why", [
    ("telegram:555", "never seen"),
    ("telegram:777", "has written to us before"),
    ("telegram:-100500", "a group or channel"),
])
def test_anyone_else_goes_to_a_person_with_the_reason(tmp_path, monkeypatch, target, why):
    home, _ = graduated(tmp_path)
    (home / "channel_directory.json").write_text(json.dumps({"platforms": {"telegram": [
        {"id": "777", "name": "Sam", "type": "dm"}, {"id": "-100500", "name": "fans", "type": "group"}]}}))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    directive = call(p, args={**ARGS, "target": target})
    assert directive["action"] == "approve" and why in directive["message"]


def test_the_agent_cannot_vouch_for_its_own_recipient(tmp_path, monkeypatch):
    """A recipient type in the call's own arguments is ignored."""
    home, _ = graduated(tmp_path)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    directive = call(p, args={**ARGS, "target": "telegram:555", "recipient_type": "existing_customer"})
    assert directive["action"] == "approve" and "never seen" in directive["message"]


# -- the editor ------------------------------------------------------------------------------


ADMIN = Principal(name="priya-ops", role="admin", via="test")
VIEWER = Principal(name="vic", role="viewer", via="test")


def test_the_editor_saves_applies_and_changes_the_outcome(tmp_path, monkeypatch):
    home, audit = graduated(tmp_path, contacts="")
    monkeypatch.setenv("HERMES_HOME", str(home))
    api = ControlAPI(load_bundle(tmp_path / "b"), HermesRuntime(home=home, tenant_id="acme"), audit=audit)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p)["action"] == "approve", "not listed yet"

    assert api.write("/platform/v1/settings/contacts", VIEWER, {"internal": [], "customers": ["telegram:123456789"]}).status == 403
    bad = api.write("/platform/v1/settings/contacts", ADMIN, {"internal": [], "customers": ["not a target"]})
    assert bad.status == 400 and "send target" in bad.body["error"]["message"]
    saved = api.write("/platform/v1/settings/contacts", ADMIN,
                      {"internal": ["# our team", "slack:C0TEAM"], "customers": ["telegram:123456789"]})
    assert saved.status == 200, saved.body
    assert load_contacts(tmp_path / "b").customers == ("telegram:123456789",)
    assert api.handle("/platform/v1/contacts").body == {"internal": ["slack:C0TEAM"], "customers": ["telegram:123456789"]}

    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    assert call(p) is None, "listed now, and the apply carried the list to the worker"
    changed = [json.loads(l) for l in audit.path.read_text().splitlines() if "settings.contacts_changed" in l]
    assert [e["phase"] for e in changed] == ["intent", "committed"]
    assert changed[0]["detail"]["customers"] == 1 and "123456789" not in json.dumps(changed)
    assert ADMIN.may("/contacts") and not VIEWER.may("/contacts")
