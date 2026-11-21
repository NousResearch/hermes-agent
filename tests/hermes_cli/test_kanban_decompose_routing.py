"""Routing gates for `kanban.auto_decompose` (the decomposition routing contract).

Two entry points share ONE backend function (``kanban_decompose.decompose_task``):
the CLI (`hermes kanban decompose`) and the dashboard
``POST /tasks/{task_id}/decompose``. These tests pin the contract at that seam:

* auto ON  -> the existing auxiliary path is preserved, byte for byte.
* auto OFF -> the auxiliary model is NEVER called. The existing structured
  prompt is handed to exactly one eligible subscriber as a wake/instruction,
  and the task graph is left untouched until an explicit ``kanban_decompose``
  tool call.
* auto OFF + no eligible subscriber -> a clear result, no auxiliary call, the
  task is unchanged (never an auxiliary fallback).
* auto OFF + several distinct eligible profiles -> a clear ambiguity, no
  broadcast and no graph mutation.
* multiple destinations for the SAME profile count as one profile.

The auxiliary client is mocked to raise, so any unexpected call fails loudly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _roster(names):
    """Pretend these profiles exist (established decompose-test convention)."""
    from types import SimpleNamespace
    fake = [
        SimpleNamespace(name=n, is_default=(i == 0), description=f"desc for {n}",
                        description_auto=False, model="m", provider="p", skill_count=1)
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name",
              return_value=names[0] if names else "default"),
    ]


def _auto_off():
    """kanban.auto_decompose = False, read through the module the gate imports."""
    return patch("hermes_cli.config.load_config_readonly",
                 return_value={"kanban": {"auto_decompose": False}})


def _no_aux(monkeypatch):
    """Any auxiliary call is a hard failure while auto-decompose is off."""
    def _boom(*_a, **_k):
        raise AssertionError("auxiliary model must NOT be called when auto_decompose is off")
    monkeypatch.setattr(decomp, "_call_aux", _boom)


def _capable(monkeypatch, names):
    monkeypatch.setattr(decomp, "profile_has_kanban_toolset",
                        lambda p: p in set(names))


class _Transport:
    """Fake gateway wake transport; records every instruction it is handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, profile, task_id, text, subs):
        self.calls.append({"profile": profile, "task_id": task_id,
                           "text": text, "subs": subs})
        return True, "instruction delivered to 1 destination(s)"


@pytest.fixture
def transport(monkeypatch):
    fake = _Transport()
    decomp.set_instruction_transport(fake)
    yield fake
    decomp.set_instruction_transport(None)


def _snapshot(conn, tid):
    """Everything a notification/routing call must not disturb."""
    return {
        "status": kb.get_task(conn, tid).status,
        "assignee": kb.get_task(conn, tid).assignee,
        "parents": kb.parent_ids(conn, tid),
        "children": kb.child_ids(conn, tid),
        "events": conn.execute(
            "SELECT kind, COUNT(*) FROM task_events WHERE task_id=? GROUP BY kind",
            (tid,)).fetchall(),
        "task_total": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
        "link_total": conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0],
    }


def _triage_task(assignee="orchestrator"):
    with kbc.connect() as conn:
        return kb.create_task(conn, title="triage me", assignee=assignee, triage=True)


def _sub(tid, *, profile, platform="telegram", chat_id="1001"):
    with kbc.connect() as conn:
        kbn.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id,
                           notifier_profile=profile)


# --------------------------------------------------------------------------- auto ON


def test_auto_decompose_enabled_preserves_the_auxiliary_path(kanban_home, monkeypatch, transport):
    """With auto ON the gate is transparent: the auxiliary model still runs."""
    tid = _triage_task()

    seen = {}

    def _fake_aux(*_args, **_kwargs):
        seen["called"] = True
        # _call_aux returns (raw, reason); a 2-tuple keeps the real unpack shape.
        return ('{"fanout": false, "title": "split", "body": "b", '
                '"assignee": "orchestrator"}', None)

    monkeypatch.setattr(decomp, "_call_aux", _fake_aux)
    patches = _roster(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert seen.get("called") is True, "auto ON must still reach the auxiliary model"
    # No instruction wake was routed — that path belongs to auto OFF only.
    assert transport.calls == []
    assert outcome.routed_to is None


# --------------------------------------------------------------------------- auto OFF + one subscriber


def test_auto_off_routes_prompt_to_the_single_eligible_subscriber(kanban_home, monkeypatch, transport):
    """Exactly one eligible profile gets the EXISTING prompt, and nothing is written."""
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False          # nothing was decomposed
    assert outcome.routed_to == "worker"
    assert "NOT called" in outcome.reason
    assert "unchanged" in outcome.reason

    # Exactly one destination, and it carries the byte-for-byte existing prompt.
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["profile"] == "worker"
    assert call["task_id"] == tid
    assert decomp._SYSTEM_PROMPT in call["text"]
    assert "=== USER ===" in call["text"]
    assert tid in call["text"]
    # The instruction must tell the receiver the board did not move.
    assert "changes NOTHING on the board" in call["text"]

    # Notification alone never mutates the graph.
    assert after == before
    assert after["status"] == "triage"
    assert after["children"] == [] and after["parents"] == []


# --------------------------------------------------------------------------- auto OFF + no subscriber


def test_auto_off_with_no_subscriber_never_falls_back_to_auxiliary(kanban_home, monkeypatch, transport):
    """No eligible subscriber -> clear result, no auxiliary call, task unchanged."""
    tid = _triage_task()
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])     # capable profiles exist, none subscribed
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.routed_to is None
    assert "no eligible Kanban-toolset-capable subscriber" in outcome.reason
    assert "stays in" in outcome.reason and "triage" in outcome.reason
    assert transport.calls == []          # no wake was pushed either
    assert after == before
    assert after["status"] == "triage"


def test_auto_off_subscriptions_without_a_profile_are_not_eligible(kanban_home, monkeypatch, transport):
    """A legacy sub with no notifier_profile stamp cannot name a profile."""
    tid = _triage_task()
    _sub(tid, profile=None, chat_id="222")   # unstamped
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False and outcome.routed_to is None
    assert "no eligible" in outcome.reason
    assert "but none is eligible" in outcome.reason
    assert "1 carry no notifier profile" in outcome.reason
    assert transport.calls == []
    assert after == before


# --------------------------------------------------------------------------- multiple destinations, one profile


def test_multiple_destinations_for_one_profile_collapse_to_a_single_route(kanban_home, monkeypatch, transport):
    """Two chats for the same profile are ONE profile — routing must not fan out."""
    tid = _triage_task()
    _sub(tid, profile="worker", platform="telegram", chat_id="1")
    _sub(tid, profile="worker", platform="discord", chat_id="2")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.routed_to == "worker", "same profile on 2 destinations is one profile"
    assert "Refusing to broadcast" not in outcome.reason
    # One instruction for the profile, carrying both of its destinations.
    assert len(transport.calls) == 1
    assert len(transport.calls[0]["subs"]) == 2
    assert after == before


# --------------------------------------------------------------------------- several distinct profiles


def test_multiple_distinct_eligible_profiles_are_rejected_as_ambiguous(kanban_home, monkeypatch, transport):
    """Several distinct capable profiles -> clear ambiguity, no broadcast, no write."""
    tid = _triage_task()
    _sub(tid, profile="alpha", platform="telegram", chat_id="1")
    _sub(tid, profile="beta", platform="discord", chat_id="2")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["alpha", "beta"])
    patches = _roster(["orchestrator", "alpha", "beta"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.routed_to is None, "ambiguity must not pick a winner"
    assert "2 distinct eligible Kanban profiles" in outcome.reason
    assert "alpha" in outcome.reason and "beta" in outcome.reason
    assert "Refusing to broadcast" in outcome.reason
    assert transport.calls == [], "must not broadcast to either profile"
    assert after == before
    assert after["status"] == "triage"


# --------------------------------------------------------------------------- dashboard entry point


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_routing_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def api(kanban_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_dashboard_decompose_no_subscriber_is_clear_and_leaves_task_unchanged(
        kanban_home, api, monkeypatch, transport):
    """POST /tasks/{id}/decompose shares the same gate: clear, no aux, no write."""
    tid = _triage_task()
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            resp = api.post(f"/api/plugins/kanban/tasks/{tid}/decompose", json={})
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()

    # Non-OK is NOT an HTTP error — the UI renders the reason inline.
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["task_id"] == tid
    assert body["child_ids"] == []
    assert body["fanout"] is False
    assert body["routed_to"] is None
    assert "no eligible Kanban-toolset-capable subscriber" in body["reason"]
    assert "graph is unchanged" in body["reason"] or "stays in" in body["reason"]
    assert transport.calls == []
    assert after == before
    assert after["status"] == "triage"


# --------------------------------------------------------------------------- delivery legs
# A process that does NOT host the gateway (a plain CLI invocation, or an
# out-of-process dashboard thread) has no in-process wake transport: it reaches
# a running gateway over the ``deliver-decompose-instruction`` control-socket
# verb, and ``routed_to`` may only ever name a CONFIRMED wake. These tests pin
# every leg of that fallback, including the real no-transport path.


def _no_transport(monkeypatch):
    """This process hosts no gateway: the in-process leg is absent."""
    monkeypatch.setattr(decomp, "_INSTRUCTION_TRANSPORT", None)


def _control_socket(monkeypatch, answer):
    """Make one control socket reachable, answering ``answer``; record the queries.

    Patching BOTH the resolver and the client keeps the test hermetic: no real
    gateway on this host can ever be queried (or woken) from the suite.
    """
    import gateway.control_socket as cs

    calls: list[dict] = []
    monkeypatch.setattr(
        cs, "resolve_client_socket_path",
        lambda home: Path(str(home)) / "gateway.sock")
    monkeypatch.setattr(
        cs, "query_gateway_control",
        lambda home, verb, *, params=None, timeout=None: (
            calls.append({"home": Path(home), "verb": verb,
                          "params": params, "timeout": timeout})
            or answer))
    return calls


def _no_socket(monkeypatch):
    """No gateway control socket is reachable from this process."""
    import gateway.control_socket as cs

    calls: list[dict] = []
    monkeypatch.setattr(cs, "resolve_client_socket_path", lambda home: None)
    monkeypatch.setattr(
        cs, "query_gateway_control",
        lambda *a, **k: calls.append({"args": a, "kwargs": k}) or None)
    return calls


def _routed_run(monkeypatch, tid):
    """Run the auto-off gate the way CLI/dashboard entry points do."""
    patches = _roster(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            before = _snapshot(conn, tid)
        with _auto_off():
            outcome = decomp.decompose_task(tid, author="me")
        with kbc.connect() as conn:
            after = _snapshot(conn, tid)
    finally:
        for p in patches:
            p.stop()
    return outcome, before, after


def test_auto_off_without_transport_or_socket_reports_no_delivery(kanban_home, monkeypatch):
    """The real no-transport path: neither leg can confirm a wake, so the result
    says so plainly, ``routed_to`` stays unset and the task is untouched."""
    _no_transport(monkeypatch)
    queries = _no_socket(monkeypatch)
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, before, after = _routed_run(monkeypatch, tid)

    assert outcome.ok is False
    assert outcome.routed_to is None, "an unconfirmed wake must never be reported as routed"
    assert "could NOT be delivered" in outcome.reason
    assert "no gateway control socket is reachable" in outcome.reason
    assert "NOT called" in outcome.reason
    assert "stays in" in outcome.reason and "triage" in outcome.reason
    assert queries == [], "with no reachable socket no gateway may be queried"
    assert after == before
    assert after["status"] == "triage"


def test_auto_off_control_socket_confirms_delivery_and_reports_routed_to(
        kanban_home, monkeypatch):
    """A confirmed gateway answer is the reachable CLI/dashboard delivery path:
    exactly one control query carrying the prompt, then a truthful routed_to."""
    _no_transport(monkeypatch)
    calls = _control_socket(monkeypatch, {
        "delivered": True,
        "detail": "instruction delivered to 'worker' as a single push wake on telegram/111",
    })
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, before, after = _routed_run(monkeypatch, tid)

    assert outcome.ok is False                  # nothing was decomposed
    assert outcome.routed_to == "worker"
    assert "handed to the single eligible" in outcome.reason
    assert "single push wake on telegram/111" in outcome.reason
    assert "NOT called" in outcome.reason

    assert len(calls) == 1, "exactly one control-socket query"
    assert calls[0]["verb"] == "deliver-decompose-instruction"
    assert calls[0]["timeout"] == decomp._CONTROL_DELIVERY_TIMEOUT
    params = calls[0]["params"]
    assert params["profile"] == "worker"
    assert params["task_id"] == tid
    assert decomp._SYSTEM_PROMPT in params["text"]
    assert "=== USER ===" in params["text"]
    assert len(params["subs"]) == 1 and str(params["subs"][0]["chat_id"]) == "111"

    # Notification alone never mutates the graph.
    assert after == before
    assert after["status"] == "triage"


@pytest.mark.parametrize("answer, marker", [
    ({"delivered": False,
      "detail": "the gateway could not confirm the instruction wake"},
     "the gateway could not confirm the instruction wake"),
    (None, "gave no usable"),
], ids=["gateway-says-not-delivered", "socket-answer-unusable"])
def test_auto_off_unconfirmed_socket_answer_leaves_routed_to_unset(
        kanban_home, monkeypatch, answer, marker):
    """A reachable socket that does not confirm the wake is NOT a handoff:
    routed_to stays unset and the prompt's failure is reported verbatim."""
    _no_transport(monkeypatch)
    calls = _control_socket(monkeypatch, answer)
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, before, after = _routed_run(monkeypatch, tid)

    assert outcome.ok is False
    assert outcome.routed_to is None, "only a CONFIRMED wake may be reported as routed"
    assert "could NOT be delivered" in outcome.reason
    assert marker in outcome.reason
    assert "NOT called" in outcome.reason
    assert len(calls) == 1
    assert calls[0]["verb"] == "deliver-decompose-instruction"
    assert after == before
    assert after["status"] == "triage"


def test_in_process_transport_takes_precedence_over_the_control_socket(
        kanban_home, monkeypatch, transport):
    """Leg ordering: a gateway hosting the transport in-process never consults
    the control socket (so a wake can never be issued twice)."""
    calls = _control_socket(monkeypatch, {"delivered": True, "detail": "never asked"})
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, _before, after = _routed_run(monkeypatch, tid)

    assert outcome.routed_to == "worker"
    assert len(transport.calls) == 1
    assert calls == [], "the control-socket leg must not be consulted in-process"
    assert after["status"] == "triage"


# ------------------------------------------------------- confirmed-delivery typing
# `routed_to` may only ever name a CONFIRMED wake, and a wake is confirmed by an
# exact boolean ``True`` — nothing else. Both delivery boundaries used to run
# their flag through ``bool()``, so a malformed truthy answer (the string
# "false" the review reproduced, or the integer 1) was coerced into a confirmed
# delivery. These tests pin the refusal on both legs, and that a real boolean
# True still routes.


@pytest.mark.parametrize("flag", ["false", 1], ids=["string-false", "integer-one"])
def test_auto_off_truthy_nonboolean_socket_answer_is_refused_and_unrouted(
        kanban_home, monkeypatch, flag):
    """A reachable socket answering ``delivered`` with a truthy NON-boolean is
    not a confirmation: routed_to stays unset and the malformed value is named."""
    _no_transport(monkeypatch)
    calls = _control_socket(monkeypatch, {
        "delivered": flag,
        "detail": "gateway says false",
    })
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, before, after = _routed_run(monkeypatch, tid)

    assert outcome.ok is False
    assert outcome.routed_to is None, f"delivered={flag!r} must never be routed"
    assert "could NOT be delivered" in outcome.reason
    assert "gateway says false" in outcome.reason, "the gateway's own detail stays visible"
    assert f"delivered={flag!r}" in outcome.reason, "the malformed value is named"
    assert "NOT confirmed" in outcome.reason
    assert "NOT called" in outcome.reason
    # The first (only) gateway answering is authoritative — a malformed answer
    # is never retried against a second gateway, so exactly one query is made.
    assert len(calls) == 1
    assert calls[0]["verb"] == "deliver-decompose-instruction"
    assert after == before
    assert after["status"] == "triage"


@pytest.mark.parametrize("flag", ["false", 1], ids=["string-false", "integer-one"])
def test_in_process_transport_truthy_nonboolean_flag_is_never_a_wake(
        kanban_home, monkeypatch, flag):
    """The in-process leg converts its transport's flag by the same rule:
    only an exact boolean True is a confirmed wake."""
    calls: list[dict] = []

    def _lying_transport(*, profile, task_id, text, subs):
        calls.append({"profile": profile, "task_id": task_id, "subs": subs})
        return flag, "gateway says false"

    monkeypatch.setattr(decomp, "_INSTRUCTION_TRANSPORT", _lying_transport)
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    outcome, before, after = _routed_run(monkeypatch, tid)

    assert outcome.ok is False
    assert outcome.routed_to is None, f"in-process delivered={flag!r} must never be routed"
    assert "could NOT be delivered" in outcome.reason
    assert f"delivered={flag!r}" in outcome.reason
    assert "NOT confirmed" in outcome.reason
    assert "NOT called" in outcome.reason
    assert len(calls) == 1, "the transport was consulted exactly once"
    assert after == before
    assert after["status"] == "triage"


def test_exact_boolean_true_still_routes_on_both_delivery_legs(kanban_home, monkeypatch):
    """The strictness is about TYPE, not pessimism: a genuine boolean ``True``
    still routes — through the in-process transport first, then through the
    control socket when that leg is the only one present."""
    tid = _triage_task()
    _sub(tid, profile="worker", chat_id="111")
    _no_aux(monkeypatch)
    _capable(monkeypatch, ["worker"])

    def _transport(**kwargs):
        assert kwargs["profile"] == "worker"
        return True, "instruction delivered to 1 destination(s)"

    # Leg 1: in-process transport answering boolean True.
    monkeypatch.setattr(decomp, "_INSTRUCTION_TRANSPORT", _transport)
    in_process, _, _after_1 = _routed_run(monkeypatch, tid)

    # Leg 2: no in-process transport, control socket answering boolean True.
    _no_transport(monkeypatch)
    calls = _control_socket(monkeypatch, {"delivered": True, "detail": "single push wake"})
    via_socket, _before, after = _routed_run(monkeypatch, tid)

    assert in_process.routed_to == "worker"
    assert "instruction delivered to 1 destination(s)" in in_process.reason
    assert via_socket.routed_to == "worker"
    assert "single push wake" in via_socket.reason
    assert len(calls) == 1
    # Neither leg ever mutated the graph.
    assert after["status"] == "triage"
