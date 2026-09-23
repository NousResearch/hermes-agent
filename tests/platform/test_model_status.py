"""Whether the workforce can reach its model, and how its failures are explained."""

from __future__ import annotations

import json
import sqlite3

from nova.control import ControlAPI
from nova.runtime.model_errors import classify_model_error, summarize_task_error


# -- classification ----------------------------------------------------------


def test_an_unentitled_account_is_not_mistaken_for_a_permissions_error():
    """Bedrock's "Operation not allowed" means the account lacks the model. Reading it as
    IAM sends an operator to widen a grant that was already right."""
    error = classify_model_error("Error code: 400 - {'message': 'Operation not allowed'}")
    assert error.category == "account_not_authorized"
    assert error.owner == "account admin"
    assert "Operation not allowed" in error.raw


def test_an_iam_refusal_is_the_operators_to_fix():
    error = classify_model_error("AccessDeniedException: User is not authorized to perform bedrock:InvokeModel")
    assert error.category == "permission_denied" and error.owner == "operator"


def test_unknown_text_is_reported_raw_rather_than_guessed():
    error = classify_model_error("something nobody has seen before")
    assert error.category == "error" and error.raw == "something nobody has seen before"


def test_the_workers_protocol_violation_reads_as_what_happened():
    summary = summarize_task_error(
        "worker exited cleanly (rc=0) without calling kanban_complete or kanban_block — protocol violation."
    )
    assert "without reporting a result" in summary["headline"]


# -- evidence from the runtime's own files -----------------------------------


FAILURE_LINE = (
    "{ts},142 WARNING [20260921_154240_a1f49c] agent.conversation_loop: API call failed "
    "(attempt 1/3) error_type=BadRequestError thread=Thread-4 provider=bedrock "
    "base_url=https://bedrock-runtime.eu-west-2.amazonaws.com model=anthropic.claude-sonnet-4-6 "
    "summary=HTTP 400: Operation not allowed\n"
)


def _fail(home, profile, ts):
    logs = home / "profiles" / profile / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(FAILURE_LINE.format(ts=ts))


def _succeed(home, profile, started_at):
    db = home / "profiles" / profile / "state.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, started_at REAL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS session_model_usage (session_id TEXT, model TEXT, "
        "api_call_count INTEGER, output_tokens INTEGER)"
    )
    connection.execute("INSERT INTO sessions VALUES (?, ?)", (f"s{started_at}", started_at))
    connection.execute(
        "INSERT INTO session_model_usage VALUES (?, 'eu.anthropic.claude-sonnet-4-6', 1, 12)",
        (f"s{started_at}",),
    )
    connection.commit()
    connection.close()


def test_nothing_tried_is_unknown_not_healthy(runtime):
    assert runtime.model_status()["state"] == "unknown"


def test_the_newest_evidence_decides(home, runtime):
    _fail(home, "operations", "2026-09-21 15:42:48")
    status = runtime.model_status()
    assert status["state"] == "failing"
    assert status["last_failure"]["profile"] == "operations"
    assert status["last_failure"]["error"]["category"] == "account_not_authorized"

    _succeed(home, "operations", 1790100000.0)  # after the failure
    assert runtime.model_status()["state"] == "working"


def test_reading_the_status_creates_nothing(home, runtime):
    """The control plane is a reader. A probe that created a database on a customer host
    — which an earlier investigation did — is a write nobody asked for."""
    (home / "profiles" / "operations").mkdir(parents=True)
    before = sorted(p.relative_to(home) for p in home.rglob("*"))
    runtime.model_status()
    assert sorted(p.relative_to(home) for p in home.rglob("*")) == before


# -- surfaced through the Control API ----------------------------------------


def test_the_model_route_says_what_is_configured_and_whether_it_works(bundle, runtime, home):
    _fail(home, "customer-support", "2026-09-21 15:42:48")
    body = ControlAPI(bundle, runtime).handle("/platform/v1/model").body
    assert body["configured"]["model"] == bundle.deployment.provider.model
    assert body["state"] == "failing"
    assert {a["id"] for a in body["agents"]} >= {"customer-support", "operations"}


def test_a_channel_is_live_only_when_the_gateway_says_so(bundle, runtime, home):
    api = ControlAPI(bundle, runtime)
    channel = api.handle("/platform/v1/channels").body["channels"][0]
    assert channel["live"]["state"] == "unknown"  # no gateway record at all

    (home / "gateway_state.json").write_text(json.dumps({
        "gateway_state": "running",
        "platforms": {f"customer-support:{channel['provider']}": {"state": "connected"}},
    }), encoding="utf-8")
    channel = api.handle("/platform/v1/channels").body["channels"][0]
    assert channel["live"]["state"] == "connected"
    assert channel["capabilities"]  # carried for the card, never invented


def test_a_platform_the_state_file_omits_is_read_from_the_gateway_log(bundle, runtime, home):
    """Seen live: a secondary profile's Slack connected and was logged, but never got an
    entry in gateway_state.json. Reading absence as "down" called a working bot broken."""
    (home / "gateway_state.json").write_text(json.dumps({
        "gateway_state": "running",
        "platforms": {"customer-support:discord": {"state": "connected"}},
    }), encoding="utf-8")
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "gateway.log"
    log.write_text(
        "2026-09-23 15:42:21,344 INFO gateway.run: ✓ telegram disconnected (0.00s) (profile: customer-support)\n"
        "2026-09-23 15:42:50,667 INFO gateway.run: ✓ telegram connected (profile: customer-support)\n",
        encoding="utf-8",
    )
    api = ControlAPI(bundle, runtime)
    live = {c["provider"]: c["live"]["state"] for c in api.handle("/platform/v1/channels").body["channels"]}
    assert live["telegram"] == "connected"

    with log.open("a", encoding="utf-8") as handle:
        handle.write("2026-09-23 16:10:00,000 INFO gateway.run: ✓ telegram disconnected (0.00s) (profile: customer-support)\n")
    live = {c["provider"]: c["live"]["state"] for c in api.handle("/platform/v1/channels").body["channels"]}
    assert live["telegram"] == "disconnected"


def test_a_platform_nothing_mentions_is_unknown_not_down(bundle, runtime, home):
    (home / "gateway_state.json").write_text(json.dumps({"gateway_state": "running", "platforms": {}}),
                                             encoding="utf-8")
    channels = ControlAPI(bundle, runtime).handle("/platform/v1/channels").body["channels"]
    assert {c["live"]["state"] for c in channels} == {"unknown"}
