"""Loud model fallback: an unhonored model/provider request must be visible.

Regression cover for the silent-fallback failure mode reported 2026-08-17:

    hermes -m "openrouter/zai/glm-5.3" --provider openrouter -z "ping"

answered "pong" with **glm-5.2 via zai** — a different model from a different
provider — with no error, no warning, and a session row that claimed the served
model had been the requested one. Three things conspired:

1. ``hermes -z`` redirects stdout AND stderr to devnull for the whole run and
   sets ``suppress_status_output``, which short-circuits ``_vprint`` *before*
   its ``force`` check — so even the forced fallback notice was swallowed.
2. ``sessions.model`` holds a single (model, provider) pair, and
   ``update_token_counts``' first-accounted-route reconciliation overwrites it
   with the route that actually billed. The request was destroyed on the first
   API call.
3. Nothing anywhere persisted "what was asked for".

The tests below pin each half of the fix: the state layer keeps the request
beside the delivery, and one-shot says so out loud on the real stderr.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# State layer: the request survives the served route overwriting `model`
# ---------------------------------------------------------------------------

def test_requested_route_survives_first_accounted_route_overwrite(db):
    """The exact reported case: request openrouter, get billed by zai."""
    db.create_session(
        "s_fallback",
        source="cli",
        model="openrouter/zai/glm-5.3",
        requested_model="openrouter/zai/glm-5.3",
        requested_provider="openrouter",
    )

    # First accounted API call comes from the fallback route, not the request.
    db.update_token_counts(
        "s_fallback",
        input_tokens=10,
        output_tokens=5,
        model="glm-5.2",
        billing_provider="zai",
        api_call_count=1,
    )
    db.flush_token_counts()

    row = db.get_session("s_fallback")
    # Served route won the aggregate columns — that part is intended.
    assert row["model"] == "glm-5.2"
    assert row["billing_provider"] == "zai"
    # ...and the request is still recoverable, which is the whole point.
    assert row["requested_model"] == "openrouter/zai/glm-5.3"
    assert row["requested_provider"] == "openrouter"


def test_requested_route_is_never_rewritten_by_a_later_observation(db):
    """A second writer must not be able to relabel the original request."""
    db.create_session(
        "s_keep",
        source="cli",
        model="glm-5.3",
        requested_model="glm-5.3",
        requested_provider="zai",
    )
    # A lazy writer (update_token_counts' self-healing insert) re-enters the
    # row with a different model; COALESCE must leave the request alone.
    db.update_token_counts(
        "s_keep", input_tokens=1, model="deepseek-v4-flash",
        billing_provider="deepseek", api_call_count=1,
    )
    db.record_session_fallback(
        "s_keep", requested_model="something-else", requested_provider="nowhere",
    )
    db.flush_token_counts()

    row = db.get_session("s_keep")
    assert row["requested_model"] == "glm-5.3"
    assert row["requested_provider"] == "zai"


def test_fallback_flag_defaults_off_and_is_sticky(db):
    db.create_session(
        "s_flag", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    assert db.get_session("s_flag")["fallback_activated"] == 0

    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1
    # Idempotent — a second chain hop must not corrupt the flag.
    db.record_session_fallback("s_flag")
    assert db.get_session("s_flag")["fallback_activated"] == 1


def test_record_session_fallback_backfills_a_row_created_before_the_columns(db):
    """Rows that predate the audit columns still get a usable request."""
    db.create_session("s_backfill", source="cli", model="glm-5.2")
    db.record_session_fallback(
        "s_backfill", requested_model="gpt-5.6-sol", requested_provider="openrouter",
    )
    row = db.get_session("s_backfill")
    assert row["fallback_activated"] == 1
    assert row["requested_model"] == "gpt-5.6-sol"
    assert row["requested_provider"] == "openrouter"


def test_record_session_fallback_tolerates_a_missing_row(db):
    """Never raise on the recovery path — the row is created lazily."""
    db.record_session_fallback("s_does_not_exist", requested_model="x")
    assert db.get_session("s_does_not_exist") is None


def test_explicit_model_switch_resets_the_request_audit(db):
    """A /model switch is a NEW request, so the stale flag must clear."""
    db.create_session(
        "s_switch", source="cli", model="glm-5.3",
        requested_model="glm-5.3", requested_provider="zai",
    )
    db.record_session_fallback("s_switch")
    db.update_session_model("s_switch", "deepseek-v4-flash", provider="deepseek")

    row = db.get_session("s_switch")
    assert row["requested_model"] == "deepseek-v4-flash"
    assert row["requested_provider"] == "deepseek"
    assert row["fallback_activated"] == 0


def test_audit_columns_are_declared_so_existing_dbs_reconcile(tmp_path):
    """The columns are declarative: an older DB gains them on next open."""
    import sqlite3

    from hermes_state_common import SCHEMA_SQL

    declared = SessionDB._parse_schema_columns(SCHEMA_SQL)["sessions"]
    for column in ("requested_model", "requested_provider", "fallback_activated"):
        assert column in declared, column

    # Build a DB, drop the columns out of the picture by recreating the table
    # without them, then reopen: _reconcile_columns must ADD them back.
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
        "model TEXT, started_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = SessionDB(path)
    try:
        live = {
            row[1]
            for row in db._conn.execute('PRAGMA table_info("sessions")').fetchall()
        }
        assert {"requested_model", "requested_provider", "fallback_activated"} <= live
    finally:
        db.close()


# ---------------------------------------------------------------------------
# One-shot: the warning reaches the real stderr, stdout stays clean
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, requested_model, requested_provider, fallback):
        self.origin_requested_model = requested_model
        self.origin_requested_provider = requested_provider
        self._fallback_activated = fallback


def test_annotate_requested_route_reads_the_immutable_snapshot():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    # try_activate_fallback reassigns requested_provider to the fallback; the
    # audit must not read that attribute.
    agent.requested_provider = "zai"
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    assert result["requested_model"] == "openrouter/zai/glm-5.3"
    assert result["requested_provider"] == "openrouter"
    assert result["fallback_activated"] is True


def test_no_warning_when_the_requested_model_answered():
    from hermes_cli import oneshot

    agent = _FakeAgent("glm-5.3", "zai", False)
    result = {"model": "glm-5.3", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)
    assert oneshot._fallback_warning_line(result) is None


def test_warning_names_both_routes():
    from hermes_cli import oneshot

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    result = {"model": "glm-5.2", "provider": "zai"}
    oneshot._annotate_requested_route(agent, result)

    line = oneshot._fallback_warning_line(result)
    assert line is not None
    assert line.endswith("\n")
    assert "openrouter/zai/glm-5.3 via openrouter" in line
    assert "glm-5.2 via zai" in line
    # The whole failure mode was that a wrong-model answer looked normal.
    assert "SERVED" in line


def test_warning_survives_a_missing_request_half():
    from hermes_cli import oneshot

    line = oneshot._fallback_warning_line(
        {"model": "glm-5.2", "provider": "zai", "fallback_activated": True}
    )
    assert line is not None
    assert "an unknown model" in line


def test_oneshot_writes_the_warning_to_the_real_stderr(monkeypatch, capsys):
    """The end-to-end guarantee: -z can no longer answer 200-quiet.

    ``run_oneshot`` swallows every byte the agent writes; only the final
    response reaches stdout. This asserts the fallback notice takes the real
    stderr path instead, and that stdout stays exactly the response.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="openrouter/zai/glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == "pong\n"
    assert "openrouter/zai/glm-5.3" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_nonexistent_requested_model_still_names_it_in_the_warning(
    monkeypatch, capsys
):
    """Requesting a model that does not exist may not answer quietly either.

    The 2026-08-17 reproduction included ``openai/gpt-5.6-sol`` — a model id
    that does not exist — which was answered by glm-5.2/zai with no signal.
    Whatever the reason a request cannot be honored (unreachable provider,
    bad id, exhausted quota), the fallback notice must name the id exactly as
    typed, so a pipeline pinning a model can see the pin was not honored.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openai/gpt-5.6-sol", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot(
        "ping", model="openai/gpt-5.6-sol", provider="openrouter"
    )
    captured = capsys.readouterr()

    assert rc == 0  # a fallback that answered is not an error...
    assert captured.out == "pong\n"  # ...but stdout stays machine-readable
    # and the nonexistent id is named verbatim on the real stderr.
    assert "openai/gpt-5.6-sol via openrouter" in captured.err
    assert "glm-5.2 via zai" in captured.err


def test_provider_without_credentials_fails_loudly(monkeypatch, capsys):
    """No credentials must never become a quiet answer from an unrelated model.

    When there is no fallback chain to absorb the failure, the run must exit
    non-zero with an explicit error — the same loud contract as the fallback
    warning, on the branch where nothing answered at all.
    """
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        raise RuntimeError(
            "openrouter: AuthenticationError: no API key configured"
        )

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    rc = oneshot.run_oneshot("ping", model="glm-5.3", provider="openrouter")
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""  # no answer at all — never an unrelated model
    assert "agent failed" in captured.err
    assert "AuthenticationError" in captured.err


def test_oneshot_usage_file_records_request_and_delivery(tmp_path, monkeypatch):
    """Pipelines get the audit in machine-readable form."""
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.2",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(
            _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True), result
        )
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot(
        "ping",
        model="openrouter/zai/glm-5.3",
        provider="openrouter",
        usage_file=str(usage),
    )

    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["requested_model"] == "openrouter/zai/glm-5.3"
    assert report["requested_provider"] == "openrouter"
    assert report["fallback_activated"] is True
    assert report["model"] == "glm-5.2"


def test_usage_file_marks_an_honored_request_as_not_fallen_back(tmp_path, monkeypatch):
    from hermes_cli import oneshot

    def _fake_run_agent(prompt, **kwargs):
        result = {
            "final_response": "pong",
            "model": "glm-5.3",
            "provider": "zai",
            "completed": True,
        }
        oneshot._annotate_requested_route(_FakeAgent("glm-5.3", "zai", False), result)
        return "pong", result

    monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(oneshot, "declare_stateless_channel", lambda *a, **k: None)

    usage = tmp_path / "usage.json"
    oneshot.run_oneshot("ping", model="glm-5.3", provider="zai", usage_file=str(usage))
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["fallback_activated"] is False


# ---------------------------------------------------------------------------
# Agent init: the snapshot is taken, and fallback cannot overwrite it
# ---------------------------------------------------------------------------

def test_fallback_swap_leaves_the_origin_snapshot_intact():
    """try_activate_fallback rewrites requested_provider; not the audit."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record_session_fallback(self, session_id, **kwargs):
            self.calls.append((session_id, kwargs))

    agent = _FakeAgent("openrouter/zai/glm-5.3", "openrouter", True)
    agent.session_id = "s1"
    agent._session_db = _Recorder()
    _record_fallback_on_session(agent)

    assert agent._session_db.calls == [
        (
            "s1",
            {
                "requested_model": "openrouter/zai/glm-5.3",
                "requested_provider": "openrouter",
            },
        )
    ]


def test_record_fallback_on_session_never_raises():
    """A bookkeeping failure must not abort provider recovery."""
    from agent.chat_completion_helpers import _record_fallback_on_session

    class _Exploding:
        def record_session_fallback(self, *a, **k):
            raise RuntimeError("db is locked")

    agent = _FakeAgent("glm-5.3", "zai", True)
    agent.session_id = "s1"
    agent._session_db = _Exploding()
    _record_fallback_on_session(agent)  # must not raise

    # No session_db / no session_id are also non-events.
    bare = _FakeAgent("glm-5.3", "zai", True)
    bare.session_id = None
    bare._session_db = None
    _record_fallback_on_session(bare)


# ---------------------------------------------------------------------------
# `hermes sessions list` names the divergence
# ---------------------------------------------------------------------------

def test_sessions_list_reports_flagged_rows(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings([
        {
            "id": "20260817_191805_ec4afa",
            "model": "glm-5.2",
            "billing_provider": "zai",
            "requested_model": "openrouter/zai/glm-5.3",
            "requested_provider": "openrouter",
            "fallback_activated": 1,
        },
        {"id": "ok", "model": "glm-5.3", "fallback_activated": 0},
    ])
    out = capsys.readouterr().out
    assert "20260817_191805_ec4afa" in out
    assert "openrouter/zai/glm-5.3 (openrouter)" in out
    assert "glm-5.2 (zai)" in out
    assert "ok" not in out.replace("20260817_191805_ec4afa", "")


def test_sessions_list_stays_quiet_when_nothing_fell_back(capsys):
    from hermes_cli import sessions_cmd

    sessions_cmd._print_fallback_warnings(
        [{"id": "s1", "model": "glm-5.3", "fallback_activated": 0}]
    )
    assert capsys.readouterr().out == ""
