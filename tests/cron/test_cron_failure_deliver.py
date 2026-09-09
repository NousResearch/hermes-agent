"""Per-job ``failure_deliver`` routing (NS-788).

A job's FAILURE notices (run failed, escaped scheduler exception, drift-skip /
blocked-config alerts) resolve their delivery targets from ``failure_deliver``
when the job sets it, falling back to ``deliver`` when unset — so existing
jobs behave byte-identically. ``failure_deliver: local`` is structural silence
for failures: nothing is sent, but state (last_status, run history, output
file) is still recorded. Success-path delivery never reads ``failure_deliver``.

The grammar is exactly the ``deliver`` grammar — same normalization, same
validation — reused, not duplicated.
"""

import json

import pytest

import cron.scheduler as s
from cron import scheduler_delivery as sched_delivery
from cron import scheduler_preflight as sched_preflight
from cron.scheduler import _resolve_delivery_targets


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    """Drive run_one_job with the REAL delivery path down to a fake sender.

    Bookkeeping primitives are stubbed (recorded), but _deliver_result and
    _resolve_delivery_targets are the genuine articles — the send that would
    leave the process is captured at the platform-registry sender seam,
    exactly where a real slack delivery exits.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    send_calls = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        send_calls.append({"chat_id": chat_id, "message": message})
        return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

    import gateway.platform_registry as reg
    import hermes_cli.plugins as hp

    entry = reg.platform_registry.get("slack")
    if entry is None:
        hp.discover_plugins()
        entry = reg.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

    state = {"send": send_calls, "marked": [], "saved": [], "finished": []}

    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-t"})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(
        s, "save_job_output",
        lambda jid, out: state["saved"].append(jid) or f"/tmp/{jid}.txt",
    )
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda *a, **kw: state["marked"].append((a, kw)) or True,
    )
    monkeypatch.setattr(
        s, "finish_execution",
        lambda *a, **kw: state["finished"].append((a, kw)),
    )
    # No durable incident store in play: never acked, no id.
    monkeypatch.setattr(
        s, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, None)
    )
    monkeypatch.setattr(s, "load_config", lambda: {})
    return state


def _failing_run_job(error="provider exploded"):
    def _fake(job, **_kw):
        return (False, "raw output", "", error)
    return _fake


def _succeeding_run_job(final="all good, here is the brief"):
    def _fake(job, **_kw):
        return (True, "raw output", final, None)
    return _fake


class TestFailureDeliverRouting:
    def test_failure_without_failure_deliver_goes_to_deliver_targets(
        self, run_env, monkeypatch
    ):
        """(a) Unset failure_deliver = today's behavior: failure summary to
        the job's deliver targets."""
        monkeypatch.setattr(s, "run_job", _failing_run_job())

        s.run_one_job({"id": "j1", "name": "scout", "deliver": "slack:D0MAIN"})

        assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]
        assert "failed" in run_env["send"][0]["message"].lower()

    def test_failure_deliver_local_is_silent_but_state_is_recorded(
        self, run_env, monkeypatch
    ):
        """(b) failure_deliver: local — no delivery leaves the process, but
        the run is still saved and marked failed."""
        monkeypatch.setattr(s, "run_job", _failing_run_job())

        s.run_one_job({
            "id": "j2", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "local",
        })

        assert run_env["send"] == []
        # State recording is untouched by the silence.
        assert run_env["saved"] == ["j2"]
        assert len(run_env["marked"]) == 1
        args, _kw = run_env["marked"][0]
        assert args[0] == "j2" and args[1] is False
        assert "provider exploded" in args[2]

    def test_failure_deliver_explicit_target_wins_over_deliver(
        self, run_env, monkeypatch
    ):
        """(c) failure_deliver set to a different target: the failure notice
        goes THERE, and nothing goes to the deliver target."""
        monkeypatch.setattr(s, "run_job", _failing_run_job())

        s.run_one_job({
            "id": "j3", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "slack:D0ALERTS",
        })

        assert [c["chat_id"] for c in run_env["send"]] == ["D0ALERTS"]
        assert "failed" in run_env["send"][0]["message"].lower()

    def test_success_ignores_failure_deliver(self, run_env, monkeypatch):
        """(d) Success output still goes to deliver — failure_deliver is
        never consulted on the success path."""
        monkeypatch.setattr(s, "run_job", _succeeding_run_job())

        ok = s.run_one_job({
            "id": "j4", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "slack:D0ALERTS",
        })

        assert ok is True
        assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]
        assert "all good, here is the brief" in run_env["send"][0]["message"]


class TestEscapedExceptionPath:
    """The scheduler-layer exception handler is the second failure-delivery
    site — it must honor failure_deliver identically."""

    def _raise_run_job(self, monkeypatch):
        monkeypatch.setattr(
            s, "run_job",
            lambda *_a, **_kw: (_ for _ in ()).throw(
                RuntimeError("cannot import name X")
            ),
        )

    def test_escaped_failure_honors_failure_deliver_target(
        self, run_env, monkeypatch
    ):
        self._raise_run_job(monkeypatch)

        ok = s.run_one_job({
            "id": "j5", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "slack:D0ALERTS",
        })

        assert ok is False
        assert [c["chat_id"] for c in run_env["send"]] == ["D0ALERTS"]

    def test_escaped_failure_with_failure_deliver_local_is_silent(
        self, run_env, monkeypatch
    ):
        self._raise_run_job(monkeypatch)

        ok = s.run_one_job({
            "id": "j6", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "local",
        })

        assert ok is False
        assert run_env["send"] == []
        # Failure is still recorded.
        assert len(run_env["marked"]) == 1
        args, _kw = run_env["marked"][0]
        assert args[1] is False and "cannot import name X" in args[2]


class TestResolutionGrammar:
    """(e) failure_deliver shares deliver's exact value grammar — the same
    normalization/expansion path, not a parallel one."""

    def test_for_failure_resolves_failure_deliver_value(self):
        job = {"deliver": "local", "failure_deliver": "slack:D0ALERTS"}
        targets = _resolve_delivery_targets(job, for_failure=True)
        assert [(t["platform"], t["chat_id"]) for t in targets] == [
            ("slack", "D0ALERTS")
        ]

    def test_for_failure_falls_back_to_deliver_when_unset(self):
        job = {"deliver": "slack:D0MAIN"}
        targets = _resolve_delivery_targets(job, for_failure=True)
        assert [(t["platform"], t["chat_id"]) for t in targets] == [
            ("slack", "D0MAIN")
        ]

    def test_success_resolution_never_reads_failure_deliver(self):
        job = {"deliver": "slack:D0MAIN", "failure_deliver": "slack:D0ALERTS"}
        targets = _resolve_delivery_targets(job)
        assert [(t["platform"], t["chat_id"]) for t in targets] == [
            ("slack", "D0MAIN")
        ]

    def test_local_yields_zero_failure_targets(self):
        job = {"deliver": "slack:D0MAIN", "failure_deliver": "local"}
        assert _resolve_delivery_targets(job, for_failure=True) == []

    def test_comma_list_and_thread_grammar(self):
        """The comma-combine + platform:chat:thread forms deliver's grammar
        supports work identically for failure_deliver."""
        job = {
            "deliver": "local",
            "failure_deliver": "slack:D0ALERTS,telegram:-1001:17",
        }
        targets = _resolve_delivery_targets(job, for_failure=True)
        assert [(t["platform"], t["chat_id"], t.get("thread_id")) for t in targets] == [
            ("slack", "D0ALERTS", None),
            ("telegram", "-1001", "17"),
        ]

    def test_legacy_list_value_is_flattened_like_deliver(self):
        """Same list/tuple tolerance _normalize_deliver_value grants deliver."""
        job = {"deliver": "local", "failure_deliver": ["slack:D0ALERTS"]}
        targets = _resolve_delivery_targets(job, for_failure=True)
        assert [(t["platform"], t["chat_id"]) for t in targets] == [
            ("slack", "D0ALERTS")
        ]


class TestToolSurface:
    """cronjob(action=create/update) accepts failure_deliver with deliver's
    validation — reusing the same normalize/validate helpers."""

    def test_create_stores_failure_deliver(self, cron_env):
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job

        result = json.loads(cronjob(
            action="create",
            prompt="scan",
            schedule="every 1h",
            deliver="slack:D0MAIN",
            failure_deliver="local",
        ))
        assert result["success"] is True
        assert get_job(result["job_id"])["failure_deliver"] == "local"

    def test_create_without_failure_deliver_does_not_persist_the_key(self, cron_env):
        """Existing-job byte-identity: the field only exists when set."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job

        result = json.loads(cronjob(
            action="create", prompt="scan", schedule="every 1h",
        ))
        assert result["success"] is True
        assert "failure_deliver" not in get_job(result["job_id"])

    def test_create_flattens_list_value_like_deliver(self, cron_env):
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job

        result = json.loads(cronjob(
            action="create",
            prompt="scan",
            schedule="every 1h",
            failure_deliver=["slack", "telegram"],
        ))
        assert result["success"] is True
        assert get_job(result["job_id"])["failure_deliver"] == "slack,telegram"

    def test_create_rejects_bad_bot_chat_profile_same_as_deliver(self, cron_env):
        from tools.cronjob_tools import cronjob

        via_failure = json.loads(cronjob(
            action="create", prompt="scan", schedule="every 1h",
            failure_deliver="bot-chat:no-such-profile-xyz",
        ))
        via_deliver = json.loads(cronjob(
            action="create", prompt="scan", schedule="every 1h",
            deliver="bot-chat:no-such-profile-xyz",
        ))
        assert via_failure["success"] is False
        assert via_deliver["success"] is False
        # Same validator, same message.
        assert via_failure["error"] == via_deliver["error"]

    def test_update_sets_and_clears_failure_deliver(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob

        job = create_job(prompt="scan", schedule="every 1h")
        result = json.loads(cronjob(
            action="update", job_id=job["id"], failure_deliver="slack:D0ALERTS",
        ))
        assert result["success"] is True
        assert get_job(job["id"])["failure_deliver"] == "slack:D0ALERTS"

        # '' clears — job falls back to deliver on failures again.
        result = json.loads(cronjob(
            action="update", job_id=job["id"], failure_deliver="",
        ))
        assert result["success"] is True
        assert not get_job(job["id"]).get("failure_deliver")


class TestOutcomeBookkeeping:
    """Review finding B1 (NS-788): delivery bookkeeping — outcome
    classification, unresolved-origin, incident 'alerted' marking — must
    read the SAME lane the notice was actually routed through, or the
    execution history and incident store record lies (silenced failures
    logged 'delivered'; delivered failures logged 'not_configured')."""

    @staticmethod
    def _outcome(state):
        assert state["finished"], "finish_execution never called"
        _a, kw = state["finished"][-1]
        return kw.get("delivery_outcome")

    def test_fd_local_failure_records_suppressed_not_delivered(
        self, run_env, monkeypatch
    ):
        alerted = []
        monkeypatch.setattr(s, "_mark_incident_alerted", alerted.append)
        monkeypatch.setattr(s, "run_job", _failing_run_job())

        s.run_one_job({
            "id": "b1a", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "local",
        })

        assert run_env["send"] == []
        assert self._outcome(run_env) == "suppressed"
        assert alerted == [], "silenced failure must NOT mark incident alerted"

    def test_fd_explicit_target_failure_records_delivered(
        self, run_env, monkeypatch
    ):
        """deliver=origin (unresolvable) + failure_deliver=explicit target:
        the notice IS delivered — outcome must say so, not 'not_configured'."""
        alerted = []
        monkeypatch.setattr(s, "_mark_incident_alerted", alerted.append)
        monkeypatch.setattr(
            s, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, "inc-b1")
        )
        monkeypatch.setattr(s, "run_job", _failing_run_job())

        s.run_one_job({
            "id": "b1b", "name": "scout",
            "deliver": "origin", "failure_deliver": "slack:D0OPS",
        })

        assert [c["chat_id"] for c in run_env["send"]] == ["D0OPS"]
        assert self._outcome(run_env) == "delivered"
        assert alerted == ["inc-b1"], "delivered failure ping must mark incident alerted"

    def test_success_outcome_still_reads_deliver_lane(self, run_env, monkeypatch):
        """Success bookkeeping is untouched: fd set, success delivers to
        deliver and records 'delivered'."""
        monkeypatch.setattr(s, "run_job", _succeeding_run_job())

        s.run_one_job({
            "id": "b1c", "name": "scout",
            "deliver": "slack:D0MAIN", "failure_deliver": "local",
        })

        assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]
        assert self._outcome(run_env) == "delivered"


class TestPreflightAndDashboardLanes:
    """Follow-up (salvage): the failure lane is validated everywhere the
    deliver lane is — preflight config checks and the dashboard update
    normalizer — so a typo'd failure target is caught before a failure
    needs it."""

    def test_preflight_blocks_unknown_failure_platform(self, monkeypatch):
        """A bogus failure_deliver platform blocks at preflight, exactly
        like a bogus deliver platform would."""
        monkeypatch.setattr(sched_delivery, "_is_known_delivery_platform", lambda _p: False)
        err = sched_preflight._preflight_check_delivery({
            "id": "p1", "deliver": "local",
            "failure_deliver": "nonexistent-platform:C1",
        })
        assert err is not None and "not a known" in err

    def test_preflight_failure_deliver_local_adds_no_platforms(self):
        """failure_deliver: local adds nothing to check — a deliver=local
        job with suppressed failures stays zero-cost at preflight."""
        assert sched_preflight._preflight_check_delivery({
            "id": "p2", "deliver": "local", "failure_deliver": "local",
        }) is None

    def test_preflight_duplicate_lane_not_checked_twice(self, monkeypatch):
        """failure_deliver equal to deliver must not double-check (or
        double-report) the same platform."""
        seen = []

        def _known(p):
            seen.append(p)
            return False

        monkeypatch.setattr(sched_delivery, "_is_known_delivery_platform", _known)
        sched_preflight._preflight_check_delivery({
            "id": "p3", "deliver": "ghost:C1", "failure_deliver": "ghost:C1",
        })
        assert seen == ["ghost"]

    def test_dashboard_update_normalizes_failure_deliver(self, tmp_path):
        """The dashboard update lane normalizes failure_deliver like
        deliver: text stripped, empty clears (None) instead of
        coalescing to a target."""
        from hermes_cli.web_routers.cron import _normalize_dashboard_cron_updates

        out = _normalize_dashboard_cron_updates(
            {"failure_deliver": "  slack:D0ALERTS  "}, tmp_path
        )
        assert out["failure_deliver"] == "slack:D0ALERTS"

        cleared = _normalize_dashboard_cron_updates(
            {"failure_deliver": ""}, tmp_path
        )
        assert cleared["failure_deliver"] is None
