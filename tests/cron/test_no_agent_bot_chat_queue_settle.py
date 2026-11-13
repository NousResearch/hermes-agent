"""A no_agent cron job's Bot Chat send settles like an agent job's queued one.

The enqueue gate used to skip no_agent runs, so their Bot Chat sends never got a
queue-ledger row and the run stayed ``delivery_queued``/ledger ``queued`` forever
(#144694). They now ride the same queue → drain → write-back path, and the settle's
job-record fence no longer depends on the canonical completion that a no_agent run
never records.
"""
import pytest

from cron import executions, jobs, scheduler
from gateway.config import GatewayConfig


@pytest.fixture()
def bot_chat_mailbox(tmp_path, monkeypatch):
    """Stub the live Bot Chat owner: find_canonical_live_owner + deliver_to_live_owner."""
    from tools import bot_live_delivery as mailbox

    owner = {"profile_home": str(tmp_path), "session_id": "s", "lease_id": "l",
             "live_session_id": "ls"}
    admitted: list[str] = []
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)

    def fake_deliver(home, owner_, message, *, delivery_id, notification_category="result",
                     author=None):
        admitted.append(delivery_id)
        if isinstance(owner_.get("error"), str):
            raise RuntimeError(owner_["error"])
        return {"status": "queued", "message": message, "delivery_id": delivery_id}

    monkeypatch.setattr(mailbox, "deliver_to_live_owner", fake_deliver)
    return admitted


@pytest.fixture()
def script_run(monkeypatch):
    """no_agent script runs succeed with a fixed stdout, no subprocess."""
    monkeypatch.setattr(
        scheduler, "_run_job_script_with_claim_heartbeat",
        lambda job, script_path, workdir=None, cancel_event=None: (True, "watchdog ok"))


def _create_job():
    return jobs.create_job(
        prompt=None, schedule="every 1h", name="watchdog", script="probe.sh",
        no_agent=True, deliver="bot-chat")


def test_no_agent_bot_chat_send_settles_after_drain(tmp_path, monkeypatch, bot_chat_mailbox,
                                                    script_run):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())

    job = _create_job()
    assert scheduler.run_one_job(job) is True
    # The run hands the send to the durable queue; nothing is admitted inline.
    assert bot_chat_mailbox == []
    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "delivery_queued"
    row = executions.latest_execution(job["id"])
    assert row["delivery_outcome"] == "queued"

    assert scheduler.drain_delivery_queue({}, None) == 1
    assert len(bot_chat_mailbox) == 1
    saved = jobs.get_job(job["id"])
    row = executions.latest_execution(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
    assert row["delivery_outcome"] == "delivered"


def test_no_agent_bot_chat_send_failure_settles_failed(tmp_path, monkeypatch, script_run):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())
    from tools import bot_live_delivery as mailbox

    owner = {"profile_home": str(tmp_path), "session_id": "s", "lease_id": "l",
             "live_session_id": "ls", "error": "owner refused"}
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)

    def refusing_deliver(home, owner_, message, *, delivery_id, notification_category="result",
                         author=None):
        raise RuntimeError(owner_["error"])

    monkeypatch.setattr(mailbox, "deliver_to_live_owner", refusing_deliver)

    job = _create_job()
    assert scheduler.run_one_job(job) is True
    assert jobs.get_job(job["id"])["last_status"] == "delivery_queued"

    assert scheduler.drain_delivery_queue({}, None) == 1
    saved = jobs.get_job(job["id"])
    row = executions.latest_execution(job["id"])
    assert saved["last_status"] == "delivery_failed"
    assert "owner refused" in saved["last_delivery_error"]
    assert row["delivery_outcome"] == "failed"


def test_no_agent_settle_stays_fenced_on_the_latest_attempt(tmp_path, monkeypatch, script_run):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())
    from cron.delivery_outcome import settle_queued_delivery
    from cron.jobs import update_job
    from tools import bot_live_delivery as mailbox

    owner = {"profile_home": str(tmp_path), "session_id": "s", "lease_id": "l",
             "live_session_id": "ls", "error": "owner refused"}
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)

    def refusing_deliver(home, owner_, message, *, delivery_id, notification_category="result",
                         author=None):
        raise RuntimeError(owner_["error"])

    monkeypatch.setattr(mailbox, "deliver_to_live_owner", refusing_deliver)

    job = _create_job()
    assert scheduler.run_one_job(job) is True
    settled_execution = str(job["execution_id"])
    assert scheduler.drain_delivery_queue({}, None) == 1
    assert jobs.get_job(job["id"])["last_status"] == "delivery_failed"

    # A newer attempt is now the job's latest and its bookkeeping owns the record (say it
    # recovered): replaying the failed attempt's settle must not rewrite the job record.
    executions.create_execution(job["id"], source="direct")
    update_job(job["id"], {"last_status": "ok", "last_delivery_error": None})
    settle_queued_delivery(job["id"], settled_execution)
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
