"""A Bot Chat cron delivery whose target owner is not running is parked, never booked failed:
a later tick admits it to the owner once, under the same delivery id."""
from unittest.mock import Mock

from cron import scheduler_delivery as delivery
from tools import bot_live_delivery as mailbox


def test_bot_chat_target_without_live_owner_is_parked_and_admitted_later(tmp_path, monkeypatch):
    from cron import jobs
    from cron.bot_chat_legacy import drain_legacy_pending
    from gateway import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
    monkeypatch.setattr(delivery._sched, "load_config", dict)
    monkeypatch.setattr(config, "load_gateway_config", lambda: None)
    monkeypatch.setattr(jobs, "update_job", lambda key, values: None)
    (tmp_path / "state.db").write_text("")
    monkeypatch.setattr(mailbox, "find_canonical_live_owner",
                        Mock(side_effect=ValueError("profile authority is not ready")))
    job = dict(id="digest", name="Digest", execution_id="run-1", deliver="bot-chat")
    assert delivery._deliver_result(job, "the report") is None  # not a failed delivery
    drain_legacy_pending()  # owner still down: stays parked

    admitted = []

    def deliver(home, owner, message, *, delivery_id, notification_category="result"):
        admitted.append((delivery_id, message))
        return {"status": "queued", "message": message, "delivery_id": delivery_id}

    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", deliver)
    drain_legacy_pending()
    drain_legacy_pending()
    key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
    assert [(k, "the report" in m) for k, m in admitted] == [(key, True)]
