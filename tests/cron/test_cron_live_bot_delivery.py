"""Cron admission must not become a second writer or a completed-delivery claim."""
from pathlib import Path
from unittest.mock import Mock

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store_runtime_readiness import trap_state_db_opens
from cron import scheduler_delivery as delivery
from tools import bot_live_delivery as mailbox


def test_live_delivery_retry_keeps_receipt_across_owner_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = tmp_path / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(source))
    subprocess_run = Mock(side_effect=AssertionError("live owner must not spawn CLI"))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", subprocess_run)
    from hermes_cli.profiles import get_profile_dir

    for profile, home in [("", source), ("research", get_profile_dir("research"))]:
        owner = dict(profile_home=str(home.resolve()), session_id="bot", lease_id="lease",
                     live_session_id="live")
        discovery = Mock(return_value=owner)
        monkeypatch.setattr(mailbox, "find_canonical_live_owner", discovery)
        job = dict(id="digest", name="Digest", execution_id="first-run")
        pending = delivery._deliver_to_bot_chat(job, "payload", profile)
        assert pending and "queued" in pending
        records = list((home / "runtime/bot_live_delivery").glob("*.json"))
        assert len(records) == 1
        key = records[0].stem
        record = mailbox.read_delivery_result(home, key)
        assert record and record["message"].endswith("payload")
        discovery.side_effect = AssertionError("receipt must precede discovery")
        assert delivery._deliver_to_bot_chat(dict(job), "payload", profile) == pending
        mailbox.claim_pending_delivery(home, owner)
        mailbox.complete_delivery(home, key, status="ambiguous", error="owner died")
        outcome = delivery._deliver_to_bot_chat(dict(job), "payload", profile)
        assert outcome and "ambiguous" in outcome
        discovery.side_effect = None
        next_job = dict(job, execution_id="next-run")
        outcome = delivery._deliver_to_bot_chat(next_job, "payload", profile)
        assert outcome and "queued" in outcome
        assert len(list((home / "runtime/bot_live_delivery").glob("*.json"))) == 2
    subprocess_run.assert_not_called()


def test_result_records_pending_until_terminal_receipt(tmp_path, monkeypatch):
    from cron import jobs
    from gateway import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
    owner = dict(profile_home=str(tmp_path.resolve()), session_id="bot", lease_id="lease",
                 live_session_id="live")
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)
    monkeypatch.setattr(delivery._sched, "load_config", lambda: {})
    monkeypatch.setattr(config, "load_gateway_config", lambda: None)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", Mock(side_effect=AssertionError("CLI")))
    updates = []
    monkeypatch.setattr(jobs, "update_job", lambda key, values: updates.append(values))
    job = dict(id="digest", execution_id="run", deliver="bot-chat")
    error = delivery._deliver_result(job, "payload")
    assert error is None
    queued = updates[-1]["last_delivery_queued"]
    assert queued and next(iter(queued.values()))["status"] == "queued"
    assert delivery._sched._classify_delivery_outcome(
        delivery_error=error, delivery_queued=queued, should_deliver=True, unresolved_origin=False,
        normalized_deliver="bot-chat", incident_acked=False, success=True) == "queued"
    record = mailbox.claim_pending_delivery(tmp_path, owner)
    assert record is not None
    mailbox.complete_delivery(tmp_path, record["delivery_id"], status="settled", reply="done")
    assert delivery._deliver_result(job, "payload") is None
    assert updates[-1]["last_delivery_queued"] is None


def test_selected_postgresql_bot_chat_delivery_refuses_before_cli_or_mailbox(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    subprocess_run = Mock(side_effect=AssertionError("selected PG must not invoke the CLI fallback"))
    monkeypatch.setattr(delivery.subprocess, "run", subprocess_run)
    token = set_hermes_home_override(str(home))
    try:
        with trap_state_db_opens(home) as opens:
            result = delivery._deliver_to_bot_chat({"id": "digest", "name": "Digest"}, "payload", "")
    finally:
        reset_hermes_home_override(token)

    assert result is not None
    assert "unverified" in result
    assert "cron-session-transcript-lifecycle" in result
    assert opens == []
    assert not (home / "runtime" / mailbox.DELIVERY_DIR_NAME).exists()
    assert not (home / "state.db").exists()
    subprocess_run.assert_not_called()
