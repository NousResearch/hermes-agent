"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: data-layer semantics
# ---------------------------------------------------------------------------


def test_create_job_no_agent_requires_script(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


def test_create_job_no_agent_stores_field(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "watchdog.sh"
    script_path.write_text("#!/bin/bash\necho hi\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["script"] == "watchdog.sh"
    # Prompt can be empty/None for no_agent jobs.
    assert job["prompt"] in {None, ""}


def test_create_job_default_is_not_no_agent(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="say hi", schedule="every 5m", deliver="local")
    assert job.get("no_agent") is False


def test_create_job_delivery_dedup_is_explicit_and_no_agent_only(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "watchdog.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")
    default_job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deliver="local",
    )
    assert default_job["deduplicate_delivery"] is False

    dedup_job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="local",
    )
    assert dedup_job["deduplicate_delivery"] is True

    with pytest.raises(ValueError, match="requires no_agent=True"):
        create_job(
            prompt="model task",
            schedule="every 5m",
            deduplicate_delivery=True,
            deliver="local",
        )


def test_update_job_roundtrips_no_agent_flag(hermes_env):
    from cron.jobs import create_job, update_job, get_job

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local")

    update_job(job["id"], {"no_agent": False})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False

    update_job(job["id"], {"no_agent": True})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool: API-layer validation
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_no_agent_without_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "no_agent=True requires a script" in result.get("error", "")


def test_cronjob_tool_create_no_agent_with_script_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["no_agent"] is True
    assert result["job"]["script"] == "alert.sh"


def test_cronjob_tool_roundtrips_delivery_dedup_and_rejects_model_job(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deduplicate_delivery=True,
            deliver="local",
        )
    )
    assert created["success"] is True
    assert created["job"]["deduplicate_delivery"] is True

    disabled = json.loads(
        cronjob(
            action="update",
            job_id=created["job_id"],
            deduplicate_delivery=False,
        )
    )
    assert disabled["success"] is True
    assert disabled["job"]["deduplicate_delivery"] is False

    rejected = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt="model task",
            no_agent=False,
            deduplicate_delivery=True,
            deliver="local",
        )
    )
    assert rejected["success"] is False
    assert "requires no_agent=True" in rejected["error"]


def test_cronjob_tool_update_toggles_no_agent(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    off = json.loads(cronjob(action="update", job_id=job_id, no_agent=False, prompt="run"))
    assert off["success"] is True
    assert off["job"].get("no_agent") in {False, None}

    on = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert on["success"] is True
    assert on["job"]["no_agent"] is True


def test_cronjob_tool_update_no_agent_without_script_errors(hermes_env):
    """Flipping no_agent=True on a job that has no script must fail."""
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(action="create", schedule="every 5m", prompt="do a thing", deliver="local")
    )
    job_id = created["job_id"]

    result = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert result.get("success") is False
    assert "without a script" in result.get("error", "")


def test_cronjob_tool_create_does_not_require_prompt_when_no_agent(hermes_env):
    """The 'prompt or skill required' rule is relaxed for no_agent jobs."""
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True


# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


def test_run_job_no_agent_empty_output_is_silent(hermes_env):
    """Empty stdout → SILENT_MARKER, which suppresses delivery downstream."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\n# nothing to say\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_wake_gate_is_silent(hermes_env):
    """wakeAgent=false gate in stdout triggers a silent run."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "gated.sh"
    script_path.write_text('#!/bin/bash\necho \'{"wakeAgent": false}\'\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="gated.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_script_failure_delivers_error(hermes_env):
    """Non-zero exit → success=False, error alert is the delivered message."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "broken.sh"
    script_path.write_text("#!/bin/bash\necho oops >&2\nexit 3\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="broken.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is False
    assert error is not None
    assert "oops" in final_response or "exited with code 3" in final_response
    assert "Cron watchdog" in final_response  # alert header


def test_run_job_no_agent_never_invokes_aiagent(hermes_env):
    """no_agent jobs must NOT import/construct the AIAgent."""
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )

    with patch("run_agent.AIAgent") as ai_mock:
        from cron.scheduler import run_job

        run_job(job)

    ai_mock.assert_not_called()


def _install_dedup_pipeline(monkeypatch, scheduler, *, content="same alert"):
    deliveries = []

    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", content, None),
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {
                "platform": "telegram",
                "chat_id": "raw-private-chat-id",
                "thread_id": "raw-private-thread-id",
            }
        ],
    )

    def deliver(job, output, adapters=None, loop=None, receipt_out=None):
        deliveries.append(output)
        receipt_out.update(
            {
                "confirmation": "confirmed",
                "dedup_holds_key": True,
                "message_id_hashes": ["sha256:" + ("a" * 64)],
                "attempt_counts": [2],
                "thread_fallback": False,
                "error_kind": None,
            }
        )
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    return deliveries


def test_no_agent_confirmed_delivery_advances_key_and_suppresses_identical(
    hermes_env, monkeypatch
):
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    deliveries = _install_dedup_pipeline(monkeypatch, scheduler)

    assert scheduler.run_one_job(job) is True
    after_first = get_job(job["id"])
    assert after_first["last_delivery_key"].startswith("sha256:")
    assert after_first["last_delivery_receipt"]["confirmation"] == "confirmed"
    assert after_first["last_delivery_receipt"]["dedup_holds_key"] is True

    assert scheduler.run_one_job(after_first) is True
    after_second = get_job(job["id"])
    assert deliveries == ["same alert"]
    assert after_second["last_delivery_receipt"]["confirmation"] == "suppressed"
    persisted = json.dumps(after_second["last_delivery_receipt"], sort_keys=True)
    assert "raw-private-chat-id" not in persisted
    assert "raw-private-thread-id" not in persisted


def test_no_agent_silence_clears_key_so_same_alert_can_recur(
    hermes_env, monkeypatch
):
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    deliveries = _install_dedup_pipeline(monkeypatch, scheduler)

    assert scheduler.run_one_job(job) is True
    active = get_job(job["id"])
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (
            True,
            "doc",
            scheduler.SILENT_MARKER,
            None,
        ),
    )
    assert scheduler.run_one_job(active) is True
    recovered = get_job(job["id"])
    assert recovered["last_delivery_key"] is None
    assert recovered["last_delivery_hold_key"] is None

    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", "same alert", None),
    )
    assert scheduler.run_one_job(recovered) is True
    assert deliveries == ["same alert", "same alert"]


def test_no_agent_dedup_toggle_clears_stale_keys_before_reenable(
    hermes_env, monkeypatch
):
    """A disabled interval must break the prior delivery-dedup sequence."""
    from cron.jobs import create_job, get_job, update_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    outputs = iter(["alert A", "alert B", "alert A"])
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (
            True,
            "doc",
            next(outputs),
            None,
        ),
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "telegram", "chat_id": "raw-private-chat-id"}],
    )
    deliveries = []

    def deliver(job, output, adapters=None, loop=None, receipt_out=None):
        deliveries.append(output)
        if receipt_out is not None:
            receipt_out.update(
                {
                    "confirmation": "confirmed",
                    "dedup_holds_key": True,
                    "message_id_hashes": [],
                    "attempt_counts": [1],
                    "thread_fallback": False,
                    "error_kind": None,
                }
            )
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    assert scheduler.run_one_job(job) is True
    after_a = get_job(job["id"])
    assert after_a["last_delivery_key"].startswith("sha256:")

    dedup_disabled = update_job(job["id"], {"deduplicate_delivery": False})
    assert scheduler.run_one_job(dedup_disabled) is True
    dedup_reenabled = update_job(job["id"], {"deduplicate_delivery": True})

    assert scheduler.run_one_job(dedup_reenabled) is True
    after_a_recurrence = get_job(job["id"])
    assert deliveries == ["alert A", "alert B", "alert A"]
    assert after_a_recurrence["last_delivery_receipt"]["confirmation"] == "confirmed"


@pytest.mark.parametrize("confirmation", ["partial", "assumed", "unconfirmed"])
def test_no_agent_newer_delivery_hold_replaces_obsolete_confirmed_key(
    hermes_env, monkeypatch, confirmation
):
    """A non-confirmed B hold must not keep obsolete confirmed A active."""
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    outputs = iter(["alert A", "alert B", "alert B", "alert A"])
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", next(outputs), None),
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "telegram", "chat_id": "raw-private-chat-id"}],
    )
    deliveries = []
    delivery_confirmations = iter(["confirmed", confirmation, "confirmed"])

    def deliver(job, output, adapters=None, loop=None, receipt_out=None):
        deliveries.append(output)
        outcome = next(delivery_confirmations)
        receipt_out.update(
            {
                "confirmation": outcome,
                "dedup_holds_key": True,
                "message_id_hashes": [],
                "attempt_counts": [1],
                "thread_fallback": False,
                "error_kind": None,
            }
        )
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    assert scheduler.run_one_job(job) is True
    after_a = get_job(job["id"])
    confirmed_a_key = after_a["last_delivery_key"]

    assert scheduler.run_one_job(after_a) is True
    after_b = get_job(job["id"])
    held_b_key = after_b["last_delivery_hold_key"]
    assert held_b_key != confirmed_a_key
    assert after_b["last_delivery_key"] is None

    assert scheduler.run_one_job(after_b) is True
    after_b_repeat = get_job(job["id"])
    assert after_b_repeat["last_delivery_receipt"]["confirmation"] == "suppressed"
    assert after_b_repeat["last_delivery_hold_key"] == held_b_key

    assert scheduler.run_one_job(after_b_repeat) is True
    after_a_recurrence = get_job(job["id"])
    assert deliveries == ["alert A", "alert B", "alert A"]
    assert after_a_recurrence["last_delivery_receipt"]["confirmation"] == "confirmed"
    assert after_a_recurrence["last_delivery_key"] == confirmed_a_key


@pytest.mark.parametrize(
    ("confirmation", "holds_key", "expected_delivery_calls", "expected_last_key"),
    [
        ("failed", False, 2, None),
        ("assumed", True, 1, None),
    ],
)
def test_no_agent_failed_retries_but_assumed_suppresses_without_confirmed_key(
    hermes_env,
    monkeypatch,
    confirmation,
    holds_key,
    expected_delivery_calls,
    expected_last_key,
):
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", "same alert", None),
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {"platform": "telegram", "chat_id": "raw-private-chat-id"}
        ],
    )
    deliveries = []

    def deliver(job, output, adapters=None, loop=None, receipt_out=None):
        deliveries.append(output)
        receipt_out.update(
            {
                "confirmation": confirmation,
                "dedup_holds_key": holds_key,
                "message_id_hashes": [],
                "attempt_counts": [1],
                "thread_fallback": False,
                "error_kind": "delivery_failed" if confirmation == "failed" else None,
            }
        )
        return "network failure" if confirmation == "failed" else None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    assert scheduler.run_one_job(job) is True
    first = get_job(job["id"])
    assert first.get("last_delivery_key") == expected_last_key
    assert scheduler.run_one_job(first) is True
    second = get_job(job["id"])
    assert len(deliveries) == expected_delivery_calls
    if confirmation == "assumed":
        assert second["last_delivery_receipt"]["confirmation"] == "suppressed"
        assert second["last_delivery_receipt"]["dedup_holds_key"] is True
    else:
        assert second["last_delivery_receipt"]["confirmation"] == "failed"
        assert second["last_delivery_receipt"]["dedup_holds_key"] is False


def test_no_agent_silent_local_and_multi_target_receipt_states(
    hermes_env, monkeypatch
):
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="local",
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (
            True,
            "doc",
            scheduler.SILENT_MARKER,
            None,
        ),
    )
    assert scheduler.run_one_job(job) is True
    assert get_job(job["id"])["last_delivery_receipt"]["confirmation"] == "silent"

    local = get_job(job["id"])
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", "alert", None),
    )
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda job: [])
    assert scheduler.run_one_job(local) is True
    assert get_job(job["id"])["last_delivery_receipt"]["confirmation"] == "local"

    multi = get_job(job["id"])
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {"platform": "telegram", "chat_id": "one"},
            {"platform": "telegram", "chat_id": "two"},
        ],
    )
    delivered = []

    def multi_deliver(job, output, adapters=None, loop=None, receipt_out=None):
        delivered.append(output)
        receipt_out.update(
            {
                "confirmation": "ineligible",
                "dedup_holds_key": False,
                "message_id_hashes": [],
                "attempt_counts": [],
                "thread_fallback": False,
                "error_kind": None,
            }
        )
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", multi_deliver)
    assert scheduler.run_one_job(multi) is True
    assert delivered == ["alert"]
    assert get_job(job["id"])["last_delivery_receipt"]["confirmation"] == "ineligible"


def test_deliver_result_receipt_hashes_ids_and_omits_raw_destination(
    hermes_env,
):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: pconfig}
    receipt = {}
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {
            "platform": "telegram",
            "chat_id": "raw-private-chat-id",
            "thread_id": "raw-private-thread-id",
        },
    }

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config",
        return_value={"cron": {"wrap_response": False}},
    ), patch(
        "tools.send_message_tool._send_to_platform",
        new=AsyncMock(
            return_value={"success": True, "message_id": "raw-private-message-id"}
        ),
    ):
        assert _deliver_result(job, "alert", receipt_out=receipt) is None

    assert receipt["confirmation"] == "confirmed"
    assert receipt["dedup_holds_key"] is True
    assert receipt["message_id_hashes"][0].startswith("sha256:")
    persisted = json.dumps(receipt, sort_keys=True)
    assert "raw-private-chat-id" not in persisted
    assert "raw-private-thread-id" not in persisted
    assert "raw-private-message-id" not in persisted


def _live_delivery_fixture(result):
    from concurrent.futures import Future

    future = Future()
    future.set_result(result)

    def schedule(coro, _loop):
        coro.close()
        return future

    return schedule


def test_partial_live_delivery_never_replays_full_payload_standalone(hermes_env):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform
    from gateway.platforms.base import SendResult

    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True
    receipt = {}
    partial = SendResult(
        success=False,
        message_id="private-prefix-id",
        error="partial_send after 1/2 chunks",
        raw_response={
            "partial_send": True,
            "message_ids": ["private-prefix-id"],
            "delivered_chunks": 1,
            "total_chunks": 2,
        },
        retryable=False,
    )
    standalone = AsyncMock(return_value={"success": True, "message_id": "duplicate"})
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=_live_delivery_fixture(partial),
        ),
        patch("tools.send_message_tool._send_to_platform", new=standalone),
    ):
        error = _deliver_result(
            job,
            "A" * 5000,
            adapters={Platform.TELEGRAM: MagicMock()},
            loop=loop,
            receipt_out=receipt,
        )

    standalone.assert_not_awaited()
    assert error and "partial" in error
    assert receipt["confirmation"] == "partial"
    assert receipt["dedup_holds_key"] is True
    assert receipt["message_id_hashes"][0].startswith("sha256:")


def test_filtered_live_delivery_is_not_confirmed_or_replayed(hermes_env):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform

    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True
    receipt = {}
    standalone = AsyncMock(return_value={"success": True})
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=_live_delivery_fixture(
                {"success": True, "delivered": False, "reason": "filtered"}
            ),
        ),
        patch("tools.send_message_tool._send_to_platform", new=standalone),
    ):
        assert (
            _deliver_result(
                job,
                "filtered narration",
                adapters={Platform.TELEGRAM: MagicMock()},
                loop=loop,
                receipt_out=receipt,
            )
            is None
        )

    standalone.assert_not_awaited()
    assert receipt["confirmation"] == "filtered"
    assert receipt["dedup_holds_key"] is False


@pytest.mark.parametrize(
    ("standalone_result", "confirmation", "holds_key", "has_error"),
    [
        ({"success": True, "message_id": "m1"}, "confirmed", True, False),
        ({"success": False}, "failed", False, True),
        (
            {"success": False, "error": "later chunk", "message_id": "m1"},
            "partial",
            True,
            True,
        ),
        ({"error": "network"}, "failed", False, True),
        (None, "unconfirmed", True, True),
        (
            {"success": True, "message_id": "m1", "warnings": ["media failed"]},
            "partial",
            False,
            True,
        ),
        ({"success": True, "delivered": False}, "filtered", False, False),
    ],
)
def test_standalone_receipt_requires_explicit_whole_payload_confirmation(
    hermes_env,
    standalone_result,
    confirmation,
    holds_key,
    has_error,
):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform

    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    receipt = {}
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "tools.send_message_tool._send_to_platform",
            new=AsyncMock(return_value=standalone_result),
        ),
    ):
        error = _deliver_result(job, "alert", receipt_out=receipt)

    assert bool(error) is has_error
    assert receipt["confirmation"] == confirmation
    assert receipt["dedup_holds_key"] is holds_key


def test_error_only_standalone_failure_is_retried_next_run(
    hermes_env,
    monkeypatch,
):
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler
    from gateway.config import Platform

    (hermes_env / "scripts" / "alert.sh").write_text("echo alert\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deduplicate_delivery=True,
        deliver="telegram:raw-private-chat-id",
    )
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, defer_agent_teardown=None: (True, "doc", "same alert", None),
    )
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda job_id, output: f"/tmp/{job_id}.md"
    )
    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    sender = AsyncMock(
        side_effect=[
            {"error": "network"},
            {"success": True, "message_id": "m2"},
        ]
    )

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", new=sender),
    ):
        assert scheduler.run_one_job(job) is True
        first = get_job(job["id"])
        assert first["last_delivery_receipt"]["confirmation"] == "failed"
        assert first["last_delivery_hold_key"] is None

        assert scheduler.run_one_job(first) is True
        second = get_job(job["id"])

    assert sender.await_count == 2
    assert second["last_delivery_receipt"]["confirmation"] == "confirmed"


def _live_delivery_sequence(*results):
    from concurrent.futures import Future

    pending = iter(results)

    def schedule(coro, _loop):
        coro.close()
        result = next(pending)
        if hasattr(result, "result") and hasattr(result, "cancel"):
            return result
        future = Future()
        future.set_result(result)
        return future

    return schedule


def test_live_media_failure_downgrades_whole_payload_receipt(
    hermes_env,
    monkeypatch,
):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform
    from gateway.platforms.base import SendResult

    media_path = hermes_env / "media" / "report.pdf"
    media_path.parent.mkdir()
    media_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (hermes_env,),
    )
    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    adapter = MagicMock()
    adapter.send_document = AsyncMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    receipt = {}
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=_live_delivery_sequence(
                SendResult(success=True, message_id="text-id"),
                SendResult(success=False, error="media failed"),
            ),
        ),
    ):
        error = _deliver_result(
            job,
            f"alert\nMEDIA:{media_path}",
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
            receipt_out=receipt,
        )

    assert error and "media" in error
    assert receipt["confirmation"] == "partial"
    assert receipt["dedup_holds_key"] is False
    assert receipt["error_kind"] == "media_delivery_incomplete"


def test_live_text_timeout_with_media_is_not_whole_payload_confirmation(
    hermes_env,
    monkeypatch,
):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform

    media_path = hermes_env / "media" / "report.pdf"
    media_path.parent.mkdir()
    media_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (hermes_env,),
    )
    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True
    timed_out_future = MagicMock()
    timed_out_future.result.side_effect = TimeoutError
    timed_out_future.cancel.return_value = False
    receipt = {}
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "agent.async_utils.safe_schedule_threadsafe",
            return_value=timed_out_future,
        ),
    ):
        error = _deliver_result(
            job,
            f"alert\nMEDIA:{media_path}",
            adapters={Platform.TELEGRAM: MagicMock()},
            loop=loop,
            receipt_out=receipt,
        )

    assert error and "media" in error
    assert receipt["confirmation"] == "partial"
    assert receipt["dedup_holds_key"] is False
    assert receipt["error_kind"] == "media_skipped_after_timeout"


def test_rejected_media_path_downgrades_whole_payload_receipt(hermes_env):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform
    from gateway.platforms.base import SendResult

    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True
    receipt = {}
    missing_path = "/definitely/missing/private-report.pdf"
    job = {
        "id": "watchdog",
        "name": "watchdog",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "private-chat"},
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=_live_delivery_sequence(
                SendResult(success=True, message_id="text-id"),
            ),
        ),
    ):
        error = _deliver_result(
            job,
            f"alert\nMEDIA:{missing_path}",
            adapters={Platform.TELEGRAM: MagicMock()},
            loop=loop,
            receipt_out=receipt,
        )

    assert error and "safe-path" in error
    assert missing_path not in error
    assert receipt["confirmation"] == "partial"
    assert receipt["dedup_holds_key"] is False
    assert receipt["error_kind"] == "media_path_rejected"


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_shell_script_runs_via_bash(hermes_env):
    """.sh files should execute under /bin/bash even without a shebang line."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "shelly.sh"
    # No shebang — relies on the interpreter-by-extension rule.
    script_path.write_text('echo "shell: $BASH_VERSION" | head -c 7\n')

    ok, output = _run_job_script("shelly.sh")
    assert ok is True
    assert output.startswith("shell:")


def test_run_job_script_bash_extension_also_runs_via_bash(hermes_env):
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "thing.bash"
    script_path.write_text('printf "via bash\\n"\n')

    ok, output = _run_job_script("thing.bash")
    assert ok is True
    assert output == "via bash"


def test_run_job_script_python_still_runs_via_python(hermes_env):
    """Regression: .py files must keep running via sys.executable."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "py.py"
    script_path.write_text("import sys\nprint(f'python {sys.version_info.major}')\n")

    ok, output = _run_job_script("py.py")
    assert ok is True
    assert output.startswith("python ")


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


def test_run_job_script_without_workdir_keeps_scripts_directory_cwd(hermes_env):
    from cron.scheduler import _run_job_script

    scripts_dir = hermes_env / "scripts"
    (scripts_dir / "marker.txt").write_text("scripts marker\n")
    (scripts_dir / "default-cwd.py").write_text(
        "from pathlib import Path\n"
        "print(f'{Path.cwd()}|{Path(\"marker.txt\").read_text().strip()}')\n"
    )

    ok, output = _run_job_script("default-cwd.py")

    assert ok is True
    assert output == f"{scripts_dir}|scripts marker"


def test_run_job_no_agent_uses_workdir_for_pwd_and_relative_files(hermes_env, tmp_path):
    from cron.scheduler import run_job

    workdir = tmp_path / "project"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("project marker\n")
    (hermes_env / "scripts" / "relative.py").write_text(
        "from pathlib import Path\n"
        "print(f'{Path.cwd()}|{Path(\"marker.txt\").read_text().strip()}')\n"
    )
    job = {
        "id": "relative-workdir",
        "name": "relative workdir",
        "script": "relative.py",
        "no_agent": True,
        "workdir": str(workdir),
    }

    success, _doc, output, error = run_job(job)

    assert success is True
    assert error is None
    assert output == f"{workdir}|project marker"


def test_run_job_no_agent_invalid_workdir_fails_clearly(hermes_env, tmp_path):
    from cron.scheduler import run_job

    missing = tmp_path / "missing-project"
    (hermes_env / "scripts" / "relative.py").write_text("print('should not run')\n")
    job = {
        "id": "missing-workdir",
        "name": "missing workdir",
        "script": "relative.py",
        "no_agent": True,
        "workdir": str(missing),
    }

    success, _doc, output, error = run_job(job)

    assert success is False
    assert error is not None
    assert "workdir" in error.lower()
    assert "does not exist" in error.lower()
    assert "should not run" not in output


def test_overlapping_no_agent_workdirs_never_change_process_cwd(hermes_env, tmp_path):
    from cron.scheduler import run_job

    initial_cwd = os.getcwd()
    workdirs = [tmp_path / "project-a", tmp_path / "project-b"]
    for index, workdir in enumerate(workdirs):
        workdir.mkdir()
        (workdir / "marker.txt").write_text(f"marker-{index}\n")
    (hermes_env / "scripts" / "overlap.py").write_text(
        "import time\n"
        "from pathlib import Path\n"
        "Path('started').write_text('ready')\n"
        "deadline = time.monotonic() + 3\n"
        "while not Path('release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "print(f'{Path.cwd()}|{Path(\"marker.txt\").read_text().strip()}')\n"
    )
    jobs = [
        {
            "id": f"overlap-{index}",
            "name": f"overlap {index}",
            "script": "overlap.py",
            "no_agent": True,
            "workdir": str(workdir),
        }
        for index, workdir in enumerate(workdirs)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        deadline = time.monotonic() + 2
        while not all((workdir / "started").exists() for workdir in workdirs):
            assert os.getcwd() == initial_cwd
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert all((workdir / "started").exists() for workdir in workdirs)
        assert os.getcwd() == initial_cwd
        for workdir in workdirs:
            (workdir / "release").write_text("go\n")
        results = [future.result(timeout=5) for future in futures]

    assert os.getcwd() == initial_cwd
    assert all(result[0] is True and result[3] is None for result in results)
    assert {result[2] for result in results} == {
        f"{workdirs[0]}|marker-0",
        f"{workdirs[1]}|marker-1",
    }
