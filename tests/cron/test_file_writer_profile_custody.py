"""Private TEST-FIRST proposal for suggestions, scheduler, authority, and Bot file writers.

The child starts with Hermes home A while every command also binds the jobs ContextVar to B.  The
real EX lease is acquired at the source-owned root for each operation, so a destination assertion
cannot pass if the producer uses the wrong profile.  Authority save and remove remain separate
finite writer cases with immediate raw/typed readbacks.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.cron.writer_test_support import WriterChild, journal_bytes, profile_env


def _homes(tmp_path: Path) -> tuple[Path, Path]:
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


def _last_jsonl(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    return rows[-1]


def test_each_a_owned_json_writer_waits_for_a_ex_while_jobs_are_b(
    tmp_path: Path,
) -> None:
    home_a, home_b = _homes(tmp_path)
    store_b = {"store_home": str(home_b)}
    authority_job = "opaque-authority-job"
    authority_request = "opaque-authority-request"

    with WriterChild(home_a) as child:
        child.run_unfenced("seed_job", {**store_b, "name": "opaque B job"})

        # Group 4: add, status resolution, and cleanup each enter the actual A-owned writer.
        accepted = child.run_under_exclusive(
            "suggestion_add",
            {**store_b, "dedup_key": "opaque-accepted-suggestion"},
            lease_home=home_a,
        )
        suggestion_path = home_a / "cron" / "suggestions.json"
        accepted_rows = json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
        assert accepted_rows[0]["id"] == accepted["id"]
        assert accepted_rows[0]["status"] == "pending"
        child.run_under_exclusive(
            "suggestion_set_status",
            {**store_b, "suggestion_id": accepted["id"], "status": "accepted"},
            lease_home=home_a,
        )
        accepted_rows = json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
        assert accepted_rows[0]["id"] == accepted["id"]
        assert accepted_rows[0]["status"] == "accepted"
        cleared = child.run_under_exclusive(
            "suggestion_clear", store_b, lease_home=home_a
        )
        assert cleared["removed"] == 1
        assert accepted["id"] not in {
            row["id"]
            for row in json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
        }
        dismissed = child.run_under_exclusive(
            "suggestion_add",
            {**store_b, "dedup_key": "opaque-dismissed-suggestion"},
            lease_home=home_a,
        )
        dismissed_rows = json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
        assert dismissed_rows[0]["id"] == dismissed["id"]
        assert dismissed_rows[0]["status"] == "pending"
        child.run_under_exclusive(
            "suggestion_dismiss",
            {**store_b, "suggestion_id": dismissed["id"]},
            lease_home=home_a,
        )
        dismissed_rows = json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
        assert dismissed_rows[0]["id"] == dismissed["id"]
        assert dismissed_rows[0]["status"] == "dismissed"

        # Groups 6 and 7: the two scheduler JSONL append entrypoints are independent cases.
        child.run_under_exclusive(
            "scheduler_forced_release", store_b, lease_home=home_a
        )
        child.run_under_exclusive(
            "scheduler_usage_audit", store_b, lease_home=home_a
        )

        # Group 8: save and remove are independent finite authority writers.
        child.run_under_exclusive(
            "authority_save",
            {
                **store_b,
                "job_id": authority_job,
                "request_id": authority_request,
            },
            lease_home=home_a,
        )
        with profile_env(home_a):
            from cron import scheduler_authority

            authority_path = scheduler_authority.journal_path(authority_job, authority_request)
        saved_authority = json.loads(journal_bytes(home_a, authority_job, authority_request))
        assert saved_authority == {
            "params": {"job_id": authority_job, "request_id": authority_request},
            "receipt": None,
            "opaque": "opaque authority value",
        }
        child.run_under_exclusive(
            "authority_remove",
            {
                **store_b,
                "job_id": authority_job,
                "request_id": authority_request,
            },
            lease_home=home_a,
        )
        assert not authority_path.exists()

        # Group 9: malformed drain runs before defer so the drain's real queue scan cannot
        # consume the separate valid queued-record oracle.
        child.run_unfenced("seed_bot_malformed", store_b)
        child.run_under_exclusive(
            "bot_drain_malformed", store_b, lease_home=home_a
        )
        malformed_path = home_a / "cron" / "bot_chat_pending" / "opaque-malformed.json"
        malformed_now = json.loads(malformed_path.read_text(encoding="utf-8"))
        assert malformed_now["status"] == "ambiguous"
        assert "missing" in malformed_now["error"]
        child.run_under_exclusive(
            "bot_defer",
            {**store_b, "target_home": str(home_b)},
            lease_home=home_a,
        )
        pending_path = home_a / "cron" / "bot_chat_pending" / "opaque-bot-pending.json"
        pending_now = json.loads(pending_path.read_text(encoding="utf-8"))
        assert pending_now["id"] == "opaque-bot-pending"
        assert pending_now["status"] == "queued"
        assert pending_now["job"] == {"id": "opaque-bot-job", "name": "opaque bot job"}
        assert pending_now["content"] == "opaque bot pending content"
        assert pending_now["home"] == str(home_b)
        assert pending_now["for_failure"] is False
        assert isinstance(pending_now["sequence"], int)

    suggestions = json.loads(suggestion_path.read_text(encoding="utf-8"))["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["id"] == dismissed["id"]
    assert suggestions[0]["status"] == "dismissed"
    assert suggestions[0]["job_spec"]["prompt"] == "opaque suggested prompt"
    assert not (home_b / "cron" / "suggestions.json").exists()

    forced = _last_jsonl(home_a / "cron" / "inflight_forced_releases.jsonl")
    assert forced["job_id"] == "opaque-forced-job"
    assert forced["name"] == "opaque forced job"
    assert forced["age_seconds"] == 7.5
    assert forced["allowance_seconds"] == 60.0
    assert isinstance(forced["at"], str) and forced["at"]
    assert _last_jsonl(home_a / "cron" / "usage_audit.jsonl") == {
        "event": "opaque usage audit",
        "job_id": "opaque-audit-job",
        "result": {"status": "opaque"},
    }
    assert not (home_b / "cron" / "usage_audit.jsonl").exists()

    assert not authority_path.exists()

    pending = pending_path
    pending_record = json.loads(pending.read_text(encoding="utf-8"))
    assert pending_record["id"] == "opaque-bot-pending"
    assert pending_record["status"] == "queued"
    assert pending_record["job"] == {"id": "opaque-bot-job", "name": "opaque bot job"}
    assert pending_record["content"] == "opaque bot pending content"
    assert pending_record["home"] == str(home_b)
    assert pending_record["for_failure"] is False
    assert isinstance(pending_record["sequence"], int)

    malformed = json.loads(
        (home_a / "cron" / "bot_chat_pending" / "opaque-malformed.json").read_text(
            encoding="utf-8"
        )
    )
    assert malformed["status"] == "ambiguous"
    assert "missing" in malformed["error"]
    assert not (home_b / "cron" / "bot_chat_pending").exists()
    assert (home_b / "cron" / "jobs.json").is_file()
    assert not (home_a / "cron" / "jobs.json").exists()
