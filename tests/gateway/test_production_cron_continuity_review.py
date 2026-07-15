from __future__ import annotations

import copy
import hashlib
import json

import pytest

from gateway import production_cron_continuity_review as review


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _store(job: dict) -> bytes:
    return json.dumps({"jobs": [job]}, ensure_ascii=False).encode()


def _script_job(**overrides) -> dict:
    job = {
        "id": "fixture-script",
        "name": "Review fixture",
        "enabled": True,
        "no_agent": True,
        "prompt": "private prompt body must never be published",
        "script": "/srv/collector.py --label private-script-argument",
        "workdir": "/srv/private-working-directory",
        "deliver": "local",
        "origin": {
            "platform": "discord",
            "chat_id": "1500000000000000001",
            "thread_id": "1500000000000000001",
            "chat_name": "private channel title",
        },
        "provider": None,
        "model": None,
        "base_url": None,
        "enabled_toolsets": None,
        "repeat": {"completed": 7, "times": None},
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "state": "scheduled",
        "last_run_at": "2026-07-14T20:00:00Z",
        "next_run_at": "2026-07-14T20:05:00Z",
    }
    job.update(overrides)
    return job


def _synthetic_record(entry: review.ReviewDisposition, index: int) -> dict:
    schedule = {"kind": "interval", "minutes": index + 1}
    is_collector = entry.disposition == review.DISPOSITION_COLLECTOR
    is_agent = entry.disposition in {
        review.DISPOSITION_AGENT,
        review.DISPOSITION_KEEP,
    }
    return {
        "index": index,
        "job_id": entry.job_id,
        "name": f"reviewed-{entry.job_id}",
        "record_sha256": _sha(f"record:{entry.job_id}".encode()),
        "definition_sha256": entry.expected_definition_sha256,
        "validation_code": entry.expected_validation_code,
        "schedule": schedule,
        "schedule_sha256": _sha(_canonical(schedule)),
        "repeat_times": None,
        "no_agent": is_collector,
        "prompt_sha256": _sha(f"prompt:{entry.job_id}".encode()),
        "script_present": is_collector,
        "script_sha256": (
            _sha(f"script:{entry.job_id}".encode()) if is_collector else None
        ),
        "script_basename": "collector.py" if is_collector else None,
        "workdir_present": entry.job_id == "fecd0675f91e",
        "workdir_sha256": (
            _sha(b"/local/workdir") if entry.job_id == "fecd0675f91e" else None
        ),
        "deliver": "origin" if is_agent else "local",
        "origin_binding_sha256": (
            _sha(f"origin:{entry.job_id}".encode()) if is_agent else None
        ),
        "provider": review.PRIMARY_PROVIDER if is_agent else None,
        "model": review.PRIMARY_MODEL if is_agent else None,
        "base_url_present": False,
        "enabled_toolsets": None,
    }


def _synthetic_observation() -> dict:
    records = [
        _synthetic_record(entry, index)
        for index, entry in enumerate(review.REVIEW_CATALOG)
    ]
    unsigned = {
        "schema": review.OBSERVATION_SCHEMA,
        "source_store_sha256": "a" * 64,
        "source_job_count": 40,
        "enabled_count": len(records),
        "records": records,
        "prompt_or_script_content_recorded": False,
        "job_executed": False,
    }
    return {
        **unsigned,
        "observation_sha256": _sha(_canonical(unsigned)),
    }


def test_observation_is_redaction_safe_and_uses_static_definition_identity() -> None:
    first_job = _script_job()
    second_job = {
        **first_job,
        "repeat": {"completed": 99, "times": None},
        "state": "running",
        "last_run_at": "2026-07-14T21:00:00Z",
        "next_run_at": "2026-07-14T21:05:00Z",
        "last_delivery_status": "sent",
        "last_delivery_confirmed_at": "2026-07-14T21:00:01Z",
    }
    first = review.observe_enabled_jobs_bytes(_store(first_job))
    second = review.observe_enabled_jobs_bytes(_store(second_job))
    encoded = json.dumps(first, ensure_ascii=False)

    assert first["records"][0]["definition_sha256"] == (
        second["records"][0]["definition_sha256"]
    )
    assert first["records"][0]["record_sha256"] != (
        second["records"][0]["record_sha256"]
    )
    assert first["records"][0]["script_basename"] == "collector.py"
    assert first["records"][0]["validation_code"] == (
        "production_cron_local_script_forbidden"
    )
    assert first["prompt_or_script_content_recorded"] is False
    assert first["job_executed"] is False
    for forbidden in (
        "private prompt body",
        "private-script-argument",
        "/srv/private-working-directory",
        "private channel title",
    ):
        assert forbidden not in encoded

    changed = review.observe_enabled_jobs_bytes(
        _store({**first_job, "prompt": "a genuinely changed prompt"})
    )
    assert changed["records"][0]["definition_sha256"] != (
        first["records"][0]["definition_sha256"]
    )


def test_observation_rejects_credential_shaped_job_fields_without_echo() -> None:
    with pytest.raises(
        review.ProductionCronContinuityReviewError,
        match="sensitive_field_forbidden",
    ):
        review.observe_enabled_jobs_bytes(
            _store({**_script_job(), "api_key": "must-not-appear"})
        )


def test_catalog_accounts_for_every_enabled_record_without_blanket_inert() -> None:
    plan = review.build_owner_review_plan(_synthetic_observation())

    assert plan["enabled_count"] == 28
    assert plan["incompatible_count"] == 27
    assert plan["disposition_counts"] == {
        review.DISPOSITION_KEEP: 1,
        review.DISPOSITION_AGENT: 5,
        review.DISPOSITION_COLLECTOR: 16,
        review.DISPOSITION_BLOCK: 6,
    }
    assert plan["owner_semantic_decision_job_ids"] == [
        "fecd0675f91e",
    ]
    assert plan["objectively_inferable_deep_review_job_ids"] == [
        "27ab4a64f8ad",
        "2b2035630202",
        "2a9f6be53fec",
        "90857403360d",
        "e62f55ca93ca",
    ]
    assert plan["retire_stale_count"] == 0
    assert plan["catalog_complete"] is True
    assert plan["owner_review_complete"] is False
    assert plan["cutover_executable"] is False
    assert plan["blanket_inert_migration_allowed"] is False


def test_deep_review_keeps_only_genuine_business_choice_with_owner() -> None:
    plan = review.build_owner_review_plan(_synthetic_observation())
    rows = {
        row["job_id"]: row
        for row in plan["records"]
        if row["deep_review"] is not None
    }

    assert set(rows) == {
        "fecd0675f91e",
        "27ab4a64f8ad",
        "2b2035630202",
        "2a9f6be53fec",
        "90857403360d",
        "e62f55ca93ca",
    }
    assert rows["fecd0675f91e"]["owner_semantic_decision_required"] is True
    assert rows["fecd0675f91e"]["deep_review"]["safest_option_kind"] == (
        "owner_choice"
    )
    for job_id in set(rows) - {"fecd0675f91e"}:
        assert rows[job_id]["owner_semantic_decision_required"] is False
        assert rows[job_id]["objectively_inferable"] is True
        assert rows[job_id]["inferred_target_id"] is not None

    assert rows["2a9f6be53fec"]["semantic_logic_outside_model_observed"] is True
    assert rows["90857403360d"]["semantic_logic_outside_model_observed"] is True
    assert rows["e62f55ca93ca"]["semantic_logic_outside_model_observed"] is False
    assert (
        "guild_connector_send_and_readback_receipt_commits_checkpoint"
        in rows["e62f55ca93ca"]["deep_review"]["safest_option_codes"]
    )


def test_business_followups_keep_primary_model_and_use_authorized_guild_reader() -> None:
    plan = review.build_owner_review_plan(_synthetic_observation())
    agent_rows = {
        row["job_id"]: row
        for row in plan["records"]
        if row["disposition"] == review.DISPOSITION_AGENT
    }

    assert set(agent_rows) == {
        "2c9a05136051",
        "e873367f6019",
        "cd778104fc92",
        "a1dfd5c2a7ab",
        "969248a7da45",
    }
    for row in agent_rows.values():
        target = row["target"]
        assert target["provider"] == "openai-codex"
        assert target["model"] == "gpt-5.6-sol"
        assert target["fallback_route_allowed"] is False
        assert target["enabled_toolsets"][0] == "discord_guild_read"
        assert "discord" not in target["enabled_toolsets"]
        assert target["deliver"] == "origin"
        assert target["origin_binding_sha256"] is not None
        assert target["authorized_guild_history_tool"] == "discord_guild_history"
        assert target["authorized_guild_history_binding_source"] == (
            "internal_reviewed_cron_context"
        )
        assert (
            target["authorized_guild_history_known_message_cursor_required"]
            is True
        )
        assert target["script"] is None
        assert target["workdir"] is None
        assert target["replacement_record_sha256"] is None


def test_collector_replacements_are_specific_and_not_yet_cutover_ready() -> None:
    plan = review.build_owner_review_plan(_synthetic_observation())
    collectors = [
        row
        for row in plan["records"]
        if row["disposition"] == review.DISPOSITION_COLLECTOR
    ]
    rail_ids = {row["target"]["systemd_rail_id"] for row in collectors}

    assert len(collectors) == len(rail_ids) == 16
    assert all(row["target"]["source_script_sha256"] for row in collectors)
    assert all(
        row["target"]["model_or_provider_allowed"] is False
        for row in collectors
    )
    assert all(
        row["target"]["direct_discord_credential_allowed"] is False
        for row in collectors
    )
    assert all(
        row["target"]["package_manifest_sha256"] is None
        for row in collectors
    )


def test_unknown_or_drifted_record_fails_closed() -> None:
    observation = _synthetic_observation()
    observation["records"][0]["job_id"] = "unknown-production-job"
    unsigned = {
        key: value
        for key, value in observation.items()
        if key != "observation_sha256"
    }
    observation["observation_sha256"] = _sha(_canonical(unsigned))
    with pytest.raises(
        review.ProductionCronContinuityReviewError,
        match="catalog_not_exhaustive",
    ):
        review.build_owner_review_plan(observation)

    observation = _synthetic_observation()
    observation["records"][0]["definition_sha256"] = "f" * 64
    unsigned = {
        key: value
        for key, value in observation.items()
        if key != "observation_sha256"
    }
    observation["observation_sha256"] = _sha(_canonical(unsigned))
    with pytest.raises(
        review.ProductionCronContinuityReviewError,
        match="definition_drifted",
    ):
        review.build_owner_review_plan(observation)


def test_observation_shape_and_self_digest_are_exact() -> None:
    observation = _synthetic_observation()
    tampered = copy.deepcopy(observation)
    tampered["enabled_count"] = 27
    with pytest.raises(
        review.ProductionCronContinuityReviewError,
        match="observation_invalid",
    ):
        review.build_owner_review_plan(tampered)

    tampered = copy.deepcopy(observation)
    tampered["records"][0]["prompt_body"] = "forged"
    unsigned = {
        key: value
        for key, value in tampered.items()
        if key != "observation_sha256"
    }
    tampered["observation_sha256"] = _sha(_canonical(unsigned))
    with pytest.raises(
        review.ProductionCronContinuityReviewError,
        match="observation_invalid",
    ):
        review.build_owner_review_plan(tampered)
