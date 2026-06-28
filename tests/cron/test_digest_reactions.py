import json
from pathlib import Path
from types import SimpleNamespace


def test_register_digest_reaction_resolves_single_source_response(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import (
        format_digest_detail_response,
        register_digest_delivery,
        resolve_digest_delivery,
    )

    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    source_output = source_dir / "2026-06-28_08-00-00.md"
    source_output.write_text(
        "# Cron Job: Source Job\n\n"
        "## Prompt\n"
        "internal collection prompt that should not be sent\n\n"
        "## Response\n"
        "**⚠️ Source finding**\n\n"
        "📌 **Befund**\n"
        "- actionable detail\n",
        encoding="utf-8",
    )

    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-job"],
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
        source_names={"source-job": "Source Job"},
        now=1000.0,
    )

    record = resolve_digest_delivery("!room:example.org", "$digest", now=1001.0)
    assert record is not None
    assert record["digest_job_id"] == "digest-job"
    assert record["sources"][0]["job_id"] == "source-job"

    text = format_digest_detail_response(record, source_index=0)
    assert "**🧾 Einzelbericht: Source Job**" in text
    assert "actionable detail" in text
    assert "internal collection prompt" not in text


def test_register_digest_reaction_ignores_unsafe_source_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response

    record = {
        "room_id": "!room:example.org",
        "event_id": "$digest",
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(tmp_path / ".." / "outside.md"),
            }
        ],
    }

    text = format_digest_detail_response(record, source_index=0)
    assert "detail output is no longer available" in text
    assert "outside.md" not in text


def test_latest_output_fallback_ignores_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response, register_digest_delivery, resolve_digest_delivery

    outside = tmp_path / "outside.md"
    outside.write_text("## Response\nSECRET OUTSIDE DETAIL", encoding="utf-8")
    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    (source_dir / "latest.md").symlink_to(outside)

    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-job"],
        source_names={"source-job": "Source Job"},
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert record["sources"][0]["output_path"] == ""
    text = format_digest_detail_response(record, source_index=0)
    assert "SECRET OUTSIDE DETAIL" not in text
    assert "detail output is no longer available" in text


def test_scheduler_registers_matrix_digest_metadata_only_after_confirmed_send(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=True, message_id="$digest")
    job = {"id": "digest-job", "name": "Morning Digest", "context_from": ["source-a", "source-b"]}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert [src["job_id"] for src in record["sources"]] == ["source-a", "source-b"]


def test_scheduler_does_not_register_failed_matrix_send_with_event_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import _registry_path
    from cron.scheduler import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=False, message_id="$digest")
    job = {"id": "digest-job", "name": "Morning Digest", "context_from": ["source-a"]}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
    )

    assert not _registry_path().exists()


def test_scheduler_does_not_register_non_digest_matrix_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import _registry_path
    from cron.scheduler import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=True, message_id="$normal")
    job = {"id": "normal-job", "name": "Normal Job"}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "normal-job" / "latest.md",
    )

    assert not _registry_path().exists()
