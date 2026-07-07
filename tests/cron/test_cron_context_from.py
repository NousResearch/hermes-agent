"""Tests for cron job context_from feature (issue #5439 Option C)."""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


class TestJobContextFromField:
    """Test that context_from is stored and retrieved correctly."""

    def test_create_job_with_context_from_string(self, cron_env):
        from cron.jobs import create_job, get_job

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize findings",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        assert job_b["context_from"] == [job_a["id"]]
        loaded = get_job(job_b["id"])
        assert loaded["context_from"] == [job_a["id"]]

    def test_create_job_with_context_from_list(self, cron_env):
        from cron.jobs import create_job

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Find weather", schedule="every 1h")
        job_c = create_job(
            prompt="Summarize everything",
            schedule="every 2h",
            context_from=[job_a["id"], job_b["id"]],
        )

        assert job_c["context_from"] == [job_a["id"], job_b["id"]]

    def test_create_job_without_context_from(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("context_from") is None

    def test_context_from_empty_string_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", context_from="")
        assert job.get("context_from") is None

    def test_context_from_empty_list_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", context_from=[])
        assert job.get("context_from") is None


class TestBuildJobPromptContextFrom:
    """Test that _build_job_prompt() injects context from referenced jobs."""

    def test_injects_latest_output(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")

        # Записываем output для job_a
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-04-22_10-00-00.md").write_text(
            "Today's top story: AI is everywhere.", encoding="utf-8"
        )

        job_b = create_job(
            prompt="Summarize the news",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "Today's top story: AI is everywhere." in prompt
        assert f"Output from job '{job_a['id']}'" in prompt

    def test_injects_output_from_custom_safe_job_id(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        source_job_id = "pmexec20260602"
        output_dir = OUTPUT_DIR / source_job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-03_09-33-53.md").write_text(
            "Morning PM plan context", encoding="utf-8"
        )

        job_b = create_job(
            prompt="Noon check-in",
            schedule="every 2h",
            context_from=source_job_id,
        )

        prompt = _build_job_prompt(job_b)
        assert "Morning PM plan context" in prompt
        assert "Output from job 'pmexec20260602'" in prompt

    def test_injects_final_response_section_from_saved_cron_doc(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Research", schedule="every 1h")
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-07_10-00-00.md").write_text(
            "# Cron Job: Research\n\n"
            "## Prompt\n\n"
            "Internal prompt instructions that should not become context.\n\n"
            "## Response\n\n"
            "Final answer: change the launch plan.",
            encoding="utf-8",
        )

        job_b = create_job(
            prompt="Use the prior answer",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "Final answer: change the launch plan." in prompt
        assert "Internal prompt instructions" not in prompt
        assert "Context extracted from the preceding job's final response." in prompt

    def test_trims_scheduler_runtime_from_extracted_response_context(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Research", schedule="every 1h")
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-07_10-30-00.md").write_text(
            "# Cron Job: Research\n\n"
            "## Prompt\n\n"
            "Internal prompt instructions.\n\n"
            "## Response\n\n"
            "Final answer only.\n\n"
            "## Runtime\n\n"
            "Elapsed seconds: 12.3\n",
            encoding="utf-8",
        )

        job_b = create_job(
            prompt="Use the prior answer",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "Final answer only." in prompt
        assert "Elapsed seconds" not in prompt
        assert "## Runtime" not in prompt

    def test_injects_last_response_section_when_transcript_has_multiple(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Research", schedule="every 1h")
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-07_11-00-00.md").write_text(
            "# Cron Job: Research\n\n"
            "## Response\n\n"
            "Earlier draft.\n\n"
            "## Response\n\n"
            "Final answer wins.",
            encoding="utf-8",
        )

        job_b = create_job(
            prompt="Use the prior answer",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "Final answer wins." in prompt
        assert "Earlier draft" not in prompt

    def test_skips_delivery_quality_gated_output(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Research", schedule="every 1h")
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-07_12-00-00.md").write_text(
            "# Cron Job: Research\n\n"
            "## Response\n\n"
            "This should not become context.\n\n"
            "## Delivery Quality Gate\n\n"
            "External delivery blocked: prompt tail leaked into response\n",
            encoding="utf-8",
        )

        job_b = create_job(
            prompt="Use only safe prior context",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "This should not become context." not in prompt
        assert "Use only safe prior context" in prompt

    def test_uses_most_recent_output(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt
        import time

        job_a = create_job(prompt="Find news", schedule="every 1h")
        output_dir = OUTPUT_DIR / job_a["id"]
        output_dir.mkdir(parents=True, exist_ok=True)

        old_file = output_dir / "2026-04-22_08-00-00.md"
        old_file.write_text("Old output", encoding="utf-8")
        time.sleep(0.01)
        new_file = output_dir / "2026-04-22_10-00-00.md"
        new_file.write_text("New output", encoding="utf-8")

        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)
        assert "New output" in prompt
        assert "Old output" not in prompt

    def test_graceful_when_no_output_yet(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"]
        )

        # job_a never ran — output dir does not exist
        # expect silent skip: no placeholder injected, base prompt intact
        prompt = _build_job_prompt(job_b)
        assert "no output" not in prompt.lower()
        assert "not found" not in prompt.lower()
        assert "Summarize" in prompt

    def test_injects_multiple_context_jobs(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Find weather", schedule="every 1h")

        for job, content in [(job_a, "News: AI boom"), (job_b, "Weather: Sunny")]:
            out_dir = OUTPUT_DIR / job["id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "2026-04-22_10-00-00.md").write_text(content, encoding="utf-8")

        job_c = create_job(
            prompt="Daily briefing",
            schedule="every 2h",
            context_from=[job_a["id"], job_b["id"]],
        )
        prompt = _build_job_prompt(job_c)
        assert "News: AI boom" in prompt
        assert "Weather: Sunny" in prompt

    def test_context_injected_before_prompt(self, cron_env):
        """Context should appear before the job's own prompt."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-04-22_10-00-00.md").write_text("Context data", encoding="utf-8")

        job_b = create_job(
            prompt="Process the data above",
            schedule="every 2h",
            context_from=job_a["id"],
        )
        prompt = _build_job_prompt(job_b)
        context_pos = prompt.find("Context data")
        prompt_pos = prompt.find("Process the data above")
        assert context_pos < prompt_pos

    def test_output_truncated_at_8k_chars(self, cron_env):
        """Output longer than 8000 chars should be truncated."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        big_output = "x" * 10000
        (out_dir / "2026-04-22_10-00-00.md").write_text(big_output, encoding="utf-8")

        job_b = create_job(
            prompt="Process", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)
        assert "truncated" in prompt
        assert "x" * 10000 not in prompt

    def test_graceful_when_file_deleted_between_listing_and_reading(self, cron_env):
        """Job should not crash if output file is deleted mid-read."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt
        from unittest.mock import patch

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-04-22_10-00-00.md").write_text("Some output", encoding="utf-8")

        job_b = create_job(
            prompt="Process", schedule="every 2h", context_from=job_a["id"]
        )

        # Simulate file deleted between glob() and read_text()
        original_read = Path.read_text
        def mock_read_text(self, *args, **kwargs):
            if self.suffix == ".md":
                raise FileNotFoundError("file deleted mid-read")
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read_text):
            prompt = _build_job_prompt(job_b)

        # Job should not crash, prompt should still contain the base prompt
        assert "Process" in prompt

    def test_graceful_when_permission_error(self, cron_env):
        """Job should not crash if output directory is not readable."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt
        from unittest.mock import patch

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-04-22_10-00-00.md").write_text("Some output", encoding="utf-8")

        job_b = create_job(
            prompt="Process", schedule="every 2h", context_from=job_a["id"]
        )

        # Simulate permission error on read
        original_read = Path.read_text
        def mock_read_text(self, *args, **kwargs):
            if self.suffix == ".md":
                raise PermissionError("permission denied")
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read_text):
            prompt = _build_job_prompt(job_b)

        # Job should not crash, prompt should still contain the base prompt
        assert "Process" in prompt

    def test_invalid_job_id_skipped(self, cron_env):
        """context_from with path traversal job_id should be skipped."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Process", schedule="every 2h")
        # Manually inject invalid context_from (simulating tampered jobs.json)
        job["context_from"] = ["../../../etc/passwd"]
        prompt = _build_job_prompt(job)
        # Should not crash and should not inject anything malicious
        assert "Process" in prompt
        assert "etc/passwd" not in prompt

    def test_invalid_job_id_log_includes_job_origin(self, cron_env, caplog):
        """Invalid stored context_from refs log job/source provenance."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Process",
            schedule="every 2h",
            name="suspicious-chain",
            origin={
                "platform": "api_server",
                "chat_id": "api",
                "source_ip": "203.0.113.10",
                "forwarded_for": "198.51.100.7",
            },
        )
        job["context_from"] = ["../../../etc/passwd"]

        caplog.set_level(logging.WARNING, logger="cron.scheduler")
        prompt = _build_job_prompt(job)

        assert "Process" in prompt
        message = caplog.text
        assert "context_from: skipping invalid job_id" in message
        assert job["id"] in message
        assert "suspicious-chain" in message
        assert "203.0.113.10" in message
        assert "198.51.100.7" in message



class TestUpdateContextFrom:
    """Verify the cronjob tool's `update` action wires context_from through.

    Without this, the create-path stores the field but users can never modify
    or clear it via the tool (schema promises "pass an empty array to clear").
    """

    def test_update_adds_context_from_to_existing_job(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Summarize", schedule="every 2h")
        assert job_b.get("context_from") is None

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=job_a["id"],
        ))
        assert result["success"] is True

        reloaded = get_job(job_b["id"])
        assert reloaded["context_from"] == [job_a["id"]]

    def test_update_changes_context_from_reference(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_a2 = create_job(prompt="Find weather", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )
        assert job_b["context_from"] == [job_a["id"]]

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=[job_a2["id"]],
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] == [job_a2["id"]]

    def test_update_clears_context_from_with_empty_list(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )
        assert get_job(job_b["id"])["context_from"] == [job_a["id"]]

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=[],
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] is None

    def test_update_clears_context_from_with_empty_string(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from="",
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] is None

    def test_update_rejects_unknown_job_reference(self, cron_env):
        from cron.jobs import create_job
        from tools.cronjob_tools import cronjob
        import json

        job_b = create_job(prompt="Summarize", schedule="every 2h")

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=["deadbeef0000"],
        ))
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_update_preserves_context_from_when_not_passed(self, cron_env):
        """Updating other fields must not clobber context_from."""
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )

        # Update an unrelated field
        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            prompt="Summarize v2",
        ))
        assert result["success"] is True
        reloaded = get_job(job_b["id"])
        assert reloaded["prompt"] == "Summarize v2"
        assert reloaded["context_from"] == [job_a["id"]]


class TestArtifactExpectationsToolContract:
    def test_create_lists_and_clears_artifact_expectations(self, cron_env):
        from tools.cronjob_tools import cronjob
        import json

        expectation = {
            "path": "vault:sources/{date}-ai-research-digest.md",
            "mode": "required_for_delivery",
        }
        created = json.loads(cronjob(
            action="create",
            prompt="Research",
            schedule="every 1h",
            deliver="local",
            artifact_expectations=[expectation],
        ))
        assert created["success"] is True
        assert created["job"]["artifact_expectations"] == [expectation]

        listing = json.loads(cronjob(action="list", include_disabled=True))
        listed = next(job for job in listing["jobs"] if job["job_id"] == created["job_id"])
        assert listed["artifact_expectations"] == [expectation]

        cleared = json.loads(cronjob(
            action="update",
            job_id=created["job_id"],
            artifact_expectations=[],
        ))
        assert cleared["success"] is True
        assert "artifact_expectations" not in cleared["job"]

    def test_update_rejects_invalid_artifact_expectation_mode(self, cron_env):
        from cron.jobs import create_job
        from tools.cronjob_tools import cronjob
        import json

        job = create_job(prompt="Research", schedule="every 1h", deliver="local")

        result = json.loads(cronjob(
            action="update",
            job_id=job["id"],
            artifact_expectations=[
                {"path": "vault:sources/{date}.md", "mode": "required"}
            ],
        ))

        assert result["success"] is False
        assert "required_for_delivery" in result["error"]

    def test_create_rejects_artifact_expectation_with_path_and_glob(self, cron_env):
        from tools.cronjob_tools import cronjob
        import json

        result = json.loads(cronjob(
            action="create",
            prompt="Research",
            schedule="every 1h",
            deliver="local",
            artifact_expectations=[
                {"path": "vault:sources/{date}.md", "glob": "vault:sources/*.md"}
            ],
        ))

        assert result["success"] is False
        assert "exactly one of path or glob" in result["error"]

    def test_direct_create_job_rejects_unprefixed_artifact_path(self, cron_env):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="vault: or hermes:"):
            create_job(
                prompt="Research",
                schedule="every 1h",
                deliver="local",
                artifact_expectations=[
                    {"path": "/Users/byron/.hermes/.env", "mode": "required_for_delivery"}
                ],
            )
