"""Tests for cron job context_from feature (issue #5439 Option C)."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


    def test_context_from_empty_string_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", context_from="")
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


class TestSelfContext:
    """The special 'self' value injects the job's OWN previous output.

    Inspired by Amp's "Right on Schedule" (agents wake up with their saved
    context and continue where they left off): recurring jobs get run-to-run
    continuity without touching session history.
    """

    def test_self_injects_own_previous_output(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Scan for news", schedule="every 1h", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text(
            "Reported: story A, story B", encoding="utf-8"
        )

        prompt = _build_job_prompt(job)
        assert "Reported: story A, story B" in prompt
        assert "previous run" in prompt.lower()
        # Self-context uses continuity framing, not the upstream-job framing.
        assert f"Output from job '{job['id']}'" not in prompt

    def test_self_case_insensitive(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Scan", schedule="every 1h", context_from="SELF"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text("prev", encoding="utf-8")
        prompt = _build_job_prompt(job)
        assert "prev" in prompt

    def test_self_silent_skip_on_first_run(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Scan for news", schedule="every 1h", context_from="self"
        )
        # No output yet (first run) — base prompt intact, no placeholder.
        prompt = _build_job_prompt(job)
        assert "Scan for news" in prompt
        assert "previous run" not in prompt.lower()

    def test_own_id_treated_as_self(self, cron_env):
        """Passing the job's literal id gets the continuity framing too."""
        from cron.jobs import create_job, update_job, get_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Scan", schedule="every 1h")
        update_job(job["id"], {"context_from": [job["id"]]})
        job = get_job(job["id"])
        assert job is not None

        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text("prev output", encoding="utf-8")

        prompt = _build_job_prompt(job)
        assert "prev output" in prompt
        assert "previous run" in prompt.lower()

    def test_self_prefers_previous_response_section(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Summarize what changed", schedule="every 1h", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text(
            "# Cron Job: nested\n\n"
            "## Prompt\n\n"
            "PROMPT-NESTING-SHOULD-NOT-REAPPEAR\n\n"
            "## Response\n\n"
            "carry forward only this response",
            encoding="utf-8",
        )

        prompt = _build_job_prompt(job)
        assert "carry forward only this response" in prompt
        assert "PROMPT-NESTING-SHOULD-NOT-REAPPEAR" not in prompt

    def test_tool_create_accepts_self(self, cron_env):
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job
        import json

        result = json.loads(cronjob(
            action="create",
            prompt="Scan for news",
            schedule="every 1h",
            context_from="self",
        ))
        assert result["success"] is True
        job_id = result["job_id"]
        assert get_job(job_id)["context_from"] == ["self"]

    def test_tool_update_accepts_self(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job = create_job(prompt="Scan", schedule="every 1h")
        result = json.loads(cronjob(
            action="update",
            job_id=job["id"],
            context_from="self",
        ))
        assert result["success"] is True
        assert get_job(job["id"])["context_from"] == ["self"]


class TestRunJobContinuityOutput:
    def test_output_doc_keeps_declared_prompt_not_augmented_context(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import run_job

        job = create_job(
            prompt="Summarize what changed", schedule="every 1h", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text(
            "# Cron Job: previous\n\n"
            "## Prompt\n\n"
            "OLD-PROMPT-SHOULD-NOT-BE-RECORDED\n\n"
            "## Response\n\n"
            "previous response context",
            encoding="utf-8",
        )

        fake_db = MagicMock()
        fake_db.get_compression_tip.side_effect = lambda session_id: session_id
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "fresh response"}

        with patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.get_shared_session_db", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent", return_value=mock_agent):
            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "fresh response"
        assert "## Prompt\n\nSummarize what changed" in output
        assert "OLD-PROMPT-SHOULD-NOT-BE-RECORDED" not in output


class TestContinuityFlag:
    """continuity=true/false is the user-facing surface for self-context.

    It translates to the reserved 'self' entry in context_from internally.
    """

    def test_create_with_continuity_true(self, cron_env):
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job
        import json

        result = json.loads(cronjob(
            action="create",
            prompt="Scan for news",
            schedule="every 1h",
            continuity=True,
        ))
        assert result["success"] is True
        assert get_job(result["job_id"])["context_from"] == ["self"]

    def test_create_continuity_false_is_noop(self, cron_env):
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job
        import json

        result = json.loads(cronjob(
            action="create",
            prompt="Scan",
            schedule="every 1h",
            continuity=False,
        ))
        assert result["success"] is True
        assert get_job(result["job_id"]).get("context_from") is None

    def test_create_continuity_combines_with_context_from(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        upstream = create_job(prompt="Collect", schedule="every 1h")
        result = json.loads(cronjob(
            action="create",
            prompt="Digest",
            schedule="every 2h",
            context_from=upstream["id"],
            continuity=True,
        ))
        assert result["success"] is True
        stored = get_job(result["job_id"])["context_from"]
        assert upstream["id"] in stored
        assert "self" in stored

    def test_update_continuity_true_adds_self(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job = create_job(prompt="Scan", schedule="every 1h")
        result = json.loads(cronjob(
            action="update",
            job_id=job["id"],
            continuity=True,
        ))
        assert result["success"] is True
        assert get_job(job["id"])["context_from"] == ["self"]

    def test_update_continuity_false_removes_self_preserves_others(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        upstream = create_job(prompt="Collect", schedule="every 1h")
        job = create_job(
            prompt="Digest",
            schedule="every 2h",
            context_from=["self", upstream["id"]],
        )
        result = json.loads(cronjob(
            action="update",
            job_id=job["id"],
            continuity=False,
        ))
        assert result["success"] is True
        assert get_job(job["id"])["context_from"] == [upstream["id"]]

    def test_update_continuity_true_idempotent(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job = create_job(prompt="Scan", schedule="every 1h", context_from="self")
        result = json.loads(cronjob(
            action="update",
            job_id=job["id"],
            continuity=True,
        ))
        assert result["success"] is True
        assert get_job(job["id"])["context_from"] == ["self"]

    def test_continuity_job_gets_previous_output(self, cron_env):
        """End-to-end: a continuity-created job injects its own prior output."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt
        import json

        result = json.loads(cronjob(
            action="create",
            prompt="Scan for news",
            schedule="every 1h",
            continuity=True,
        ))
        job = get_job(result["job_id"])
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-08-01_10-00-00.md").write_text(
            "Reported: story A", encoding="utf-8"
        )
        prompt = _build_job_prompt(job)
        assert "Reported: story A" in prompt
        assert "previous run" in prompt.lower()

