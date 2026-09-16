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


@pytest.mark.parametrize("self_context", [False, True])
def test_saved_run_payload_survives_repeated_chaining(cron_env, monkeypatch, self_context):
    """Real run documents must not recursively inject prompts or lose payload headings."""
    from unittest.mock import MagicMock
    from cron import jobs, scheduler
    from cron.scheduler_prompt import _inject_context_from

    job = jobs.create_job(prompt="wrapper sentinel " + "P" * 9000, schedule="every 1h")
    consumer = dict(job, context_from=["self"] if self_context else [job["id"]])
    if not self_context:
        consumer["id"] = "abcdef123456"
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *a: scheduler._CronAgentSetup())
    monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *a, **kw: None)
    for run in range(3):
        response = f"report {run}\n\n## Response\n\n## Error\n\nbody with trailing space  \n"
        monkeypatch.setattr(scheduler, "_run_agent_with_watchdog", lambda *a, **kw: {"final_response": response})
        success, output, _, error = scheduler.run_job(job)
        assert success, error
        jobs.save_job_output(job["id"], output)
        injected, found = _inject_context_from(consumer, "next task")
        assert found and response in injected
        assert "wrapper sentinel" not in injected
        job["context_from"] = ["self"]

    def fail_setup(*args):
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", fail_setup)
    success, output, _, _ = scheduler.run_job(job)
    assert not success
    jobs.save_job_output(job["id"], output)
    injected, found = _inject_context_from(consumer, "recover")
    assert found and "RuntimeError: upstream failed" in injected
    assert "wrapper sentinel" not in injected


def test_oversized_result_length_falls_back_during_context_injection(cron_env):
    from cron.jobs import create_job, save_job_output
    from cron.scheduler_prompt import _inject_context_from

    source = create_job(prompt="upstream", schedule="every 1h")
    save_job_output(
        source["id"],
        "**Result Chars:** " + "9" * 5000 + "\n\n## Response\n\nusable result\n",
    )
    prompt, injected = _inject_context_from(
        {"id": "abcdef123456", "context_from": [source["id"]]}, "downstream task"
    )
    assert injected
    assert "usable result" in prompt
    assert "Result Chars" not in prompt
    assert "downstream task" in prompt


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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt
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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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

    def test_cron_output_prefers_response_section_over_prompt_wrapper(self, cron_env):
        """Cron output chaining should inject the upstream response body, not the wrapper preamble."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cron_doc = (
            "# Cron Job: Example\n\n"
            "## Prompt\n\n" + ("P" * 9000) +
            "\n\n## Response\n\n"
            "status: candidate_unverified\n"
            "generated_at: 2026-05-29T08:30:25+02:00\n"
            "fresh payload\n"
        )
        (out_dir / "2026-04-22_10-00-00.md").write_text(cron_doc, encoding="utf-8")

        job_b = create_job(
            prompt="Verify upstream draft", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)

        assert "fresh payload" in prompt
        assert "generated_at: 2026-05-29T08:30:25+02:00" in prompt
        assert "P" * 8500 not in prompt

    def test_cron_output_uses_final_structural_heading_when_preamble_contains_response(self, cron_env):
        """A heading copied into prompt data must not eclipse the final artifact response."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cron_doc = (
            "# Cron Job: Example\n\n"
            "## Prompt\n\n"
            "Script emitted this literal heading:\n"
            "## Response\n\n"
            "preamble data that must not be injected\n\n"
            "## Response\n\n"
            "authoritative final payload\n"
        )
        (out_dir / "2026-04-22_10-00-00.md").write_text(cron_doc, encoding="utf-8")

        job_b = create_job(
            prompt="Verify upstream draft", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)

        assert "authoritative final payload" in prompt
        assert "preamble data that must not be injected" not in prompt

    def test_cron_output_preserves_structural_headings_inside_result_body(self, cron_env):
        """Length metadata prevents result-body headings from becoming delimiters."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        response = "answer intro\n\n## Error\n\nquoted subheading"
        cron_doc = (
            f"**Result Chars:** {len(response)}\n"
            "# Cron Job: Example\n**Result Chars:** 1\n\n"
            "## Prompt\n\n"
            "preamble data that must not be injected\n\n"
            "## Response\n\n"
            f"{response}\n"
        )
        (out_dir / "2026-04-22_10-00-00.md").write_text(cron_doc, encoding="utf-8")

        job_b = create_job(
            prompt="Verify upstream draft", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)

        assert response in prompt
        assert "preamble data that must not be injected" not in prompt

    def test_cron_output_truncation_preserves_response_header_when_wrapper_is_huge(self, cron_env):
        """If a cron wrapper is huge, truncation should still preserve the response start."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        big_response = "Y" * 9000
        cron_doc = (
            "# Cron Job: Example\n\n"
            "## Prompt\n\n" + ("P" * 12000) +
            "\n\n## Response\n\n"
            "**Candidate Report**\n"
            "**Generated at:** `2026-05-29T08:30:25+02:00`\n" +
            big_response
        )
        (out_dir / "2026-04-22_10-00-00.md").write_text(cron_doc, encoding="utf-8")

        job_b = create_job(
            prompt="Verify upstream draft", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)

        assert "**Candidate Report**" in prompt
        assert "**Generated at:** `2026-05-29T08:30:25+02:00`" in prompt
        assert "[... output truncated ...]" in prompt

    def test_cron_output_prefers_error_section_when_response_absent(self, cron_env):
        """Failed upstream cron runs should pass the useful error body, not the whole wrapper."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        out_dir = OUTPUT_DIR / job_a["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cron_doc = (
            "# Cron Job: Example\n\n"
            "## Prompt\n\n" + ("P" * 9000) +
            "\n\n## Error\n\n"
            "Script exited with code 2\n"
            "missing feed URL\n"
        )
        (out_dir / "2026-04-22_10-00-00.md").write_text(cron_doc, encoding="utf-8")

        job_b = create_job(
            prompt="Verify upstream draft", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)

        assert "missing feed URL" in prompt
        assert "Script exited with code 2" in prompt
        assert "P" * 8500 not in prompt

    def test_output_truncated_at_8k_chars(self, cron_env):
        """Output longer than 8000 chars should be truncated."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt

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
        from cron.scheduler_prompt import _build_job_prompt
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


