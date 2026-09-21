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


class TestContinuityStripsPersistedPromptScaffolding:
    """Regression for the learn-daily 413 incident (2026-09-21).

    ``context_from: self`` used to feed the WHOLE persisted run document back
    in — including that run's own ``## Prompt`` section (skill body + any
    context it itself injected). Since each generation's document embeds the
    document before it, size compounds every run: a real production doc
    measured 15255 bytes, of which only 1353 (9%) was the actual response —
    the rest was N nested copies of the same injected skill text. This
    starved a job's real per-minute token budget for zero new information,
    and was the proximate trigger for a Groq 413 (\"tokens per minute limit
    8000, requested 16885\") that then hit a SEPARATE classifier bug (see
    #118274) and aborted the whole cron run instead of falling back.

    Continuity should carry forward only what the run actually produced.
    """

    @staticmethod
    def _run_doc(prompt_body: str, response_body: str) -> str:
        """Build a realistic persisted doc, same shape as
        ``cron.scheduler._run_doc_header`` + ``## Response`` produces."""
        return (
            "# Cron Job: learn-daily\n\n"
            "**Job ID:** abc123\n"
            "**Run Time:** 2026-09-21 08:30:47\n"
            "**Schedule:** 30 08 * * *\n\n"
            f"## Prompt\n\n{prompt_body}\n\n"
            f"## Response\n\n{response_body}\n"
        )

    def test_self_continuity_carries_only_the_response(self, cron_env):
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Scan for news", schedule="every 1h", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        big_skill_text = "SKILL INSTRUCTIONS " * 200  # stand-in for an injected skill
        (out_dir / "2026-08-01_10-00-00.md").write_text(
            self._run_doc(prompt_body=big_skill_text, response_body="Reported: story A"),
            encoding="utf-8",
        )

        prompt = _build_job_prompt(job)
        assert "Reported: story A" in prompt
        assert "SKILL INSTRUCTIONS" not in prompt

    def test_nested_prompts_across_runs_do_not_compound(self, cron_env):
        """The core reproduction: generation N's persisted doc already embeds
        generation N-1's whole doc (because continuity fed it into N's own
        ## Prompt). Confirm generation N+1 does not inherit that nesting."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Daily lesson", schedule="every 1d", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        skill_body = "SKILL: automation/newton-bot instructions here"
        gen1_response = "Day 1 lesson: X"
        gen1_doc = self._run_doc(prompt_body=skill_body, response_body=gen1_response)

        # Generation 2's prompt (as actually built) nests generation 1's WHOLE
        # doc under "## Your previous run's output" plus its own fresh skill
        # injection — this is what a real second run's persisted doc looks like.
        gen2_prompt_body = (
            f"{skill_body}\n\n## Your previous run's output\n```\n{gen1_doc}\n```"
        )
        gen2_response = "Day 2 lesson: Y"
        gen2_doc = self._run_doc(prompt_body=gen2_prompt_body, response_body=gen2_response)
        (out_dir / "2026-08-02_10-00-00.md").write_text(gen2_doc, encoding="utf-8")

        # Generation 3's prompt should carry forward ONLY "Day 2 lesson: Y" —
        # not gen2's skill text, not gen1's doc nested two levels deep inside it.
        prompt = _build_job_prompt(job)
        assert "Day 2 lesson: Y" in prompt
        assert "Day 1 lesson: X" not in prompt
        assert skill_body not in prompt
        assert "## Prompt" not in prompt

    def test_failed_run_doc_has_no_response_section_keeps_full_doc(self, cron_env):
        """A FAILED-run doc has no '## Response' — the error text itself is
        the useful continuity signal, so the fallback keeps the whole doc."""
        from cron.jobs import create_job, OUTPUT_DIR
        from cron.scheduler import _build_job_prompt

        job = create_job(
            prompt="Scan for news", schedule="every 1h", context_from="self"
        )
        out_dir = OUTPUT_DIR / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        failed_doc = (
            "# Cron Job: learn-daily (FAILED)\n\n"
            "**Job ID:** abc123\n**Run Time:** 2026-09-21 08:30:47\n"
            "**Schedule:** 30 08 * * *\n\n## Prompt\n\nSome prompt\n\n"
            "RateLimitError: tokens per minute limit exceeded"
        )
        (out_dir / "2026-08-01_10-00-00.md").write_text(failed_doc, encoding="utf-8")

        prompt = _build_job_prompt(job)
        assert "RateLimitError: tokens per minute limit exceeded" in prompt

    def test_extract_continuity_payload_uses_last_response_marker(self):
        """A model's own response can legitimately contain the literal string
        '## Response' (e.g. writing markdown about its own output format) —
        the extraction must anchor on the FINAL marker, not the first."""
        from cron.scheduler_prompt import _extract_continuity_payload

        doc = (
            "# Cron Job: x\n\n## Prompt\n\nExample doc with\n## Response\nheading "
            "inside the prompt body\n\n## Response\n\nActual answer here"
        )
        assert _extract_continuity_payload(doc) == "Actual answer here"

    def test_extract_continuity_payload_no_marker_returns_whole_doc(self):
        from cron.scheduler_prompt import _extract_continuity_payload

        assert _extract_continuity_payload("plain text, no sections") == (
            "plain text, no sections"
        )



