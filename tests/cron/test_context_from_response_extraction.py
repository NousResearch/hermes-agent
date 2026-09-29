"""context_from must inject the previous run's answer, not its prompt (#117290).

Stored agent-run archives are ``# Cron Job`` / ``## Prompt`` / ``## Response``
documents; skill-bearing prompts routinely exceed the 8000-char injection
budget, so head-truncation amputated the ``## Response`` section and
self-continuity silently became a no-op.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.scheduler  # noqa: E402  (top-level so home_io_guard sees no git-dir probe)
import run_agent  # noqa: E402


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


def _write_archive(cron_env, job_id: str, filename: str, body: str) -> None:
    from cron.jobs import OUTPUT_DIR

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(body, encoding="utf-8")


class TestResponseSurvivesLongPrompt:
    """The answer, not the prompt, is the part continuity needs."""

    def test_long_prompt_archive_keeps_response(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Run the daily check", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\n" + "SKILL LINE\n" * 1800 +
            "\n\n## Response\n\nCONCLUSION-MARKER-42\n",
        )

        prompt = _build_job_prompt(job)

        assert "CONCLUSION-MARKER-42" in prompt
        assert "SKILL LINE" not in prompt  # the prompt half is dropped, not the answer


class TestUnusableAnswersFallThrough:
    """A [SILENT] or blank response is not usable continuity."""

    def test_silent_response_falls_through_to_older_archive(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-18_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\ndo the thing\n\n## Response\n\nOLDER-REAL-ANSWER\n",
        )
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\ndo the thing\n\n## Response\n\n[SILENT]\n",
        )

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "OLDER-REAL-ANSWER" in prompt
        assert "[SILENT]" not in prompt

class TestScriptModeArchives:
    """Archives without a ## Response heading (script-mode) stay whole-document."""

    def test_headingless_archive_injects_whole_document(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(cron_env, job["id"], "2026-09-19_08-00-00.md", "plain script payload\nline two")

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "plain script payload" in prompt
        assert "line two" in prompt


class TestStampedArchives:
    """The writer stamps the response length; extraction is bounded by it (#128543)."""

    @staticmethod
    def _stamped_archive(prompt: str, answer: str) -> str:
        # Mirrors the writer's assembly in cron/scheduler.py::run_job.
        return (
            "# Cron Job: probe\n\n"
            "**Job ID:** j1\n**Run Time:** t\n**Schedule:** N/A\n\n"
            f"## Prompt\n\n{prompt}\n\n"
            f"**Response Length:** {len(answer)}\n\n## Response\n\n{answer}\n"
        )

    def test_answer_with_embedded_heading_survives(self, cron_env):
        """The issue's reproduction: an answer containing its own ## Response heading."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        answer = ("Summary before embedded heading\n\n## Response\n"
                  "This is an answer subsection.\nConclusion.")
        job = create_job(prompt="Run the daily check", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            self._stamped_archive("Synthetic prompt.", answer))

        prompt = _build_job_prompt(job)

        assert "Summary before embedded heading" in prompt
        assert "This is an answer subsection." in prompt
        assert "Conclusion." in prompt

    def test_inline_mention_is_not_a_boundary(self, cron_env):
        """"## Response" mentioned mid-line stays payload, not a heading."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        answer = "We document the ## Response format below.\nDone."
        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            self._stamped_archive("Explain the archive format.", answer))

        prompt = _build_job_prompt(job)

        assert "We document the ## Response format below." in prompt
        assert "Done." in prompt

    def test_stamp_quoted_in_the_prompt_does_not_hijack(self, cron_env):
        """A prompt quoting the stamp+heading pair (a skill documenting the format) must
        not satisfy the declared count before the writer's own stamp does."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        quoting_prompt = (
            "Skill docs say the archive looks like:\n\n"
            "**Response Length:** 4\n\n## Response\n\nbody\n\nNow do the work."
        )
        answer = "REAL-ANSWER-42"
        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            self._stamped_archive(quoting_prompt, answer))

        prompt = _build_job_prompt(job)

        assert "REAL-ANSWER-42" in prompt
        assert "Skill docs say" not in prompt  # the prompt half stays dropped

    def test_edited_stamp_falls_through_to_older_archive(self, cron_env):
        """A stamp that no longer bounds the tail (edited/truncated archive) is not a
        guessed boundary: the archive is skipped, the older one is used."""
        from cron.jobs import create_job
        from cron.scheduler_prompt import _inject_context_from

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-18_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\ndo the thing\n\n## Response\n\nOLDER-REAL-ANSWER\n")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            self._stamped_archive("do the thing", "TRUNCATED-HEAD")[:-20] + "\n")

        prompt, injected = _inject_context_from(job, "Report")

        assert injected is True
        assert "OLDER-REAL-ANSWER" in prompt
        assert "TRUNCATED-HEAD" not in prompt

    def test_unicode_answer_round_trips_by_character_count(self, cron_env):
        """The stamp counts characters (not bytes): a CJK answer re-extracts whole."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        answer = "摘要：库存清点完成。\n\n## Response\n\n明细见下表。"
        job = create_job(prompt="盘点", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            self._stamped_archive("盘点", answer))

        prompt = _build_job_prompt(job)

        assert "摘要：库存清点完成。" in prompt
        assert "明细见下表。" in prompt


class TestLegacyLastHeadingSplit:
    """Unstamped archives keep the last-occurrence boundary (guards current behaviour)."""

    def test_prompt_embedded_heading_keeps_last_boundary(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
        _write_archive(
            cron_env, job["id"], "2026-09-19_08-00-00.md",
            "# Cron Job: probe\n\n## Prompt\n\nsee ## Response docs\n\n"
            "## Response\n\nLEGACY-REAL-ANSWER\n")

        prompt = _build_job_prompt(job)

        assert "LEGACY-REAL-ANSWER" in prompt
        assert "see ## Response docs" not in prompt


class TestWriterStampsTheResponse:
    """run_job persists the length stamp, so the writer and reader agree (#128543)."""

    def test_run_job_output_is_stamped_and_round_trips(self, cron_env, monkeypatch):
        answer = ("Stock count done.\n\n## Response\n\nTotals: 42 units.")

        class _StubAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, user_message, conversation_history=None, task_id=None):
                return {"final_response": answer, "completed": True, "failed": False}

        monkeypatch.setattr(run_agent, "AIAgent", _StubAgent)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **kwargs: {"provider": "openai", "api_key": "test-key"})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))

        job = {"id": "0" * 32, "name": "stamp-probe", "prompt": "Count the stock"}
        success, output, final_response, error = cron.scheduler.run_job(job)

        assert success is True
        assert final_response == answer
        assert f"**Response Length:** {len(answer)}" in output
        # The stored document re-extracts the whole answer, embedded heading included.
        from cron.scheduler_prompt import _archive_answer
        assert _archive_answer(output) == answer
