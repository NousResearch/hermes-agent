"""context_from must inject the previous run's answer, not its prompt (#117290).

Stored agent-run archives are ``# Cron Job`` / ``## Prompt`` / ``## Response``
documents; skill-bearing prompts routinely exceed the 8000-char injection
budget, so head-truncation amputated the ``## Response`` section and
self-continuity silently became a no-op.
"""

import sys
from pathlib import Path

import pytest
import cron.scheduler
import run_agent

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


def test_writer_reader_preserve_response_with_nested_frames(cron_env, monkeypatch):
    from cron.jobs import create_job, save_job_output
    from cron.scheduler_prompt import _inject_context_from

    answer = "摘要 before heading\r\n\r\n## Response\nsubsection\n**Response Characters:** 4\n## Response\n\nbody\n\n  "

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            return {"final_response": answer, "completed": True, "failed": False}

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kwargs: {"provider": "openai", "api_key": "fixture"})
    job = create_job(prompt="Original prompt noise\r\n**Response Characters:** 4\n## Response\n\nbody",
                     schedule="0 8 * * *", context_from="self")
    success, archive, final, error = cron.scheduler.run_job(job)
    assert success, error
    assert final == answer
    save_job_output(job["id"], archive)
    prompt, injected = _inject_context_from(job, "Next task")
    assert injected
    assert answer.replace("\r\n", "\n").strip() in prompt
    assert "Original prompt noise" not in prompt


def test_truncated_framed_archive_falls_back_to_older_answer(cron_env):
    import os
    from cron.jobs import create_job, OUTPUT_DIR
    from cron.scheduler_prompt import _inject_context_from

    job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
    _write_archive(cron_env, job["id"], "older.md", "## Response\n\nOLDER ANSWER\n")
    _write_archive(cron_env, job["id"], "newer.md",
                   "**Response Characters:** 80\n## Response\n\nTRUNCATED\n")
    os.utime(OUTPUT_DIR / job["id"] / "older.md", (1, 1))
    os.utime(OUTPUT_DIR / job["id"] / "newer.md", (2, 2))
    prompt, injected = _inject_context_from(job, "Report")
    assert injected
    assert "OLDER ANSWER" in prompt
    assert "TRUNCATED" not in prompt


def test_truncated_outer_frame_cannot_promote_a_quoted_inner_frame(cron_env, monkeypatch):
    import os
    from cron.jobs import create_job, save_job_output, OUTPUT_DIR
    from cron.scheduler_prompt import _inject_context_from

    quoted = "QUOTED INNER ANSWER"
    suffix = "\nThis tail will be lost."
    answer = ("Outer response introduction\n"
              f"**Response Characters:** {len(quoted)}\n## Response\n\n{quoted}"
              + suffix)

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            return {"final_response": answer, "completed": True, "failed": False}

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kwargs: {"provider": "openai", "api_key": "fixture"})
    job = create_job(prompt="Report", schedule="0 8 * * *", context_from="self")
    success, archive, final, error = cron.scheduler.run_job(job)
    assert success, error
    assert final == answer
    save_job_output(job["id"], archive)
    saved = next((OUTPUT_DIR / job["id"]).glob("*.md"))
    complete = saved.read_text(encoding="utf-8")
    assert complete.endswith(suffix + "\n")
    # Simulate a partial write: the quoted inner frame now reaches EOF exactly,
    # but the enclosing writer-owned response is missing its declared suffix.
    saved.write_text(complete[:-len(suffix + "\n")] + "\n", encoding="utf-8")
    os.utime(saved, (2, 2))
    prompt, injected = _inject_context_from(job, "Next task")
    assert not injected
    assert prompt == "Next task"

    _write_archive(cron_env, job["id"], "older.md", "## Response\n\nOLDER COMPLETE ANSWER\n")
    os.utime(OUTPUT_DIR / job["id"] / "older.md", (1, 1))
    prompt, injected = _inject_context_from(job, "Next task")
    assert injected
    assert "OLDER COMPLETE ANSWER" in prompt
    assert quoted not in prompt

    # Archives produced before prompt-length framing get the same conservative
    # treatment: an invalid first response frame never promotes a later one.
    saved.write_text(f"**Response Characters:** {len(answer)}\n## Response\n\n"
                     + answer[:-len(suffix)] + "\n", encoding="utf-8")
    os.utime(saved, (2, 2))
    prompt, injected = _inject_context_from(job, "Next task")
    assert injected and "OLDER COMPLETE ANSWER" in prompt
    assert quoted not in prompt

    # Losing the response boundary itself is also unusable, not a script archive.
    saved.write_text(complete.split("**Response Characters:**", 1)[0], encoding="utf-8")
    os.utime(saved, (2, 2))
    prompt, injected = _inject_context_from(job, "Next task")
    assert injected and "OLDER COMPLETE ANSWER" in prompt
    assert "## Prompt" not in prompt
