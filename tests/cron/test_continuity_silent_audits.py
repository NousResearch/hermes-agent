"""Silent audit records must not replace useful continuity output (#104541)."""
import os

from cron import jobs
from cron.scheduler_prompt import _inject_context_from


def test_silent_audits_preserve_latest_payload(tmp_path):
    with jobs.use_cron_store(tmp_path):
        directory = jobs.get_cron_output_dir() / "abcdef"
        directory.mkdir(parents=True)
        header = "# Cron Job: " + "long name " * 80 + "\n\n**Job ID:** abcdef\n**Run Time:** now\n"
        payload = header + "**Mode:** no_agent (script)\n\n---\n\n**Status:** silent but useful payload\n"
        records = [payload, header + "**Mode:** monitor\n**Status:** no_change (agent run suppressed)\n",
                   header + "**Mode:** no_agent (script)\n**Status:** silent (empty output)\n",
                   header + "\nScript gate returned `wakeAgent=false` — agent skipped.\n", ""]
        for index, text in enumerate(records):
            path = directory / f"{index}.md"
            path.write_text(text, encoding="utf-8")
            os.utime(path, (index + 1, index + 1))
        for source in ("self", "abcdef"):
            prompt, injected = _inject_context_from({"id": "abcdef", "context_from": [source]}, "next")
            assert injected and "silent but useful payload" in prompt
            assert "agent skipped" not in prompt and "agent run suppressed" not in prompt
        assert len(list(directory.glob("*.md"))) == len(records)


def test_audit_only_history_is_empty_but_errors_remain_context(tmp_path):
    with jobs.use_cron_store(tmp_path):
        directory = jobs.get_cron_output_dir() / "abcdef"
        directory.mkdir(parents=True)
        path = directory / "audit.md"
        path.write_text("# Cron Job: monitor\n**Status:** no_change (agent run suppressed)\n", encoding="utf-8")
        job = {"id": "abcdef", "context_from": ["self"]}
        assert _inject_context_from(job, "next") == ("next", False)
        path.write_text("# Cron Job: monitor\n**Status:** monitor source failed\n\nConnection refused\n", encoding="utf-8")
        prompt, injected = _inject_context_from(job, "next")
        assert injected and "Connection refused" in prompt
