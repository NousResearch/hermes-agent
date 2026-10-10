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


def test_undecodable_output_skips_file_not_run(tmp_path):
    """Binary strays matching the `*.md` glob (e.g. macOS AppleDouble `._*.md`) must be
    skipped per-file, falling through to older archives, instead of crashing the run
    with UnicodeDecodeError (#105582 class, seen in the wild 2026-10-08/09)."""
    with jobs.use_cron_store(tmp_path):
        directory = jobs.get_cron_output_dir() / "abcdef"
        directory.mkdir(parents=True)
        # Good older archive — must be the one injected.
        good = directory / "2026-10-07_09-00-00.md"
        good.write_text(
            "# Cron Job: email check\n\n**Status:** all clear, nothing new\n\n---\n\nEmail check complete.\n",
            encoding="utf-8")
        os.utime(good, (10, 10))
        # Undecodable newest file — binary header like AppleDouble xattr payload.
        stray = directory / "2026-10-08_09-00-00.md"
        stray.write_bytes(b"\x00\x05\x16\x07Microsoft Office Mac\x00\xa3\x03\x00\x00\x00")
        os.utime(stray, (20, 20))
        job = {"id": "abcdef", "context_from": ["self"]}
        prompt, injected = _inject_context_from(job, "next")
        assert injected and "Email check complete." in prompt
        assert "Microsoft Office Mac" not in prompt
