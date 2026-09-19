import os


def test_latest_job_output_excerpt_uses_most_recent_file_mtime(tmp_path, monkeypatch):
    """The completion summary and its source path must describe the same newest output."""
    output_dir = tmp_path / "output"
    job_dir = output_dir / "job-1"
    job_dir.mkdir(parents=True)

    lexically_last = job_dir / "z-old.md"
    lexically_last.write_text("older output", encoding="utf-8")
    latest = job_dir / "a-new.md"
    latest.write_text("newest output that exceeds the excerpt limit", encoding="utf-8")
    os.utime(lexically_last, (1_000_000_000, 1_000_000_000))
    os.utime(latest, (1_000_000_100, 1_000_000_100))

    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "get_cron_output_dir", lambda: output_dir)

    from tools.cronjob_tools import _latest_job_output_excerpt

    excerpt = _latest_job_output_excerpt("job-1", max_chars=10)

    assert excerpt is not None
    assert excerpt.startswith("newest out")
    assert "older output" not in excerpt
    assert f"full output: {latest}" in excerpt
