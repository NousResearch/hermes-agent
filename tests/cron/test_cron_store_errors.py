"""Storage failures remain fail-closed and distinguishable from handler bugs."""
import builtins

import pytest

from cron import jobs


@pytest.mark.parametrize(("content", "message"), [
    ('{"jobs": ', "corrupted and unrepairable"),
    ('42', "expected"),
])
def test_corrupt_load_raises_store_error_without_mutating_file(tmp_path, content, message):
    with jobs.use_cron_store(tmp_path):
        jobs_file = tmp_path / "cron" / "jobs.json"
        jobs_file.parent.mkdir()
        jobs_file.write_text(content)
        with pytest.raises(jobs.CronStoreError, match=message):
            jobs.load_jobs()
        assert jobs_file.read_text() == content


def test_unreadable_load_raises_store_error(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path):
        jobs_file = tmp_path / "cron" / "jobs.json"
        jobs_file.parent.mkdir()
        jobs_file.write_text('{"jobs": []}')
        original_open = builtins.open

        def denied(path, *args, **kwargs):
            if path == jobs_file:
                raise PermissionError("test read denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", denied)
        with pytest.raises(jobs.CronStoreError, match="Failed to read cron database"):
            jobs.load_jobs()


def test_save_refuses_to_overwrite_corrupt_store(tmp_path):
    with jobs.use_cron_store(tmp_path):
        jobs_file = tmp_path / "cron" / "jobs.json"
        jobs_file.parent.mkdir()
        content = b'{"jobs": [{"id": "unfinished'
        jobs_file.write_bytes(content)
        with pytest.raises(jobs.CronStoreError, match="refusing to overwrite"):
            jobs.save_jobs([])
        assert jobs_file.read_bytes() == content
