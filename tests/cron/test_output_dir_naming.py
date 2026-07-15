"""Filesystem-level tests for path-friendly cron output directories.

Job output lives in ``OUTPUT_DIR / {name-slug}-{job_id}`` (plain
``{job_id}`` when the name slugifies to empty). All resolution goes
through ``cron.jobs.resolve_job_output_dir``, which preserves the
path-escape guard from ``_job_output_dir`` (unsafe legacy IDs fail
closed) and migrates legacy ID-only directories on the write path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.jobs import (  # noqa: E402
    _slugify_job_name,
    create_job,
    load_jobs,
    remove_job,
    resolve_job_output_dir,
    save_job_output,
    save_jobs,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def output_root(tmp_cron_dir):
    return tmp_cron_dir / "cron" / "output"


# =========================================================================
# Slugification
# =========================================================================

class TestSlugifyJobName:
    def test_basic_name(self):
        assert _slugify_job_name("Daily News Report") == "daily-news-report"

    def test_special_chars_stripped(self):
        assert _slugify_job_name("  Check: server / status!!  ") == "check-server-status"

    def test_runs_joined_by_single_hyphen(self):
        assert _slugify_job_name("a---b___c...d") == "a-b-c-d"

    def test_unicode_only_name_slugifies_to_empty(self):
        assert _slugify_job_name("日本語のジョブ") == ""

    def test_symbols_only_name_slugifies_to_empty(self):
        assert _slugify_job_name("!!! ??? ***") == ""

    def test_none_and_empty(self):
        assert _slugify_job_name(None) == ""
        assert _slugify_job_name("") == ""

    def test_length_capped_at_40_without_trailing_hyphen(self):
        slug = _slugify_job_name("word " * 20)
        assert len(slug) <= 40
        assert not slug.endswith("-")

    def test_traversal_name_reduced_to_cosmetic_slug(self):
        assert _slugify_job_name("../../etc/passwd") == "etc-passwd"


# =========================================================================
# Writer (save_job_output) naming + migration
# =========================================================================

class TestSaveJobOutputNaming:
    def test_writes_to_name_slug_dir(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Daily News")
        output_file = save_job_output(job["id"], "report", job_name=job["name"])
        assert output_file.parent == output_root / f"daily-news-{job['id']}"
        assert output_file.read_text() == "report"

    def test_looks_up_name_when_not_passed(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Weather Watch")
        output_file = save_job_output(job["id"], "report")
        assert output_file.parent == output_root / f"weather-watch-{job['id']}"

    def test_empty_slug_falls_back_to_plain_id(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="日本語")
        output_file = save_job_output(job["id"], "report", job_name=job["name"])
        assert output_file.parent == output_root / job["id"]

    def test_unknown_job_without_name_uses_plain_id(self, output_root):
        output_file = save_job_output("abc123def456", "report")
        assert output_file.parent == output_root / "abc123def456"

    def test_migrates_legacy_id_dir(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Daily News")
        legacy_dir = output_root / job["id"]
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "2026-01-01_00-00-00.md").write_text("old run", encoding="utf-8")

        save_job_output(job["id"], "new run", job_name=job["name"])

        new_dir = output_root / f"daily-news-{job['id']}"
        assert not legacy_dir.exists()
        assert (new_dir / "2026-01-01_00-00-00.md").read_text() == "old run"
        assert len(list(new_dir.glob("*.md"))) == 2

    def test_migrates_stale_slug_dir_after_rename(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="New Name")
        stale_dir = output_root / f"old-name-{job['id']}"
        stale_dir.mkdir(parents=True)
        (stale_dir / "2026-01-01_00-00-00.md").write_text("old run", encoding="utf-8")

        save_job_output(job["id"], "new run", job_name="New Name")

        new_dir = output_root / f"new-name-{job['id']}"
        assert not stale_dir.exists()
        assert (new_dir / "2026-01-01_00-00-00.md").read_text() == "old run"


# =========================================================================
# Resolver semantics
# =========================================================================

class TestResolveJobOutputDir:
    def test_read_resolution_does_not_rename(self, output_root):
        legacy_dir = output_root / "abc123def456"
        legacy_dir.mkdir(parents=True)

        resolved = resolve_job_output_dir("abc123def456", "Daily News")

        assert resolved == legacy_dir
        assert legacy_dir.exists()
        assert not (output_root / "daily-news-abc123def456").exists()

    def test_read_resolution_finds_slug_dir_without_name(self, output_root):
        slug_dir = output_root / "daily-news-abc123def456"
        slug_dir.mkdir(parents=True)
        assert resolve_job_output_dir("abc123def456") == slug_dir

    def test_prefers_canonical_dir_when_both_exist(self, output_root):
        (output_root / "abc123def456").mkdir(parents=True)
        canonical = output_root / "daily-news-abc123def456"
        canonical.mkdir(parents=True)
        assert resolve_job_output_dir("abc123def456", "Daily News") == canonical

    def test_resolved_dir_always_under_output_root(self, output_root):
        hostile_names = ["../../etc", "/etc/passwd", "..", "a/../../b", "C:\\evil"]
        for name in hostile_names:
            resolved = resolve_job_output_dir("abc123def456", name)
            assert resolved.parent == output_root
            assert resolved.name.endswith("abc123def456")

    @pytest.mark.parametrize("bad_id", ["..", "../escape", "a/b", "a\\b", "", "/abs"])
    def test_unsafe_job_id_fails_closed(self, output_root, bad_id):
        with pytest.raises(ValueError, match="output path"):
            resolve_job_output_dir(bad_id, "Daily News")

    def test_unsafe_job_id_fails_closed_on_save(self, output_root):
        with pytest.raises(ValueError, match="output path"):
            save_job_output("../escape", "report", job_name="Daily News")


# =========================================================================
# Reader / chaining compatibility (scheduler context_from)
# =========================================================================

class TestContextFromReadsNewNaming:
    def test_reads_output_written_under_new_naming(self, tmp_cron_dir):
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h", name="News Fetch")
        save_job_output(job_a["id"], "Top story: AI everywhere.", job_name=job_a["name"])

        job_b = create_job(
            prompt="Summarize the news",
            schedule="every 2h",
            context_from=job_a["id"],
        )
        prompt = _build_job_prompt(job_b)
        assert "Top story: AI everywhere." in prompt

    def test_still_reads_legacy_id_dir(self, output_root, tmp_cron_dir):
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h", name="News Fetch")
        legacy_dir = output_root / job_a["id"]
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "2026-01-01_00-00-00.md").write_text("Legacy story.", encoding="utf-8")

        job_b = create_job(
            prompt="Summarize the news",
            schedule="every 2h",
            context_from=job_a["id"],
        )
        prompt = _build_job_prompt(job_b)
        assert "Legacy story." in prompt
        # Read path never migrates.
        assert legacy_dir.exists()


# =========================================================================
# Removal cleans up both naming forms
# =========================================================================

class TestRemoveJobCleansBothForms:
    def test_removes_new_style_dir(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Daily News")
        save_job_output(job["id"], "report", job_name=job["name"])
        new_dir = output_root / f"daily-news-{job['id']}"
        assert new_dir.exists()

        assert remove_job(job["id"]) is True
        assert not new_dir.exists()

    def test_removes_legacy_and_stale_slug_dirs(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Daily News")
        legacy_dir = output_root / job["id"]
        legacy_dir.mkdir(parents=True)
        stale_dir = output_root / f"old-name-{job['id']}"
        stale_dir.mkdir(parents=True)

        assert remove_job(job["id"]) is True
        assert not legacy_dir.exists()
        assert not stale_dir.exists()

    def test_does_not_remove_other_jobs_dirs(self, output_root):
        job = create_job(prompt="x", schedule="every 1h", name="Daily News")
        other_dir = output_root / "other-job-ffffffffffff"
        other_dir.mkdir(parents=True)

        assert remove_job(job["id"]) is True
        assert other_dir.exists()

    def test_unsafe_legacy_id_fails_closed_without_removal(self, tmp_cron_dir, output_root):
        job = create_job(prompt="Legacy unsafe", schedule="every 1h")
        job["id"] = "../escape"
        save_jobs([job])
        outside = tmp_cron_dir / "escape"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(ValueError, match="output path"):
            remove_job("../escape")

        assert load_jobs()[0]["id"] == "../escape"
        assert (outside / "keep.txt").exists()
