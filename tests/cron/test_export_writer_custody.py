"""Causal contracts for cron writers participating in export custody."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cron import jobs
from hermes_constants import (
    get_hermes_home,
    mark_named_profile_deleted,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cron_export_custody import CronCustodyRefused, cron_capture_custody


@pytest.fixture()
def profile_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _assert_writer_finished(
    writer: threading.Thread,
    release_writer: threading.Event,
    errors: list[BaseException],
) -> None:
    release_writer.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    assert errors == []


def test_capture_refuses_during_actual_output_write_and_preserves_bytes(
    profile_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = jobs.create_job(prompt="output job", schedule="every 1h")
    expected = "writer output preserved exactly"
    writer_started = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    original_write = jobs.atomic_write_text

    def blocked_write(path: Path, content: str, **kwargs: object) -> None:
        if content == expected:
            writer_started.set()
            assert release_writer.wait(timeout=5.0)
        original_write(path, content, **kwargs)

    monkeypatch.setattr(jobs, "atomic_write_text", blocked_write)

    def save_output() -> None:
        try:
            jobs.save_job_output(job["id"], expected)
        except BaseException as exc:  # preserve the real writer failure for the assertion below
            errors.append(exc)

    writer = threading.Thread(target=save_output)
    writer.start()
    try:
        assert writer_started.wait(timeout=5.0)
        with pytest.raises(CronCustodyRefused) as refusal:
            with cron_capture_custody(timeout_seconds=2.0):
                pass
        assert refusal.value.code == "cron_custody_busy"
    finally:
        _assert_writer_finished(writer, release_writer, errors)

    output_files = sorted((profile_home / "cron" / "output" / job["id"]).glob("*.md"))
    assert len(output_files) == 1
    assert output_files[0].read_text(encoding="utf-8") == expected


def test_capture_refuses_during_actual_jobs_update_and_preserves_record(
    profile_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = jobs.create_job(prompt="before update", schedule="every 1h", name="before")
    before = jobs.get_job(job["id"])
    assert before is not None
    updates = {"name": "after", "prompt": "after update"}
    writer_started = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    original_save = jobs._save_jobs_unlocked

    def blocked_save(payload: list[dict[str, object]], **kwargs: object) -> None:
        writer_started.set()
        assert release_writer.wait(timeout=5.0)
        original_save(payload, **kwargs)

    monkeypatch.setattr(jobs, "_save_jobs_unlocked", blocked_save)

    def update_job() -> None:
        try:
            jobs.update_job(job["id"], updates)
        except BaseException as exc:  # preserve the real writer failure for the assertion below
            errors.append(exc)

    writer = threading.Thread(target=update_job)
    writer.start()
    try:
        assert writer_started.wait(timeout=5.0)
        with pytest.raises(CronCustodyRefused) as refusal:
            with cron_capture_custody(timeout_seconds=2.0):
                pass
        assert refusal.value.code == "cron_custody_busy"
    finally:
        _assert_writer_finished(writer, release_writer, errors)

    after = jobs.get_job(job["id"])
    assert after is not None
    expected = dict(before)
    expected.update(updates)
    assert after == expected


def test_use_cron_store_binds_writer_custody_to_store_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_a))

    with jobs.use_cron_store(home_b):
        job = jobs.create_job(prompt="store B", schedule="every 1h")
    assert (home_b / "cron" / "jobs.json").is_file()
    assert not (home_a / "cron" / "jobs.json").exists()

    expected = "store B output"
    writer_started = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    original_write = jobs.atomic_write_text

    def blocked_write(path: Path, content: str, **kwargs: object) -> None:
        if content == expected:
            writer_started.set()
            assert release_writer.wait(timeout=5.0)
        original_write(path, content, **kwargs)

    monkeypatch.setattr(jobs, "atomic_write_text", blocked_write)

    def save_store_b_output() -> None:
        try:
            with jobs.use_cron_store(home_b):
                jobs.save_job_output(job["id"], expected)
        except BaseException as exc:  # preserve the real writer failure for the assertion below
            errors.append(exc)

    writer = threading.Thread(target=save_store_b_output)
    writer.start()
    try:
        assert writer_started.wait(timeout=5.0)

        # A capture must remain independent while the producer owns B.
        with cron_capture_custody(timeout_seconds=2.0):
            pass

        # Bind the capture caller to the producer's actual store before probing B.
        token = set_hermes_home_override(home_b)
        try:
            with pytest.raises(CronCustodyRefused) as refusal:
                with cron_capture_custody(timeout_seconds=2.0):
                    pass
            assert refusal.value.code == "cron_custody_busy"
        finally:
            reset_hermes_home_override(token)
    finally:
        _assert_writer_finished(writer, release_writer, errors)

    output_files = sorted((home_b / "cron" / "output" / job["id"]).glob("*.md"))
    assert len(output_files) == 1
    assert output_files[0].read_text(encoding="utf-8") == expected
    assert not list((home_a / "cron" / "output").glob("**/*.md"))
    assert get_hermes_home() == home_a


def test_default_and_custom_first_creation_preserve_their_own_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_home = tmp_path / "default-home"
    custom_home = tmp_path / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    default_job = jobs.create_job(prompt="default home", schedule="every 1h")
    with jobs.use_cron_store(custom_home):
        custom_job = jobs.create_job(prompt="custom home", schedule="every 1h")

    assert (default_home / "cron" / "jobs.json").is_file()
    assert (custom_home / "cron" / "jobs.json").is_file()
    assert jobs.load_jobs()[0]["id"] == default_job["id"]
    with jobs.use_cron_store(custom_home):
        assert jobs.load_jobs()[0]["id"] == custom_job["id"]


def test_missing_named_profile_is_not_resurrected_by_first_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    deleted_home = profiles_root / "deleted"
    mark_named_profile_deleted(deleted_home)
    monkeypatch.setenv("HERMES_HOME", str(deleted_home))

    with pytest.raises(FileNotFoundError):
        jobs.create_job(prompt="must not create", schedule="every 1h")

    assert not deleted_home.exists()
    assert (profiles_root / ".deleted" / deleted_home.name).read_text(encoding="utf-8") == "deleted\n"
