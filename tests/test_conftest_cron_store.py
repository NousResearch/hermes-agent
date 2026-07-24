"""Compatibility coverage for the suite's imported cron-store isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from tests.conftest import _hermetic_environment


class _CronStorePaths(NamedTuple):
    cron_dir: Path
    jobs_file: Path
    output_dir: Path


@pytest.mark.parametrize(
    "supports_import_store",
    [True, False],
    ids=["newer-module", "lightweight-stub"],
)
def test_hermetic_fixture_only_aligns_existing_import_store(
    tmp_path, monkeypatch, supports_import_store
):
    cron_jobs = SimpleNamespace()
    if supports_import_store:
        import_dir = tmp_path / "imported" / "cron"
        cron_jobs._CronStorePaths = _CronStorePaths
        cron_jobs._IMPORT_STORE = _CronStorePaths(
            import_dir,
            import_dir / "jobs.json",
            import_dir / "output",
        )

    monkeypatch.setitem(sys.modules, "cron.jobs", cron_jobs)
    fixture_tmp = tmp_path / "fixture"
    fixture_tmp.mkdir()
    _hermetic_environment.__wrapped__(fixture_tmp, monkeypatch)

    if supports_import_store:
        cron_dir = fixture_tmp / "hermes_test" / "cron"
        assert cron_jobs._IMPORT_STORE == _CronStorePaths(
            cron_dir,
            cron_dir / "jobs.json",
            cron_dir / "output",
        )
    else:
        assert not hasattr(cron_jobs, "_CronStorePaths")
        assert not hasattr(cron_jobs, "_IMPORT_STORE")
