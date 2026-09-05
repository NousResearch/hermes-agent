"""Regression tests for cron imports in environments with a third-party utils module."""

import importlib
import sys
import types


def test_cron_jobs_does_not_use_unrelated_utils_module(monkeypatch):
    unrelated_utils = types.ModuleType("utils")
    unrelated_utils.atomic_replace = None
    monkeypatch.setitem(sys.modules, "utils", unrelated_utils)
    for name in ("cron.jobs", "cron._hermes_utils"):
        sys.modules.pop(name, None)

    jobs = importlib.import_module("cron.jobs")

    assert callable(jobs.atomic_replace)
    assert callable(jobs.atomic_write_text)