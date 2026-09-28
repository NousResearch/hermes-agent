"""Tests for cron.scheduler_worker_env.pin_hermes_tree_on_pythonpath.

Covers:
  - repo_root is prepended to PYTHONPATH, existing entries preserved
  - the PM-managed dependency environment's site-packages is also prepended
    when one is selected for repo_root (#112729 follow-up: a worker spawned
    via ``-m cron.scheduler`` never runs ``pm.environments.activate_dependencies``,
    so third-party imports fail under the PM-managed environment layout)
  - resolution failures (no PM environment committed, etc.) are swallowed and
    only repo_root gets pinned, same as before this existed
  - repo_root is skipped when it equals the interpreter's own purelib (wheel /
    pipx / uv-tool installs)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cron import scheduler_worker_env as swe


def test_pin_prepends_repo_root_and_keeps_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(swe, "_installed_purelib", lambda: None)
    monkeypatch.setattr(swe, "_pm_dependency_site_packages", lambda repo_root: None)
    env = swe.pin_hermes_tree_on_pythonpath({"PYTHONPATH": "/existing"}, tmp_path)
    assert env["PYTHONPATH"].split(":") == [str(tmp_path), "/existing"]


def test_pin_adds_pm_dependency_site_packages(monkeypatch, tmp_path):
    dep_site_packages = tmp_path / "pm-venv" / "site-packages"
    monkeypatch.setattr(swe, "_installed_purelib", lambda: None)
    monkeypatch.setattr(swe, "_pm_dependency_site_packages", lambda repo_root: dep_site_packages)
    env = swe.pin_hermes_tree_on_pythonpath({}, tmp_path)
    assert env["PYTHONPATH"].split(":") == [str(tmp_path), str(dep_site_packages)]


def test_pin_skips_pm_dependency_when_unresolvable(monkeypatch, tmp_path):
    monkeypatch.setattr(swe, "_installed_purelib", lambda: None)
    monkeypatch.setattr(swe, "_pm_dependency_site_packages", lambda repo_root: None)
    env = swe.pin_hermes_tree_on_pythonpath({}, tmp_path)
    assert env["PYTHONPATH"] == str(tmp_path)


def test_pin_skips_repo_root_when_it_is_the_installed_purelib(monkeypatch, tmp_path):
    monkeypatch.setattr(swe, "_installed_purelib", lambda: tmp_path.resolve())
    monkeypatch.setattr(swe, "_pm_dependency_site_packages", lambda repo_root: None)
    env = swe.pin_hermes_tree_on_pythonpath({}, tmp_path)
    assert env["PYTHONPATH"] == ""


def test_pm_dependency_site_packages_swallows_lookup_failure(monkeypatch, tmp_path):
    def _boom(project_root):
        raise RuntimeError("no dependency environment is committed for this install")

    monkeypatch.setattr("pm.environments.selected_venv", _boom)
    assert swe._pm_dependency_site_packages(tmp_path) is None
