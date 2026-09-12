"""Invariants for the dashboard/serve web-stack import guard.

Two behaviours are pinned here:

1. ``_drop_cwd_from_sys_path`` keeps the working directory off ``sys.path`` for
   the dashboard/serve backend. Desktop spawns that backend with ``cwd`` set to
   the user's home directory and ``python -m`` puts it first on ``sys.path``, so
   a stray module there shadowed an installed package and the backend died
   before it could bind (a leftover ``~/email_validator.py`` shadowed pydantic's
   optional email dependency).
2. ``_web_stack_import_error_report`` names what actually failed. The guard
   around ``import fastapi`` / ``import uvicorn`` catches every ``ImportError``
   the chain raises, so a transitive failure used to be reported as "fastapi and
   uvicorn are missing" — sending the user to reinstall what they already had.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

import pytest

from hermes_cli.main_dashboard import (
    _drop_cwd_from_sys_path,
    _import_outside_environment,
    _web_stack_import_error_report,
)

REPORT_KWARGS = {"project_root": "/repo", "python_executable": "/repo/venv/bin/python"}


def test_drop_cwd_from_sys_path_removes_cwd_entries_only(monkeypatch, tmp_path):
    """Every entry resolving to the cwd goes; other absolute entries stay."""
    monkeypatch.chdir(tmp_path)
    cwd = os.path.normcase(os.path.abspath(str(tmp_path)))
    monkeypatch.setattr(
        sys, "path", ["", "/keep/me", cwd, str(tmp_path / "sub"), ""]
    )
    _drop_cwd_from_sys_path()
    assert sys.path == ["/keep/me", str(tmp_path / "sub")]


def test_drop_cwd_from_sys_path_removes_the_absolute_cwd(monkeypatch, tmp_path):
    """``python -m`` puts the absolute cwd on ``sys.path[0]``, not ``""``.

    The desktop spawns the backend as ``python -m hermes_cli.main serve`` with
    cwd set to the home directory, so this is the entry that actually shadows
    an installed package in the reported repro. Stripping only ``""`` leaves it
    in place and the backend still dies on the ``import fastapi`` chain.
    """
    monkeypatch.chdir(tmp_path)
    cwd = os.path.normcase(os.path.abspath(str(tmp_path)))
    monkeypatch.setattr(sys, "path", [cwd, "/keep/me"])
    _drop_cwd_from_sys_path()
    assert sys.path == ["/keep/me"]


def test_import_outside_environment_flags_a_stray_file(tmp_path, monkeypatch):
    """A module resolving outside every installed root is a shadowing file."""
    stray = tmp_path / "email_validator.py"
    stray.write_text("SHADOW = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("email_validator", None)

    assert _import_outside_environment("email-validator") == str(stray)

    sys.modules.pop("email_validator", None)


def test_report_names_the_shadowing_file_instead_of_blaming_fastapi(tmp_path, monkeypatch):
    """A transitive failure must not be reported as a missing web stack."""
    stray = tmp_path / "email_validator.py"
    stray.write_text("SHADOW = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("email_validator", None)

    # What pydantic's import_email_validator() raises when the stray file wins
    # the import but owns no distribution metadata.
    exc = importlib.metadata.PackageNotFoundError("email-validator")

    report = _web_stack_import_error_report(exc, **REPORT_KWARGS)

    assert "Web UI dependencies not installed" not in report
    assert "fastapi and uvicorn are installed" in report
    assert str(stray) in report
    assert "PackageNotFoundError: No package metadata was found for email-validator" in report

    sys.modules.pop("email_validator", None)


def test_report_keeps_reinstall_guidance_when_a_web_distribution_is_missing():
    """The genuine missing-fastapi case still gets the reinstall instructions."""
    exc = ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    report = _web_stack_import_error_report(exc, **REPORT_KWARGS)

    assert "Web UI dependencies not installed (need fastapi + uvicorn)." in report
    assert "cd /repo" in report
    assert "/repo/venv/bin/python -m pip install -e ." in report
    assert "Import error: ModuleNotFoundError: No module named 'fastapi'" in report


def test_report_is_honest_about_an_unshadowed_transitive_failure():
    """No fastapi/uvicorn claim and no bogus reinstall hint without evidence."""
    exc = ModuleNotFoundError("No module named 'totally_absent_thing'", name="totally_absent_thing")

    report = _web_stack_import_error_report(exc, **REPORT_KWARGS)

    assert "Web UI dependencies not installed" not in report
    assert "pip install -e ." not in report
    assert "Import error: ModuleNotFoundError: No module named 'totally_absent_thing'" in report


@pytest.mark.parametrize("name", ["fastapi", "uvicorn"])
def test_report_treats_each_web_distribution_as_the_install_case(name):
    exc = ModuleNotFoundError(f"No module named '{name}'", name=name)
    assert "Web UI dependencies not installed" in _web_stack_import_error_report(
        exc, **REPORT_KWARGS
    )
