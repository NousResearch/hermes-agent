"""`hermes pm environment`: the shell entry skills use for a named side environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import pm
from pm import cli, runtime

PROJECT = Path(cli.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _in_pm_runtime(monkeypatch):
    # The command body as it runs once `hermes pm` is inside PM's runtime;
    # repo_root() would consult the worktree manifest under the real home.
    monkeypatch.setattr(runtime, "is_runtime", lambda: True)
    monkeypatch.setattr(cli, "repo_root", lambda: PROJECT)


def _noisy(result: Path, calls: list):
    def ensure(*args, **kwargs):
        calls.append((args, kwargs))
        # Progress from PM (and its uv children) is written to fd 1 directly.
        os.write(1, b"  -> Installing Python dependencies\n")
        print("  ok Installing Python dependencies")
        return result
    return ensure


def test_stdout_is_only_the_interpreter_for_command_substitution(monkeypatch, capfd, tmp_path):
    calls: list = []
    python = tmp_path / "gen-x" / "venv" / "bin" / "python"
    monkeypatch.setattr(pm, "ensure_environment", _noisy(python, calls))

    assert cli.main(["environment", "documents", "--extra", "documents", "--extra", "documents-pdf"]) == 0

    out, err = capfd.readouterr()
    assert out == f"{python}\n"
    assert "Installing Python dependencies" in err
    (name, requirements), kwargs = calls[0]
    assert name == "documents"
    # An agent asking for it is an explicit install, even where lazy installs are off.
    assert kwargs == {"explicit": True}
    # Both extras' pins: every skill passing the same flags shares one generation.
    pinned = {requirement.split("==")[0] for requirement in requirements}
    assert {"python-docx", "openpyxl", "python-pptx", "reportlab", "pypdf", "pdfplumber", "pypdfium2"} <= pinned
    assert all("==" in requirement for requirement in requirements)


def test_extra_expands_self_references_and_rejects_unknown(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n'
        '[project.optional-dependencies]\n'
        'a = ["pkg-a==1.0"]\nb = ["demo[a]", "pkg-b==2.0"]\n', encoding="utf-8")

    assert sorted(cli._extra_requirements(tmp_path, ["b"])) == ["pkg-a==1.0", "pkg-b==2.0"]
    with pytest.raises(ValueError, match="not an extra"):
        cli._extra_requirements(tmp_path, ["missing"])


def test_requirements_select_an_ad_hoc_environment(monkeypatch, capfd, tmp_path):
    calls: list = []
    python = tmp_path / "python"
    monkeypatch.setattr(pm, "ensure_environment", _noisy(python, calls))

    assert cli.main(["environment", "pdf-pymupdf", "--requirement", "pymupdf>=1.26,<3"]) == 0

    assert capfd.readouterr().out == f"{python}\n"
    assert calls == [(("pdf-pymupdf", ["pymupdf>=1.26,<3"]), {"explicit": True})]


@pytest.mark.parametrize("flags", [[], ["--extra", "documents", "--requirement", "pypdf>=6,<7"]])
def test_exactly_one_source_of_requirements(monkeypatch, capfd, flags):
    def refuse(*_args, **_kwargs):
        raise AssertionError("must not build")
    monkeypatch.setattr(pm, "ensure_environment", refuse)

    assert cli.main(["environment", "documents", *flags]) == 2
    assert capfd.readouterr().out == ""


def test_a_failed_build_reports_on_stderr_and_restores_stdout(monkeypatch, capfd):
    def fail(*_args, **_kwargs):
        raise pm.InstallError("documents", "no network")
    monkeypatch.setattr(pm, "ensure_environment", fail)

    assert cli.main(["environment", "documents", "--extra", "documents"]) == 1
    print("after")
    out, err = capfd.readouterr()
    assert out == "after\n"
    assert "no network" in err
