"""A launch on dependencies without google-re2 refuses to start; the repair and diagnostics still run.

When a post-update dependency sync fails, ``hermes_bootstrap`` boots the new tree on the previous
dependency generation. Once product code matches through ``hermes_re``, a generation without the
engine makes every match raise mid-turn, so ``guard_regex_engine`` (called by the bootstrap after
activation) refuses that launch with the repair hint (``hermes pm install``, which needs no regex).
``hermes update`` stays startable (where its launch retries the sync it is a repair), as do its
probes and completion tails, the package manager, the diagnostics and the commands that stop a
gateway; ``hermes update`` itself stops before its first step while the engine is still missing.
An engine that cannot be located (a finder refusing it, a stub, a bare directory) counts as
missing; the guard never raises. Plan decision D29 (c).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, venv_sync


@pytest.fixture
def no_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "re2", None)  # Python's spelling of "this module cannot be imported"


@pytest.fixture
def engine(monkeypatch):
    installed = types.ModuleType("re2")
    installed.__spec__ = importlib.machinery.ModuleSpec("re2", None, origin="site-packages/re2.py")
    monkeypatch.setitem(sys.modules, "re2", installed)


def _launch(root: Path, argv: list[str], monkeypatch, *, argv0: str = "hermes") -> int:
    """Exit status of the guard for a process started as ``argv0 *argv`` (0 = it boots)."""
    monkeypatch.setattr(sys, "argv", [argv0, *argv])
    try:
        venv_sync.guard_regex_engine(root, argv)
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.mark.parametrize("argv", [
    pytest.param([], id="chat-repl-and-non-cli-entry-points"),
    pytest.param(["chat", "-q", "hello"], id="one-shot-chat"),
    pytest.param(["gateway", "run"], id="gateway-service"),
    pytest.param(["-p", "work", "gateway", "run", "--replace"], id="profile-gateway"),
    pytest.param(["gateway", "start"], id="gateway-start"),
    pytest.param(["gateway", "restart"], id="gateway-restart"),
    pytest.param(["serve"], id="desktop-backend"),
    pytest.param(["cron", "list"], id="cron"),
])
def test_runtime_launches_refuse_with_the_repair_hint(tmp_path, no_engine, monkeypatch, capsys, argv):
    assert _launch(tmp_path, argv, monkeypatch) == 1
    err = capsys.readouterr().err
    assert "cannot start" in err and "google-re2" in err
    assert "`hermes pm install`" in err and "`hermes update`" in err  # a repair that needs no regex first


@pytest.mark.parametrize("argv", [
    pytest.param(["update"], id="update-and-the-import-health-probe-argv-contract"),
    pytest.param(["update", "--check"], id="report-only-update"),
    pytest.param(["-p", "work", "update", "--yes"], id="profile-update"),
    pytest.param(["update", "--gateway"], id="chat-update"),
])
def test_the_repair_starts_silently(tmp_path, no_engine, monkeypatch, capsys, argv):
    """``hermes update`` re-runs the sync at launch and judges its own run; shipped updaters import
    the CLI as ``['hermes', 'update']`` and read any exit as a broken checkout."""
    assert _launch(tmp_path, argv, monkeypatch) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("tail", ["update_completion.py", "source_completion.py"])
def test_update_completion_tails_are_never_refused(tmp_path, no_engine, monkeypatch, capsys, tail):
    root = tmp_path.resolve()
    script = str(root / "hermes_cli" / tail)
    assert _launch(root, ["request.json", "result.json", "--prepared"], monkeypatch, argv0=script) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("argv", [
    pytest.param(["doctor"], id="doctor"),
    pytest.param(["-p", "work", "doctor"], id="profile-doctor"),
    pytest.param(["pm", "sync"], id="package-manager"),
    pytest.param(["gateway", "status"], id="gateway-status"),
    pytest.param(["gateway", "stop"], id="gateway-stop-ends-a-supervisor-restart-loop"),
    pytest.param(["-p", "work", "gateway", "uninstall"], id="gateway-uninstall"),
    pytest.param(["--version"], id="version"),
    pytest.param(["chat", "--help"], id="help"),
])
def test_diagnostics_start_with_a_warning(tmp_path, no_engine, monkeypatch, capsys, argv):
    assert _launch(tmp_path, argv, monkeypatch) == 0
    err = capsys.readouterr().err
    assert "warning" in err and "google-re2" in err and "`hermes pm install`" in err


@pytest.mark.parametrize("argv", [[], ["gateway", "run"], ["doctor"], ["update"]])
def test_an_installed_engine_changes_nothing(tmp_path, engine, monkeypatch, capsys, argv):
    assert _launch(tmp_path, argv, monkeypatch) == 0
    assert capsys.readouterr().err == ""


class _RefuseEngine(importlib.abc.MetaPathFinder):
    """The finder ``tests/test_regex_engine_optional.py`` blocks the engine with: it raises."""

    def find_spec(self, name, path=None, target=None):
        if name == "re2":
            raise ImportError("re2 is blocked")


def _refusing_finder(monkeypatch):
    monkeypatch.delitem(sys.modules, "re2", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_RefuseEngine(), *sys.meta_path])


def _stub_without_spec(monkeypatch):
    stub = types.ModuleType("re2")
    stub.__spec__ = None
    monkeypatch.setitem(sys.modules, "re2", stub)


def _namespace_portion(monkeypatch):
    # What a bare re2/ directory on sys.path is located as: a package spec with no code (origin None).
    portion = types.ModuleType("re2")
    portion.__spec__ = importlib.machinery.ModuleSpec("re2", None, is_package=True)
    monkeypatch.setitem(sys.modules, "re2", portion)


@pytest.mark.parametrize("unusable", [
    pytest.param(_refusing_finder, id="finder-raises-importerror"),
    pytest.param(_stub_without_spec, id="module-without-spec"),
    pytest.param(_namespace_portion, id="bare-directory"),
])
def test_an_engine_that_cannot_be_located_counts_as_missing(tmp_path, monkeypatch, capsys, unusable):
    """Never an exception out of the guard: it runs before the ``update`` exemption, and the
    import-health probe imports the CLI as ``['hermes', 'update']``."""
    unusable(monkeypatch)
    assert venv_sync.regex_engine_missing()
    assert _launch(tmp_path, ["update"], monkeypatch) == 0
    assert capsys.readouterr().err == ""
    assert _launch(tmp_path, ["gateway", "run"], monkeypatch) == 1
    assert "cannot start" in capsys.readouterr().err


def test_hermes_update_stops_before_its_first_step_without_the_engine(no_engine, monkeypatch, capsys):
    steps = []
    monkeypatch.setattr(update_cmd._commit, "begin_update_attempt", lambda: steps.append("began"))
    with pytest.raises(SystemExit) as stopped:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)
    assert stopped.value.code == 1
    assert steps == []
    out = capsys.readouterr().out
    assert "Cannot update" in out and "google-re2" in out and "re-run `hermes update`" in out
    assert "`hermes pm install`" in out  # installs without a launch that retries the sync (ZIP install)


def test_hermes_update_proceeds_with_the_engine(engine, monkeypatch):
    class Began(Exception):
        pass

    def began():
        raise Began

    monkeypatch.setattr(update_cmd._commit, "begin_update_attempt", began)
    with pytest.raises(Began):
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)
