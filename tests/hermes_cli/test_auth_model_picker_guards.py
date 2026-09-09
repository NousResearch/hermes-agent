"""Fail-closed behaviour of the post-login model picker's selection-guard gate.

``_confirm_selection_guards`` must not treat an unexpected guard-registry error as
"no warnings" — doing so silently let an unvetted model change through. On such an
error it now logs with context, tells the user, and returns ``False`` so the model
is left unchanged. A *missing* guard module stays tolerated (feature not built in).
"""

from __future__ import annotations

import logging
import sys

import pytest

from hermes_cli import auth_model_picker


def _no_input(*_args, **_kwargs):
    raise AssertionError("input() must not be reached on this path")


def test_guard_exception_blocks_selection_and_logs(monkeypatch, capsys, caplog):
    """An unexpected guard error fails closed: no prompt, model unchanged, useful log."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("guard registry exploded")

    monkeypatch.setattr("hermes_cli.model_selection_guards.selection_warnings", _boom)
    monkeypatch.setattr("builtins.input", _no_input)

    with caplog.at_level(logging.ERROR, logger="hermes_cli.auth"):
        ok = auth_model_picker._confirm_selection_guards(
            "vendor/some-model", provider="vendor"
        )

    assert ok is False
    out = capsys.readouterr().out
    assert "was not changed" in out
    assert "check the log" in out.lower()

    rec = next(r for r in caplog.records if r.levelno >= logging.ERROR)
    assert rec.exc_info is not None, "the traceback must be logged, not just a message"
    assert "vendor/some-model" in rec.getMessage(), "the log must carry the model context"
    assert "RuntimeError" in caplog.text or rec.exc_info[0] is RuntimeError


def test_no_warnings_success_path_unchanged(monkeypatch, capsys):
    """The existing 'guards ran, nothing fired' path still returns True with no prompt."""
    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.selection_warnings", lambda *_a, **_k: []
    )
    monkeypatch.setattr("builtins.input", _no_input)

    ok = auth_model_picker._confirm_selection_guards(
        "vendor/ordinary-model", provider="vendor"
    )

    assert ok is True
    assert capsys.readouterr().out == ""


def test_missing_guard_module_is_tolerated(monkeypatch):
    """ImportError == feature not in this build: nothing to verify, selection proceeds."""
    monkeypatch.setitem(sys.modules, "hermes_cli.model_selection_guards", None)
    monkeypatch.setattr("builtins.input", _no_input)

    ok = auth_model_picker._confirm_selection_guards("vendor/model", provider="vendor")

    assert ok is True


def test_warnings_still_prompt_and_decline(monkeypatch, capsys):
    """A fired warning still shows the confirm prompt; 'n' leaves the model unchanged."""
    from hermes_cli.model_selection_guards import SelectionWarning

    warning = SelectionWarning(
        kind="cost", title="Expensive Model Warning", model="m", provider="p",
        message="EXPENSIVE MODEL WARNING",
    )
    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.selection_warnings",
        lambda *_a, **_k: [warning],
    )
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

    ok = auth_model_picker._confirm_selection_guards("vendor/pricey", provider="vendor")

    assert ok is False
    assert "EXPENSIVE MODEL WARNING" in capsys.readouterr().out
