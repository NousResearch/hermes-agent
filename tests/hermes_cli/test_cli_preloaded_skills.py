from __future__ import annotations

import importlib
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


class _DummyCLI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = "session-123"
        self.system_prompt = "base prompt"
        self.preloaded_skills = []
        # Preload-machinery attributes the real finalize reads (getattr defaults).
        self._preload_skills_finalized = False
        self._preload_skills_thread: _JoinableThread | None = None
        self._preload_skills_error: Exception | None = None
        self._auto_load_skills_result: tuple | None = None
        self._preload_skills_result: tuple | None = None

    def show_banner(self):
        return None

    def show_tools(self):
        return None

    def show_toolsets(self):
        return None

    def run(self):
        return None


def _real_finalize(cli_obj):
    """Call the real HermesCLI.finalize_preloaded_skills on a dummy object."""
    return _REAL_FINALIZE(cli_obj)


class _JoinableThread:
    """Minimal stand-in for the preload thread: join() returns immediately."""

    def join(self, timeout=None):
        return None


def _finalize_stub(monkeypatch, preload_result, *, kanban_task=None, auto_result=None):
    """A _DummyCLI prepped so the REAL finalize reaches the result triple.

    ``kanban_task`` controls the dispatched-worker provenance env: None clears
    HERMES_KANBAN_TASK (human CLI), a task id sets it (dispatcher spawn).
    """
    cli_obj = _DummyCLI()
    cli_obj._preload_skills_finalized = False
    cli_obj._preload_skills_thread = _JoinableThread()
    cli_obj._preload_skills_error = None
    cli_obj._auto_load_skills_result = auto_result
    cli_obj._preload_skills_result = preload_result
    if kanban_task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)
    return cli_obj


def _capture_real_finalize():
    import cli as cli_mod
    return cli_mod.HermesCLI.__dict__["finalize_preloaded_skills"]


_REAL_FINALIZE = _capture_real_finalize()


def test_main_applies_preloaded_skills_to_system_prompt(monkeypatch):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: ("skill prompt", ["hermes-agent-dev", "github-auth"], []),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="hermes-agent-dev,github-auth", list_tools=True)

    cli_obj = created["cli"]
    # The preload now runs in a background thread and is folded in at agent
    # init via finalize_preloaded_skills() (startup-latency change). Drive
    # the finalize explicitly — the same call _init_agent makes.
    _real_finalize(cli_obj)
    assert cli_obj.system_prompt == "base prompt\n\nskill prompt"
    assert cli_obj.preloaded_skills == ["hermes-agent-dev", "github-auth"]


def test_main_raises_for_unknown_preloaded_skill(monkeypatch):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: ("", [], ["missing-skill"]),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="missing-skill", list_tools=True)

    # The all-skills-unknown hard failure now surfaces when the preload is
    # finalized (agent init), preserving the fail-loud contract.
    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-skill"):
        _real_finalize(created["cli"])


def test_finalize_strict_raise_without_kanban_provenance(monkeypatch):
    """Invariant (human CLI, t_ba6a1061): without dispatched-kanban provenance
    (HERMES_KANBAN_TASK unset), an ALL-unknown --skills set still fails loudly
    at finalize — the typo-protection contract is preserved."""
    cli_obj = _finalize_stub(monkeypatch, ("", [], ["missing-skill"]))
    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-skill"):
        _real_finalize(cli_obj)


def test_finalize_dispatched_kanban_worker_degrades_instead_of_dying(monkeypatch, caplog):
    """Invariant (t_ba6a1061, live defect t_11f61c5a): a dispatched kanban
    worker (HERMES_KANBAN_TASK set) must not die startup-fatal on an
    unresolvable machine-built --skills set — a startup-fatal spawn retries
    identically every tick until gave_up parks the card. It logs the skip and
    starts without preloaded skills instead."""
    cli_obj = _finalize_stub(
        monkeypatch, ("", [], ["sdlc-review"]), kanban_task="t_11f61c5a"
    )
    with caplog.at_level(logging.WARNING, logger="cli"):
        _real_finalize(cli_obj)  # must NOT raise
    assert cli_obj.system_prompt == "base prompt"
    assert cli_obj.preloaded_skills == []
    assert any(
        "Continuing without preloaded skills" in record.getMessage()
        for record in caplog.records
    )


def test_finalize_partial_success_still_folds_loaded_prompt(monkeypatch, caplog):
    """Positive control: when SOME requested skills resolve, the loaded prompt
    is folded in and only the unknown names are skipped — unchanged behavior,
    regardless of kanban provenance."""
    cli_obj = _finalize_stub(
        monkeypatch, ("skill prompt", ["github-auth"], ["sdlc-review"]), kanban_task=None
    )
    with caplog.at_level(logging.WARNING, logger="cli"):
        _real_finalize(cli_obj)
    assert cli_obj.system_prompt == "base prompt\n\nskill prompt"
    assert cli_obj.preloaded_skills == ["github-auth"]
    assert any(
        "Unknown skill(s) requested, skipping" in record.getMessage()
        for record in caplog.records
    )
