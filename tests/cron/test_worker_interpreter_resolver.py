"""Cron worker interpreter resolver: no rejected candidate goes silent.

The silent-drop class this pins: the resolver probes candidates in order and
returned early on the first healthy one, discarding every rejected candidate
without a word — including HERMES_CRON_PYTHON, the operator's own pin. An
override that fails its probe and vanishes is indistinguishable from an
override that ran and found nothing; an instrument that never ran leaves no
evidence behind. Silence must never read as positive attestation.
"""

from __future__ import annotations

import logging
import sys

import pytest


@pytest.fixture
def resolver(tmp_path, monkeypatch):
    """The resolver with a hermetic candidate set.

    Repo root is redirected at tmp_path so the venv candidate is a real file
    the test controls; the memo cache is reset per test; probe results are
    decided by membership in ``healthy``, set by each test.
    """
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_CRON_WORKER_PYTHON_OK", [])
    monkeypatch.setattr(scheduler, "_cron_repo_root", lambda: str(tmp_path))
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("")

    probed: list[str] = []
    state = {"healthy": set()}

    def fake_probe(candidate: str) -> bool:
        probed.append(candidate)
        return candidate in state["healthy"]

    monkeypatch.setattr(scheduler, "_interpreter_is_healthy", fake_probe)
    return scheduler, str(venv_python), probed, state


def _override_file(tmp_path, monkeypatch):
    override = tmp_path / "operator-pin"
    override.write_text("")
    monkeypatch.setenv("HERMES_CRON_PYTHON", str(override))
    return str(override)


def test_override_failure_is_named_when_a_later_candidate_passes(
    tmp_path, monkeypatch, caplog, resolver
):
    """Acceptance case: override fails its probe, the next candidate passes —
    the rejected override must be logged BY NAME, flagged as the operator
    override, and the chosen interpreter stated. Never a silent drop."""
    scheduler, _venv, _probed, state = resolver
    override = _override_file(tmp_path, monkeypatch)
    state["healthy"] = {sys.executable}

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        chosen = scheduler._cron_worker_python()

    assert chosen == sys.executable
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        override in m and "HERMES_CRON_PYTHON" in m and "NOT in effect" in m
        for m in msgs
    ), f"override drop was silent; warnings were: {msgs}"


def test_missing_override_is_also_surfaced(tmp_path, monkeypatch, caplog, resolver):
    """An override pointing at a path that does not exist (a deleted wrapper)
    is the same silence class: skipped, not spoken about, would read as
    'the instrument never needed to say anything.'"""
    scheduler, _venv, _probed, state = resolver
    monkeypatch.setenv("HERMES_CRON_PYTHON", str(tmp_path / "gone" / "python"))
    state["healthy"] = {sys.executable}

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        chosen = scheduler._cron_worker_python()

    assert chosen == sys.executable
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("gone/python" in m for m in msgs), f"skipped override was silent: {msgs}"


def test_healthy_override_selects_without_noise(tmp_path, monkeypatch, caplog, resolver):
    """A passing override means later candidates were never probed — nothing
    was rejected, so nothing is (or should be) logged."""
    scheduler, _venv, probed, state = resolver
    override = _override_file(tmp_path, monkeypatch)
    state["healthy"] = {override}

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        chosen = scheduler._cron_worker_python()

    assert chosen == override
    assert probed == [override]
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_all_candidates_fail_names_every_rejected(
    tmp_path, monkeypatch, caplog, resolver
):
    """The raise was always loud; it stays loud and every rejected path is in
    it, override explicitly flagged."""
    scheduler, _venv, _probed, state = resolver
    override = _override_file(tmp_path, monkeypatch)
    state["healthy"] = set()

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        with pytest.raises(RuntimeError) as excinfo:
            scheduler._cron_worker_python()

    assert override in str(excinfo.value)
    assert sys.executable in str(excinfo.value)


def test_non_override_rejection_is_named(tmp_path, monkeypatch, caplog, resolver):
    """sys.executable unhealthy, venv healthy: the rejected candidate is named
    with the chosen one, without the override flag."""
    scheduler, venv, _probed, state = resolver
    monkeypatch.delenv("HERMES_CRON_PYTHON", raising=False)
    state["healthy"] = {venv}

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        chosen = scheduler._cron_worker_python()

    assert chosen == venv
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        sys.executable in m and "HERMES_CRON_PYTHON" not in m and venv in m
        for m in msgs
    ), f"plain rejection silent or misflagged: {msgs}"
