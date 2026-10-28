"""`gateway run` refuses a scratch runtime; --allow-scratch is the logged test override (t_0012c421)."""

import logging

import pytest

from hermes_cli import gateway_run_guard as guard

SCRATCH_VENV = "/Users/x/.hermes/profiles/cto/cache/scratch/t_c7ce02ae/venv/bin/python"
NORMAL_TREE = "/Users/x/.hermes/hermes-agent/venv/bin/python"


@pytest.mark.parametrize(
    "path, scratch",
    [
        (SCRATCH_VENV, True),
        ("/Users/x/.hermes/profiles/hub-dev/cache/scratch/t_1", True),
        (NORMAL_TREE, False),
        ("/Users/x/.hermes/profiles/cto/cache", False),
        ("/Users/x/scratch-notes/cache", False),
    ],
)
def test_is_scratch_path(path, scratch):
    assert guard.is_scratch_path(path) is scratch


def test_run_from_scratch_is_refused(capsys):
    with pytest.raises(SystemExit) as refused:
        guard.enforce_production_runtime(candidates=[SCRATCH_VENV, NORMAL_TREE])

    assert refused.value.code != 0
    assert capsys.readouterr().err.strip() == (
        f"refusing to run production gateway from scratch dir {SCRATCH_VENV}"
    )


def test_normal_runtime_passes(capsys):
    guard.enforce_production_runtime(
        candidates=[NORMAL_TREE, "/Users/x/.hermes/hermes-agent"]
    )
    assert capsys.readouterr().err == ""


def test_allow_scratch_overrides_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.enforce_production_runtime(allow_scratch=True, candidates=[SCRATCH_VENV])

    assert any(
        r.levelno == logging.WARNING and SCRATCH_VENV in r.getMessage()
        for r in caplog.records
    )


def test_runtime_candidates_cover_interpreter_prefix_and_package():
    import sys

    candidates = guard.runtime_candidates()
    assert sys.executable in candidates and sys.prefix in candidates
    assert any(c.endswith("hermes_cli") for c in candidates)
