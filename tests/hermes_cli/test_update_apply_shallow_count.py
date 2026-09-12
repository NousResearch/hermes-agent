"""The real apply planner must not trust ancestry counts from shallow Git."""

import subprocess
from unittest.mock import Mock

import pytest

from hermes_cli import banner, main, update_cmd

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.mark.parametrize("shallow,raw_count,api_count,expected", [
    (False, 7, None, 7),
    (True, 9980, 12, 12),
    (True, 9980, None, -1),
    (True, 3, 0, 0),
    (True, 0, None, 0),
], ids=["full", "shallow-api", "shallow-offline", "local-ahead", "equal-tips"])
def test_apply_plan_counts_supplied_git_results(monkeypatch, tmp_path, shallow, raw_count, api_count, expected):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    responses = {
        ("status", "--porcelain"): "",
        ("rev-list", "HEAD..origin/main", "--count"): str(raw_count),
        ("rev-parse", "--is-shallow-repository"): "true" if shallow else "false",
        ("rev-parse", "HEAD"): SHA_A,
        ("rev-parse", "origin/main"): SHA_B,
    }

    def git(cmd, **kwargs):
        assert cmd[0] == "git"
        return subprocess.CompletedProcess(cmd, 0, responses[tuple(cmd[1:])] + "\n", "")

    monkeypatch.setattr(subprocess, "run", git)
    compare = Mock(return_value=api_count)
    monkeypatch.setattr(banner, "_github_compare_behind", compare)
    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=False,
        gateway_mode=False, gw_input_fn=None, switch_branch=False,
        _windows_gateway_resume=None,
    )
    assert plan.commit_count == expected
    if shallow and raw_count:
        compare.assert_called_once_with(SHA_A, SHA_B)
    else:
        compare.assert_not_called()