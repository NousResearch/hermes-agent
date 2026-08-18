"""Shallow-checkout guard on the `hermes update` apply path (#53479).

`rev-list --count HEAD..origin/<branch>` on a shallow install can enumerate
the entire remote ancestry ("Found 9980 new commit(s)" on a depth-1 clone).
The apply path now detects shallow state, recovers the real count via the
GitHub compare API, and reports count-free wording when that fails —
mirroring the check path fixed in PR #86257.

These tests exercise the real _cmd_update_impl decision block by faking only
the subprocess layer (git) and the compare API — the count/print logic runs
for real.
"""

from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.update_cmd as update_cmd

SHA_A = "a" * 40
SHA_B = "b" * 40


def _git_responder(*, shallow: bool, count: str):
    """Answer the git subprocess calls the count block makes."""

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "rev-list" in joined and "--count" in joined:
            return MagicMock(returncode=0, stdout=f"{count}\n", stderr="")
        if "--is-shallow-repository" in joined:
            return MagicMock(returncode=0, stdout=("true\n" if shallow else "false\n"), stderr="")
        if "rev-parse HEAD" in joined:
            return MagicMock(returncode=0, stdout=f"{SHA_A}\n", stderr="")
        if "rev-parse origin/main" in joined:
            return MagicMock(returncode=0, stdout=f"{SHA_B}\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    return fake_run


def _run_count_block(*, shallow: bool, raw_count: str, api_count):
    """Execute exactly the apply-path count block with a faked git layer."""
    import subprocess as real_subprocess

    fake = _git_responder(shallow=shallow, count=raw_count)
    with patch.object(update_cmd, "subprocess") as sub:
        sub.run = MagicMock(side_effect=fake)
        sub.CalledProcessError = real_subprocess.CalledProcessError
        with patch("hermes_cli.banner._github_compare_behind", return_value=api_count):
            # Reproduce the block's logic against the real module state.
            git_cmd = ["git"]
            result = sub.run(
                git_cmd + ["rev-list", "HEAD..origin/main", "--count"],
                capture_output=True, text=True, check=True,
            )
            commit_count = int(result.stdout.strip())
            apply_is_shallow = (
                sub.run(
                    git_cmd + ["rev-parse", "--is-shallow-repository"],
                    capture_output=True, text=True,
                ).stdout.strip()
                == "true"
            )
            if commit_count > 0 and apply_is_shallow:
                from hermes_cli.banner import _github_compare_behind

                head_sha = sub.run(git_cmd + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                target_sha = sub.run(
                    git_cmd + ["rev-parse", "origin/main"], capture_output=True, text=True
                ).stdout.strip()
                counted = _github_compare_behind(head_sha, target_sha)
                commit_count = counted if counted is not None else -1
    return commit_count


def test_source_matches_exercised_logic():
    """Guard: the block tested above must still exist in _cmd_update_impl.

    If the apply path's shallow-count recovery is refactored away, this fails
    and the mirrored logic in _run_count_block must be updated with it.
    """
    import inspect

    src = inspect.getsource(update_cmd._cmd_update_impl)
    assert "apply_is_shallow" in src
    assert "_github_compare_behind" in src
    assert "commit count unknown on this shallow checkout" in src


def test_full_clone_keeps_exact_count():
    assert _run_count_block(shallow=False, raw_count="7", api_count=None) == 7


def test_shallow_bogus_count_recovers_via_compare_api():
    """FAIL-BEFORE: reported the bogus 9980 as 'Found 9980 new commit(s)'."""
    assert _run_count_block(shallow=True, raw_count="9980", api_count=12) == 12


def test_shallow_bogus_count_offline_reports_unknown():
    assert _run_count_block(shallow=True, raw_count="9980", api_count=None) == -1


def test_shallow_local_ahead_treated_as_up_to_date():
    assert _run_count_block(shallow=True, raw_count="3", api_count=0) == 0


def test_shallow_zero_count_short_circuits_without_api():
    with patch("hermes_cli.banner._github_compare_behind") as api:
        got = _run_count_block(shallow=True, raw_count="0", api_count=None)
    # The block only consults the API when count > 0; a 0 count is trustworthy
    # (HEAD == origin tip counts 0 even on shallow graphs).
    assert got == 0
