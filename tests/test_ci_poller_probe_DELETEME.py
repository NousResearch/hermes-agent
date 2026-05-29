"""TEMPORARY CI-poller probe — DO NOT MERGE.

This test deliberately fails so we can observe how the CI status poller
(scripts/wait_for_pr_green.sh) behaves when a failure lands while other
jobs are still pending (the two-wave trap). It will be removed before this
branch is ever considered for merge. The PR exists only to exercise CI.
"""


def test_deliberate_failure_for_poller_probe():
    assert False, "intentional failure — CI poller probe, see docstring"
