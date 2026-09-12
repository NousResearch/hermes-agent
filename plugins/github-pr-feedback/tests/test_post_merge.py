import pytest
from pathlib import Path

from github_pr_feedback.post_merge import (
    DeploymentError,
    ProcessRecord,
    _require_package_provenance,
    _wait_for_processes_to_exit,
)


def test_package_provenance_requires_the_exact_full_source_sha():
    source_sha = "a" * 40

    _require_package_provenance({"source_sha": source_sha}, source_sha)

    with pytest.raises(DeploymentError, match="package_provenance_missing"):
        _require_package_provenance({}, source_sha)
    with pytest.raises(DeploymentError, match="package_provenance_mismatch"):
        _require_package_provenance({"source_sha": "b" * 40}, source_sha)


def test_process_shutdown_wait_rechecks_the_current_census():
    process = ProcessRecord(123, Path("/Applications/Hermes.app/Contents/MacOS/Hermes"), (), None)

    class Controller:
        def __init__(self):
            self.censuses = [[process], []]

        def census(self):
            return tuple(self.censuses.pop(0))

    controller = Controller()
    _wait_for_processes_to_exit([process], controller, timeout=0.2)
    assert controller.censuses == []
