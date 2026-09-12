import pytest

from github_pr_feedback.post_merge import DeploymentError, _require_package_provenance


def test_package_provenance_requires_the_exact_full_source_sha():
    source_sha = "a" * 40

    _require_package_provenance({"source_sha": source_sha}, source_sha)

    with pytest.raises(DeploymentError, match="package_provenance_missing"):
        _require_package_provenance({}, source_sha)
    with pytest.raises(DeploymentError, match="package_provenance_mismatch"):
        _require_package_provenance({"source_sha": "b" * 40}, source_sha)
