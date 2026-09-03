import os
import subprocess
from pathlib import Path

import pytest

from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.preparation import (
    RepositoryFacts,
    inspect_repository,
    prepare_bundle,
    write_bundle,
)


def _facts(clean=True):
    return RepositoryFacts(
        root="/srv/project",
        remote="https://example.invalid/project.git",
        head="a" * 40,
        branch="main",
        clean=clean,
    )


def _bundle(tmp_path, **overrides):
    values = dict(
        cycle_id="FEATURE_EXAMPLE_001",
        contract_id="FEATURE-EXAMPLE-001",
        repository_id="my-project",
        repository=_facts(),
        goal="Add the requested behavior",
        acceptance_criteria=["Focused tests pass"],
        allowed_paths=["src/example.py", "tests/test_example.py"],
        planned_branch="feat/example-001",
        planned_worktree=str(tmp_path / "worktree"),
        validation_profile_id="strict.v1",
        max_runtime_seconds=1800,
        heartbeat_timeout_seconds=180,
        registered_remote="https://example.invalid/project.git",
    )
    values.update(overrides)
    return prepare_bundle(**values)


def test_prepare_bundle_is_hash_stamped_and_non_activating(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle["activation_state"] == "READY_FOR_GOVERNANCE_REVIEW"
    assert len(bundle["bundle_sha256"]) == 64
    assert bundle["retry_policy"]["max_attempts"] == 1
    assert bundle["repository"]["base_sha"] == "a" * 40


def test_prepare_rejects_dirty_repository_and_global_write(tmp_path):
    with pytest.raises(AdapterError, match="clean"):
        _bundle(tmp_path, repository=_facts(clean=False))
    with pytest.raises(AdapterError, match="repository-wide"):
        _bundle(tmp_path, allowed_paths=["**/*"])


def test_unregistered_remote_is_explicit(tmp_path):
    bundle = _bundle(tmp_path, registered_remote=None)
    assert bundle["activation_state"] == "NEEDS_REPOSITORY_ALLOWLIST"


def test_write_bundle_is_owner_only_and_never_overwrites(tmp_path):
    destination = tmp_path / "pending" / "job.json"
    write_bundle(destination, _bundle(tmp_path))
    assert os.stat(destination).st_mode & 0o777 == 0o600
    with pytest.raises(AdapterError, match="already exists"):
        write_bundle(destination, _bundle(tmp_path))


def test_inspect_repository_pins_clean_local_git_identity(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "remote", "add", "origin", "https://example.invalid/project.git"],
        check=True,
    )
    (repository / "README.md").write_text("example\n")
    subprocess.run(["/usr/bin/git", "-C", str(repository), "add", "README.md"], check=True)
    subprocess.run(
        [
            "/usr/bin/git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "initial",
        ],
        check=True,
    )
    facts = inspect_repository(repository)
    assert facts.root == str(repository)
    assert facts.remote == "https://example.invalid/project.git"
    assert len(facts.head) == 40
    assert facts.clean is True
