import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

from plugins.builder_adapter.completion import CompletionAttestor
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.gitops import AllowedPathManifest, GitVerifier


def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "plugins/builder_adapter").mkdir(parents=True)
    (tmp_path / "plugins/builder_adapter/base.py").write_text("x\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    return subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def manifest():
    return AllowedPathManifest(
        {
            "default_access": "forbidden",
            "base_sha": "0" * 40,
            "read_policy": {
                "source": "git_tracked_regular_files",
                "snapshot": "base_sha",
                "deny_patterns": [".env", ".git/**"],
            },
            "symlinks": "reject",
            "submodules": "reject",
            "rules": [{"pattern": "plugins/builder_adapter/**", "access": "read_write"}],
        }
    )


def test_changed_paths_are_derived_from_git_not_evidence(tmp_path):
    base = repo(tmp_path)
    (tmp_path / "plugins/builder_adapter/new.py").write_text("safe\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "change"], check=True)
    verifier = GitVerifier({})
    paths = verifier.changed_paths(tmp_path, base)
    assert paths == ["plugins/builder_adapter/new.py"]
    verifier.verify_paths(tmp_path, paths, manifest())


def test_disallowed_git_change_blocks_completion(tmp_path):
    base = repo(tmp_path)
    (tmp_path / "run_agent.py").write_text("bad\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "bad"], check=True)
    verifier = GitVerifier({})
    paths = verifier.changed_paths(tmp_path, base)
    with pytest.raises(AdapterError) as raised:
        verifier.verify_paths(tmp_path, paths, manifest())
    assert raised.value.code == "MANIFEST_MISMATCH"


def test_completion_independently_validates_commits_and_emits_evidence(tmp_path):
    base = repo(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "remote", "add", "origin", "git@example.invalid:hermes.git"],
        check=True,
    )
    (tmp_path / "plugins/builder_adapter/new.py").write_text("safe\n")
    hooks = tmp_path.parent / f"{tmp_path.name}-hostile-hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 99\n")
    hook.chmod(0o755)
    checkout_marker = tmp_path.parent / f"{tmp_path.name}-post-checkout-fired"
    checkout_hook = hooks / "post-checkout"
    checkout_hook.write_text(f"#!/bin/sh\ntouch {checkout_marker}\n")
    checkout_hook.chmod(0o755)
    filter_marker = tmp_path.parent / f"{tmp_path.name}-filter-fired"
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "filter.hostile.smudge",
            f"sh -c 'touch {filter_marker}; cat'",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.hooksPath", str(hooks)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "commit.gpgSign", "true"],
        check=True,
    )
    hostile_markers = {}
    for key in ("fsmonitor", "gpg", "credential", "external-diff"):
        marker = tmp_path.parent / f"{tmp_path.name}-{key}-fired"
        probe = tmp_path.parent / f"{tmp_path.name}-{key}-probe"
        probe.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        probe.chmod(0o755)
        hostile_markers[key] = marker
        config_key = {
            "fsmonitor": "core.fsmonitor",
            "gpg": "gpg.program",
            "credential": "credential.helper",
            "external-diff": "diff.external",
        }[key]
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", config_key, str(probe)],
            check=True,
        )

    class Validation:
        def run(
            self,
            profile,
            worktree,
            expected_sha,
            *,
            materialized_sha=None,
            scope_id="validation",
        ):
            if materialized_sha is not None:
                assert expected_sha == materialized_sha
                assert not (worktree / ".git").exists()
            return {
                "profile": profile,
                "commands": [{
                    "command_id": "focused",
                    "argv": ["python", "-m", "pytest"],
                    "exit_status": 0,
                    "started_at": "2026-01-01T00:00:00Z",
                    "finished_at": "2026-01-01T00:00:01Z",
                    "stdout_sha256": "0" * 64,
                    "stderr_sha256": "0" * 64,
                }],
                "overall_status": "PASSED",
            }

    class Schemas:
        def validate(self, name, value):
            assert name == "completion_evidence"
            assert value["terminal_execution"] == "SUCCEEDED"

    policy = {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "profile": "deepseek-builder",
    }
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    request = SimpleNamespace(
        dispatch_id=str(uuid4()),
        cycle_id="FEAT_TEST_001",
        worktree_path=str(tmp_path),
        validation_profile="strict.v1",
        expected_head_sha=base,
        branch=branch,
        repository=SimpleNamespace(
            repository_id="hermes-agent",
            canonical_remote="git@example.invalid:hermes.git",
        ),
    )
    snapshot = SimpleNamespace(task_id="t_123", run_ids=["r_123"])
    effective = SimpleNamespace(
        evidence=lambda: {
            "provider": policy["provider"],
            "model": policy["model"],
            "profile": policy["profile"],
            "profile_configuration_sha256": "a" * 64,
            "fallback_chain": [],
            "fallback_used": False,
            "attested_by": "hermes.builder_dispatch.v1",
        }
    )
    attestor = CompletionAttestor(GitVerifier({}), Validation(), Schemas(), effective)
    evidence = attestor.complete(
        request, snapshot, "orchestrator-mcp", "b" * 64, manifest()
    )
    assert evidence["git"]["starting_sha"] == base
    assert evidence["git"]["resulting_sha"] != base
    assert evidence["git"]["final_dirty_state"] == "CLEAN"
    assert evidence["changed_files"][0]["path"] == "plugins/builder_adapter/new.py"
    assert evidence["validation"]["overall_status"] == "PASSED"
    assert not checkout_marker.exists()
    assert not filter_marker.exists()
    assert all(not marker.exists() for marker in hostile_markers.values())
    reconciled = attestor.complete(
        request, snapshot, "orchestrator-mcp", "b" * 64, manifest()
    )
    assert reconciled["git"]["resulting_sha"] == evidence["git"]["resulting_sha"]


def test_completion_detects_mutation_during_snapshot_validation(tmp_path):
    base = repo(tmp_path)
    (tmp_path / "plugins/builder_adapter/new.py").write_text("safe\n")
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class RacingValidation:
        def run(
            self,
            profile,
            worktree,
            expected_sha,
            *,
            materialized_sha=None,
            scope_id="validation",
        ):
            (tmp_path / "plugins/builder_adapter/new.py").write_text("raced\n")
            return {"overall_status": "PASSED", "commands": []}

    request = SimpleNamespace(
        dispatch_id=str(uuid4()),
        cycle_id="FEAT_TEST_001",
        worktree_path=str(tmp_path),
        validation_profile="strict.v1",
        expected_head_sha=base,
        branch=branch,
        repository=SimpleNamespace(
            repository_id="hermes-agent",
            canonical_remote="git@example.invalid:hermes.git",
        ),
    )
    effective = SimpleNamespace(evidence=lambda: {})
    attestor = CompletionAttestor(
        GitVerifier({}),
        RacingValidation(),
        SimpleNamespace(validate=lambda *_: None),
        effective,
    )
    with pytest.raises(AdapterError) as raised:
        attestor.complete(
            request,
            SimpleNamespace(task_id="task", run_ids=[]),
            "principal",
            "b" * 64,
            manifest(),
        )
    assert raised.value.code == "WORKTREE_RACE"
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", branch],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == base


def test_repository_verifier_requires_a_clean_linked_worktree(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    base = repo(primary)
    remote = "git@example.invalid:hermes.git"
    subprocess.run(
        ["git", "-C", str(primary), "remote", "add", "origin", remote], check=True
    )
    linked = tmp_path / "linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(primary),
            "worktree",
            "add",
            "-qb",
            "feat/linked",
            str(linked),
            base,
        ],
        check=True,
    )
    verifier = GitVerifier({"hermes-agent": remote})
    request = SimpleNamespace(
        repository=SimpleNamespace(
            repository_id="hermes-agent", canonical_remote=remote
        ),
        worktree_path=str(linked),
        branch="feat/linked",
        expected_head_sha=base,
    )
    assert verifier.verify_worktree(request) == linked
    request.worktree_path = str(primary)
    request.branch = "main"
    with pytest.raises(AdapterError) as raised:
        verifier.verify_worktree(request)
    assert raised.value.code == "WORKTREE_MISMATCH"


def test_repository_verifier_rejects_tracked_symlink(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    base = repo(primary)
    remote = "git@example.invalid:hermes.git"
    subprocess.run(
        ["git", "-C", str(primary), "remote", "add", "origin", remote], check=True
    )
    linked = tmp_path / "linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(primary),
            "worktree",
            "add",
            "-qb",
            "feat/symlink",
            str(linked),
            base,
        ],
        check=True,
    )
    (linked / "plugins/builder_adapter/link.py").symlink_to("base.py")
    subprocess.run(
        ["git", "-C", str(linked), "add", "plugins/builder_adapter/link.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(linked), "commit", "-qm", "symlink"], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    request = SimpleNamespace(
        repository=SimpleNamespace(
            repository_id="hermes-agent", canonical_remote=remote
        ),
        worktree_path=str(linked),
        branch="feat/symlink",
        expected_head_sha=head,
    )
    with pytest.raises(AdapterError) as raised:
        GitVerifier({"hermes-agent": remote}).verify_worktree(request)
    assert raised.value.code == "MANIFEST_MISMATCH"
