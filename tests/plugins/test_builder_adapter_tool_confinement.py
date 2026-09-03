import os
import hashlib
import json
import subprocess

import pytest

from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.gitops import (
    AllowedPathManifest,
    GitVerifier,
    safe_relative_path,
)
from plugins.builder_adapter.tools import ConfinedTools
from plugins.builder_adapter.canonical import canonical_sha256
from plugins.builder_adapter.models import ResolvedDispatchRequest
from plugins.builder_adapter.native import BUILDER_WORKER_POLICY
from plugins.builder_adapter.store import DispatchStore
from plugins.builder_adapter import plugin_tools


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
            "rules": [
                {"pattern": "plugins/builder_adapter/**", "access": "read_write"}
            ],
        }
    )


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "a/../../b", r"a\..\b"])
def test_unsafe_paths_fail_closed(path):
    with pytest.raises(AdapterError):
        safe_relative_path(path)


def test_confined_write_and_symlink_escape(tmp_path):
    root = tmp_path / "worktree"
    (root / "plugins/builder_adapter").mkdir(parents=True)
    tools = ConfinedTools(
        root,
        manifest(),
        {"plugins/builder_adapter/existing.py"},
    )
    (root / "plugins/builder_adapter/existing.py").write_text("old\n")
    tools.write_file("plugins/builder_adapter/new.py", "safe\n")
    with pytest.raises(AdapterError, match="not readable"):
        tools.read_file("plugins/builder_adapter/new.py")
    assert tools.read_file("plugins/builder_adapter/existing.py") == "old\n"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "plugins/builder_adapter/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AdapterError, match="symlink"):
        tools.write_file("plugins/builder_adapter/link/escape.py", "bad")


def test_forbidden_but_in_worktree_write_is_rejected(tmp_path):
    root = tmp_path / "worktree"
    root.mkdir()
    tools = ConfinedTools(root, manifest(), set())
    with pytest.raises(AdapterError, match="not permitted"):
        tools.write_file("run_agent.py", "bad")


def test_reads_and_searches_expose_only_tracked_non_denied_regular_files(tmp_path):
    root = tmp_path / "worktree"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"], check=True
    )
    (root / "src").mkdir()
    (root / "src/allowed.txt").write_text("allowed\n")
    (root / "src/denied-secret.txt").write_text("governed secret\n")
    (root / ".env").write_text("TRACKED_SECRET=1\n")
    (root / ".gitignore").write_text(".env.local\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (root / ".env.local").write_text("IGNORED_SECRET=1\n")
    (root / "src/untracked.txt").write_text("untracked\n")
    policy = AllowedPathManifest(
        {
            "default_access": "forbidden",
            "base_sha": head,
            "read_policy": {
                "source": "git_tracked_regular_files",
                "snapshot": "base_sha",
                "deny_patterns": [
                    ".git",
                    ".git/**",
                    ".env",
                    ".env.*",
                    "**/*secret*",
                ],
            },
            "symlinks": "reject",
            "submodules": "reject",
            "rules": [{"pattern": "src/**", "access": "read_write"}],
        }
    )
    readable = GitVerifier({}).tracked_readable_paths(root, head, policy)
    assert readable == frozenset({"src/allowed.txt", ".gitignore"})
    tools = ConfinedTools(root, policy, readable)
    assert tools.read_file("src/allowed.txt") == "allowed\n"
    for forbidden in (
        "src/untracked.txt",
        ".env.local",
        ".env",
        "src/denied-secret.txt",
        ".git/config",
        "../outside",
    ):
        with pytest.raises(AdapterError):
            tools.read_file(forbidden)
    assert tools.search_files("*.txt") == ["src/allowed.txt"]

    (root / "src/allowed.txt").unlink()
    (root / "src/allowed.txt").symlink_to(root / ".env.local")
    with pytest.raises(AdapterError, match="read rejected"):
        tools.read_file("src/allowed.txt")
    assert tools.search_files("*.txt") == []


def test_real_governed_tool_context_exercises_complete_surface(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
    )
    (repository / "src").mkdir()
    (repository / "src/base.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(repository), "add", "src/base.txt"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "base"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "remote", "add", "origin", "local:test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "-q", "-b", "feat/test", str(worktree)],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_value = {
        "default_access": "forbidden",
        "base_sha": head,
        "read_policy": {
            "source": "git_tracked_regular_files",
            "snapshot": "base_sha",
            "deny_patterns": [".env", ".git/**"],
        },
        "symlinks": "reject",
        "submodules": "reject",
        "rules": [{"pattern": "src/**", "access": "read_write"}],
    }
    profile = {
        "profile_id": "strict.v1",
        "environment_policy": {"allow": ["PATH"], "deny": ["*_API_KEY"]},
        "commands": [
            {
                "command_id": "syntax",
                "argv": ["{python}", "-c", "print('validated')"],
                "timeout_seconds": 10,
                "required": True,
            }
        ],
    }
    request = ResolvedDispatchRequest.model_validate(
        {
            "schema_version": "1.0.0",
            "dispatch_id": "11111111-1111-4111-8111-111111111111",
            "idempotency_key": "context-" + "a" * 32,
            "cycle_id": "FEAT_CONTEXT_001",
            "contract": {
                "contract_id": "FEAT_CONTEXT_001",
                "repository_id": "orchestrator",
                "path": "contracts/active/context.json",
                "commit": "1" * 40,
                "sha256": "2" * 64,
            },
            "repository": {
                "repository_id": "hermes-agent",
                "canonical_remote": "local:test",
            },
            "worktree_path": str(worktree),
            "branch": "feat/test",
            "expected_head_sha": head,
            "allowed_path_manifest": {
                "repository_id": "orchestrator",
                "path": "contracts/manifests/context.json",
                "commit": "1" * 40,
                "sha256": "3" * 64,
            },
            "validation_profile": "strict.v1",
            "builder_role": "primary_builder",
            "timeout_policy": {
                "max_runtime_seconds": 60,
                "heartbeat_timeout_seconds": 30,
            },
            "retry_policy": {
                "max_attempts": 1,
                "retryable_terminal_states": [],
            },
            "completion_schema_version": "1.0.0",
        }
    )
    packet_body = {"dispatch_id": request.dispatch_id, "cycle_id": request.cycle_id}
    packet = {"packet": packet_body, "sha256": canonical_sha256(packet_body)}
    store = DispatchStore(tmp_path / "dispatch.db")
    record, _ = store.reserve(
        request.dispatch_id,
        request.idempotency_key,
        "4" * 64,
        request.cycle_id,
        "principal",
        request.model_dump(mode="json"),
        packet,
    )
    store.update(request.dispatch_id, phase="TASK_CREATED", task_id="t_context")

    class Snapshot:
        def __init__(self, *_args, **_kwargs):
            pass

        def raw(self, artifact_id):
            assert artifact_id == "allowed_path_manifest"
            return json.dumps(manifest_value).encode()

        def value(self, artifact_id):
            assert artifact_id == "validation_profile"
            return profile

    monkeypatch.setattr(plugin_tools, "GovernanceSnapshot", Snapshot, raising=False)
    monkeypatch.setattr(
        "plugins.builder_adapter.attestation.GovernanceSnapshot", Snapshot
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "builder_dispatch": {
                "state_path": str(store.path),
                "repository_allowlist": {"hermes-agent": "local:test"},
                "governance_repo": str(tmp_path),
                "governance_commit": "1" * 40,
            }
        },
    )
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_POLICY", "hermes.builder_dispatch.v1"
    )
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
        json.dumps(BUILDER_WORKER_POLICY["tool_allowlist"]),
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_context")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(worktree))

    assert json.loads(plugin_tools.handle_packet({}))["ok"]
    assert "base" in json.loads(
        plugin_tools.handle_read({"path": "src/base.txt"})
    )["result"]["content"]
    preimage = hashlib.sha256(b"base\n").hexdigest()
    assert json.loads(
        plugin_tools.handle_patch(
            {
                "path": "src/base.txt",
                "expected_sha256": preimage,
                "content": "two\n",
            }
        )
    )["ok"]
    assert json.loads(
        plugin_tools.handle_write({"path": "src/new.txt", "content": "one\n"})
    )["ok"]
    assert "src/base.txt" in json.loads(
        plugin_tools.handle_search({"pattern": "*.txt"})
    )["result"]["paths"]
    assert "src/new.txt" not in json.loads(
        plugin_tools.handle_search({"pattern": "*.txt"})
    )["result"]["paths"]
    validation = json.loads(
        plugin_tools.handle_validation(
            {"profile_id": "strict.v1", "expected_sha": head}
        )
    )
    assert validation["ok"] is False
    assert (
        validation["errors"][0]["code"]
        == "VALIDATION_CONTAINMENT_UNAVAILABLE"
    )
