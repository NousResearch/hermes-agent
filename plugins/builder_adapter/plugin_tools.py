"""Task-scoped Hermes plugin tools for governed builder workers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .completion import CompletionAttestor
from .canonical import canonical_sha256
from .errors import AdapterError
from .gitops import GitVerifier, _run_git
from .models import ResolvedDispatchRequest
from .native import BUILDER_WORKER_POLICY
from .store import DispatchStore
from .tools import ConfinedTools
from .validation import ValidationRunner


@dataclass
class ToolContext:
    request: ResolvedDispatchRequest
    root: Path
    manifest: object
    tools: ConfinedTools
    validation: ValidationRunner
    packet: dict


def _context() -> ToolContext:
    try:
        worker_tools = json.loads(
            os.environ.get("HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST", "")
        )
    except (TypeError, ValueError):
        worker_tools = None
    if (
        os.environ.get("HERMES_INTERNAL_WORKER_POLICY")
        != BUILDER_WORKER_POLICY["policy_id"]
        or worker_tools != BUILDER_WORKER_POLICY["tool_allowlist"]
    ):
        raise AdapterError(
            "AUTHORIZATION_FAILED", "exact governed worker policy is missing"
        )
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    if not task_id or not workspace:
        raise AdapterError("AUTHORIZATION_FAILED", "task identity is unavailable")

    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    adapter_config = config.get("builder_dispatch")
    if not isinstance(adapter_config, dict):
        raise AdapterError("PROFILE_POLICY_MISMATCH", "adapter profile config missing")
    store = DispatchStore(adapter_config["state_path"])
    record = store.get_by_task(task_id)
    if not record or not record.get("request_json"):
        raise AdapterError("AUTHORIZATION_FAILED", "task is not adapter-correlated")
    request = ResolvedDispatchRequest.model_validate(json.loads(record["request_json"]))
    root = Path(workspace)
    if root != Path(request.worktree_path):
        raise AdapterError("WORKTREE_MISMATCH", "task workspace identity mismatch")

    git = GitVerifier(dict(adapter_config["repository_allowlist"]))
    git.verify_worktree(request, require_clean=False)
    governance_root = Path(adapter_config["governance_repo"])
    from .attestation import GovernanceSnapshot

    snapshot = GovernanceSnapshot(
        governance_root,
        request.contract.commit,
        registered_contract_path=request.contract.path,
    )
    raw = snapshot.raw("allowed_path_manifest")
    manifest = git.manifest_from_artifact(raw)
    if manifest.base_sha != request.expected_head_sha:
        raise AdapterError("MANIFEST_MISMATCH", "manifest base SHA mismatch")
    readable_paths = git.tracked_readable_paths(
        root, request.expected_head_sha, manifest
    )
    profile = snapshot.value("validation_profile")
    if profile.get("profile_id") != request.validation_profile:
        raise AdapterError("PROFILE_POLICY_MISMATCH", "validation profile unavailable")
    record = store.assert_packet_identity(str(request.dispatch_id))
    packet = json.loads(record["packet_json"])
    if canonical_sha256(packet["packet"]) != packet["sha256"]:
        raise AdapterError("CONTRACT_MISMATCH", "execution packet hash mismatch")
    return ToolContext(
        request=request,
        root=root,
        manifest=manifest,
        tools=ConfinedTools(root, manifest, readable_paths),
        validation=ValidationRunner(
            {request.validation_profile: profile}, python=os.sys.executable
        ),
        packet=packet,
    )


def _result(function):
    def wrapped(args: dict, **_kwargs) -> str:
        try:
            return json.dumps({"ok": True, "result": function(_context(), args)})
        except AdapterError as error:
            return json.dumps({"ok": False, "errors": [error.as_dict()]})
        except Exception:
            error = AdapterError("INTERNAL_ERROR", "builder tool failed closed")
            return json.dumps({"ok": False, "errors": [error.as_dict()]})

    return wrapped


@_result
def handle_read(context: ToolContext, args: dict):
    return {"content": context.tools.read_file(args["path"])}


@_result
def handle_write(context: ToolContext, args: dict):
    context.tools.write_file(args["path"], args["content"])
    return {"path": args["path"], "sha256": hashlib.sha256(args["content"].encode()).hexdigest()}


@_result
def handle_patch(context: ToolContext, args: dict):
    current = context.tools.read_file(args["path"])
    if hashlib.sha256(current.encode()).hexdigest() != args["expected_sha256"]:
        raise AdapterError("WORKTREE_RACE", "patch preimage hash mismatch")
    context.tools.write_file(args["path"], args["content"])
    return {"path": args["path"], "sha256": hashlib.sha256(args["content"].encode()).hexdigest()}


@_result
def handle_search(context: ToolContext, args: dict):
    return {"paths": context.tools.search_files(args["pattern"])}


@_result
def handle_packet(context: ToolContext, _args: dict):
    return context.packet


def _validation_snapshot(context: ToolContext) -> tuple[tempfile.TemporaryDirectory, Path, str]:
    root = context.root
    base = context.request.expected_head_sha
    tracked = _run_git(root, "diff", "--name-only", "-z").stdout
    untracked = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    paths = sorted({part.decode() for part in (tracked + untracked).split(b"\0") if part})
    GitVerifier({}).verify_paths(root, paths, context.manifest)
    temporary = tempfile.TemporaryDirectory(prefix="builder-tool-validation-")
    index = Path(temporary.name) / "index"
    env = CompletionAttestor._git_identity_env(index)
    _run_git(root, "read-tree", base, env=env)
    for path in paths:
        try:
            data, mode = CompletionAttestor._safe_blob(root, path)
        except AdapterError:
            if (root / path).exists() or (root / path).is_symlink():
                temporary.cleanup()
                raise
            _run_git(root, "update-index", "--remove", "--", path, env=env)
            continue
        blob = _run_git(
            root, "hash-object", "-w", "--stdin", "--no-filters", input=data, env=env
        ).stdout.decode().strip()
        _run_git(root, "update-index", "--add", "--cacheinfo", mode, blob, path, env=env)
    tree = _run_git(root, "write-tree", env=env).stdout.decode().strip()
    commit = _run_git(root, "commit-tree", tree, "-p", base, env=env).stdout.decode().strip()
    checkout = Path(temporary.name) / "checkout"
    GitVerifier({}).materialize_tree(root, commit, checkout)
    return temporary, checkout, commit


@_result
def handle_validation(context: ToolContext, args: dict):
    if (
        args["profile_id"] != context.request.validation_profile
        or args["expected_sha"] != context.request.expected_head_sha
    ):
        raise AdapterError("VALIDATION_FAILED", "validation binding mismatch")
    temporary, checkout, commit = _validation_snapshot(context)
    try:
        return context.validation.run(
            args["profile_id"],
            checkout,
            commit,
            materialized_sha=commit,
            scope_id=str(context.request.dispatch_id),
        )
    finally:
        temporary.cleanup()


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
    }


TOOLS = (
    (
        "builder_read_execution_packet",
        schema(
            "builder_read_execution_packet",
            "Read the immutable hash-bound task execution packet.",
            {},
            [],
        ),
        handle_packet,
    ),
    (
        "builder_read_file",
        schema("builder_read_file", "Read one governed file.", {"path": {"type": "string"}}, ["path"]),
        handle_read,
    ),
    (
        "builder_write_file",
        schema(
            "builder_write_file",
            "Atomically replace one governed file.",
            {"path": {"type": "string"}, "content": {"type": "string", "maxLength": 1000000}},
            ["path", "content"],
        ),
        handle_write,
    ),
    (
        "builder_patch",
        schema(
            "builder_patch",
            "Hash-guarded atomic replacement of one governed file.",
            {
                "path": {"type": "string"},
                "expected_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "content": {"type": "string", "maxLength": 1000000},
            },
            ["path", "expected_sha256", "content"],
        ),
        handle_patch,
    ),
    (
        "builder_search_files",
        schema(
            "builder_search_files",
            "Search governed workspace file names.",
            {"pattern": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            ["pattern"],
        ),
        handle_search,
    ),
    (
        "builder_run_validation_profile",
        schema(
            "builder_run_validation_profile",
            "Run the registered validation profile against a snapshot.",
            {
                "profile_id": {"type": "string"},
                "expected_sha": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{40}([0-9a-f]{24})?$",
                },
            },
            ["profile_id", "expected_sha"],
        ),
        handle_validation,
    ),
)
