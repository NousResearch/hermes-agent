#!/usr/bin/env python3
"""Read-only evidence that recovery changes reached a merge target.

This is intentionally a Git observer. It never applies a patch, changes refs,
or shells out through a command interpreter. Exact patch identity is the only
automatic coverage proof; transformed/manual ports remain review-required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


POLICY_TESTS: dict[str, list[str]] = {
    "wave1-verification": [
        "tests/ci/test_verification_bundle.py",
        "tests/scripts/test_scaffold_ulw_ledger.py",
    ],
    "worktree-quarantine": [
        "tests/cli/test_worktree.py",
        "tests/cli/test_worktree_security.py",
    ],
    "runtime-contract": [
        "tests/agent/test_system_prompt_restore.py",
        "tests/run_agent/test_message_sequence_repair.py",
        "tests/agent/test_codex_runtime_model_overrides.py",
        "tests/agent/transports/test_codex_app_server_runtime.py",
    ],
}


class GitError(RuntimeError):
    pass


def _git(repo: Path, args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def _git_stdout(repo: Path, args: list[str]) -> str:
    result = _git(repo, args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise GitError(f"{' '.join(args)}: {detail}")
    return result.stdout


def _resolve_commit(repo: Path, ref: str) -> str:
    return _git_stdout(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = _git(repo, ["merge-base", "--is-ancestor", older, newer])
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    detail = result.stderr.strip() or "merge-base failed"
    raise GitError(detail)


def _patch_id_from_text(repo: Path, patch_text: str) -> str | None:
    if not patch_text.strip():
        return None
    result = _git(repo, ["patch-id", "--stable"], input_text=patch_text)
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or "git patch-id failed")
    tokens = result.stdout.split()
    return tokens[0] if tokens else None


def _commit_patch_id(repo: Path, commit: str) -> str | None:
    parent = _git(repo, ["rev-parse", "--verify", f"{commit}^"])
    if parent.returncode != 0:
        return None
    patch = _git_stdout(repo, ["show", "--format=", "--no-ext-diff", commit])
    return _patch_id_from_text(repo, patch)


def _paths_for_commit(repo: Path, commit: str) -> list[str]:
    return [
        line for line in _git_stdout(repo, ["diff-tree", "--no-commit-id", "--name-only", "-r", commit]).splitlines()
        if line
    ]


def _paths_for_patch(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        parts = line.split(" ", 3)
        if len(parts) == 4 and parts[3].startswith("b/"):
            paths.append(parts[3][2:])
    return sorted(set(paths))


def _range_patch_ids(repo: Path, pre_target: str, post_target: str) -> dict[str, list[str]]:
    commit_ids = [
        item for item in _git_stdout(repo, ["rev-list", "--reverse", "--no-merges", f"{pre_target}..{post_target}"]).splitlines()
        if item
    ]
    patch_ids: dict[str, list[str]] = {}
    for commit in commit_ids:
        patch_id = _commit_patch_id(repo, commit)
        if patch_id is not None:
            patch_ids.setdefault(patch_id, []).append(commit)
    return patch_ids


def _paths_for_range(repo: Path, pre_target: str, post_target: str) -> list[str]:
    return [
        line
        for line in _git_stdout(repo, ["diff", "--name-only", pre_target, post_target]).splitlines()
        if line
    ]


def _commit_effect_present_at_target(repo: Path, commit: str, post_target: str) -> bool | None:
    parent = _git(repo, ["rev-parse", "--verify", f"{commit}^"])
    if parent.returncode != 0:
        return None
    paths = _paths_for_commit(repo, commit)
    if not paths:
        return None
    result = _git(repo, ["diff", "--quiet", parent.stdout.strip(), post_target, "--", *paths])
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    detail = result.stderr.strip() or "git diff --quiet failed"
    raise GitError(detail)


def _paths_changed_after_commit(repo: Path, commit: str, post_target: str, paths: list[str]) -> list[str]:
    if not paths:
        return []
    return [
        line
        for line in _git_stdout(repo, ["diff", "--name-only", commit, post_target, "--", *paths]).splitlines()
        if line
    ]


def _check_diff(repo: Path, pre_target: str, post_target: str) -> dict[str, Any]:
    result = _git(repo, ["diff", "--check", pre_target, post_target])
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "output": result.stdout.strip() or result.stderr.strip(),
    }


def _check_conflict_markers(repo: Path, pre_target: str, post_target: str) -> dict[str, Any]:
    diff = _git_stdout(repo, ["diff", "--no-ext-diff", "--unified=0", pre_target, post_target])
    markers = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+<<<<<<< ") or line.startswith("+>>>>>>> ")
    ]
    return {
        "status": "failed" if markers else "passed",
        "count": len(markers),
        "markers": markers,
    }


def _check_target_scope(
    repo: Path,
    pre_target: str,
    post_target: str,
    source_paths: list[str],
    allowed_paths: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    target_paths = _paths_for_range(repo, pre_target, post_target)
    unexpected_paths = sorted(set(target_paths) - set(source_paths) - set(allowed_paths))
    return {
        "status": "failed" if strict and unexpected_paths else "passed",
        "enforced": strict,
        "target_paths": target_paths,
        "source_paths": sorted(set(source_paths)),
        "allowed_paths": sorted(set(allowed_paths)),
        "unexpected_paths": unexpected_paths,
    }


def _run_policy_tests(repo: Path, policy_tests: list[str], run_tests: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for policy_id in policy_tests:
        paths = POLICY_TESTS.get(policy_id)
        if paths is None:
            results.append({"id": policy_id, "status": "invalid_policy_id"})
            continue
        if not run_tests:
            results.append({"id": policy_id, "status": "not_run", "paths": paths})
            continue
        command = [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ]
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
        results.append(
            {
                "id": policy_id,
                "status": "passed" if result.returncode == 0 else "failed",
                "command": command,
                "exit_code": result.returncode,
                "output_tail": (result.stdout + result.stderr).splitlines()[-20:],
            }
        )
    return results


def _candidate_record(
    *,
    kind: str,
    source: str,
    patch_id: str | None,
    source_paths: list[str],
    preexisting: bool,
    range_patch_ids: dict[str, list[str]],
    manual_port: bool = False,
    effect_present_at_post: bool | None = None,
    post_source_path_mutations: list[str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "patch_id": patch_id,
        "source_paths": source_paths,
    }
    if effect_present_at_post is not None:
        record["effect_present_at_post"] = effect_present_at_post
    if post_source_path_mutations:
        record["post_source_path_mutations"] = post_source_path_mutations
    if patch_id is None:
        record["status"] = "unverifiable"
    elif kind == "patch" or manual_port:
        record["status"] = "manual_review_required"
    elif preexisting:
        record["status"] = "reverted_in_target" if effect_present_at_post is False else "preexisting_ancestry"
    elif patch_id in range_patch_ids:
        if effect_present_at_post is False:
            record["status"] = "reverted_in_target"
        elif post_source_path_mutations:
            record["status"] = "manual_review_required"
            record["target_commits"] = range_patch_ids[patch_id]
        else:
            record["status"] = "introduced_exact"
            record["target_commits"] = range_patch_ids[patch_id]
    else:
        record["status"] = "missing"
    return record


def _summary(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "introduced_exact": sum(item["status"] == "introduced_exact" for item in candidates),
        "preexisting": sum(item["status"] == "preexisting_ancestry" for item in candidates),
        "manual_review_required": sum(item["status"] == "manual_review_required" for item in candidates),
        "reverted_in_target": sum(item["status"] == "reverted_in_target" for item in candidates),
        "missing": sum(item["status"] == "missing" for item in candidates),
        "unverifiable": sum(item["status"] == "unverifiable" for item in candidates),
    }


def _manifest_list(manifest: dict[str, Any], key: str) -> list[str]:
    value = manifest.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise GitError(f"manifest field {key!r} must be a list of non-empty strings")
    return value


def _load_manifest(path_name: str) -> tuple[Path, list[str], list[str], list[str], list[str], list[str]]:
    path = Path(path_name).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GitError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise GitError("manifest must be a JSON object with version=1")
    commits = _manifest_list(manifest, "candidate_commits")
    manual_port_commits = _manifest_list(manifest, "manual_port_commits")
    patches = _manifest_list(manifest, "candidate_patches")
    policy_tests = _manifest_list(manifest, "policy_tests")
    allowed_target_paths = _manifest_list(manifest, "allowed_target_paths")
    return path, commits, manual_port_commits, patches, policy_tests, allowed_target_paths


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _manifest_patch_path(manifest_path: Path, patch_name: str) -> Path:
    contract_dir = manifest_path.parent.resolve()
    patch_path = (contract_dir / patch_name).resolve()
    if not patch_path.is_relative_to(contract_dir):
        raise GitError("manifest candidate_patches must stay within the manifest directory")
    return patch_path


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    pre_target = _resolve_commit(repo, args.pre_target)
    post_target = _resolve_commit(repo, args.post_target)
    ancestry = _is_ancestor(repo, pre_target, post_target)
    checks: dict[str, Any] = {
        "ancestry": {"status": "passed" if ancestry else "failed"},
        "diff_check": _check_diff(repo, pre_target, post_target),
        "conflict_markers": _check_conflict_markers(repo, pre_target, post_target),
    }
    range_patch_ids = _range_patch_ids(repo, pre_target, post_target) if ancestry else {}
    manifest_path: Path | None = None
    manifest_commits: list[str] = []
    manifest_manual_port_commits: list[str] = []
    manifest_patches: list[str] = []
    manifest_policy_tests: list[str] = []
    manifest_allowed_target_paths: list[str] = []
    if args.manifest:
        (
            manifest_path,
            manifest_commits,
            manifest_manual_port_commits,
            manifest_patches,
            manifest_policy_tests,
            manifest_allowed_target_paths,
        ) = _load_manifest(args.manifest)
    manual_port_commits = _dedupe([*args.manual_port_commit, *manifest_manual_port_commits])
    candidate_commits = _dedupe([*args.candidate_commit, *manifest_commits, *manual_port_commits])
    candidate_patch_specs = [(patch_name, None) for patch_name in args.candidate_patch]
    candidate_patch_specs.extend((patch_name, manifest_path.parent) for patch_name in manifest_patches)
    policy_test_ids = _dedupe([*args.policy_test, *manifest_policy_tests])
    allowed_target_paths = _dedupe([*args.allow_target_path, *manifest_allowed_target_paths])
    candidates: list[dict[str, Any]] = []
    for ref in candidate_commits:
        commit = _resolve_commit(repo, ref)
        source_paths = _paths_for_commit(repo, commit)
        target_commits = range_patch_ids.get(_commit_patch_id(repo, commit) or "", [])
        post_source_path_mutations = sorted(
            {
                path
                for target_commit in target_commits
                for path in _paths_changed_after_commit(repo, target_commit, post_target, source_paths)
            }
        )
        candidates.append(
            _candidate_record(
                kind="commit",
                source=commit,
                patch_id=_commit_patch_id(repo, commit),
                source_paths=source_paths,
                preexisting=_is_ancestor(repo, commit, pre_target),
                range_patch_ids=range_patch_ids,
                manual_port=ref in manual_port_commits,
                effect_present_at_post=_commit_effect_present_at_target(repo, commit, post_target),
                post_source_path_mutations=post_source_path_mutations,
            )
        )
    seen_patch_paths: set[Path] = set()
    for patch_name, base_path in candidate_patch_specs:
        patch_path = _manifest_patch_path(manifest_path, patch_name) if base_path else Path(patch_name).resolve()
        if patch_path in seen_patch_paths:
            continue
        seen_patch_paths.add(patch_path)
        patch_text = patch_path.read_text(encoding="utf-8")
        candidates.append(
            _candidate_record(
                kind="patch",
                source=str(patch_path),
                patch_id=_patch_id_from_text(repo, patch_text),
                source_paths=_paths_for_patch(patch_text),
                preexisting=False,
                range_patch_ids=range_patch_ids,
            )
        )
    source_paths = [path for candidate in candidates for path in candidate["source_paths"]]
    checks["target_scope"] = _check_target_scope(
        repo,
        pre_target,
        post_target,
        source_paths,
        allowed_target_paths,
        strict=args.strict_paths,
    )
    policy_results = _run_policy_tests(repo, policy_test_ids, args.run_policy_tests)
    candidate_bad = {"missing", "unverifiable", "manual_review_required", "reverted_in_target"}
    policy_bad = {"failed", "invalid_policy_id", "not_run"}
    ok = (
        all(item["status"] == "passed" for item in checks.values())
        and not any(item["status"] in candidate_bad for item in candidates)
        and not any(item["status"] in policy_bad for item in policy_results)
    )
    return {
        "ok": ok,
        "repo": str(repo),
        "pre_target": pre_target,
        "post_target": post_target,
        "manifest": str(manifest_path) if manifest_path else None,
        "checks": checks,
        "candidates": candidates,
        "policy_tests": policy_results,
        "summary": _summary(candidates),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Git repository to observe (default: cwd)")
    parser.add_argument("--pre-target", required=True, help="Target commit before merge")
    parser.add_argument("--post-target", required=True, help="Target commit after merge")
    parser.add_argument("--candidate-commit", action="append", default=[], help="Expected source commit")
    parser.add_argument(
        "--manual-port-commit",
        action="append",
        default=[],
        help="Source commit intentionally ported with changed hunks; remains review-required",
    )
    parser.add_argument("--candidate-patch", action="append", default=[], help="Expected source patch file")
    parser.add_argument("--manifest", help="Versioned JSON recovery coverage contract")
    parser.add_argument(
        "--allow-target-path",
        action="append",
        default=[],
        help="Exact additional target path permitted by --strict-paths",
    )
    parser.add_argument("--policy-test", action="append", default=[], choices=sorted(POLICY_TESTS))
    parser.add_argument("--run-policy-tests", action="store_true")
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail when the target range changes paths outside sources and allowlist",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless all evidence passes")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(args)
    except (GitError, OSError, UnicodeError) as exc:
        report = {"ok": False, "error": str(exc)}
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
