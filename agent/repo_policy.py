from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

POLICY_RELATIVE_PATH = Path(".hermes/repo-policy.yaml")
REQUIRED_TOP_LEVEL = {
    "version",
    "repo",
    "authority",
    "roles",
    "workflow",
    "branches",
    "gates",
    "runtime",
    "closeout",
}
REQUIRED_CLOSEOUT = {
    "결론",
    "실제 반영",
    "아직 안 한 것",
    "다음 판단",
    "Policy check",
    "Green 완료",
    "Yellow 대기",
    "Red 필요",
    "검증",
    "Git 상태",
    "Live 상태",
}
RUNTIME_CLOSEOUT = {
    "Gateway restart 필요",
    "Live runtime 반영됨",
    "대기열 포함됨",
}
RED_EXTERNAL_EFFECTS = {
    "push",
    "upstream_pr",
    "merge",
    "release",
    "deploy",
    "prod_mutation",
    "env_secret_change",
    "billing_pricing_change",
    "customer_visible_change",
    "gateway_restart_reload",
    "live_runtime_apply",
    "destructive_cleanup",
}


@dataclass
class RepoPolicyIssue:
    code: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class RepoPolicyCheckResult:
    ok: bool
    status: str
    repo_path: str
    policy_path: str
    repo_class: str | None = None
    issues: list[RepoPolicyIssue] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    authority: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "repo_path": self.repo_path,
            "policy_path": self.policy_path,
            "repo_class": self.repo_class,
            "issues": [issue.as_dict() for issue in self.issues],
            "observed": self.observed,
            "authority": self.authority,
        }


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _load_policy(path: Path) -> tuple[dict[str, Any] | None, list[RepoPolicyIssue]]:
    if not path.exists():
        return None, [RepoPolicyIssue("missing_policy", "Policy drift detected: .hermes/repo-policy.yaml is missing")]
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [RepoPolicyIssue("malformed_yaml", f"Policy drift detected: invalid YAML: {exc}")]
    if not isinstance(loaded, dict):
        return None, [RepoPolicyIssue("malformed_policy", "Policy drift detected: policy must be a YAML mapping")]
    return loaded, []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _add_contract_issues(policy: dict[str, Any], issues: list[RepoPolicyIssue]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(policy))
    if missing:
        issues.append(RepoPolicyIssue("missing_required_fields", f"Policy drift detected: missing top-level keys: {missing}"))

    if policy.get("version") != 1:
        issues.append(RepoPolicyIssue("unsupported_version", "Policy drift detected: unsupported repo policy version"))

    repo = _mapping(policy.get("repo"))
    repo_class = repo.get("class")
    if repo_class not in {"product", "runtime_tooling", "other"}:
        issues.append(RepoPolicyIssue("invalid_repo_class", "Policy drift detected: repo.class must be product, runtime_tooling, or other"))

    authority = _mapping(policy.get("authority"))
    if authority.get("policy_source") != str(POLICY_RELATIVE_PATH):
        issues.append(RepoPolicyIssue("invalid_policy_source", "Policy drift detected: authority.policy_source must be .hermes/repo-policy.yaml"))

    roles = _mapping(policy.get("roles"))
    workflow = _mapping(policy.get("workflow"))
    branches = _mapping(policy.get("branches"))
    runtime = _mapping(policy.get("runtime"))
    gates = _mapping(policy.get("gates"))
    closeout = _mapping(policy.get("closeout"))
    closeout_sections = set(closeout.get("required_sections") or [])

    missing_closeout = sorted(REQUIRED_CLOSEOUT - closeout_sections)
    if missing_closeout:
        issues.append(RepoPolicyIssue("missing_closeout_sections", f"Policy drift detected: missing closeout sections: {missing_closeout}"))

    red = set(gates.get("red_requires_explicit_approval") or [])
    missing_red = sorted((RED_EXTERNAL_EFFECTS & {"merge", "deploy", "env_secret_change", "destructive_cleanup"}) - red)
    if missing_red:
        issues.append(RepoPolicyIssue("missing_red_gates", f"Policy drift detected: missing baseline Red gates: {missing_red}"))

    if repo_class == "product":
        if workflow.get("canonical_landing") != "develop":
            issues.append(RepoPolicyIssue("product_workflow_landing_not_develop", "Policy drift detected: product workflow.canonical_landing must be develop"))
        if workflow.get("work_done_means") != "pushed_to_develop":
            issues.append(RepoPolicyIssue("product_work_done_not_develop", "Policy drift detected: product work_done_means must be pushed_to_develop"))
        release_target = workflow.get("release_target")
        if workflow.get("release_source") != "develop" or release_target not in {"main", "master"}:
            issues.append(RepoPolicyIssue("product_release_path_not_develop_to_release_target", "Policy drift detected: product release path must be develop-><repo release target main|master>"))
        if branches.get("release_base") != release_target:
            issues.append(RepoPolicyIssue("product_release_base_mismatch", "Policy drift detected: product branches.release_base must match workflow.release_target"))
        if workflow.get("live_apply") != "deploy_gate":
            issues.append(RepoPolicyIssue("product_workflow_live_apply_not_deploy_gate", "Policy drift detected: product workflow.live_apply must be deploy_gate"))
        if roles.get("landing") != "develop":
            issues.append(RepoPolicyIssue("product_landing_not_develop", "Policy drift detected: product roles.landing must be develop"))
        if branches.get("landing") != "develop":
            issues.append(RepoPolicyIssue("product_branch_not_develop", "Policy drift detected: product branches.landing must be develop"))
        if roles.get("release") not in {"release_pr_to_main", "release_gate_to_release_target"}:
            issues.append(RepoPolicyIssue("product_release_not_standard", "Policy drift detected: product roles.release must be release_pr_to_main or release_gate_to_release_target"))
        if roles.get("live_apply") != "deploy_gate":
            issues.append(RepoPolicyIssue("product_live_apply_not_deploy_gate", "Policy drift detected: product roles.live_apply must be deploy_gate"))

    if repo_class == "runtime_tooling":
        if workflow.get("canonical_landing") not in {"verified_landing", "verified_fork_main"}:
            issues.append(RepoPolicyIssue("runtime_workflow_landing_not_verified", "Policy drift detected: runtime_tooling workflow.canonical_landing must be verified_landing or verified_fork_main"))
        if workflow.get("live_apply") != "gateway_runtime_queue":
            issues.append(RepoPolicyIssue("runtime_workflow_live_apply_missing_queue", "Policy drift detected: runtime_tooling workflow.live_apply must be gateway_runtime_queue"))
        if roles.get("live_apply") != "gateway_runtime_queue":
            issues.append(RepoPolicyIssue("runtime_live_apply_missing_queue", "Policy drift detected: runtime_tooling roles.live_apply must be gateway_runtime_queue"))
        if runtime.get("live_apply_queue") != "required":
            issues.append(RepoPolicyIssue("runtime_queue_not_required", "Policy drift detected: runtime_tooling runtime.live_apply_queue must be required"))
        if runtime.get("restart_policy") not in {"queue_apply_one_restart", "manual_explicit_only"}:
            issues.append(RepoPolicyIssue("runtime_restart_policy_missing", "Policy drift detected: runtime_tooling runtime.restart_policy must be explicit"))
        missing_runtime_closeout = sorted(RUNTIME_CLOSEOUT - closeout_sections)
        if missing_runtime_closeout:
            issues.append(RepoPolicyIssue("missing_runtime_closeout_sections", f"Policy drift detected: missing runtime closeout sections: {missing_runtime_closeout}"))


def _observed_git(repo_path: Path) -> dict[str, Any]:
    inside = _run_git(repo_path, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        return {"is_git_repo": False, "git_error": inside.stderr.strip()}
    head = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    refs = _run_git(repo_path, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes")
    ref_lines = [line.strip() for line in refs.stdout.splitlines() if line.strip()] if refs.returncode == 0 else []
    return {
        "is_git_repo": True,
        "head_branch": head.stdout.strip() if head.returncode == 0 else None,
        "refs": ref_lines,
        "has_develop": any(ref == "refs/heads/develop" or ref.endswith("/develop") for ref in ref_lines),
    }


def _agents_conflicts(repo_path: Path) -> list[RepoPolicyIssue]:
    agents_path = repo_path / "AGENTS.md"
    if not agents_path.exists():
        return []
    text = agents_path.read_text(encoding="utf-8", errors="replace").lower()
    conflict_needles = [
        "ignore .hermes/repo-policy.yaml",
        "ignore repo-policy",
        "repo policy does not apply",
        "push directly to main",
        "always push to main",
        "restart gateway after every change",
        "restart/reload after every change",
    ]
    for needle in conflict_needles:
        if needle in text:
            return [RepoPolicyIssue("agents_pointer_conflict", f"Policy drift detected: AGENTS.md conflicts with repo policy pointer ({needle})")]
    return []


def check_repo_policy(repo_path: str | Path) -> RepoPolicyCheckResult:
    repo = Path(repo_path).resolve()
    policy_path = repo / POLICY_RELATIVE_PATH
    policy, issues = _load_policy(policy_path)
    observed = _observed_git(repo)

    repo_class: str | None = None
    authority: dict[str, Any] = {
        "external_effects_allowed": False,
        "fail_closed_behavior": "local_only_on_any_issue",
    }

    if policy is not None:
        _add_contract_issues(policy, issues)
        repo_data = _mapping(policy.get("repo"))
        repo_class = repo_data.get("class")
        if repo_class == "product" and not observed.get("has_develop"):
            issues.append(RepoPolicyIssue("product_develop_not_observed", "Policy drift detected: product policy requires develop, but no local or remote develop ref was observed"))
        workflow = _mapping(policy.get("workflow"))
        if workflow:
            authority["workflow"] = {
                "canonical_landing": workflow.get("canonical_landing"),
                "work_done_means": workflow.get("work_done_means"),
                "release_path": f"{workflow.get('release_source')}->{workflow.get('release_target')}" if workflow.get("release_source") and workflow.get("release_target") else None,
                "live_apply": workflow.get("live_apply"),
            }
        issues.extend(_agents_conflicts(repo))

    ok = not issues
    if ok:
        authority["external_effects_allowed"] = True
        authority["green_only"] = True
        authority["red_still_requires_explicit_approval"] = True
    else:
        authority["external_effects_allowed"] = False
        authority["blocked_external_effects"] = sorted(RED_EXTERNAL_EFFECTS)

    return RepoPolicyCheckResult(
        ok=ok,
        status="pass" if ok else "fail_closed",
        repo_path=str(repo),
        policy_path=str(policy_path),
        repo_class=repo_class,
        issues=issues,
        observed=observed,
        authority=authority,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check .hermes/repo-policy.yaml and observable repo drift.")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path to check")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    result = check_repo_policy(args.repo)
    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Repo policy check: {payload['status']}")
        for issue in payload["issues"]:
            print(f"- {issue['code']}: {issue['message']}")
    return 0 if result.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
