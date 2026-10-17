"""Small, testable policy helpers for the CI orchestrator workflow."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Mapping

FORK = "mekenthompson/hermes-agent"

# These products are not shipped in the fork's Linux runtime image. All other
# selected lanes, including Python/runtime E2E and bundled web/TUI JS, still gate it.
_NON_IMAGE_JOBS = frozenset({
    "native-install-tests",
    "tests-os", "installer-tests", "rust-tests", "bootstrap-installer",
    "e2e-desktop", "e2e-desktop-core", "e2e-desktop-update", "docs-site",
    # Advisory PR summary and always-on blob hygiene are not image runtime.
    "prompt-surface-diff", "blob-size-check",
})

_FORK_IMAGE_WORKFLOW = ".github/workflows/fork-agent-image.yml"
_FORK_UPGRADE_UNRELATED_PREFIXES = (
    "gateway/platforms/", "plugins/kanban/", "tests/agent/test_",
    "tests/gateway/platforms/test_", "website/docs/",
)


def is_fork(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Explicit fork mode, or the maintained fork's Actions repository identity."""
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if environ is None else environ
    return "--fork" in argv or (env.get("REPO") or env.get("GITHUB_REPOSITORY") or "") == FORK


def apply_fork_classification(files: list[str], lanes: Mapping[str, bool]) -> dict[str, bool]:
    """Apply the maintained fork's image and upgrade policy to shared lanes.

    Shared path classification stays in classify_changes. These decisions are
    deployment policy for the fork's smaller runners and image contents.
    """
    result = dict(lanes)
    result["upgrade"] = not files or any(
        not path.startswith(_FORK_UPGRADE_UNRELATED_PREFIXES) for path in files
    )
    ci_change = any(path.startswith("scripts/ci/") for path in files)
    if ci_change:
        # A classifier/policy edit runs every lane except unrelated MCP catalog review.
        for lane in result:
            if lane not in {"mcp_catalog", "upgrade"}:
                result[lane] = True
    if files and (ci_change or any(path.startswith(".github/") for path in files)):
        # .github is ignored by the image; its publication workflow is the exception.
        result["docker"] = any(
            path == _FORK_IMAGE_WORKFLOW or not path.startswith(".github/") for path in files
        )
    if any(path.startswith(("apps/shared/", "web/", "skills/")) for path in files):
        result["docker"] = True
    return result


@dataclass(frozen=True)
class NeedsSummary:
    """The compact job-result map and jobs that prevent a passing gate."""

    compact: dict[str, str]
    failed: list[str]


def should_cancel_in_progress(event_name: str, repository: str, ref: str) -> bool:
    """Supersede PRs everywhere and only main pushes in the maintained fork."""
    return event_name == "pull_request" or (
        event_name == "push"
        and repository == "mekenthompson/hermes-agent"
        and ref == "refs/heads/main"
    )


def selected_jobs(
    lanes: Mapping[str, str], repository: str, event_name: str, *, release: bool = False,
) -> dict[str, bool]:
    """One applicability decision consumed by both CI jobs and their gate."""

    def on(lane: str) -> bool:
        return lanes.get(lane) == "true"

    fork = repository == FORK
    pr = event_name == "pull_request"
    product = on("python_prod") or on("frontend")
    # Unshipped surfaces stay upstream-owned. A full sync that turns every
    # classifier lane true must not schedule them on the fork. ``release`` does
    # not opt the fork back in; image publication is its own dispatch.
    _ = release
    return {
        "detect": True,
        "tests": on("python"),
        "catalog-tests": on("catalog"),
        "native-install-tests": False,
        "tests-os": on("python") and not fork,
        "lint": on("python"),
        "js-tests": on("frontend"),
        "installer-tests": on("installer") and not fork,
        "rust-tests": on("rust") and not fork,
        "bootstrap-installer": on("bootstrap") and not fork,
        "e2e-desktop": False,  # existing intentional disablement
        "e2e-desktop-core": product and not fork,
        "e2e-desktop-update": product and not fork,
        "docs-site": on("site") and not fork,
        "history-check": pr,
        "contributor-check": on("python") and not fork,
        "uv-lockfile": on("uv_lock"),
        "infographic-check": not fork,
        "profile-artifact-check": not fork or on("binary_artifacts"),
        "icons-freshness-check": not fork,
        "case-collision-check": not fork,
        "lazy-deps-guard": not fork,
        "lockfile-diff": pr and on("npm_lock"),
        "prompt-surface-diff": pr and on("python"),
        "docker-lint": on("docker_meta"),
        "supply-chain": pr and (on("scan") or on("deps")),
        "review-labels": pr and (on("ci_review") or on("mcp_catalog")),
        "osv-scanner": not fork or on("uv_lock") or on("npm_lock") or on("deps"),
        "remote-backends": on("remote_backends"),
        # Always-on hygiene check. It has no detect if, so a skip is a failure.
        "blob-size-check": True,
    }


def image_selected_jobs(selected: Mapping[str, bool]) -> dict[str, bool]:
    return {name: required for name, required in selected.items() if name not in _NON_IMAGE_JOBS}


def evaluate_needs(
    needs: Mapping[str, Mapping[str, str]],
    selected: Mapping[str, bool],
    *,
    critical_findings: bool = False,
) -> NeedsSummary:
    """Every expected job must report; a selected one must succeed."""
    compact = {name: info.get("result", "missing") for name, info in needs.items()}
    if not selected or any(type(value) is not bool for value in selected.values()):
        return NeedsSummary(compact=compact, failed=["selected_jobs"])
    applicable = dict(selected)
    if critical_findings and applicable.get("review-labels") is False:
        applicable["review-labels"] = True
    failed = [
        name
        for name, required in applicable.items()
        if compact.get(name)
        not in ({"success"} if required else {"success", "skipped"})
    ]
    failed.extend(
        name
        for name, result in compact.items()
        if name not in applicable and result != "success"
    )
    return NeedsSummary(compact=compact, failed=failed)


def main() -> int:
    if "--select" in sys.argv[1:]:
        lanes = json.loads(os.environ.get("LANES", "{}"))
        selected = selected_jobs(
            lanes, os.environ.get("REPO", ""), os.environ.get("EVENT_NAME", ""),
            release=os.environ.get("RELEASE") == "true",
        )
        out = "selected_jobs=" + json.dumps(selected, separators=(",", ":"))
        print(out)
        if path := os.environ.get("GITHUB_OUTPUT"):
            with open(path, "a", encoding="utf-8") as output:
                output.write(out + "\n")
        return 0
    needs = json.load(sys.stdin)
    try:
        selected = json.loads(os.environ.get("SELECTED_JOBS", "{}"))
    except json.JSONDecodeError:
        selected = {}
    if "--image" in sys.argv[1:]:
        selected = image_selected_jobs(selected)
    summary = evaluate_needs(
        needs, selected, critical_findings=os.environ.get("CRITICAL_FINDINGS") == "true"
    )
    needs_json = json.dumps(summary.compact, separators=(",", ":"))
    print(f"needs-json={needs_json}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"needs-json={needs_json}\n")
    for name, result in sorted(summary.compact.items()):
        icon = "✅" if name not in summary.failed else "❌"
        print(f"{icon} {name}: {result}")
    if summary.failed:
        print(
            f"::error::{len(summary.failed)} job(s) did not pass: "
            f"{', '.join(summary.failed)}"
        )
        return 1
    print("All selected checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
