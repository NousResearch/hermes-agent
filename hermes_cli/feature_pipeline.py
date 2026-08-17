#!/usr/bin/env python3
"""Feature Pipeline: Phase A front-half input-presence checks.

Each gate in this file answers a single question: "has the required
artifact / section / data been produced?"  It does NOT judge quality.
Quality judgement is the job of the review stages (Quan fleet, kensei
review, etc.) further down the pipeline.

Provides:
- Artifact storage helpers (get_artifact_path, write_artifact, read_artifact)
- Presence-check functions for each pipeline stage (intake, research, prd, spec)
- Pipeline state machine (advance_pipeline, get_pipeline_status)
"""
import hashlib
import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Artifact storage helpers
# ---------------------------------------------------------------------------

def get_artifact_dir(base_dir: str, task_id: str) -> str:
    """Return the artifact directory path for a task."""
    return os.path.join(base_dir, task_id)


def get_artifact_path(base_dir: str, task_id: str, filename: str) -> str:
    """Return the full path to a specific artifact file."""
    return os.path.join(get_artifact_dir(base_dir, task_id), filename)


def write_artifact(base_dir: str, task_id: str, filename: str, content: str) -> None:
    """Write content to an artifact file, creating directories as needed."""
    artifact_dir = get_artifact_dir(base_dir, task_id)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, filename), "w") as f:
        f.write(content)


def read_artifact(base_dir: str, task_id: str, filename: str) -> Optional[str]:
    """Read content from an artifact file. Returns None if not found."""
    path = get_artifact_path(base_dir, task_id, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Gate functions — return None if gate passes, reason string if blocked
# ---------------------------------------------------------------------------

# Section markers (case-insensitive matching)
_INTAKE_MARKERS = {
    "problem": ("## problem", "# problem", "problem:", "**problem**"),
    "success_criteria": (
        "## success criteria", "# success criteria",
        "success criteria:", "**success criteria**",
        "## success metric", "# success metric",
    ),
}

_RESEARCH_MARKERS = {
    "findings": ("## findings", "# findings", "findings:", "**findings**"),
}

_PRD_MARKERS = {
    "problem": ("## problem", "# problem", "problem:", "**problem**"),
    "users": ("## users", "# users", "users:", "**users**"),
    "scope": ("## scope", "# scope", "scope:", "**scope**"),
    "out_of_scope": (
        "## out of scope", "# out of scope",
        "out of scope:", "**out of scope**",
    ),
    "metrics": ("## metrics", "# metrics", "metrics:", "**metrics**"),
}

_SPEC_MARKERS = {
    "architecture": ("## architecture", "# architecture", "architecture:", "**architecture**"),
    "interfaces": ("## interfaces", "# interfaces", "interfaces:", "**interfaces**"),
    "test_strategy": (
        "## test strategy", "# test strategy",
        "test strategy:", "**test strategy**",
    ),
}


def _check_body_markers(body: Optional[str], markers: dict) -> Optional[str]:
    """Check that body contains all required section markers.

    Returns None if all present, or a human-readable reason string.
    """
    if not body:
        missing = [name.replace("_", " ").title() for name in markers]
        return f"Body is empty. Required sections: {', '.join(missing)}"

    body_lower = body.lower()
    missing = []
    for name, variants in markers.items():
        if not any(v in body_lower for v in variants):
            missing.append(name.replace("_", " ").title())

    if missing:
        return f"Missing required sections: {', '.join(missing)}"
    return None


def validate_intake_brief(body: Optional[str]) -> Optional[str]:
    """Input-presence check: intake brief must have Problem and Success Criteria sections.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist."""
    return _check_body_markers(body, _INTAKE_MARKERS)


def validate_research_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: research-brief.md must exist and have a Findings section.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks artifact exists and has required section."""
    path = os.path.join(artifact_dir, "research-brief.md")
    if not os.path.exists(path):
        return "Missing research-brief.md artifact"
    with open(path) as f:
        content = f.read()

    if not content or not content.strip():
        return "research-brief.md is empty"

    return _check_body_markers(content, _RESEARCH_MARKERS)


def validate_prd_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: prd.md must exist with Problem, Users, Scope, Out of Scope, Metrics.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist."""
    path = os.path.join(artifact_dir, "prd.md")
    if not os.path.exists(path):
        return "Missing prd.md artifact"
    with open(path) as f:
        content = f.read()

    if not content or not content.strip():
        return "prd.md is empty"

    return _check_body_markers(content, _PRD_MARKERS)


def validate_spec_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: spec.md must exist with Architecture, Interfaces, Test Strategy.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist."""
    path = os.path.join(artifact_dir, "spec.md")
    if not os.path.exists(path):
        return "Missing spec.md artifact"
    with open(path) as f:
        content = f.read()

    if not content or not content.strip():
        return "spec.md is empty"

    return _check_body_markers(content, _SPEC_MARKERS)


# Sentinel returned while the council is still deliberating. The
# dispatcher recognises this to launch/await the background run rather than
# bouncing the task to spec (which a normal REVISE reason would do).
# S2: proper sentinel type instead of a bare string so `is` checks work
# and accidental string collisions cannot produce false matches.
class _GatePending:
    """Sentinel: gate cannot evaluate yet; retry next tick."""
    __slots__ = ()
    def __repr__(self) -> str:
        return "GatePending"
    def __bool__(self) -> bool:
        return False  # truthy enough for `if result:` to treat as non-pass

COUNCIL_PENDING = _GatePending()


def validate_council_artifact(artifact_dir: str) -> Optional[str] | _GatePending:
    """Input-presence check: council-verdict.json must exist and contain APPROVED.

    Returns None if approved, or a human-readable reason string.
    Does NOT judge quality; only checks the verdict is present.

    This gate is PURE: it never runs the (expensive, multi-LLM) deliberation
    itself, so it is safe to call inside the dispatcher tick. When the verdict
    is missing it returns the ``COUNCIL_PENDING`` sentinel; the dispatcher is
    responsible for launching the deliberation in the background.

    Reads the machine-readable JSON verdict (C-a). The markdown artifact
    (council-verdict.md) is kept as a human-readable companion.

    Returns None if APPROVED, reason string if REVISE/pending/error.
    """
    json_path = os.path.join(artifact_dir, "council-verdict.json")

    if not os.path.exists(json_path):
        return COUNCIL_PENDING

    # Verdict artifact exists — read and check
    try:
        with open(json_path) as f:
            data = json.loads(f.read())
    except (OSError, json.JSONDecodeError) as exc:
        return f"Cannot read council-verdict.json: {exc}"

    verdict = data.get("verdict", "").upper()
    if verdict == "APPROVED":
        return None

    if verdict == "REVISE":
        issues = data.get("issues", [])
        if issues:
            issue_lines = [
                f"[{i.get('severity', 'medium').upper()}] {i.get('description', '')}"
                for i in issues
            ]
            return "Council REVISE. Issues:\n" + "\n".join(issue_lines)
        return "Council REVISE; see council-verdict.json for details"

    # Verdict unclear — treat as REVISE
    return f"Council verdict unclear ({verdict}); see council-verdict.json"


def validate_tech_review_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: tech-review.md must exist with Architecture, Risk Assessment sections.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist."""
    path = os.path.join(artifact_dir, "tech-review.md")
    if not os.path.exists(path):
        return "Missing tech-review.md artifact"
    with open(path) as f:
        content = f.read()

    if not content or not content.strip():
        return "tech-review.md is empty"

    _MARKERS = {
        "architecture": ("## architecture", "# architecture", "architecture:", "**architecture**"),
        "risks": ("## risks", "# risks", "risks:", "**risks**", "## risk assessment", "# risk assessment"),
    }
    return _check_body_markers(content, _MARKERS)


_DECOMPOSE_ROLES = {"implementation", "qa", "audit"}
_DECOMPOSE_WORKSPACE_KINDS = {"scratch", "worktree", "none"}


def load_decompose_manifest(artifact_dir: str) -> dict:
    """Load the executable decomposition contract.

    ``decompose-output.md`` remains the human-readable design.  This JSON
    sidecar is the machine contract used to create real Kanban tasks; prose is
    deliberately never parsed into task rows.
    """
    path = os.path.join(artifact_dir, "decompose-tasks.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("decompose-tasks.json must contain a JSON object")
    return payload


def _validate_decompose_manifest(payload: dict) -> Optional[str]:
    if payload.get("schema_version") != 1:
        return "decompose-tasks.json schema_version must be 1"
    parent_id = payload.get("parent_task_id")
    if not isinstance(parent_id, str) or not parent_id.strip():
        return "decompose-tasks.json parent_task_id is required"
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "decompose-tasks.json tasks must be a non-empty list"

    keys: list[str] = []
    for index, task in enumerate(tasks):
        label = f"task[{index}]"
        if not isinstance(task, dict):
            return f"{label} must be an object"
        key = task.get("key")
        if not isinstance(key, str) or not key.strip():
            return f"{label}.key is required"
        if key in keys:
            return f"duplicate task key: {key}"
        keys.append(key)
        for field in ("title", "owner", "body"):
            value = task.get(field)
            if not isinstance(value, str) or not value.strip():
                return f"{label}.{field} is required"
        role = task.get("role")
        if role not in _DECOMPOSE_ROLES:
            return f"{label}.role must be one of {sorted(_DECOMPOSE_ROLES)}"
        workspace_kind = task.get("workspace_kind", "scratch")
        if workspace_kind not in _DECOMPOSE_WORKSPACE_KINDS:
            return (
                f"{label}.workspace_kind must be one of "
                f"{sorted(_DECOMPOSE_WORKSPACE_KINDS)}"
            )
        skills = task.get("skills")
        if skills is not None and (
            not isinstance(skills, list)
            or any(not isinstance(skill, str) or not skill.strip() for skill in skills)
            or len(set(skills)) != len(skills)
        ):
            return f"{label}.skills must be a unique list of non-empty skill names"
        dependencies = task.get("dependencies")
        if not isinstance(dependencies, list) or any(
            not isinstance(dep, str) or not dep.strip() for dep in dependencies
        ):
            return f"{label}.dependencies must be a list of task keys"
        if len(set(dependencies)) != len(dependencies):
            return f"{label}.dependencies contains duplicates"

    key_set = set(keys)
    graph: dict[str, list[str]] = {}
    for task in tasks:
        key = task["key"]
        dependencies = task["dependencies"]
        unknown = sorted(set(dependencies) - key_set)
        if unknown:
            return f"task {key} has unknown dependency: {', '.join(unknown)}"
        if key in dependencies:
            return f"task {key} cannot depend on itself"
        graph[key] = dependencies

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> bool:
        if key in visiting:
            return False
        if key in visited:
            return True
        visiting.add(key)
        if any(not visit(dep) for dep in graph[key]):
            return False
        visiting.remove(key)
        visited.add(key)
        return True

    if any(not visit(key) for key in keys):
        return "decompose-tasks.json dependency graph contains a cycle"
    if not any(task["role"] == "implementation" for task in tasks):
        return "decompose-tasks.json requires at least one implementation task"
    if not any(task["role"] == "qa" for task in tasks):
        return "decompose-tasks.json requires at least one qa task"
    if not any(task["role"] == "audit" for task in tasks):
        return "decompose-tasks.json requires at least one audit task"
    return None


def validate_decompose_artifact(artifact_dir: str) -> Optional[str]:
    """Validate human-readable decomposition plus executable task manifest."""
    path = os.path.join(artifact_dir, "decompose-output.md")
    if not os.path.exists(path):
        return "Missing decompose-output.md artifact"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return f"Cannot read decompose-output.md: {exc}"
    if not content or not content.strip():
        return "decompose-output.md is empty"

    content_lower = content.lower()
    if not any(
        marker in content_lower
        for marker in ("## child tasks", "# child tasks", "child tasks:", "**child tasks**")
    ):
        return "Missing required section: Child Tasks"
    missing = []
    if not any(
        marker in content_lower
        for marker in ("## acceptance criteria", "acceptance criteria:", "## ac", "# ac")
    ):
        missing.append("Acceptance Criteria")
    if not any(
        marker in content_lower
        for marker in ("## test plan", "## test strategy", "test plan:", "**test plan**")
    ):
        missing.append("Test Plan")
    if missing:
        return f"Decomposition missing required sections: {', '.join(missing)}"

    manifest_path = os.path.join(artifact_dir, "decompose-tasks.json")
    if not os.path.exists(manifest_path):
        return "Missing decompose-tasks.json artifact"
    try:
        payload = load_decompose_manifest(artifact_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"Invalid decompose-tasks.json: {exc}"
    return _validate_decompose_manifest(payload)


def _load_json_artifact(artifact_dir: str, filename: str) -> tuple[Optional[dict], Optional[str]]:
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        return None, f"Missing {filename} artifact"
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Invalid {filename}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{filename} must contain a JSON object"
    return payload, None


def validate_execute_artifact(artifact_dir: str) -> Optional[str]:
    """Require task-bound, digest-bearing evidence for implementation children."""
    payload, error = _load_json_artifact(artifact_dir, "execution-evidence.json")
    if error:
        return error
    assert payload is not None
    if payload.get("schema_version") != 1:
        return "execution-evidence.json schema_version must be 1"
    if not isinstance(payload.get("parent_task_id"), str) or not payload["parent_task_id"].strip():
        return "execution-evidence.json parent_task_id is required"
    children = payload.get("children")
    if not isinstance(children, list) or not children:
        return "execution-evidence.json children must be a non-empty list"
    seen: set[str] = set()
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            return f"execution child[{index}] must be an object"
        task_id = child.get("task_id")
        key = child.get("key")
        digest = child.get("result_digest")
        if not isinstance(task_id, str) or not task_id.strip():
            return f"execution child[{index}].task_id is required"
        if task_id in seen:
            return f"execution evidence duplicates child task {task_id}"
        seen.add(task_id)
        if not isinstance(key, str) or not key.strip():
            return f"execution child[{index}].key is required"
        if child.get("status") != "done":
            return f"execution child {key} is not done"
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            return f"execution child {key} result_digest must be a SHA-256 hex digest"
    return None


def _safe_report_path(artifact_dir: str, relative: object) -> Optional[str]:
    if not isinstance(relative, str) or not relative.strip() or os.path.isabs(relative):
        return None
    base = os.path.realpath(artifact_dir)
    candidate = os.path.realpath(os.path.join(base, relative))
    if candidate != base and not candidate.startswith(base + os.sep):
        return None
    if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
        return None
    return candidate


def _version_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, str):
        return ()
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def validate_pr_qa_artifact(artifact_dir: str) -> Optional[str]:
    """Require PR, tests, and current Hermaguard/Simplify Swarm evidence."""
    payload, error = _load_json_artifact(artifact_dir, "pr-qa-evidence.json")
    if error:
        return error
    assert payload is not None
    if payload.get("schema_version") != 1:
        return "pr-qa-evidence.json schema_version must be 1"
    if not isinstance(payload.get("parent_task_id"), str) or not payload["parent_task_id"].strip():
        return "pr-qa-evidence.json parent_task_id is required"
    commit = payload.get("commit_sha")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        return "pr-qa-evidence.json commit_sha must be a 40-character hex SHA"
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return "pr-qa-evidence.json pull_request object is required"
    url = pull_request.get("url")
    if not isinstance(url, str) or not url.startswith("https://github.com/") or "/pull/" not in url:
        return "pr-qa-evidence.json pull_request.url must be a GitHub PR URL"
    if pull_request.get("state") not in {"open", "merged"}:
        return "pr-qa-evidence.json pull_request.state must be open or merged"
    tests = payload.get("tests")
    if not isinstance(tests, list) or not tests:
        return "pr-qa-evidence.json tests must be a non-empty list"
    for index, test in enumerate(tests):
        if not isinstance(test, dict) or not isinstance(test.get("command"), str) or not test["command"].strip():
            return f"pr-qa test[{index}].command is required"
        if test.get("exit_code") != 0:
            return f"pr-qa test[{index}] did not pass"
        if not isinstance(test.get("summary"), str) or not test["summary"].strip():
            return f"pr-qa test[{index}].summary is required"
    gates = payload.get("quality_gates")
    if not isinstance(gates, dict):
        return "pr-qa-evidence.json quality_gates object is required"
    minimums = {"hermaguard": (2, 1, 0), "simplify_swarm": (2, 0, 0)}
    for name, minimum in minimums.items():
        gate = gates.get(name)
        if not isinstance(gate, dict):
            return f"missing quality gate: {name}"
        if gate.get("status") != "pass":
            return f"quality gate {name} did not pass"
        if _version_tuple(gate.get("version")) < minimum:
            return f"quality gate {name} must use version {'.'.join(map(str, minimum))} or newer"
        report_path = _safe_report_path(artifact_dir, gate.get("report"))
        if report_path is None:
            return f"quality gate {name} report is missing, empty, or outside the artifact directory"
        report_digest = gate.get("report_sha256")
        if not isinstance(report_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", report_digest):
            return f"quality gate {name} report_sha256 must be a lowercase SHA-256 digest"
        with open(report_path, "rb") as report_handle:
            actual_digest = hashlib.sha256(report_handle.read()).hexdigest()
        if actual_digest != report_digest:
            return f"quality gate {name} report SHA-256 mismatch"
    return None


# ---------------------------------------------------------------------------
# Audit gate (Phase D)
# ---------------------------------------------------------------------------

# Quan's six-worker fleet writes one sub-section per reviewer. The aggregated
# verdict is parsed from the **Verdict:** line. kensei-review adds its own
# section independently — both must be present for the gate to pass.
_AUDIT_FLEET_REVIEWERS = ("code", "arch", "perf", "security", "ux")
_AUDIT_REQUIRED_SECTIONS = (
    "## quan-fleet", "## kensei-review", "## verdict",
)


def _parse_audit_verdict(content: str) -> Optional[str]:
    """Extract the aggregated verdict from an audit-report.md.

    Recognised forms (case-insensitive):
        **Verdict: PASS**
        **Verdict: CONDITIONAL**
        **Verdict: BLOCKED**
    Returns the uppercased verdict string, or None if not found.
    """
    import re
    m = re.search(
        r"\*\*\s*verdict\s*:\s*(PASS|CONDITIONAL|BLOCKED)\s*\*\*",
        content, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    # Fallback: plain "Verdict: PASS" line
    m = re.search(
        r"^verdict\s*:\s*(PASS|CONDITIONAL|BLOCKED)\s*$",
        content, re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return m.group(1).upper()
    return None


def _check_audit_fleet_section(content: str) -> Optional[str]:
    """Verify the quan-fleet block contains all five reviewer sub-verdicts."""
    import re
    fleet_block = re.search(
        r"##\s*quan-fleet\s*(.+?)(?=^##\s|\Z)",
        content, re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not fleet_block:
        return "Missing quan-fleet section"
    body = fleet_block.group(1)
    # Strip all bold markers before matching so both "**code:** PASS" and
    # "code: PASS" match identically. This is a lenient parser that prefers
    # not to miss valid input over strict formatting requirements.
    import re
    clean = re.sub(r'\*{1,2}', '', body)
    missing = []
    for reviewer in _AUDIT_FLEET_REVIEWERS:
        if not re.search(
            rf"\b{re.escape(reviewer)}\s*:\s*(PASS|CONDITIONAL|BLOCKED)\b",
            clean, re.IGNORECASE,
        ):
            missing.append(reviewer)
    if missing:
        return f"Quan fleet missing verdicts for: {', '.join(missing)}"
    return None


def _check_kensei_review_section(content: str) -> Optional[str]:
    """Verify the kensei-review section is present and non-empty."""
    import re
    block = re.search(
        r"##\s*kensei-review\s*(.+?)(?=^##\s|\Z)",
        content, re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not block:
        return "Missing kensei-review section"
    body = block.group(1).strip()
    if len(body) < 50:
        return "kensei-review section is too short (need substantive content)"
    return None


def validate_audit_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: audit-report.md must exist with quan fleet + kensei-review + verdict.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist.

    Verdict semantics (design doc §3 [11]):
        - PASS        — gate passes; advance to final_sign_off.
        - CONDITIONAL — gate passes; auto-creates a follow-up task so the
                        conditional issues are tracked, then advance.
        - BLOCKED     — gate fails; bounces to spec (capped) so the spec
                        author can address the blockers.

    Returns None if PASS/CONDITIONAL, reason string if BLOCKED or invalid.
    """
    path = os.path.join(artifact_dir, "audit-report.md")
    if not os.path.exists(path):
        return "Missing audit-report.md artifact"
    try:
        with open(path) as f:
            content = f.read()
    except OSError as exc:
        return f"Cannot read audit-report.md: {exc}"

    if not content or not content.strip():
        return "audit-report.md is empty"

    content_lower = content.lower()
    missing_sections = [
        marker for marker in _AUDIT_REQUIRED_SECTIONS
        if marker not in content_lower
    ]
    if missing_sections:
        return f"audit-report.md missing sections: {', '.join(missing_sections)}"

    fleet_err = _check_audit_fleet_section(content)
    if fleet_err:
        return fleet_err

    review_err = _check_kensei_review_section(content)
    if review_err:
        return review_err

    verdict = _parse_audit_verdict(content)
    if verdict is None:
        return "audit-report.md has no parseable Verdict line (PASS/CONDITIONAL/BLOCKED)"

    if verdict == "BLOCKED":
        return "Audit BLOCKED — see audit-report.md for blocker list"
    # PASS or CONDITIONAL: gate passes. The dispatcher reads the verdict
    # via get_audit_verdict_for_dispatch() to decide whether to spawn a
    # follow-up task for CONDITIONAL issues.
    return None


def get_audit_verdict(artifact_dir: str) -> Optional[str]:
    """Read the audit verdict from audit-report.md. Returns None on miss."""
    path = os.path.join(artifact_dir, "audit-report.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            content = f.read()
    except OSError:
        return None
    return _parse_audit_verdict(content)


def validate_document_artifact(artifact_dir: str) -> Optional[str]:
    """Input-presence check: docs-output.md must exist with sections for the wiki entry.

    Returns None if present, or a human-readable reason string.
    Does NOT judge quality; only checks required sections exist."""
    path = os.path.join(artifact_dir, "docs-output.md")
    if not os.path.exists(path):
        return "Missing docs-output.md artifact"
    try:
        with open(path) as f:
            content = f.read()
    except OSError as exc:
        return f"Cannot read docs-output.md: {exc}"

    if not content or not content.strip():
        return "docs-output.md is empty"

    # Light is responsible for docs; require an overview + usage section so
    # the wiki entry is self-contained.
    _MARKERS = {
        "overview": ("## overview", "# overview", "overview:", "**overview**"),
        "usage": ("## usage", "# usage", "usage:", "**usage**"),
    }
    return _check_body_markers(content, _MARKERS)


def check_human_approved(conn: "sqlite3.Connection", task_id: str, stage: str) -> bool:
    """Check if a task has been approved by a human for the given stage.

    Looks for ``human_approved`` events in the events table.
    The stage field in the event payload must match the current stage.
    The approval must have been granted AFTER the task entered this stage
    (guards against stale approval from a previous cycle being reused).
    """
    # Find the timestamp when the task entered the current stage
    entered = conn.execute(
        "SELECT created_at FROM task_events "
        "WHERE task_id = ? AND kind = 'pipeline_advanced' "
        "AND json_extract(payload, '$.to_stage') = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (task_id, stage),
    ).fetchone()
    entered_at = entered[0] if entered else 0

    # Approval must exist AND be newer than the stage entry
    row = conn.execute(
        "SELECT 1 FROM task_events "
        "WHERE task_id = ? AND kind = 'human_approved' "
        "AND json_extract(payload, '$.stage') = ? "
        "AND created_at > ? "
        "ORDER BY created_at DESC LIMIT 1",
        (task_id, stage, entered_at),
    ).fetchone()
    return row is not None


def time_in_stage_hours(conn: "sqlite3.Connection", task_id: str, stage: str) -> float:
    """Return hours since the task entered the given stage."""
    row = conn.execute(
        "SELECT created_at FROM task_events "
        "WHERE task_id = ? AND kind = 'pipeline_advanced' "
        "AND json_extract(payload, '$.to_stage') = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (task_id, stage),
    ).fetchone()
    if not row or row[0] is None:
        return 0.0
    # created_at is epoch-seconds (int) in the live schema, but tests may
    # insert an ISO-8601 string. Handle both.
    import datetime
    raw = row[0]
    try:
        created_epoch = float(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.datetime.fromisoformat(str(raw))
        except ValueError:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        created_epoch = dt.timestamp()
    import time as _time
    return max(0.0, (_time.time() - created_epoch) / 3600.0)


# ---------------------------------------------------------------------------
# Pipeline state machine
# ---------------------------------------------------------------------------

# Stage order in the feature pipeline (matches design doc §3).
# Full tier=full path: 1.research 2.prd 3.spec 4.council 5.sign_off
# Pipeline stage list.  MUST stay in sync with the canonical design doc
# docs/kanban/pipeline.md (or whatever the canonical source becomes).  If
# adding / removing stages, update the design doc and the gate map below
# simultaneously.  Express stages are a subset of this list (see
# EXPRESS_PIPELINE_STAGES).
PIPELINE_STAGES = [
    "research",      # [2] research brief
    "prd",           # [3] product requirements doc
    "spec",          # [4] tech spec
    "council",       # [5] LLM council deliberation
    "sign_off",      # [6] ⏸ spec sign-off (human)
    "tech_review",   # [7] octacon architectural review
    "decompose",     # [8] child task decomposition (WS-1 contract gate)
    "execute",       # [9] workers claim and ship code
    "pr+qa",         # [10] PR open + tests green
    "audit",         # [11] multi-layer audit (quan fleet + kensei-review)
    "final_sign_off",# [12] ⏸ final sign-off (human)
    "document",      # [13] light → wiki/docs
]

# Gate function mapping per stage. Every execution-bearing stage has an
# evidence gate; no implementation or QA stage is allowed to pass through on
# elapsed time alone.
GATE_FUNCTIONS = {
    "research": validate_research_artifact,
    "prd": validate_prd_artifact,
    "spec": validate_spec_artifact,
    "council": validate_council_artifact,
    "tech_review": validate_tech_review_artifact,
    "decompose": validate_decompose_artifact,
    "execute": validate_execute_artifact,
    "pr+qa": validate_pr_qa_artifact,
    "audit": validate_audit_artifact,
    "document": validate_document_artifact,
}

# Human-gate stages: these stages require manual approval via CLI/Discord.
# The dispatcher checks for a ``human_approved`` event in the events table
# rather than running a gate function on disk artifacts.
HUMAN_GATE_STAGES = {"sign_off", "final_sign_off"}

# Retained as a compatibility export for callers that imported the symbol.
# Safety-sensitive pipeline stages are no longer pass-through.
PASS_THROUGH_STAGES: set[str] = set()

# Express path: drops PRD, Council, Tech Review; keeps the two human gates
# and the full audit. Used by ``hermes feature create --express`` and the
# /feature express skill. Skipped stages are logged as bypass-records.
EXPRESS_PIPELINE_STAGES = [
    "research", "spec", "sign_off", "decompose", "execute",
    "pr+qa", "audit", "final_sign_off", "document",
]


def get_next_stage(current_stage: str, pipeline_mode: str = "full") -> Optional[str]:
    """Return the next pipeline stage, or None if at the end.

    ``pipeline_mode`` selects which stage set to walk:
        - ``"full"`` (default): all 12 stages (design doc §3 full path).
        - ``"express"``: 9 stages (design doc §4a express path), skipping
          PRD, Council, and Tech Review.
    """
    if pipeline_mode == "express":
        stages = EXPRESS_PIPELINE_STAGES
    else:
        stages = PIPELINE_STAGES
    try:
        idx = stages.index(current_stage)
        if idx + 1 < len(stages):
            return stages[idx + 1]
    except ValueError:
        pass
    return None


def get_pipeline_mode(task_row: dict) -> str:
    """Return ``"express"`` if the task is in express mode, else ``"full"``.

    Reads the ``pipeline_mode`` column from the task row. Defaults to
    ``"full"`` when unset so the existing data is unaffected.
    """
    if not task_row:
        return "full"
    mode = task_row.get("pipeline_mode") or "full"
    return mode if mode in ("full", "express") else "full"


def get_skipped_stages(pipeline_mode: str = "full") -> list[str]:
    """Return the set of full-pipeline stages that the given mode skips.

    For express: PRD, Council, Tech Review. For full: empty.
    """
    if pipeline_mode == "express":
        return [s for s in PIPELINE_STAGES if s not in EXPRESS_PIPELINE_STAGES]
    return []


def get_pipeline_status(task_id: str, artifact_base_dir: str) -> dict:
    """Return the current pipeline status for a task (DB-backed).

    Returns dict with keys:
        task_id: str
        current_stage: Optional[str]  # None if not in pipeline
        pipeline_mode: str            # "full" or "express"
        gate_status: str  # "pass", "fail", "pending", "not_in_pipeline", "unknown"
        gate_message: Optional[str]  # reason if gate fails
        next_stage: Optional[str]  # next stage if gate passes
    """
    base = {
        "task_id": task_id,
        "current_stage": None,
        "pipeline_mode": "full",
        "gate_status": "unknown",
        "gate_message": None,
        "next_stage": None,
    }
    try:
        from hermes_cli.kanban_db import connect, get_task
    except Exception:
        return base

    try:
        with connect() as conn:
            task = get_task(conn, task_id)
    except Exception:
        return base
    if task is None:
        return base

    stage = task.pipeline_stage
    mode = get_pipeline_mode({"pipeline_mode": task.pipeline_mode})
    base["pipeline_mode"] = mode
    base["current_stage"] = stage

    if not stage:
        base["gate_status"] = "not_in_pipeline"
        return base

    base["next_stage"] = get_next_stage(stage, mode)

    if stage in HUMAN_GATE_STAGES:
        try:
            from hermes_cli.kanban_db import connect as _c
            with _c() as conn:
                approved = check_human_approved(conn, task_id, stage)
        except Exception:
            approved = False
        base["gate_status"] = "pass" if approved else "pending"
        if not approved:
            base["gate_message"] = "Awaiting human sign-off"
        return base

    if stage in PASS_THROUGH_STAGES:
        base["gate_status"] = "pending"
        base["gate_message"] = "Work in progress (no artifact gate)"
        return base

    gate_fn = GATE_FUNCTIONS.get(stage)
    if gate_fn is None:
        base["gate_status"] = "pass"
        return base

    artifact_dir = os.path.join(artifact_base_dir, task_id)
    result = gate_fn(artifact_dir)
    if result is None:
        base["gate_status"] = "pass"
    else:
        base["gate_status"] = "fail"
        base["gate_message"] = result
    return base


def advance_pipeline(
    task_id: str, current_stage: str, artifact_base_dir: str,
    pipeline_mode: str = "full",
) -> dict:
    """Try to advance a task to the next pipeline stage.

    ``pipeline_mode`` ("full"/"express") selects the stage set so express
    tasks correctly skip PRD/Council/Tech Review.

    Returns dict with keys:
        advanced: bool
        from_stage: str
        to_stage: Optional[str]
        gate_passed: bool
        gate_message: Optional[str]
    """
    gate_fn = GATE_FUNCTIONS.get(current_stage)
    if gate_fn is None:
        # No gate for this stage (e.g. council — Phase B)
        next_stage = get_next_stage(current_stage, pipeline_mode)
        return {
            "advanced": next_stage is not None,
            "from_stage": current_stage,
            "to_stage": next_stage,
            "gate_passed": True,
            "gate_message": None,
        }

    artifact_dir = os.path.join(artifact_base_dir, task_id)
    gate_result = gate_fn(artifact_dir)

    if gate_result is None:
        # Gate passed
        next_stage = get_next_stage(current_stage, pipeline_mode)
        return {
            "advanced": next_stage is not None,
            "from_stage": current_stage,
            "to_stage": next_stage,
            "gate_passed": True,
            "gate_message": None,
        }
    else:
        # Gate failed
        return {
            "advanced": False,
            "from_stage": current_stage,
            "to_stage": None,
            "gate_passed": False,
            "gate_message": gate_result,
        }
