"""Task 29 Phase I — artifact inspection tests (subset T003–T045, T113, T126)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from htr.advisory_inspection_constants import SUPPLEMENTAL_FINDING_TOKENS
from htr.advisory_inspection_models import ArtifactReferenceSelector
from htr.advisory_inspection_path import lexical_validate_artifact_path
from htr.advisory_inspection_secure import raw_sha256_digest
from htr.artifact_inspection import inspect_artifact_reference, is_supplemental_finding
from htr import io, events, paths
from htr.ids import new_attempt_id, new_run_id, new_task_id

_FORBIDDEN_NAMES = frozenset(
    {
        "read_artifact_manifest",
        "read_json",
        "evaluate_run_seal",
        "parse_strict_json_bytes",
    }
)

_ARTIFACT_AXIS_SCALARS = frozenset(
    {
        "reference_selected",
        "reference_absent_from_manifest",
        "manifest_bound",
        "path_valid_attempt_relative",
        "filesystem_observed",
        "advisory_only",
        "budget_within_limits",
    }
)


def _bootstrap_attempt() -> tuple[str, str, str]:
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    io.create_run_workspace(run_id)
    io.create_task_workspace(run_id, task_id)
    events.register_attempt(run_id, task_id, attempt_id, actor="test")
    return run_id, task_id, attempt_id


def _write_manifest(
    run_id: str,
    task_id: str,
    attempt_id: str,
    payload: dict,
    *,
    trailing_lf: bool = True,
) -> str:
    target = paths.artifact_manifest_path(run_id, task_id, attempt_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if trailing_lf:
        raw += b"\n"
    target.write_bytes(raw)
    return raw_sha256_digest(raw)


def _selector(
    run_id: str,
    task_id: str,
    attempt_id: str,
    digest: str,
    entry_index: int = 0,
) -> ArtifactReferenceSelector:
    return ArtifactReferenceSelector(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        manifest_raw_digest=digest,
        entry_index=entry_index,
    )


def test_t003_invalid_run_id_rejected_before_open():
    selector = ArtifactReferenceSelector(
        run_id="not-a-run-id",
        task_id=new_task_id(),
        attempt_id=new_attempt_id(),
        manifest_raw_digest="sha256:" + "a" * 64,
        entry_index=0,
    )
    result = inspect_artifact_reference(selector)
    assert result.authority_status == "selector_identity_invalid"
    assert result.manifest_status == "manifest_absent"


def test_t004_base_dir_kwarg_rejected():
    run_id, task_id, attempt_id = _bootstrap_attempt()
    digest = _write_manifest(
        run_id,
        task_id,
        attempt_id,
        {"schema_version": "1", "run_id": run_id, "task_id": task_id, "attempt_id": attempt_id, "artifacts": []},
    )
    selector = _selector(run_id, task_id, attempt_id, digest)
    result = inspect_artifact_reference(selector, base_dir="/tmp/evil")
    assert result.authority_status == "caller_host_root_rejected"
    assert result.aggregate_completeness == "aggregate_blocked_untrusted_scope"


def test_t023_missing_artifacts_key():
    run_id, task_id, attempt_id = _bootstrap_attempt()
    digest = _write_manifest(
        run_id,
        task_id,
        attempt_id,
        {"schema_version": "1", "run_id": run_id, "task_id": task_id, "attempt_id": attempt_id},
    )
    result = inspect_artifact_reference(_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_bound"
    assert result.reference_status == "reference_absent_from_manifest"


def test_t024_artifacts_not_list():
    run_id, task_id, attempt_id = _bootstrap_attempt()
    digest = _write_manifest(
        run_id,
        task_id,
        attempt_id,
        {
            "schema_version": "1",
            "run_id": run_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "artifacts": "not-a-list",
        },
    )
    result = inspect_artifact_reference(_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_bound"
    assert result.reference_status == "reference_absent_from_manifest"


@pytest.mark.parametrize(
    ("declared_path", "expected_status"),
    [
        ("dir/./file.txt", "path_dot_rejected"),  # T032
        ("a//b", "path_separator_rejected"),  # T033
        ("/etc/passwd", "path_absolute_rejected"),  # T036
        ("../sibling", "path_dotdot_rejected"),  # T037
        ("", "path_empty_rejected"),  # T038
        ("artifacts/out/", "path_separator_rejected"),  # T039
        ("/leading", "path_absolute_rejected"),  # T040
        (r"artifacts\out.txt", "path_backslash_rejected"),  # T041
        ("artifacts/out\x00.txt", "path_control_rejected"),  # T042
        ("a" * 4097, "path_budget_exceeded"),  # T043
        ("/".join(["c"] * 33), "path_budget_exceeded"),  # T044
        ("artifacts/" + ("x" * 256), "path_budget_exceeded"),  # T045
    ],
)
def test_t032_t045_path_lexical_cases(declared_path: str, expected_status: str):
    status, components, _ = lexical_validate_artifact_path(declared_path)
    assert status == expected_status
    assert components is None


def test_supplemental_finding_registry_has_23_tokens():
    assert len(SUPPLEMENTAL_FINDING_TOKENS) == 23


def test_findings_never_contain_axis_scalars():
    run_id, task_id, attempt_id = _bootstrap_attempt()
    digest = _write_manifest(
        run_id,
        task_id,
        attempt_id,
        {
            "schema_version": "1",
            "run_id": run_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "artifacts": [
                {
                    "path": "artifacts/output.txt",
                    "kind": "file",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "metadata": {},
                }
            ],
        },
    )
    result = inspect_artifact_reference(_selector(run_id, task_id, attempt_id, digest))
    for finding in result.findings:
        assert finding not in _ARTIFACT_AXIS_SCALARS
        assert is_supplemental_finding(finding) or finding in SUPPLEMENTAL_FINDING_TOKENS


def _collect_source_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
    return names


def test_t113_forbidden_api_spy_artifact_inspection():
    repo_root = Path(__file__).resolve().parents[2]
    for rel in ("htr/artifact_inspection.py", "htr/advisory_inspection_secure.py", "htr/advisory_inspection_decoder.py"):
        names = _collect_source_names(repo_root / rel)
        assert _FORBIDDEN_NAMES.isdisjoint(names), f"forbidden API referenced in {rel}"


def test_publication_and_atime_defaults():
    run_id, task_id, attempt_id = _bootstrap_attempt()
    digest = _write_manifest(
        run_id,
        task_id,
        attempt_id,
        {"schema_version": "1", "run_id": run_id, "task_id": task_id, "attempt_id": attempt_id, "artifacts": []},
    )
    result = inspect_artifact_reference(_selector(run_id, task_id, attempt_id, digest))
    assert result.publication == "none"
    assert result.atime_may_have_changed is True
    assert result.may_execute is False
