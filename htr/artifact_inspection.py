"""Task 29 Phase I — advisory artifact reference inspection (R5)."""

from __future__ import annotations

import os
import re
from typing import Any

from htr.advisory_inspection_aggregate import (
    artifact_item_sort_key,
    compute_aggregate_budget_status,
    compute_aggregate_completeness,
)
from htr.advisory_inspection_constants import (
    MAX_ARTIFACT_REFERENCES_PER_AGGREGATE,
    MAX_ARTIFACT_REFERENCES_PER_MANIFEST,
    MAX_MANIFESTS_PER_AGGREGATE,
    MAX_TOTAL_BYTES_HASHED,
    SUPPLEMENTAL_FINDING_TOKENS,
)
from htr.advisory_inspection_models import (
    ArtifactAggregateResult,
    ArtifactInspectionResult,
    ArtifactReferenceSelector,
    UnreferencedObservation,
    sort_findings,
)
from htr.advisory_inspection_path import (
    PATH_LEXICAL_FATAL,
    lexical_validate_artifact_path,
    path_identity_digest,
)
from htr.advisory_inspection_run_context import detect_run_context
from htr.advisory_inspection_secure import (
    HashArtifactResult,
    classify_regular_file_presence,
    hash_artifact_file,
    open_intermediate_dir,
    os_close_runs_root,
    read_regular_control_file,
    scan_unreferenced_artifacts,
    semantic_sha256_digest,
    validate_runs_root_s0,
    walk_attempt_path,
    walk_run_path,
)
from htr.bounded_action_control_paths import list_dir_names_sorted
from htr.ids import require_id

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_KWARGS = frozenset(
    {
        "base_dir",
        "runs_root",
        "project_dir",
        "path",
        "url",
        "body",
        "fd",
        "inode",
        "mime",
        "item_id",
        "link_source_record",
    }
)
_MANIFEST_NAME = "artifact_manifest.json"
_MANIFEST_KNOWN_FIELDS = frozenset({"schema_version", "run_id", "task_id", "attempt_id", "artifacts"})


def _reject_extra_kwargs(kwargs: dict[str, Any]) -> str | None:
    if any(key in kwargs for key in _FORBIDDEN_KWARGS):
        return "caller_host_root_rejected"
    if kwargs:
        return "caller_host_root_rejected"
    return None


def _validate_artifact_selector(selector: ArtifactReferenceSelector) -> str | None:
    if not isinstance(selector.run_id, str):
        return "selector_identity_invalid"
    if not isinstance(selector.task_id, str):
        return "selector_identity_invalid"
    if not isinstance(selector.attempt_id, str):
        return "selector_identity_invalid"
    if not isinstance(selector.manifest_raw_digest, str):
        return "selector_identity_invalid"
    if not isinstance(selector.entry_index, int) or isinstance(selector.entry_index, bool):
        return "selector_identity_invalid"

    try:
        require_id(selector.run_id, "run")
        require_id(selector.task_id, "task")
        require_id(selector.attempt_id, "attempt")
    except ValueError:
        return "selector_identity_invalid"

    if not _DIGEST_RE.match(selector.manifest_raw_digest):
        return "selector_identity_invalid"

    if selector.entry_index < 0:
        return "selector_entry_index_out_of_range"

    return None


def _base_artifact_result(selector: ArtifactReferenceSelector | None = None) -> ArtifactInspectionResult:
    result = ArtifactInspectionResult()
    if selector is not None:
        result.run_id = selector.run_id
        result.task_id = selector.task_id
        result.attempt_id = selector.attempt_id
        result.entry_index = selector.entry_index
        result.manifest_raw_digest = selector.manifest_raw_digest
    return result


def _is_entry_malformed(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return True
    for field in ("path", "kind", "created_at"):
        val = entry.get(field)
        if not isinstance(val, str) or not val:
            return True
    metadata = entry.get("metadata")
    if metadata is None or not isinstance(metadata, dict):
        return True
    if "sha256" in entry:
        sha = entry["sha256"]
        if sha is not None and (not isinstance(sha, str) or not sha):
            return True
    if "size_bytes" in entry:
        size = entry["size_bytes"]
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            return True
    return False


def _entry_duplicate_key(entry: dict[str, Any]) -> tuple[tuple[str, ...], str, Any, Any]:
    path_status, components, _ = lexical_validate_artifact_path(entry.get("path"))
    if components is None:
        components = ()
    return (
        components,
        str(entry.get("kind", "")),
        entry.get("sha256"),
        entry.get("size_bytes"),
    )


def _build_index_maps(
    artifacts: list[Any],
) -> tuple[dict[tuple[tuple[str, ...], str, Any, Any], list[int]], dict[tuple[str, ...], list[int]]]:
    duplicate_keys: dict[tuple[tuple[str, ...], str, Any, Any], list[int]] = {}
    conflict_keys: dict[tuple[str, ...], list[int]] = {}
    limit = min(len(artifacts), MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
    for index in range(limit):
        entry = artifacts[index]
        if not isinstance(entry, dict) or _is_entry_malformed(entry):
            continue
        dup_key = _entry_duplicate_key(entry)
        duplicate_keys.setdefault(dup_key, []).append(index)
        comp = dup_key[0]
        if comp:
            conflict_keys.setdefault(comp, []).append(index)
    return duplicate_keys, conflict_keys


def _reference_status_for_index(
    artifacts: list[Any],
    index: int,
    *,
    duplicate_keys: dict[tuple[tuple[str, ...], str, Any, Any], list[int]],
    conflict_keys: dict[tuple[str, ...], list[int]],
) -> str:
    if index >= len(artifacts):
        return "reference_index_out_of_range"
    if index >= MAX_ARTIFACT_REFERENCES_PER_MANIFEST:
        return "reference_not_processed_budget"
    entry = artifacts[index]
    if not isinstance(entry, dict) or _is_entry_malformed(entry):
        return "reference_malformed"
    dup_key = _entry_duplicate_key(entry)
    if len(duplicate_keys.get(dup_key, [])) > 1:
        return "reference_exact_duplicate_member"
    comp = dup_key[0]
    if comp and len(conflict_keys.get(comp, [])) > 1:
        path_entries = conflict_keys[comp]
        distinct = { _entry_duplicate_key(artifacts[i]) for i in path_entries if isinstance(artifacts[i], dict) and not _is_entry_malformed(artifacts[i]) }
        if len(distinct) > 1:
            return "reference_conflict_member"
    return "reference_selected"


def _manifest_scope_conflict(manifest: dict[str, Any], *, run_id: str, task_id: str, attempt_id: str) -> bool:
    for field, expected in (("run_id", run_id), ("task_id", task_id), ("attempt_id", attempt_id)):
        value = manifest.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or value != expected:
            return True
    return False


def _manifest_unknown_fields(manifest: dict[str, Any]) -> bool:
    return any(key not in _MANIFEST_KNOWN_FIELDS for key in manifest.keys())


def _apply_manifest_presence(result: ArtifactInspectionResult, presence: str) -> None:
    if presence == "absent":
        result.manifest_status = "manifest_absent"
    elif presence == "symlink":
        result.manifest_status = "manifest_symlink_blocked"
        result.file_type_status = "file_symlink"
    elif presence == "wrong_type":
        result.manifest_status = "manifest_wrong_type"
        result.file_type_status = "file_directory"
    elif presence == "hardlink":
        result.manifest_status = "manifest_hardlink_blocked"
        result.hardlink_status = "manifest_hardlink_blocked"
        result.file_type_status = "file_regular"
    elif presence == "size_budget":
        result.manifest_status = "manifest_byte_budget_exceeded"
        result.budget_status = "budget_manifest_exceeded"
        result.file_type_status = "file_regular"


def _apply_decode_failure(result: ArtifactInspectionResult, decode_status: str, *, budget: bool) -> None:
    if budget or decode_status == "budget_control_json_exceeded":
        result.manifest_status = "manifest_control_budget_exceeded"
        result.budget_status = "budget_control_json_exceeded"
        return
    mapping = {
        "manifest_utf8_malformed": "manifest_utf8_malformed",
        "manifest_duplicate_json_keys": "manifest_duplicate_json_keys",
        "manifest_top_level_schema_malformed": "manifest_top_level_schema_malformed",
    }
    result.manifest_status = mapping.get(decode_status, "manifest_json_malformed")


def _apply_l1_result(result: ArtifactInspectionResult, l1: HashArtifactResult, entry: dict[str, Any]) -> None:
    if l1.file_type_status == "file_symlink":
        result.path_status = "path_symlink_blocked"
        result.file_type_status = "file_symlink"
        result.filesystem_status = "filesystem_observed"
        return

    result.filesystem_status = l1.filesystem_status
    result.file_type_status = l1.file_type_status
    result.hardlink_status = l1.hardlink_status
    result.identity_status = l1.identity_status if l1.identity_status != "identity_not_applicable" else result.identity_status
    result.stability_status = l1.stability_status

    if l1.budget_exceeded:
        result.budget_status = "budget_artifact_exceeded"
        result.size_status = "size_not_inspected"
        result.digest_status = "digest_not_inspected"
        return

    if not l1.ok:
        if l1.digest_status == "digest_indeterminate":
            result.digest_status = "digest_indeterminate"
        return

    declared_size = entry.get("size_bytes")
    if isinstance(declared_size, int) and not isinstance(declared_size, bool):
        if l1.observed_size == declared_size:
            result.size_status = "size_matches_declared"
        else:
            result.size_status = "size_mismatch"
    else:
        result.size_status = "size_undeclared"

    declared_sha = entry.get("sha256")
    if isinstance(declared_sha, str) and declared_sha:
        if l1.computed_digest == declared_sha:
            result.digest_status = "digest_matches_declared"
        else:
            result.digest_status = "digest_mismatch"
    else:
        result.digest_status = "digest_undeclared"


def _apply_entry_path_and_l1(
    result: ArtifactInspectionResult,
    entry: dict[str, Any],
    *,
    attempt_fd: int,
    run_id: str,
    task_id: str,
    attempt_id: str,
    findings: list[str],
    hash_budget: list[int],
) -> None:
    declared_path = entry.get("path")
    path_status, components, path_findings = lexical_validate_artifact_path(declared_path)
    result.path_status = path_status
    result.declared_path = declared_path if isinstance(declared_path, str) else None
    findings.extend(path_findings)
    if components is not None and result.declared_path is not None:
        result.validated_components = components
        if path_status not in PATH_LEXICAL_FATAL:
            result.path_identity_digest = path_identity_digest(
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                declared_path=result.declared_path,
                validated_components=components,
            )
            result.identity_status = "identity_bound_lexical"
            if hash_budget[0] < MAX_TOTAL_BYTES_HASHED:
                l1 = hash_artifact_file(attempt_fd, components)
                if l1.ok and l1.observed_size is not None:
                    hash_budget[0] += l1.observed_size
                    if hash_budget[0] > MAX_TOTAL_BYTES_HASHED:
                        result.budget_status = "budget_aggregate_hash_exceeded"
                _apply_l1_result(result, l1, entry)


def _referenced_artifact_names(artifacts: list[Any]) -> set[str]:
    names: set[str] = set()
    limit = min(len(artifacts), MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
    for index in range(limit):
        entry = artifacts[index]
        if not isinstance(entry, dict) or _is_entry_malformed(entry):
            continue
        _, components, _ = lexical_validate_artifact_path(entry.get("path"))
        if components and components[0] == "artifacts" and len(components) == 2:
            names.add(components[1])
    return names


def _manifest_degraded_flags(
    artifacts: list[Any],
    *,
    scope_conflict: bool,
    unknown_field: bool,
    duplicate_keys: dict[tuple[tuple[str, ...], str, Any, Any], list[int]],
    conflict_keys: dict[tuple[str, ...], list[int]],
) -> tuple[bool, bool, bool]:
    limit = min(len(artifacts), MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
    malformed = any(
        not isinstance(artifacts[i], dict) or _is_entry_malformed(artifacts[i]) for i in range(limit)
    )
    well_formed = any(
        isinstance(artifacts[i], dict) and not _is_entry_malformed(artifacts[i]) for i in range(limit)
    )
    has_dup = any(len(v) > 1 for v in duplicate_keys.values())
    has_conflict = False
    for comp, indices in conflict_keys.items():
        if len(indices) <= 1:
            continue
        distinct = set()
        for idx in indices:
            entry = artifacts[idx]
            if isinstance(entry, dict) and not _is_entry_malformed(entry):
                distinct.add(_entry_duplicate_key(entry))
        if len(distinct) > 1:
            has_conflict = True
            break
    partially_malformed = malformed and well_formed
    return partially_malformed or scope_conflict or unknown_field, has_dup, has_conflict


def inspect_artifact_reference(
    selector: ArtifactReferenceSelector,
    **kwargs: Any,
) -> ArtifactInspectionResult:
    """Path A — inspect one manifest artifact reference (R5-01)."""
    result = _base_artifact_result(selector)
    findings: list[str] = []

    rejected = _reject_extra_kwargs(kwargs)
    if rejected is not None:
        result.authority_status = rejected
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    selector_err = _validate_artifact_selector(selector)
    if selector_err is not None:
        result.authority_status = selector_err
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    runs_root_ctx, fs_status = validate_runs_root_s0()
    if runs_root_ctx is None:
        result.filesystem_status = fs_status
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    walk, walk_err = walk_attempt_path(
        runs_root_ctx,
        selector.run_id,
        selector.task_id,
        selector.attempt_id,
    )
    if walk is None:
        result.filesystem_status = walk_err
        result.manifest_status = "manifest_absent"
        result.findings = sort_findings(findings)
        os_close_runs_root(runs_root_ctx)
        return result

    run_fd = walk.fds[1] if len(walk.fds) > 1 else walk.current_fd
    result.run_context_status = detect_run_context(run_fd, run_id=selector.run_id)

    try:
        presence, _, _ = classify_regular_file_presence(walk.current_fd, _MANIFEST_NAME)
        if presence != "regular":
            _apply_manifest_presence(result, presence)
            result.findings = sort_findings(findings)
            return result

        read_result = read_regular_control_file(
            walk.current_fd,
            _MANIFEST_NAME,
            decode_kind="manifest",
            context="artifact_manifest",
        )
        result.filesystem_status = read_result.filesystem_status
        result.file_type_status = read_result.file_type_status
        result.hardlink_status = read_result.hardlink_status

        if read_result.budget_exceeded:
            result.manifest_status = "manifest_control_budget_exceeded"
            result.budget_status = "budget_control_json_exceeded"
            result.findings = sort_findings(findings)
            return result

        if not read_result.ok or read_result.decode is None:
            result.findings = sort_findings(findings)
            return result

        if read_result.raw_digest != selector.manifest_raw_digest:
            result.authority_status = "selector_manifest_digest_mismatch"
            result.identity_status = "identity_selector_digest_mismatch"
            result.aggregate_completeness = "aggregate_indeterminate_selector_unbound"
            result.findings = sort_findings(findings)
            return result

        decode = read_result.decode
        if not decode.ok or decode.obj is None:
            _apply_decode_failure(result, decode.decode_status, budget=decode.budget_exceeded)
            result.findings = sort_findings(findings)
            return result

        manifest = decode.obj
        result.manifest_status = "manifest_bound"
        result.decoded_manifest = manifest
        result.manifest_semantic_digest = semantic_sha256_digest(read_result.raw_bytes)

        if _manifest_unknown_fields(manifest):
            findings.append("manifest_unknown_field_observed")

        scope_conflict = _manifest_scope_conflict(
            manifest,
            run_id=selector.run_id,
            task_id=selector.task_id,
            attempt_id=selector.attempt_id,
        )
        if scope_conflict:
            result.manifest_status = "manifest_scope_conflict"

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            result.reference_status = "reference_absent_from_manifest"
            result.findings = sort_findings(findings)
            return result

        extras = max(0, len(artifacts) - MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
        result.extras_unprocessed_count = extras
        if extras:
            findings.append("manifest_references_not_processed_budget")

        duplicate_keys, conflict_keys = _build_index_maps(artifacts)
        degraded, has_dup, has_conflict = _manifest_degraded_flags(
            artifacts,
            scope_conflict=scope_conflict,
            unknown_field="manifest_unknown_field_observed" in findings,
            duplicate_keys=duplicate_keys,
            conflict_keys=conflict_keys,
        )
        if degraded:
            result.manifest_status = "manifest_partially_malformed"
        if has_dup:
            result.manifest_status = "manifest_exact_duplicates_present"
        if has_conflict:
            result.manifest_status = "manifest_conflicts_present"
            findings.append("reference_same_path_distinct_kind")

        result.reference_status = _reference_status_for_index(
            artifacts,
            selector.entry_index,
            duplicate_keys=duplicate_keys,
            conflict_keys=conflict_keys,
        )
        if result.reference_status == "reference_not_processed_budget":
            result.budget_status = "budget_references_exceeded"

        if selector.entry_index < len(artifacts):
            entry = artifacts[selector.entry_index]
            result.entry = entry if isinstance(entry, dict) else None
            if isinstance(entry, dict) and result.reference_status in {
                "reference_selected",
                "reference_malformed",
                "reference_exact_duplicate_member",
                "reference_conflict_member",
            }:
                hash_budget = [0]
                _apply_entry_path_and_l1(
                    result,
                    entry,
                    attempt_fd=walk.current_fd,
                    run_id=selector.run_id,
                    task_id=selector.task_id,
                    attempt_id=selector.attempt_id,
                    findings=findings,
                    hash_budget=hash_budget,
                )

        result.aggregate_completeness = compute_aggregate_completeness(single_selector=True)
        result.findings = sort_findings(findings)
        return result
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)


def _inspect_attempt_manifest(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    attempt_fd: int,
    run_context_status: str,
    hash_budget: list[int],
    manifest_count: list[int],
) -> tuple[list[ArtifactInspectionResult], list[UnreferencedObservation], dict[str, bool]]:
    flags = {
        "partial_malformed": False,
        "partial_scope_missing": False,
        "partial_budget": False,
        "race": False,
        "dir_exceeded": False,
    }
    items: list[ArtifactInspectionResult] = []
    unreferenced: list[UnreferencedObservation] = []

    if manifest_count[0] >= MAX_MANIFESTS_PER_AGGREGATE:
        flags["partial_budget"] = True
        return items, unreferenced, flags

    presence, _, _ = classify_regular_file_presence(attempt_fd, _MANIFEST_NAME)
    if presence != "regular":
        if presence == "absent":
            flags["partial_scope_missing"] = True
        return items, unreferenced, flags

    read_result = read_regular_control_file(
        attempt_fd,
        _MANIFEST_NAME,
        decode_kind="manifest",
        context="aggregate/artifact_manifest",
    )
    if not read_result.ok or read_result.decode is None or not read_result.decode.ok or read_result.decode.obj is None:
        flags["partial_malformed"] = True
        if read_result.budget_exceeded:
            flags["partial_budget"] = True
        return items, unreferenced, flags

    manifest_count[0] += 1
    manifest = read_result.decode.obj
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        flags["partial_malformed"] = True
        return items, unreferenced, flags

    findings_common: list[str] = []
    if _manifest_unknown_fields(manifest):
        findings_common.append("manifest_unknown_field_observed")
    scope_conflict = _manifest_scope_conflict(
        manifest,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
    )
    duplicate_keys, conflict_keys = _build_index_maps(artifacts)
    degraded, has_dup, has_conflict = _manifest_degraded_flags(
        artifacts,
        scope_conflict=scope_conflict,
        unknown_field="manifest_unknown_field_observed" in findings_common,
        duplicate_keys=duplicate_keys,
        conflict_keys=conflict_keys,
    )
    if degraded:
        flags["partial_malformed"] = True
    if scope_conflict:
        flags["partial_scope_missing"] = True

    extras = max(0, len(artifacts) - MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
    limit = min(len(artifacts), MAX_ARTIFACT_REFERENCES_PER_MANIFEST)
    for entry_index in range(limit):
        if len(items) >= MAX_ARTIFACT_REFERENCES_PER_AGGREGATE:
            flags["partial_budget"] = True
            break
        entry = artifacts[entry_index]
        item = ArtifactInspectionResult(
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            entry_index=entry_index,
            manifest_raw_digest=read_result.raw_digest,
            manifest_semantic_digest=semantic_sha256_digest(read_result.raw_bytes),
            run_context_status=run_context_status,
        )
        item.manifest_status = "manifest_bound"
        item.decoded_manifest = manifest
        item.extras_unprocessed_count = extras
        item_findings = list(findings_common)
        if extras:
            item_findings.append("manifest_references_not_processed_budget")
        if has_conflict:
            item.manifest_status = "manifest_conflicts_present"
            item_findings.append("reference_same_path_distinct_kind")
        elif has_dup:
            item.manifest_status = "manifest_exact_duplicates_present"
        elif degraded:
            item.manifest_status = "manifest_partially_malformed"
        if scope_conflict:
            item.manifest_status = "manifest_scope_conflict"

        item.reference_status = _reference_status_for_index(
            artifacts,
            entry_index,
            duplicate_keys=duplicate_keys,
            conflict_keys=conflict_keys,
        )
        item.entry = entry if isinstance(entry, dict) else None
        if item.reference_status == "reference_not_processed_budget":
            item.budget_status = "budget_references_exceeded"
            flags["partial_budget"] = True
        elif isinstance(entry, dict) and item.reference_status in {
            "reference_selected",
            "reference_exact_duplicate_member",
            "reference_conflict_member",
        }:
            _apply_entry_path_and_l1(
                item,
                entry,
                attempt_fd=attempt_fd,
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                findings=item_findings,
                hash_budget=hash_budget,
            )
            if item.budget_status == "budget_aggregate_hash_exceeded":
                flags["partial_budget"] = True
            if item.stability_status == "stability_race_detected":
                flags["race"] = True
        item.findings = sort_findings(item_findings)
        items.append(item)

    referenced = _referenced_artifact_names(artifacts)
    scan_findings, dir_exceeded = scan_unreferenced_artifacts(attempt_fd, referenced)
    if dir_exceeded:
        flags["dir_exceeded"] = True
    for name, obs_findings in scan_findings:
        unreferenced.append(UnreferencedObservation(name=name, hashed=False, findings=sort_findings(obs_findings)))

    return items, unreferenced, flags


def _aggregate_from_attempts(
    *,
    run_id: str,
    task_id: str | None,
    attempt_id: str | None,
    walk,
    run_context_status: str,
) -> ArtifactAggregateResult:
    agg = ArtifactAggregateResult(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        run_context_status=run_context_status,
    )
    hash_budget = [0]
    manifest_count = [0]
    all_items: list[ArtifactInspectionResult] = []
    all_unreferenced: list[UnreferencedObservation] = []
    flags = {
        "partial_malformed": False,
        "partial_scope_missing": False,
        "partial_budget": False,
        "race": False,
        "dir_exceeded": False,
    }

    def _merge_attempt(tid: str, aid: str, att_fd: int) -> None:
        items, unref, f = _inspect_attempt_manifest(
            run_id=run_id,
            task_id=tid,
            attempt_id=aid,
            attempt_fd=att_fd,
            run_context_status=run_context_status,
            hash_budget=hash_budget,
            manifest_count=manifest_count,
        )
        all_items.extend(items)
        all_unreferenced.extend(unref)
        for key in flags:
            flags[key] = flags[key] or f.get(key, False)

    if task_id is not None and attempt_id is not None:
        _merge_attempt(task_id, attempt_id, walk.current_fd)
    else:
        tasks_fd, err = open_intermediate_dir(walk.current_fd, "tasks", context="aggregate/tasks")
        if err is not None or tasks_fd is None:
            agg.aggregate_completeness = "aggregate_partial_scope_missing"
            agg.budget_status = "budget_within_limits"
            return agg
        try:
            task_names = [task_id] if task_id is not None else sorted(list_dir_names_sorted(tasks_fd))
            for tid in task_names:
                task_fd, terr = open_intermediate_dir(tasks_fd, tid, context=f"aggregate/task/{tid}")
                if terr is not None or task_fd is None:
                    flags["partial_scope_missing"] = True
                    continue
                try:
                    attempts_fd, aerr = open_intermediate_dir(task_fd, "attempts", context=f"aggregate/{tid}/attempts")
                    if aerr is not None or attempts_fd is None:
                        flags["partial_scope_missing"] = True
                        continue
                    try:
                        attempt_names = sorted(list_dir_names_sorted(attempts_fd))
                        for aid in attempt_names:
                            att_fd, atterr = open_intermediate_dir(attempts_fd, aid, context=f"aggregate/{tid}/{aid}")
                            if atterr is not None or att_fd is None:
                                continue
                            try:
                                _merge_attempt(tid, aid, att_fd)
                            finally:
                                os.close(att_fd)
                    finally:
                        os.close(attempts_fd)
                finally:
                    os.close(task_fd)
        finally:
            os.close(tasks_fd)

    all_items.sort(
        key=lambda item: artifact_item_sort_key(
            item.run_id,
            item.task_id,
            item.attempt_id,
            item.entry_index or 0,
        )
    )
    agg.items = all_items
    agg.unreferenced = all_unreferenced
    agg.budget_status = compute_aggregate_budget_status(
        [item.budget_status for item in all_items],
        hash_exceeded=hash_budget[0] > MAX_TOTAL_BYTES_HASHED,
        dir_exceeded=flags["dir_exceeded"],
    )
    agg.aggregate_completeness = compute_aggregate_completeness(
        partial_budget_exhausted=flags["partial_budget"] or hash_budget[0] > MAX_TOTAL_BYTES_HASHED,
        partial_malformed=flags["partial_malformed"],
        partial_scope_missing=flags["partial_scope_missing"],
        partial_unreferenced_capped=flags["dir_exceeded"],
        indeterminate_race=flags["race"],
        applicable_unit_count=len(all_items),
        fully_complete_unit_count=sum(
            1
            for item in all_items
            if item.reference_status == "reference_selected"
            and item.manifest_status == "manifest_bound"
            and item.budget_status == "budget_within_limits"
        ),
    )
    return agg


def inspect_attempt_artifacts(
    run_id: str,
    task_id: str,
    attempt_id: str,
    **kwargs: Any,
) -> ArtifactAggregateResult:
    """Path B attempt scope — aggregate artifact inspection."""
    agg = ArtifactAggregateResult(run_id=run_id, task_id=task_id, attempt_id=attempt_id)
    rejected = _reject_extra_kwargs(kwargs)
    if rejected is not None:
        agg.authority_status = rejected
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    try:
        require_id(run_id, "run")
        require_id(task_id, "task")
        require_id(attempt_id, "attempt")
    except ValueError:
        agg.authority_status = "selector_identity_invalid"
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    runs_root_ctx, fs_status = validate_runs_root_s0()
    if runs_root_ctx is None:
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    walk, walk_err = walk_attempt_path(runs_root_ctx, run_id, task_id, attempt_id)
    if walk is None:
        agg.aggregate_completeness = "aggregate_partial_scope_missing"
        os_close_runs_root(runs_root_ctx)
        return agg

    run_fd = walk.fds[1] if len(walk.fds) > 1 else walk.current_fd
    run_context = detect_run_context(run_fd, run_id=run_id)
    try:
        return _aggregate_from_attempts(
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            walk=walk,
            run_context_status=run_context,
        )
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)


def inspect_task_artifacts(run_id: str, task_id: str, **kwargs: Any) -> ArtifactAggregateResult:
    agg = ArtifactAggregateResult(run_id=run_id, task_id=task_id)
    rejected = _reject_extra_kwargs(kwargs)
    if rejected is not None:
        agg.authority_status = rejected
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    try:
        require_id(run_id, "run")
        require_id(task_id, "task")
    except ValueError:
        agg.authority_status = "selector_identity_invalid"
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    runs_root_ctx, _ = validate_runs_root_s0()
    if runs_root_ctx is None:
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    walk, _ = walk_run_path(runs_root_ctx, run_id)
    if walk is None:
        agg.aggregate_completeness = "aggregate_partial_scope_missing"
        os_close_runs_root(runs_root_ctx)
        return agg

    run_fd = walk.current_fd
    run_context = detect_run_context(run_fd, run_id=run_id)
    try:
        return _aggregate_from_attempts(
            run_id=run_id,
            task_id=task_id,
            attempt_id=None,
            walk=walk,
            run_context_status=run_context,
        )
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)


def inspect_run_artifacts(run_id: str, **kwargs: Any) -> ArtifactAggregateResult:
    agg = ArtifactAggregateResult(run_id=run_id)
    rejected = _reject_extra_kwargs(kwargs)
    if rejected is not None:
        agg.authority_status = rejected
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    try:
        require_id(run_id, "run")
    except ValueError:
        agg.authority_status = "selector_identity_invalid"
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    runs_root_ctx, _ = validate_runs_root_s0()
    if runs_root_ctx is None:
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        return agg

    walk, _ = walk_run_path(runs_root_ctx, run_id)
    if walk is None:
        agg.aggregate_completeness = "aggregate_partial_scope_missing"
        os_close_runs_root(runs_root_ctx)
        return agg

    run_context = detect_run_context(walk.current_fd, run_id=run_id)
    try:
        return _aggregate_from_attempts(
            run_id=run_id,
            task_id=None,
            attempt_id=None,
            walk=walk,
            run_context_status=run_context,
        )
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)


def is_supplemental_finding(token: str) -> bool:
    return token in SUPPLEMENTAL_FINDING_TOKENS
