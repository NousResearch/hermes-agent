"""Task 29 Phase I — advisory link reference inspection (R5)."""

from __future__ import annotations

import re
from typing import Any

from htr.advisory_inspection_aggregate import compute_aggregate_budget_status, compute_aggregate_completeness
from htr.advisory_inspection_constants import (
    DERIVED_ALIGNMENT_ROLES,
    LINK_SOURCE_RECORD_FILENAMES,
    MAX_LINKS_PER_AGGREGATE,
    MAX_LINKS_PER_RECORD,
)
from htr.advisory_inspection_models import (
    DerivedAlignment,
    LinkAggregateResult,
    LinkInspectionResult,
    LinkReferenceSelector,
    RecordLoadStatus,
    sort_findings,
)
from htr.advisory_inspection_run_context import detect_run_context
from htr.advisory_inspection_secure import (
    classify_regular_file_presence,
    os_close_runs_root,
    read_regular_control_file,
    semantic_sha256_digest,
    validate_runs_root_s0,
    walk_run_path,
)
from htr.advisory_inspection_url import classify_url_full
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

_RECORD_KIND_TO_FILENAME = {
    "run_execution_request_record": "run_execution_request_record.json",
    "run_post_verification_execution_request_record": "run_post_verification_execution_request_record.json",
}

_PRIMARY_RECORD_KINDS = frozenset(_RECORD_KIND_TO_FILENAME.keys())

_URL_BEARING_KINDS = frozenset({"manual_open_link", "reopen_link_manually"})

_DERIVED_BY_ROLE: dict[str, dict[str, str]] = {
    "1a": {
        "filename": "run_execution_result_record.json",
        "items_key": "item_results",
        "primary_field": "item_id",
        "derived_field": "item_id",
    },
    "1b": {
        "filename": "run_execution_verification_record.json",
        "items_key": "item_verifications",
        "primary_field": "item_id",
        "derived_field": "source_request_item_id",
    },
    "2a": {
        "filename": "run_post_verification_execution_result_record.json",
        "items_key": "item_results",
        "primary_field": "request_item_id",
        "derived_field": "item_id",
    },
    "2b": {
        "filename": "run_post_verification_execution_verification_record.json",
        "items_key": "item_verifications",
        "primary_field": "request_item_id",
        "derived_field": "source_request_item_id",
    },
}

_ROLE_APPLICABLE = {
    "run_execution_request_record": {"1a", "1b"},
    "run_post_verification_execution_request_record": {"2a", "2b"},
}


def _reject_extra_kwargs(kwargs: dict[str, Any]) -> str | None:
    if any(key in kwargs for key in _FORBIDDEN_KWARGS):
        return "caller_host_root_rejected"
    if kwargs:
        return "caller_host_root_rejected"
    return None


def _validate_link_selector(selector: LinkReferenceSelector) -> str | None:
    if not isinstance(selector.run_id, str):
        return "selector_identity_invalid"
    if not isinstance(selector.record_kind, str):
        return "selector_identity_invalid"
    if not isinstance(selector.record_raw_digest, str):
        return "selector_identity_invalid"
    if not isinstance(selector.item_index, int) or isinstance(selector.item_index, bool):
        return "selector_identity_invalid"

    try:
        require_id(selector.run_id, "run")
    except ValueError:
        return "selector_identity_invalid"

    if not _DIGEST_RE.match(selector.record_raw_digest):
        return "selector_identity_invalid"

    if selector.item_index < 0:
        return "selector_item_index_out_of_range"

    if selector.record_kind not in _PRIMARY_RECORD_KINDS:
        return "selector_record_kind_invalid"

    return None


def _base_link_result(selector: LinkReferenceSelector | None = None) -> LinkInspectionResult:
    result = LinkInspectionResult()
    if selector is not None:
        result.run_id = selector.run_id
        result.record_kind = selector.record_kind
        result.item_index = selector.item_index
        result.record_raw_digest = selector.record_raw_digest
    return result


def _record_items_key(record_kind: str) -> str:
    return "execution_items"


def _primary_id_field(record_kind: str) -> str:
    if record_kind == "run_post_verification_execution_request_record":
        return "request_item_id"
    return "item_id"


def _is_url_bearing_item(item: dict[str, Any]) -> bool:
    kind = item.get("execution_kind")
    return isinstance(kind, str) and kind in _URL_BEARING_KINDS


def _classify_link_item(item: Any, *, duplicate: bool) -> str:
    if not isinstance(item, dict):
        return "link_item_malformed"
    if not _is_url_bearing_item(item):
        return "link_kind_not_url_bearing"
    command = item.get("command")
    if not isinstance(command, dict):
        return "link_command_malformed"
    if "url" not in command:
        return "link_url_absent"
    url = command.get("url")
    if not isinstance(url, str):
        return "link_url_not_string"
    if duplicate:
        return "link_item_id_duplicate"
    return "link_item_selected"


def _apply_record_presence(result: LinkInspectionResult, presence: str) -> None:
    if presence == "absent":
        result.link_record_status = "link_record_absent"
    elif presence == "symlink":
        result.link_record_status = "link_record_symlink_blocked"
    elif presence == "wrong_type":
        result.link_record_status = "link_record_wrong_type"
    elif presence == "hardlink":
        result.link_record_status = "link_record_hardlink_blocked"
    elif presence == "size_budget":
        result.link_record_status = "link_record_control_budget_exceeded"
        result.budget_status = "budget_control_json_exceeded"


def _apply_record_decode_failure(result: LinkInspectionResult, decode_status: str, *, budget: bool) -> None:
    if budget or decode_status == "budget_control_json_exceeded":
        result.link_record_status = "link_record_control_budget_exceeded"
        result.budget_status = "budget_control_json_exceeded"
        return
    if decode_status == "link_record_top_schema_malformed":
        result.link_record_status = "link_record_top_schema_malformed"
    else:
        result.link_record_status = "link_record_json_malformed"


def _items_from_record(record: dict[str, Any], record_kind: str) -> list[Any] | None:
    items = record.get(_items_key(record_kind))
    if not isinstance(items, list):
        return None
    return items


def _items_key(record_kind: str) -> str:
    return "execution_items"


def _primary_item_malformed(item: Any, record_kind: str) -> bool:
    if not isinstance(item, dict):
        return True
    id_field = _primary_id_field(record_kind)
    item_id = item.get(id_field)
    return not isinstance(item_id, str) or not item_id


def _item_url(item: dict[str, Any]) -> str | None:
    command = item.get("command")
    if isinstance(command, dict):
        url = command.get("url")
        if isinstance(url, str):
            return url
    output = item.get("output")
    if isinstance(output, dict):
        nested = output.get("command")
        if isinstance(nested, dict):
            url = nested.get("url")
            if isinstance(url, str):
                return url
    return None


def _build_derived_alignments(
    *,
    record_kind: str,
    primary_item: dict[str, Any] | None,
    primary_malformed: bool,
    run_fd: int,
    record_bound: bool,
) -> list[DerivedAlignment]:
    alignments: list[DerivedAlignment] = []
    applicable_roles = _ROLE_APPLICABLE.get(record_kind, set())

    for role in DERIVED_ALIGNMENT_ROLES:
        slot = DerivedAlignment(
            role=role,
            applicable=role in applicable_roles,
            match_status="link_derived_not_applicable",
            derived_index=None,
            candidate_derived_indexes=[],
            findings=[],
        )

        if not record_bound:
            alignments.append(slot)
            continue

        if role not in applicable_roles:
            slot.applicable = False
            slot.match_status = "link_derived_not_applicable"
            alignments.append(slot)
            continue

        slot.applicable = True

        if primary_malformed or primary_item is None:
            slot.match_status = "link_match_primary_item_malformed"
            alignments.append(slot)
            continue

        derived_spec = _DERIVED_BY_ROLE[role]
        derived_filename = derived_spec["filename"]
        primary_field = derived_spec["primary_field"]
        derived_field = derived_spec["derived_field"]
        items_key = derived_spec["items_key"]

        primary_id = primary_item.get(primary_field)

        presence, _, _ = classify_regular_file_presence(run_fd, derived_filename)
        if presence == "absent":
            slot.match_status = "link_derived_absent"
            alignments.append(slot)
            continue
        if presence != "regular":
            slot.match_status = "link_derived_unreadable"
            alignments.append(slot)
            continue

        read_result = read_regular_control_file(
            run_fd,
            derived_filename,
            decode_kind="link",
            context=f"derived/{derived_filename}",
        )
        if not read_result.ok or read_result.decode is None or not read_result.decode.ok:
            slot.match_status = "link_derived_unreadable"
            alignments.append(slot)
            continue

        derived_record = read_result.decode.obj
        assert derived_record is not None
        derived_items = derived_record.get(items_key)
        if not isinstance(derived_items, list):
            slot.match_status = "link_derived_unreadable"
            alignments.append(slot)
            continue

        matches = [
            idx
            for idx, derived_item in enumerate(derived_items)
            if isinstance(derived_item, dict) and derived_item.get(derived_field) == primary_id
        ]

        if len(matches) == 0:
            slot.match_status = "link_derived_unmatched"
        elif len(matches) > 1:
            slot.match_status = "link_derived_ambiguous"
            slot.candidate_derived_indexes = sorted(matches)
        else:
            idx = matches[0]
            slot.candidate_derived_indexes = [idx]
            derived_item = derived_items[idx]
            if not isinstance(derived_item, dict) or _primary_item_malformed(derived_item, record_kind):
                slot.match_status = "link_match_derived_item_malformed"
            else:
                primary_url = _item_url(primary_item)
                derived_url = _item_url(derived_item) if isinstance(derived_item, dict) else None

                if primary_url is None and derived_url is None:
                    slot.match_status = "link_match_url_both_missing"
                elif primary_url is None:
                    slot.match_status = "link_match_url_primary_missing"
                elif derived_url is None:
                    slot.match_status = "link_match_url_derived_missing"
                elif primary_url == derived_url:
                    slot.match_status = "link_match_url_equal"
                else:
                    slot.match_status = "link_match_url_conflict"
                    slot.findings.append("link_primary_derived_conflict")
                slot.derived_index = idx

        slot.findings = sort_findings(slot.findings)
        alignments.append(slot)

    return alignments


def inspect_link_reference(selector: LinkReferenceSelector, **kwargs: Any) -> LinkInspectionResult:
    """Path C — inspect one link item from a primary record (R5-08/R5-09)."""
    result = _base_link_result(selector)
    findings: list[str] = []

    rejected = _reject_extra_kwargs(kwargs)
    if rejected is not None:
        result.authority_status = rejected
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    selector_err = _validate_link_selector(selector)
    if selector_err is not None:
        result.authority_status = selector_err
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    runs_root_ctx, fs_status = validate_runs_root_s0()
    if runs_root_ctx is None:
        result.link_record_status = "link_record_not_attempted"
        result.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        result.findings = sort_findings(findings)
        return result

    walk, walk_err = walk_run_path(runs_root_ctx, selector.run_id)
    if walk is None:
        result.link_record_status = "link_record_absent"
        result.findings = sort_findings(findings)
        os_close_runs_root(runs_root_ctx)
        return result

    filename = _RECORD_KIND_TO_FILENAME[selector.record_kind]
    result.run_context_status = detect_run_context(walk.current_fd, run_id=selector.run_id)

    try:
        presence, _, _ = classify_regular_file_presence(walk.current_fd, filename)
        if presence != "regular":
            _apply_record_presence(result, presence)
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        read_result = read_regular_control_file(
            walk.current_fd,
            filename,
            decode_kind="link",
            context=f"link/{filename}",
        )

        if read_result.budget_exceeded:
            result.link_record_status = "link_record_control_budget_exceeded"
            result.budget_status = "budget_control_json_exceeded"
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        if not read_result.ok or read_result.decode is None:
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        if read_result.raw_digest != selector.record_raw_digest:
            result.authority_status = "selector_record_digest_mismatch"
            result.aggregate_completeness = "aggregate_indeterminate_selector_unbound"
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        decode = read_result.decode
        if not decode.ok or decode.obj is None:
            _apply_record_decode_failure(result, decode.decode_status, budget=decode.budget_exceeded)
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        record = decode.obj
        result.link_record_status = "link_record_bound"
        result.record_semantic_digest = semantic_sha256_digest(read_result.raw_bytes)

        items = _items_from_record(record, selector.record_kind)
        if items is None:
            result.link_record_status = "link_record_top_schema_malformed"
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        if selector.item_index >= len(items):
            result.authority_status = "selector_item_index_out_of_range"
            result.link_item_status = "link_item_not_applicable"
            result.derived_alignments = []
            result.findings = sort_findings(findings)
            return result

        item = items[selector.item_index]
        result.item = item if isinstance(item, dict) else None

        id_field = _primary_id_field(selector.record_kind)
        duplicate = False
        if isinstance(item, dict) and isinstance(item.get(id_field), str):
            item_id = item[id_field]
            duplicate = sum(
                1
                for other in items
                if isinstance(other, dict)
                and isinstance(other.get(id_field), str)
                and other.get(id_field) == item_id
                and _is_url_bearing_item(other)
            ) > 1

        result.link_item_status = _classify_link_item(item, duplicate=duplicate)

        primary_malformed = _primary_item_malformed(item, selector.record_kind)
        primary_item = item if isinstance(item, dict) else None

        if result.link_record_status == "link_record_bound" and not primary_malformed:
            result.derived_alignments = _build_derived_alignments(
                record_kind=selector.record_kind,
                primary_item=primary_item,
                primary_malformed=primary_malformed,
                run_fd=walk.current_fd,
                record_bound=True,
            )
        elif result.link_record_status == "link_record_bound":
            result.derived_alignments = _build_derived_alignments(
                record_kind=selector.record_kind,
                primary_item=primary_item,
                primary_malformed=True,
                run_fd=walk.current_fd,
                record_bound=True,
            )
        else:
            result.derived_alignments = []

        if result.link_item_status == "link_item_selected" and isinstance(item, dict):
            command = item.get("command")
            if isinstance(command, dict) and isinstance(command.get("url"), str):
                classified = classify_url_full(command["url"])
                findings.extend(classified.findings)
                if classified.budget_exceeded:
                    result.budget_status = "budget_url_exceeded"
                else:
                    result.link_scheme_status = classified.scheme_status
                    result.link_host_status = classified.host_status
                    result.link_port_status = classified.port_status
                    result.link_structure_status = classified.structure_status
                result.link_fetch_status = "link_remote_not_fetched"
            else:
                result.link_fetch_status = "link_fetch_not_applicable"

        result.aggregate_completeness = compute_aggregate_completeness(single_selector=True)
        result.findings = sort_findings(findings)
        return result
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)


def _record_load_status(run_fd: int, filename: str) -> str:
    presence, _, _ = classify_regular_file_presence(run_fd, filename)
    if presence == "absent":
        return "record_load_absent"
    if presence == "symlink":
        return "record_load_symlink_blocked"
    if presence == "wrong_type":
        return "record_load_wrong_type"
    if presence == "hardlink":
        return "record_load_hardlink_blocked"
    if presence == "size_budget":
        return "record_load_control_budget_exceeded"

    read_result = read_regular_control_file(
        run_fd,
        filename,
        decode_kind="link",
        context=f"aggregate/{filename}",
    )
    if read_result.budget_exceeded:
        return "record_load_control_budget_exceeded"
    if not read_result.ok or read_result.decode is None:
        return "record_load_unreadable"
    if not read_result.decode.ok:
        if read_result.decode.decode_status == "link_record_top_schema_malformed":
            return "record_load_top_schema_malformed"
        return "record_load_json_malformed"
    return "record_load_bound"


def _inspect_discovered_link_item(
    *,
    run_id: str,
    record_kind: str,
    item_index: int,
    record_raw_digest: str,
    record_semantic_digest: str,
    item: Any,
    items: list[Any],
    run_fd: int,
    run_context_status: str,
) -> LinkInspectionResult:
    result = LinkInspectionResult(
        run_id=run_id,
        record_kind=record_kind,
        item_index=item_index,
        record_raw_digest=record_raw_digest,
        record_semantic_digest=record_semantic_digest,
        link_record_status="link_record_bound",
        run_context_status=run_context_status,
    )
    findings: list[str] = []
    result.item = item if isinstance(item, dict) else None

    id_field = _primary_id_field(record_kind)
    duplicate = False
    if isinstance(item, dict) and isinstance(item.get(id_field), str):
        item_id = item[id_field]
        duplicate = sum(
            1
            for other in items
            if isinstance(other, dict)
            and isinstance(other.get(id_field), str)
            and other.get(id_field) == item_id
            and _is_url_bearing_item(other)
        ) > 1

    result.link_item_status = _classify_link_item(item, duplicate=duplicate)
    primary_malformed = _primary_item_malformed(item, record_kind)
    primary_item = item if isinstance(item, dict) else None
    result.derived_alignments = _build_derived_alignments(
        record_kind=record_kind,
        primary_item=primary_item,
        primary_malformed=primary_malformed,
        run_fd=run_fd,
        record_bound=True,
    )

    if result.link_item_status == "link_item_selected" and isinstance(item, dict):
        command = item.get("command")
        if isinstance(command, dict) and isinstance(command.get("url"), str):
            classified = classify_url_full(command["url"])
            findings.extend(classified.findings)
            if classified.budget_exceeded:
                result.budget_status = "budget_url_exceeded"
            else:
                result.link_scheme_status = classified.scheme_status
                result.link_host_status = classified.host_status
                result.link_port_status = classified.port_status
                result.link_structure_status = classified.structure_status
            result.link_fetch_status = "link_remote_not_fetched"
        else:
            result.link_fetch_status = "link_fetch_not_applicable"

    result.findings = sort_findings(findings)
    return result


def inspect_run_links(run_id: str, **kwargs: Any) -> LinkAggregateResult:
    """Path B run scope — link aggregate inspection."""
    agg = LinkAggregateResult(run_id=run_id)
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

    runs_root_ctx, fs_status = validate_runs_root_s0()
    if runs_root_ctx is None:
        agg.aggregate_completeness = "aggregate_blocked_untrusted_scope"
        for filename in sorted(LINK_SOURCE_RECORD_FILENAMES):
            agg.records_loaded.append(RecordLoadStatus(filename=filename, status="record_load_not_attempted"))
        return agg

    walk, walk_err = walk_run_path(runs_root_ctx, run_id)
    if walk is None:
        agg.aggregate_completeness = "aggregate_partial_scope_missing"
        for filename in sorted(LINK_SOURCE_RECORD_FILENAMES):
            agg.records_loaded.append(RecordLoadStatus(filename=filename, status="record_load_not_attempted"))
        os_close_runs_root(runs_root_ctx)
        return agg

    run_context = detect_run_context(walk.current_fd, run_id=run_id)
    agg.run_context_status = run_context

    try:
        for filename in sorted(LINK_SOURCE_RECORD_FILENAMES):
            agg.records_loaded.append(
                RecordLoadStatus(filename=filename, status=_record_load_status(walk.current_fd, filename))
            )

        items: list[LinkInspectionResult] = []
        partial_budget = False
        for record_kind in sorted(_PRIMARY_RECORD_KINDS):
            filename = _RECORD_KIND_TO_FILENAME[record_kind]
            load = next(r for r in agg.records_loaded if r.filename == filename)
            if load.status != "record_load_bound":
                continue

            read_result = read_regular_control_file(
                walk.current_fd,
                filename,
                decode_kind="link",
                context=f"aggregate/enumerate/{filename}",
            )
            if not read_result.ok or read_result.decode is None or not read_result.decode.ok:
                continue
            record = read_result.decode.obj
            assert record is not None
            execution_items = _items_from_record(record, record_kind)
            if execution_items is None:
                continue

            semantic = semantic_sha256_digest(read_result.raw_bytes)
            for item_index, item in enumerate(execution_items[:MAX_LINKS_PER_RECORD]):
                if len(items) >= MAX_LINKS_PER_AGGREGATE:
                    partial_budget = True
                    break
                if not _is_url_bearing_item(item if isinstance(item, dict) else {}):
                    continue
                items.append(
                    _inspect_discovered_link_item(
                        run_id=run_id,
                        record_kind=record_kind,
                        item_index=item_index,
                        record_raw_digest=read_result.raw_digest,
                        record_semantic_digest=semantic,
                        item=item,
                        items=execution_items,
                        run_fd=walk.current_fd,
                        run_context_status=run_context,
                    )
                )
            if len(items) >= MAX_LINKS_PER_AGGREGATE:
                partial_budget = True
                break

        agg.items = items
        agg.budget_status = compute_aggregate_budget_status(
            [item.budget_status for item in items],
            hash_exceeded=False,
            dir_exceeded=False,
        )
        agg.aggregate_completeness = compute_aggregate_completeness(
            partial_budget_exhausted=partial_budget,
            applicable_unit_count=len(items),
            fully_complete_unit_count=sum(
                1
                for item in items
                if item.link_item_status == "link_item_selected"
                and item.budget_status == "budget_within_limits"
            ),
        )
        return agg
    finally:
        walk.close_all()
        os_close_runs_root(runs_root_ctx)
