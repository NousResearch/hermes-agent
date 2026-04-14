#!/usr/bin/env python3
"""Shared continuity-now contract/evidence path resolution helpers.

Keep this module intentionally small and dependency-free so shell-embedded
Python surfaces can import it safely.
"""

from __future__ import annotations

import pathlib
from typing import Any, Mapping

DEFAULT_CONTINUITY_NOW_LATEST_REL = "state/continuity/latest/continuity_now_latest.json"
CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON = "continuity_now_contract_path_conflict"


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _resolve_candidate_path(root: pathlib.Path, raw_path: Any) -> pathlib.Path | None:
    path_txt = str(raw_path or "").strip()
    if not path_txt:
        return None
    path = pathlib.Path(path_txt)
    if path.is_absolute():
        return path.resolve()
    return (pathlib.Path(root).resolve() / path).resolve()


def to_rel_or_abs(root: pathlib.Path, path: pathlib.Path) -> str:
    """Return root-relative path when possible, otherwise absolute."""

    resolved_root = pathlib.Path(root).resolve()
    resolved_path = pathlib.Path(path).resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except Exception:
        return str(resolved_path)


def continuity_now_contract_path_conflict_reason(
    root: pathlib.Path,
    *,
    contract_obj: Any = None,
    source_refs: Any = None,
) -> str | None:
    """Return explicit contract-violation reason when path pins diverge."""

    contract_map = _as_mapping(contract_obj)
    source_map = _as_mapping(source_refs)

    contract_path = _resolve_candidate_path(root, contract_map.get("path"))
    source_path = _resolve_candidate_path(root, source_map.get("continuity_now"))

    if contract_path is None or source_path is None:
        return None
    if contract_path == source_path:
        return None
    return CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON


def resolve_continuity_now_contract_path(
    root: pathlib.Path,
    *,
    current_obj: Any = None,
    contract_obj: Any = None,
    source_refs: Any = None,
    fallback_rel: str = DEFAULT_CONTINUITY_NOW_LATEST_REL,
) -> pathlib.Path:
    """Resolve continuity_now contract path from contract/source_refs/current."""

    if current_obj is not None:
        current_map = _as_mapping(current_obj)
        if contract_obj is None:
            contract_obj = current_map.get("continuity_now_contract")
        if source_refs is None:
            source_refs = current_map.get("source_refs")

    contract_map = _as_mapping(contract_obj)
    source_map = _as_mapping(source_refs)

    path_txt = str(
        contract_map.get("path")
        or source_map.get("continuity_now")
        or fallback_rel
    ).strip() or fallback_rel

    path = pathlib.Path(path_txt)
    if path.is_absolute():
        return path.resolve()
    return (pathlib.Path(root).resolve() / path).resolve()


def resolve_continuity_now_evidence_path(
    root: pathlib.Path,
    *,
    current_obj: Any = None,
    contract_obj: Any = None,
    source_refs: Any = None,
    raw_path: Any = None,
    fallback_rel: str = DEFAULT_CONTINUITY_NOW_LATEST_REL,
) -> str:
    """Resolve continuity_now evidence path as root-relative where possible."""

    raw_txt = str(raw_path or "").strip()
    if raw_txt:
        raw_path_obj = pathlib.Path(raw_txt)
        if raw_path_obj.is_absolute():
            return to_rel_or_abs(pathlib.Path(root), raw_path_obj)
        return str(raw_path_obj)

    contract_path = resolve_continuity_now_contract_path(
        pathlib.Path(root),
        current_obj=current_obj,
        contract_obj=contract_obj,
        source_refs=source_refs,
        fallback_rel=fallback_rel,
    )
    return to_rel_or_abs(pathlib.Path(root), contract_path)
