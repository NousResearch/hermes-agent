"""Detect xAI models retired on May 15, 2026.

Source: https://docs.x.ai/developers/migration/may-15-retirement

Pure logic: walks a Hermes config dict, returns issues for any reference
to a retired xAI model. No I/O, no CLI dependencies — testable in isolation
and reusable from both `hermes doctor` and a future `hermes migrate xai`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


MIGRATION_GUIDE_URL = "https://docs.x.ai/developers/migration/may-15-retirement"
RETIREMENT_DATE = "May 15, 2026"


# Official mapping per xAI migration guide.
# Some entries set ``reasoning_effort`` because non-reasoning variants don't
# have a one-to-one replacement: ``grok-4.3`` reasons by default, so emulating
# ``*-non-reasoning`` behavior on it requires ``reasoning_effort="none"``.
_RETIRED_MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "grok-4-0709":                  {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-fast-reasoning":        {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-fast-non-reasoning":    {"replacement": "grok-4.3", "reasoning_effort": "none", "note": None},
    "grok-4-1-fast-reasoning":      {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-1-fast-non-reasoning":  {"replacement": "grok-4.3", "reasoning_effort": "none", "note": None},
    "grok-code-fast-1":             {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-3":                       {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-imagine-image-pro":       {"replacement": "grok-imagine-image-quality", "reasoning_effort": None, "note": None},
}


@dataclass(frozen=True)
class RetirementIssue:
    """A reference to a retired xAI model found in a Hermes config."""

    config_path: str            # e.g. "principal.model" or "auxiliary.vision.model"
    current_model: str          # exact value found in config (preserves casing/prefix)
    replacement: str            # recommended xAI replacement
    reasoning_effort: Optional[str] = None  # set if non-reasoning variant migration
    note: Optional[str] = None  # disambiguation note when applicable


def _normalize(model_id: str) -> str:
    """Strip provider prefix (``x-ai/grok-4`` → ``grok-4``) and lowercase."""
    m = model_id.strip().lower()
    for prefix in ("x-ai/", "xai/"):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    return m


def _looks_like_xai(model_id: Optional[str]) -> bool:
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    return _normalize(model_id).startswith("grok-")


def find_retired_xai_refs(config: Dict[str, Any]) -> List[RetirementIssue]:
    """Walk all model slots in a Hermes config and return retirement issues.

    Slots scanned:
      - ``principal.model``
      - ``auxiliary.<any>.model`` (introspective — covers future aux slots)
      - ``delegation.model``
      - ``tts.xai.model``
      - ``plugins.image_gen.xai.model``
    """
    issues: List[RetirementIssue] = []

    def _check(path: str, model: Any) -> None:
        if not _looks_like_xai(model):
            return
        norm = _normalize(model)
        entry = _RETIRED_MODELS.get(norm)
        if entry is None:
            return
        issues.append(RetirementIssue(
            config_path=path,
            current_model=model,
            replacement=entry["replacement"],
            reasoning_effort=entry.get("reasoning_effort"),
            note=entry.get("note"),
        ))

    if not isinstance(config, dict):
        return issues

    principal = config.get("principal")
    if isinstance(principal, dict):
        _check("principal.model", principal.get("model"))

    aux = config.get("auxiliary")
    if isinstance(aux, dict):
        for slot_name, slot_cfg in aux.items():
            if isinstance(slot_cfg, dict):
                _check(f"auxiliary.{slot_name}.model", slot_cfg.get("model"))

    delegation = config.get("delegation")
    if isinstance(delegation, dict):
        _check("delegation.model", delegation.get("model"))

    tts = config.get("tts")
    if isinstance(tts, dict):
        tts_xai = tts.get("xai")
        if isinstance(tts_xai, dict):
            _check("tts.xai.model", tts_xai.get("model"))

    plugins = config.get("plugins")
    if isinstance(plugins, dict):
        image_gen = plugins.get("image_gen")
        if isinstance(image_gen, dict):
            ig_xai = image_gen.get("xai")
            if isinstance(ig_xai, dict):
                _check("plugins.image_gen.xai.model", ig_xai.get("model"))

    return issues


def format_issue(issue: RetirementIssue) -> str:
    """One-line human-readable rendering of a retirement issue."""
    parts = [
        f"{issue.config_path}: {issue.current_model!r} → use {issue.replacement!r}"
    ]
    if issue.reasoning_effort:
        parts.append(f'(set reasoning_effort: "{issue.reasoning_effort}")')
    if issue.note:
        parts.append(f"[note: {issue.note}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Apply migration to config.yaml (round-trip preserves comments/order/types)
# ---------------------------------------------------------------------------

import datetime as _dt
from pathlib import Path
import shutil


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of an apply_migration call."""

    file_path: Path
    backup_path: Optional[Path]
    issues_resolved: List[RetirementIssue]
    config_changed: bool


def _walk_to_parent(yaml_doc: Any, dotted_path: str) -> "tuple[Any, str]":
    """Resolve a dotted slot path to (parent_mapping, leaf_key).

    Example: "auxiliary.vision.model" -> (yaml_doc["auxiliary"]["vision"], "model").
    Raises KeyError if any intermediate node is missing or not a mapping.
    """
    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"Path must have at least one parent: {dotted_path!r}")
    node = yaml_doc
    for segment in parts[:-1]:
        if not isinstance(node, dict) or segment not in node:
            raise KeyError(f"Path segment {segment!r} missing in {dotted_path!r}")
        node = node[segment]
    return node, parts[-1]


def apply_migration(
    config_path: Path,
    issues: List[RetirementIssue],
    backup: bool = True,
) -> ApplyResult:
    """Rewrite ``config_path`` in-place so each issue is resolved.

    For every issue, the model name is replaced by ``issue.replacement``. If the
    issue has ``reasoning_effort`` set (i.e. the migration is from a
    ``*-non-reasoning`` variant), a sibling ``reasoning_effort`` key is added
    or updated alongside the model.

    Uses ``ruamel.yaml`` round-trip mode so comments, key order, indentation,
    and type literals (booleans, ints) are preserved.

    A backup copy is written to
    ``<config_path>.bak-pre-migrate-xai-YYYYMMDD-HHMMSS`` before rewriting,
    unless ``backup=False``.
    """
    from ruamel.yaml import YAML  # local import — avoid hard dep at module load

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not issues:
        return ApplyResult(
            file_path=config_path,
            backup_path=None,
            issues_resolved=[],
            config_changed=False,
        )

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    with config_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh)

    if doc is None:
        return ApplyResult(
            file_path=config_path,
            backup_path=None,
            issues_resolved=[],
            config_changed=False,
        )

    resolved: List[RetirementIssue] = []
    for issue in issues:
        try:
            parent, leaf = _walk_to_parent(doc, issue.config_path)
        except KeyError:
            # Slot vanished between scan and apply — skip silently
            continue
        parent[leaf] = issue.replacement
        if issue.reasoning_effort:
            parent["reasoning_effort"] = issue.reasoning_effort
        resolved.append(issue)

    if not resolved:
        return ApplyResult(
            file_path=config_path,
            backup_path=None,
            issues_resolved=[],
            config_changed=False,
        )

    backup_path: Optional[Path] = None
    if backup:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = config_path.with_name(
            f"{config_path.name}.bak-pre-migrate-xai-{ts}"
        )
        shutil.copy2(config_path, backup_path)

    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(doc, fh)

    return ApplyResult(
        file_path=config_path,
        backup_path=backup_path,
        issues_resolved=resolved,
        config_changed=True,
    )
