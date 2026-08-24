"""Detect xAI models retired on May 15, 2026 and migrate config.yaml references."""
from __future__ import annotations

import datetime as _dt
import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


MIGRATION_GUIDE_URL = "https://docs.x.ai/developers/migration/may-15-retirement"
RETIREMENT_DATE = "May 15, 2026"


# Official mapping per xAI migration guide. ``grok-4.3`` reasons by default, so ``*-non-reasoning``
# variants need ``reasoning_effort="none"`` to emulate their behavior.
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
    replacement: str
    reasoning_effort: Optional[str] = None  # set for non-reasoning variant migration
    note: Optional[str] = None


def _normalize(model_id: str) -> str:
    """Strip provider prefix (``x-ai/grok-4`` → ``grok-4``) and lowercase."""
    m = model_id.strip().lower()
    for prefix in ("x-ai/", "xai/"):
        if m.startswith(prefix):
            return m[len(prefix):]
    return m


def _looks_like_xai(model_id: Optional[str]) -> bool:
    return isinstance(model_id, str) and _normalize(model_id).startswith("grok-")


def find_retired_xai_refs(config: Dict[str, Any]) -> List[RetirementIssue]:
    """Walk all model slots in a Hermes config and return retirement issues.

    Slots scanned: ``principal.model``, ``auxiliary.<any>.model`` (introspective, covers future
    aux slots), ``delegation.model``, ``tts.xai.model``, ``plugins.image_gen.xai.model``.
    """
    issues: List[RetirementIssue] = []
    if not isinstance(config, dict):
        return issues

    def _check(path: str, model: Any) -> None:
        entry = _RETIRED_MODELS.get(_normalize(model)) if _looks_like_xai(model) else None
        if entry is not None:
            issues.append(RetirementIssue(
                config_path=path,
                current_model=model,
                replacement=entry["replacement"],
                reasoning_effort=entry.get("reasoning_effort"),
                note=entry.get("note")))

    def _section(*keys: str) -> Optional[Dict[str, Any]]:
        node: Any = config
        for key in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node if isinstance(node, dict) else None

    def _check_section(*path: str) -> None:
        section = _section(*path)
        if section is not None:
            _check(".".join(path) + ".model", section.get("model"))

    _check_section("principal")
    for slot_name, slot_cfg in (_section("auxiliary") or {}).items():
        if isinstance(slot_cfg, dict):
            _check(f"auxiliary.{slot_name}.model", slot_cfg.get("model"))
    for path in (("delegation",), ("tts", "xai"), ("plugins", "image_gen", "xai")):
        _check_section(*path)
    return issues


def format_issue(issue: RetirementIssue) -> str:
    """One-line human-readable rendering of a retirement issue."""
    parts = [f"{issue.config_path}: {issue.current_model!r} → use {issue.replacement!r}"]
    if issue.reasoning_effort:
        parts.append(f'(set reasoning_effort: "{issue.reasoning_effort}")')
    if issue.note:
        parts.append(f"[note: {issue.note}]")
    return " ".join(parts)


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of an apply_migration call."""

    file_path: Path
    backup_path: Optional[Path]
    issues_resolved: List[RetirementIssue]
    config_changed: bool


def _walk_to_parent(yaml_doc: Any, dotted_path: str) -> "tuple[Any, str]":
    """Resolve a dotted slot path to (parent_mapping, leaf_key)."""
    *parents, leaf = dotted_path.split(".")
    if not parents:
        raise ValueError(f"Path must have at least one parent: {dotted_path!r}")
    node = yaml_doc
    for segment in parents:
        if not isinstance(node, dict) or segment not in node:
            raise KeyError(f"Path segment {segment!r} missing in {dotted_path!r}")
        node = node[segment]
    return node, leaf


def apply_migration(
    config_path: Path, issues: List[RetirementIssue], backup: bool = True) -> ApplyResult:
    """Rewrite ``config_path`` in place (ruamel round-trip: comments, order, type literals kept).

    Unless ``backup=False`` a copy goes to ``<config_path>.bak-pre-migrate-xai-YYYYMMDD-HHMMSS``.
    """
    from ruamel.yaml import YAML  # local import — avoid hard dep at module load
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    unchanged = ApplyResult(file_path=config_path, backup_path=None, issues_resolved=[], config_changed=False)
    if not issues:
        return unchanged

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    with config_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh)
    if doc is None:
        return unchanged

    resolved: List[RetirementIssue] = []
    for issue in issues:
        try:
            parent, leaf = _walk_to_parent(doc, issue.config_path)
        except KeyError:
            continue  # slot vanished between scan and apply
        parent[leaf] = issue.replacement
        if issue.reasoning_effort:
            parent["reasoning_effort"] = issue.reasoning_effort
        resolved.append(issue)
    if not resolved:
        return unchanged

    backup_path: Optional[Path] = None
    if backup:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = config_path.with_name(f"{config_path.name}.bak-pre-migrate-xai-{ts}")
        shutil.copy2(config_path, backup_path)

    from hermes_cli.config import require_readable_config_before_write
    from utils import atomic_write_text
    require_readable_config_before_write(config_path)
    # Dump to a buffer, then atomic-write: ``open(path, "w")`` truncates before the dump runs, so a
    # crash mid-write would leave config.yaml empty (and with ``--no-backup`` that is the only
    # copy; the ``doc is None`` early return would then hide the damage). atomic_replace also keeps
    # a symlinked config.yaml intact. preserve_mode keeps permission bits AND owner (managed NixOS
    # 0640 / container installs; a root-run migration must not flip ownership).
    buf = io.StringIO()
    yaml.dump(doc, buf)
    atomic_write_text(config_path, buf.getvalue(), preserve_mode=True)
    return ApplyResult(file_path=config_path, backup_path=backup_path, issues_resolved=resolved, config_changed=True)
