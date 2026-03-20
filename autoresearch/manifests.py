"""Manifest loading and validation for workspace AutoResearch configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from autoresearch.models import (
    AnchorConfig,
    CommandConfig,
    EditableMarker,
    FamilyConfig,
    InterestingConfig,
    InterestingRule,
    MutationConfig,
    ProjectConfig,
    SecondaryMetricRule,
    SelectionConfig,
)

PROJECT_CONFIG_RELPATH = Path(".hermes") / "autoresearch" / "project.yaml"
FAMILIES_RELPATH = Path(".hermes") / "autoresearch" / "families"

_ALLOWED_OPS = {"==", "!=", ">", ">=", "<", "<="}


class ManifestError(ValueError):
    """Raised when an AutoResearch manifest is invalid."""


def discover_project_root(start: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
    """Find the nearest workspace that declares an AutoResearch project."""
    current = Path(start or os.getcwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_CONFIG_RELPATH).exists():
            return candidate
    return None


def list_project_roots(start: Optional[str | os.PathLike[str]] = None) -> list[Path]:
    """Discover nearby workspaces that declare AutoResearch configs."""
    roots: list[Path] = []
    start_path = Path(start or os.getcwd()).resolve()
    nearest = discover_project_root(start_path)
    if nearest is not None:
        roots.append(nearest)

    try:
        for child in start_path.iterdir():
            if child.is_dir() and (child / PROJECT_CONFIG_RELPATH).exists():
                resolved = child.resolve()
                if resolved not in roots:
                    roots.append(resolved)
    except OSError:
        pass

    return roots


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"Failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ManifestError(f"Manifest must be a mapping: {path}")
    return loaded


def _load_command_config(raw: dict[str, Any], fallback: Optional[CommandConfig]) -> Optional[CommandConfig]:
    evaluation = raw.get("evaluation") or raw.get("evaluation_command")
    validation = raw.get("validation") or raw.get("validation_command")
    result_json = raw.get("result_json")

    if evaluation is None and fallback is not None:
        evaluation = fallback.evaluation
    if validation is None and fallback is not None:
        validation = fallback.validation
    if result_json is None and fallback is not None:
        result_json = fallback.result_json

    if not evaluation or not result_json:
        return None
    return CommandConfig(
        evaluation=str(evaluation),
        validation=str(validation) if validation else None,
        result_json=str(result_json),
    )


def load_project_config(project_root: str | os.PathLike[str]) -> ProjectConfig:
    """Load and normalize the workspace project manifest."""
    root = Path(project_root).resolve()
    raw = _load_yaml(root / PROJECT_CONFIG_RELPATH)
    evaluator = _load_command_config(raw.get("evaluator", {}), None)
    report = raw.get("report", {}) if isinstance(raw.get("report", {}), dict) else {}
    publish = raw.get("publish", {}) if isinstance(raw.get("publish", {}), dict) else {}
    project_id = str(raw.get("project_id") or raw.get("id") or "").strip()
    description = str(raw.get("description") or "").strip()

    if not project_id:
        raise ManifestError("project.yaml requires 'project_id'")
    if not description:
        raise ManifestError("project.yaml requires 'description'")

    return ProjectConfig(
        project_id=project_id,
        description=description,
        root=root,
        default_cwd=str(raw.get("default_cwd") or raw.get("cwd") or "."),
        datasets=list(raw.get("datasets") or []),
        benchmarks=[str(item) for item in (raw.get("benchmarks") or [])],
        evaluator=evaluator,
        report_output_dir=str(report.get("output_dir") or raw.get("report_output_dir") or "research"),
        publish_target=str(publish.get("default_target") or raw.get("publish_target") or "").strip() or None,
    )


def list_family_files(project_root: str | os.PathLike[str]) -> list[Path]:
    """Return all family manifest files for the workspace."""
    family_dir = Path(project_root).resolve() / FAMILIES_RELPATH
    if not family_dir.exists():
        return []
    return sorted(
        path for path in family_dir.iterdir()
        if path.suffix.lower() in {".yaml", ".yml"} and path.is_file()
    )


def load_family_config(
    project_root: str | os.PathLike[str],
    family_id: str,
    project: Optional[ProjectConfig] = None,
) -> FamilyConfig:
    """Load one family manifest by id."""
    root = Path(project_root).resolve()
    project_cfg = project or load_project_config(root)
    target_file: Optional[Path] = None
    for path in list_family_files(root):
        if path.stem == family_id:
            target_file = path
            break
        raw = _load_yaml(path)
        if str(raw.get("family_id") or raw.get("id") or "").strip() == family_id:
            target_file = path
            break
    if target_file is None:
        raise ManifestError(f"Unknown family_id '{family_id}' in {root}")

    raw = _load_yaml(target_file)
    commands = _load_command_config(raw.get("commands", {}), project_cfg.evaluator)
    if commands is None:
        raise ManifestError(f"Family '{family_id}' must define evaluation command and result_json")

    selection_raw = raw.get("selection", {}) if isinstance(raw.get("selection", {}), dict) else {}
    primary_metric = str(selection_raw.get("primary_metric") or "").strip()
    if not primary_metric:
        raise ManifestError(f"Family '{family_id}' must define selection.primary_metric")

    secondaries = [
        SecondaryMetricRule(
            metric=str(item.get("metric") or "").strip(),
            min_delta=float(item.get("min_delta", 0.0)),
        )
        for item in (selection_raw.get("secondary_metrics") or [])
        if isinstance(item, dict) and str(item.get("metric") or "").strip()
    ]

    interesting_raw = raw.get("interesting_if", {}) if isinstance(raw.get("interesting_if", {}), dict) else {}
    interesting_rules = [
        InterestingRule(
            metric=str(item.get("metric") or "").strip(),
            op=str(item.get("op") or "").strip(),
            value=item.get("value"),
        )
        for item in (interesting_raw.get("rules") or [])
        if isinstance(item, dict) and str(item.get("metric") or "").strip()
    ]

    mutation_raw = raw.get("mutation", {}) if isinstance(raw.get("mutation", {}), dict) else {}
    mutation_mode = str(
        mutation_raw.get("mode")
        or raw.get("mutation_mode")
        or ""
    ).strip()
    if not mutation_mode:
        raise ManifestError(f"Family '{family_id}' must define mutation.mode or mutation_mode")

    anchors_raw = raw.get("anchors") or []
    anchors = [
        AnchorConfig(
            candidate_id=str(item.get("candidate_id") or item.get("id") or "").strip(),
            label=str(item.get("label") or item.get("candidate_id") or item.get("id") or "").strip(),
            description=str(item.get("description") or "").strip(),
            parameters=dict(item.get("parameters") or {}),
        )
        for item in anchors_raw
        if isinstance(item, dict) and str(item.get("candidate_id") or item.get("id") or "").strip()
    ]

    markers = [
        EditableMarker(
            file=str(item.get("file") or "").strip(),
            start=str(item.get("start") or "").strip(),
            end=str(item.get("end") or "").strip(),
        )
        for item in (raw.get("editable_markers") or [])
        if isinstance(item, dict) and str(item.get("file") or "").strip()
    ]

    family = FamilyConfig(
        family_id=str(raw.get("family_id") or raw.get("id") or family_id).strip(),
        thesis=str(raw.get("thesis") or raw.get("description") or "").strip(),
        project_root=root,
        commands=commands,
        mutation=MutationConfig(
            mode=mutation_mode,
            population=int(mutation_raw.get("population", raw.get("population", 8))),
            survivors=int(mutation_raw.get("survivors", raw.get("survivors", 3))),
            parameter_space={
                str(key): list(value or [])
                for key, value in dict(mutation_raw.get("parameter_space") or raw.get("parameter_space") or {}).items()
            },
            max_mutations_per_candidate=int(mutation_raw.get("max_mutations_per_candidate", 2)),
            prompt=str(mutation_raw.get("prompt") or raw.get("mutation_prompt") or "").strip(),
        ),
        selection=SelectionConfig(
            primary_metric=primary_metric,
            goal=str(selection_raw.get("goal") or "maximize").strip() or "maximize",
            secondary_metrics=secondaries,
        ),
        interesting=InterestingConfig(
            mode=str(interesting_raw.get("mode") or "all").strip() or "all",
            rules=interesting_rules,
        ),
        anchors=anchors,
        editable_files=[str(item) for item in (raw.get("editable_files") or [])],
        editable_markers=markers,
    )

    errors = validate_project_family(project_cfg, family)
    if errors:
        raise ManifestError("; ".join(errors))
    return family


def validate_project_family(project: ProjectConfig, family: FamilyConfig) -> list[str]:
    """Return manifest validation errors for a project/family pair."""
    errors: list[str] = []

    if family.selection.goal not in {"maximize", "minimize"}:
        errors.append(f"Family '{family.family_id}' has invalid selection.goal '{family.selection.goal}'")

    if family.interesting.mode not in {"all", "any"}:
        errors.append(f"Family '{family.family_id}' has invalid interesting_if.mode '{family.interesting.mode}'")

    if family.mutation.mode not in {"param_mutation", "agent_patch"}:
        errors.append(f"Family '{family.family_id}' has invalid mutation mode '{family.mutation.mode}'")

    if not family.anchors:
        errors.append(f"Family '{family.family_id}' must define at least one anchor")

    anchor_ids = [anchor.candidate_id for anchor in family.anchors]
    if len(anchor_ids) != len(set(anchor_ids)):
        errors.append(f"Family '{family.family_id}' has duplicate anchor candidate_ids")

    if family.mutation.population < 1:
        errors.append(f"Family '{family.family_id}' population must be >= 1")
    if family.mutation.survivors < 1:
        errors.append(f"Family '{family.family_id}' survivors must be >= 1")

    if family.mutation.mode == "param_mutation" and not family.mutation.parameter_space:
        errors.append(f"Family '{family.family_id}' requires parameter_space for param_mutation")

    if family.mutation.mode == "agent_patch" and not family.editable_files:
        errors.append(f"Family '{family.family_id}' requires editable_files for agent_patch")

    editable_set = set(family.editable_files)
    for marker in family.editable_markers:
        if marker.file not in editable_set:
            errors.append(
                f"Family '{family.family_id}' marker for '{marker.file}' must also appear in editable_files"
            )
        if not marker.start or not marker.end:
            errors.append(f"Family '{family.family_id}' marker for '{marker.file}' needs start and end tokens")

    for rule in family.interesting.rules:
        if rule.op not in _ALLOWED_OPS:
            errors.append(f"Family '{family.family_id}' has invalid interesting_if op '{rule.op}'")

    for secondary in family.selection.secondary_metrics:
        if not secondary.metric:
            errors.append(f"Family '{family.family_id}' has empty secondary metric rule")

    if not family.commands.evaluation.strip():
        errors.append(f"Family '{family.family_id}' must define commands.evaluation")
    if not family.commands.result_json.strip():
        errors.append(f"Family '{family.family_id}' must define commands.result_json")

    if not project.report_output_dir.strip():
        errors.append("project.yaml report output directory must not be empty")

    return errors


def validate_project_manifest(project_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the project manifest and all family manifests."""
    root = Path(project_root).resolve()
    project = load_project_config(root)
    families: list[dict[str, Any]] = []
    errors: list[str] = []
    for family_file in list_family_files(root):
        try:
            family = load_family_config(root, family_file.stem, project=project)
            families.append(family.to_dict())
        except ManifestError as exc:
            errors.append(f"{family_file.name}: {exc}")

    return {
        "valid": not errors,
        "project": project.to_dict(),
        "families": families,
        "errors": errors,
    }
