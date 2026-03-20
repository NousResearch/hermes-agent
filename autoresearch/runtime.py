"""Runtime orchestration for generic bounded AutoResearch cycles."""

from __future__ import annotations

import copy
import json
import os
import random
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from autoresearch.manifests import (
    ManifestError,
    discover_project_root,
    list_family_files,
    list_project_roots,
    load_family_config,
    load_project_config,
    validate_project_manifest,
)
from autoresearch.models import FamilyConfig, ProjectConfig, WorkspaceInfo
from autoresearch.reports import build_publish_summary, write_run_report
from autoresearch.workspaces import create_candidate_workspace, list_changed_files


class AutoResearchRuntimeError(RuntimeError):
    """Raised when a research-cycle step fails in a user-visible way."""


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    parts = [part for part in text.split("-") if part]
    return "-".join(parts) or "item"


def _resolve_project_root(project_root: Optional[str] = None) -> Path:
    root = Path(project_root).resolve() if project_root else discover_project_root()
    if root is None:
        raise AutoResearchRuntimeError(
            "No AutoResearch workspace found. Create .hermes/autoresearch/project.yaml in your project."
        )
    return Path(root).resolve()


def _run_dir(project: ProjectConfig, run_id: str) -> Path:
    return project.root / ".hermes" / "autoresearch" / "runs" / run_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _temporary_env(updates: dict[str, Optional[str]], chdir: Optional[Path] = None):
    previous_env: dict[str, Optional[str]] = {}
    previous_cwd = Path.cwd()
    try:
        for key, value in updates.items():
            previous_env[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if chdir is not None:
            os.chdir(chdir)
        yield
    finally:
        if chdir is not None:
            os.chdir(previous_cwd)
        for key, old_value in previous_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _json_command_result(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutoResearchRuntimeError(f"Expected JSON tool result, got: {raw[:400]}") from exc


def _run_command(command: str, workdir: Path, task_id: Optional[str]) -> dict[str, Any]:
    from tools.terminal_tool import terminal_tool

    return _json_command_result(
        terminal_tool(
            command=command,
            workdir=str(workdir),
            task_id=task_id,
        )
    )


def _safe_format(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format_map(context)
    except KeyError as exc:
        missing = exc.args[0]
        raise AutoResearchRuntimeError(
            f"Command template references missing placeholder '{missing}': {template}"
        ) from exc


def _metric_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    for segment in dotted_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(dotted_path)
            current = current[segment]
            continue
        if isinstance(current, list):
            index = int(segment)
            current = current[index]
            continue
        raise KeyError(dotted_path)
    return current


def _metric_delta(goal: str, candidate_value: float, parent_value: float) -> float:
    if goal == "minimize":
        return parent_value - candidate_value
    return candidate_value - parent_value


def _sort_key(goal: str, value: Any) -> float:
    numeric = float(value)
    return -numeric if goal == "minimize" else numeric


def _compare(op: str, left: Any, right: Any) -> bool:
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    raise AutoResearchRuntimeError(f"Unsupported comparison op: {op}")


def _parse_marker_ranges(content: str, start_token: str, end_token: str) -> tuple[int, int]:
    lines = content.splitlines()
    start_index = next((index for index, line in enumerate(lines, start=1) if start_token in line), None)
    end_index = next((index for index, line in enumerate(lines, start=1) if end_token in line), None)
    if start_index is None or end_index is None or end_index <= start_index:
        raise AutoResearchRuntimeError(
            f"Could not resolve editable marker range between '{start_token}' and '{end_token}'"
        )
    return start_index, end_index


def _changed_line_ranges(base_content: Optional[str], new_content: Optional[str]) -> list[tuple[int, int]]:
    import difflib

    before = (base_content or "").splitlines()
    after = (new_content or "").splitlines()
    ranges: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(a=before, b=after)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        start = i1 + 1
        end = max(i2, i1 + 1)
        ranges.append((start, end))
    return ranges


def _review_agent_patch_candidate(
    family: FamilyConfig,
    workspace: WorkspaceInfo,
) -> dict[str, Any]:
    changed_files = list_changed_files(workspace)
    allowed_files = set(family.editable_files)
    forbidden = [path for path in changed_files if path not in allowed_files]
    if forbidden:
        return {
            "accepted": False,
            "reasons": [f"Edited forbidden files: {', '.join(sorted(forbidden))}"],
            "changed_files": changed_files,
        }

    marker_errors: list[str] = []
    markers_by_file = {marker.file: marker for marker in family.editable_markers}
    for relpath in changed_files:
        marker = markers_by_file.get(relpath)
        if marker is None:
            continue
        base_content = workspace.editable_base_contents.get(relpath)
        target_path = workspace.path / relpath
        new_content = target_path.read_text(encoding="utf-8") if target_path.exists() else None
        try:
            marker_start, marker_end = _parse_marker_ranges(base_content or new_content or "", marker.start, marker.end)
        except AutoResearchRuntimeError as exc:
            marker_errors.append(str(exc))
            continue
        for start, end in _changed_line_ranges(base_content, new_content):
            if start <= marker_start or end >= marker_end:
                marker_errors.append(
                    f"Changes in '{relpath}' exceeded editable marker bounds {marker.start}..{marker.end}"
                )
                break

    return {
        "accepted": not marker_errors,
        "reasons": marker_errors,
        "changed_files": changed_files,
    }


def _load_result_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AutoResearchRuntimeError(f"Evaluation command did not create result_json: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AutoResearchRuntimeError(f"Malformed result_json at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutoResearchRuntimeError(f"result_json must contain a JSON object: {path}")
    return payload


def _evaluate_candidate(
    *,
    project: ProjectConfig,
    family: FamilyConfig,
    run_id: str,
    candidate: dict[str, Any],
    workspace: WorkspaceInfo,
    task_id: Optional[str],
) -> dict[str, Any]:
    candidate_dir = _run_dir(project, run_id) / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_json = candidate_dir / f"{candidate['candidate_id']}.json"
    result_json = candidate_dir / f"{candidate['candidate_id']}--result.json"

    candidate_payload = {
        "candidate_id": candidate["candidate_id"],
        "label": candidate.get("label", candidate["candidate_id"]),
        "description": candidate.get("description", ""),
        "parent_candidate_id": candidate.get("parent_candidate_id"),
        "parameters": copy.deepcopy(candidate.get("parameters") or {}),
        "mutation_mode": family.mutation.mode,
        "mutated_fields": list(candidate.get("mutated_fields") or []),
    }
    _write_json(candidate_json, candidate_payload)

    command_ctx = {
        "project_root": str(project.root),
        "workspace": str(workspace.path),
        "workspace_path": str(workspace.path),
        "cwd": str((workspace.path / project.default_cwd).resolve()),
        "candidate_id": candidate["candidate_id"],
        "family_id": family.family_id,
        "project_id": project.project_id,
        "candidate_json": str(candidate_json),
        "result_json": str(result_json),
    }

    validation_result = None
    if family.commands.validation:
        validation_cmd = _safe_format(family.commands.validation, command_ctx)
        validation_result = _run_command(validation_cmd, workspace.path / project.default_cwd, task_id)
        if validation_result.get("exit_code") != 0:
            raise AutoResearchRuntimeError(
                f"Validation command failed for {candidate['candidate_id']}: {validation_result.get('output') or validation_result.get('error')}"
            )

    evaluation_cmd = _safe_format(family.commands.evaluation, command_ctx)
    evaluation_result = _run_command(evaluation_cmd, workspace.path / project.default_cwd, task_id)
    if evaluation_result.get("exit_code") != 0:
        raise AutoResearchRuntimeError(
            f"Evaluation command failed for {candidate['candidate_id']}: {evaluation_result.get('output') or evaluation_result.get('error')}"
        )

    payload = _load_result_payload(result_json)
    primary_metric = float(_metric_value(payload, family.selection.primary_metric))
    return {
        **candidate,
        "workspace": workspace.to_dict(),
        "workspace_path": str(workspace.path),
        "result_json": str(result_json),
        "candidate_json": str(candidate_json),
        "validation_result": validation_result,
        "evaluation_result": evaluation_result,
        "metrics": payload,
        "primary_metric": primary_metric,
    }


def _generate_param_candidates(
    family: FamilyConfig,
    *,
    seed: int,
    population: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    parents = family.anchors
    for anchor in parents:
        seen.add(json.dumps(anchor.parameters, sort_keys=True))

    dimensions = list(family.mutation.parameter_space.items())
    candidates: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(50, population * 20)

    while len(candidates) < population and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(parents)
        params = copy.deepcopy(parent.parameters)
        mutate_count = min(
            max(1, family.mutation.max_mutations_per_candidate),
            len(dimensions),
        )
        chosen = rng.sample(dimensions, k=max(1, min(len(dimensions), rng.randint(1, mutate_count))))
        mutated_fields: list[str] = []
        for field_name, values in chosen:
            options = [value for value in values if value != params.get(field_name)]
            if not options:
                continue
            params[field_name] = rng.choice(options)
            mutated_fields.append(field_name)
        if not mutated_fields:
            continue
        signature = json.dumps(params, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        candidate_index = len(candidates) + 1
        candidates.append(
            {
                "candidate_id": f"{family.family_id}_gen_{candidate_index:03d}",
                "label": f"{family.family_id} Candidate {candidate_index:03d}",
                "description": f"Generated from {parent.candidate_id} by mutating {', '.join(mutated_fields)}.",
                "parent_candidate_id": parent.candidate_id,
                "parameters": params,
                "mutated_fields": mutated_fields,
            }
        )
    return candidates


def _run_agent_patch(
    *,
    project: ProjectConfig,
    family: FamilyConfig,
    workspace: WorkspaceInfo,
    candidate: dict[str, Any],
    model: Optional[str],
) -> dict[str, Any]:
    from run_agent import AIAgent

    editable_lines = [f"- {path}" for path in family.editable_files]
    if family.editable_markers:
        editable_lines.extend(
            f"- {marker.file}: only between `{marker.start}` and `{marker.end}`"
            for marker in family.editable_markers
        )

    prompt = "\n".join(
        [
            f"You are generating an AutoResearch candidate inside {workspace.path}.",
            f"Project: {project.project_id}",
            f"Family: {family.family_id}",
            f"Thesis: {family.thesis}",
            f"Parent candidate: {candidate.get('parent_candidate_id') or 'anchor'}",
            "Only edit the allowed files below. Do not touch any other files.",
            *editable_lines,
            "",
            "Make one plausible improvement aligned with the thesis and save the files.",
            "Do not explain the plan first. Make the changes directly, then summarize them briefly.",
        ]
    )
    if family.mutation.prompt:
        prompt = f"{prompt}\n\nAdditional family guidance:\n{family.mutation.prompt}"

    env_updates = {
        "TERMINAL_CWD": str(workspace.path),
        "HERMES_WRITE_SAFE_ROOT": str(workspace.path),
    }
    with _temporary_env(env_updates, chdir=workspace.path):
        agent = AIAgent(
            model=model or os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6",
            max_iterations=20,
            enabled_toolsets=["file", "terminal", "todo"],
            quiet_mode=True,
            save_trajectories=False,
            skip_memory=True,
            platform="cli",
        )
        response = agent.chat(prompt)

    return {
        "model": model or os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6",
        "response": response,
    }


def _generate_agent_patch_candidates(
    *,
    project: ProjectConfig,
    family: FamilyConfig,
    run_id: str,
    population: int,
    model: Optional[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    parents = family.anchors
    for candidate_index in range(1, population + 1):
        parent = parents[(candidate_index - 1) % len(parents)]
        candidate = {
            "candidate_id": f"{family.family_id}_gen_{candidate_index:03d}",
            "label": f"{family.family_id} Candidate {candidate_index:03d}",
            "description": f"Agent patch candidate generated from {parent.candidate_id}.",
            "parent_candidate_id": parent.candidate_id,
            "parameters": copy.deepcopy(parent.parameters),
            "mutated_fields": [],
        }
        workspace = create_candidate_workspace(project.root, run_id, candidate["candidate_id"], family.editable_files)
        candidate["workspace"] = workspace
        candidate["agent_patch"] = _run_agent_patch(
            project=project,
            family=family,
            workspace=workspace,
            candidate=candidate,
            model=model,
        )
        candidates.append(candidate)
    return candidates


def _selector_payload(
    *,
    family: FamilyConfig,
    anchors: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    survivors: int,
) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda item: _sort_key(family.selection.goal, item["primary_metric"]),
        reverse=True,
    )
    shortlisted = ranked[:survivors]
    anchor_map = {anchor["candidate_id"]: anchor for anchor in anchors}
    kept: list[dict[str, Any]] = []

    for candidate in shortlisted:
        parent = anchor_map.get(candidate.get("parent_candidate_id"))
        if parent is None:
            continue
        primary_delta = _metric_delta(
            family.selection.goal,
            float(candidate["primary_metric"]),
            float(parent["primary_metric"]),
        )
        if primary_delta <= 0:
            candidate["selector_reasons"] = ["Primary metric did not beat parent."]
            candidate["primary_delta"] = primary_delta
            continue

        selector_errors: list[str] = []
        for rule in family.selection.secondary_metrics:
            candidate_metric = float(_metric_value(candidate["metrics"], rule.metric))
            parent_metric = float(_metric_value(parent["metrics"], rule.metric))
            delta = _metric_delta(family.selection.goal, candidate_metric, parent_metric)
            if delta < rule.min_delta:
                selector_errors.append(
                    f"Secondary metric '{rule.metric}' regressed by {delta}, below min_delta {rule.min_delta}"
                )
        candidate["primary_delta"] = primary_delta
        candidate["selector_reasons"] = selector_errors
        if not selector_errors:
            kept.append(candidate)

    champion = None
    if kept:
        champion = sorted(
            kept,
            key=lambda item: _sort_key(family.selection.goal, item["primary_metric"]),
            reverse=True,
        )[0]

    return {
        "ranked_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item.get("parent_candidate_id"),
                "primary_metric": item["primary_metric"],
                "result_json": item.get("result_json"),
                "workspace_path": item.get("workspace_path"),
            }
            for item in ranked
        ],
        "kept_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item.get("parent_candidate_id"),
                "primary_metric": item["primary_metric"],
                "primary_delta": item.get("primary_delta"),
                "result_json": item.get("result_json"),
                "workspace_path": item.get("workspace_path"),
            }
            for item in kept
        ],
        "champion": (
            {
                "candidate_id": champion["candidate_id"],
                "parent_candidate_id": champion.get("parent_candidate_id"),
                "primary_metric": champion["primary_metric"],
                "primary_delta": champion.get("primary_delta"),
                "result_json": champion.get("result_json"),
                "workspace_path": champion.get("workspace_path"),
                "metrics": champion.get("metrics"),
            }
            if champion
            else None
        ),
    }


def _interesting_payload(
    *,
    project: ProjectConfig,
    family: FamilyConfig,
    run_id: str,
    selector: dict[str, Any],
) -> dict[str, Any]:
    champion = selector.get("champion")
    if champion is None:
        return {"verdict": False, "reason": "No champion selected."}
    if not family.interesting.rules:
        return {"verdict": True, "reason": "Champion selected and no explicit interestingness rules were configured."}

    context = {
        "project": project.to_dict(),
        "family": family.to_dict(),
        "run": {"run_id": run_id},
        "selector": selector,
        "champion": champion,
    }
    verdicts: list[bool] = []
    failed_descriptions: list[str] = []
    for rule in family.interesting.rules:
        try:
            left = _metric_value(context, rule.metric)
        except (KeyError, IndexError, ValueError):
            verdicts.append(False)
            failed_descriptions.append(f"Missing metric '{rule.metric}'")
            continue
        passed = _compare(rule.op, left, rule.value)
        verdicts.append(passed)
        if not passed:
            failed_descriptions.append(f"{rule.metric} {rule.op} {rule.value} was false (left={left})")

    verdict = all(verdicts) if family.interesting.mode == "all" else any(verdicts)
    return {
        "verdict": verdict,
        "reason": "All interestingness rules passed." if verdict else "; ".join(failed_descriptions),
    }


def list_projects(project_root: Optional[str] = None) -> dict[str, Any]:
    roots = list_project_roots(project_root)
    projects = []
    for root in roots:
        try:
            project = load_project_config(root)
        except ManifestError as exc:
            projects.append({"path": str(root), "error": str(exc)})
            continue
        projects.append(
            {
                "project_id": project.project_id,
                "description": project.description,
                "path": str(root),
                "families": [path.stem for path in list_family_files(root)],
            }
        )
    return {"count": len(projects), "projects": projects}


def inspect_project(project_root: Optional[str] = None) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    project = load_project_config(root)
    families = [load_family_config(root, path.stem, project=project).to_dict() for path in list_family_files(root)]
    return {"project": project.to_dict(), "families": families}


def validate_project(project_root: Optional[str] = None) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    return validate_project_manifest(root)


def research_cycle(
    *,
    project_root: Optional[str] = None,
    family_id: str,
    population: Optional[int] = None,
    survivors: Optional[int] = None,
    seed: int = 7,
    model: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    project = load_project_config(root)
    family = load_family_config(root, family_id, project=project)
    run_id = f"ar-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_path = _run_dir(project, run_id)
    run_path.mkdir(parents=True, exist_ok=True)

    actual_population = int(population or family.mutation.population)
    actual_survivors = int(survivors or family.mutation.survivors)

    run_record: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _iso_now(),
        "project_id": project.project_id,
        "family_id": family.family_id,
        "project_root": str(project.root),
        "run_path": str(run_path),
        "status": "running",
        "phase": "anchors",
        "selection": {
            "primary_metric": family.selection.primary_metric,
            "goal": family.selection.goal,
            "survivors": actual_survivors,
        },
        "project": project.to_dict(),
        "family": family.to_dict(),
        "anchors": [],
        "candidates": [],
        "selector": {},
        "interesting": {"verdict": False, "reason": "Run has not completed yet."},
        "report_path": None,
        "summary": None,
    }
    _write_json(run_path / "run.json", run_record)

    try:
        anchors: list[dict[str, Any]] = []
        for anchor in family.anchors:
            workspace = create_candidate_workspace(project.root, run_id, anchor.candidate_id, family.editable_files)
            anchor_candidate = {
                "candidate_id": anchor.candidate_id,
                "label": anchor.label,
                "description": anchor.description,
                "parent_candidate_id": None,
                "parameters": copy.deepcopy(anchor.parameters),
                "mutated_fields": [],
            }
            evaluated = _evaluate_candidate(
                project=project,
                family=family,
                run_id=run_id,
                candidate=anchor_candidate,
                workspace=workspace,
                task_id=task_id,
            )
            anchors.append(evaluated)
        run_record["anchors"] = [
            {
                "candidate_id": item["candidate_id"],
                "primary_metric": item["primary_metric"],
                "result_json": item["result_json"],
                "workspace_path": item["workspace_path"],
                "metrics": item["metrics"],
            }
            for item in anchors
        ]
        run_record["phase"] = "generate"
        _write_json(run_path / "run.json", run_record)

        if family.mutation.mode == "param_mutation":
            generated = _generate_param_candidates(family, seed=seed, population=actual_population)
            generated_wrapped = []
            for candidate in generated:
                workspace = create_candidate_workspace(project.root, run_id, candidate["candidate_id"], family.editable_files)
                candidate["workspace"] = workspace
                generated_wrapped.append(candidate)
        else:
            generated_wrapped = _generate_agent_patch_candidates(
                project=project,
                family=family,
                run_id=run_id,
                population=actual_population,
                model=model,
            )

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        run_record["phase"] = "review"
        _write_json(run_path / "run.json", run_record)

        for candidate in generated_wrapped:
            workspace = candidate.pop("workspace")
            if family.mutation.mode == "agent_patch":
                review = _review_agent_patch_candidate(family, workspace)
            else:
                review = {"accepted": True, "reasons": [], "changed_files": []}
            candidate["review"] = review
            if not review["accepted"]:
                rejected.append(
                    {
                        **candidate,
                        "workspace": workspace.to_dict(),
                        "workspace_path": str(workspace.path),
                    }
                )
                continue

            evaluated = _evaluate_candidate(
                project=project,
                family=family,
                run_id=run_id,
                candidate=candidate,
                workspace=workspace,
                task_id=task_id,
            )
            evaluated["review"] = review
            accepted.append(evaluated)

        run_record["candidates"] = [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item.get("parent_candidate_id"),
                "primary_metric": item.get("primary_metric"),
                "result_json": item.get("result_json"),
                "workspace_path": item.get("workspace_path"),
                "review": item.get("review"),
                "agent_patch": item.get("agent_patch"),
            }
            for item in accepted + rejected
        ]
        run_record["phase"] = "select"
        _write_json(run_path / "run.json", run_record)

        selector = _selector_payload(
            family=family,
            anchors=anchors,
            candidates=accepted,
            survivors=actual_survivors,
        )
        run_record["selector"] = selector
        interesting = _interesting_payload(project=project, family=family, run_id=run_id, selector=selector)
        run_record["interesting"] = interesting

        if selector.get("champion") and interesting.get("verdict"):
            report_rel = Path(project.report_output_dir) / datetime.now().strftime("%Y-%m-%d") / f"{_slug(project.project_id)}--{run_id}.md"
            report_path = project.root / report_rel
            write_run_report(
                project=project.to_dict(),
                family=family.to_dict(),
                run={
                    **run_record,
                    "selector": selector,
                },
                output_path=report_path,
            )
            run_record["report_path"] = str(report_path)

        run_record["summary"] = build_publish_summary(run_record)
        run_record["status"] = "completed"
        run_record["phase"] = "completed"
        run_record["completed_at"] = _iso_now()
        _write_json(run_path / "run.json", run_record)
        return run_record

    except Exception as exc:
        run_record["status"] = "failed"
        run_record["phase"] = "failed"
        run_record["error"] = str(exc)
        run_record["completed_at"] = _iso_now()
        _write_json(run_path / "run.json", run_record)
        raise


def _load_run(project_root: Path, run_id: str) -> dict[str, Any]:
    path = _run_dir(load_project_config(project_root), run_id) / "run.json"
    if not path.exists():
        raise AutoResearchRuntimeError(f"Unknown AutoResearch run '{run_id}'")
    return _read_json(path)


def status(*, run_id: str, project_root: Optional[str] = None) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    run = _load_run(root, run_id)
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "phase": run["phase"],
        "error": run.get("error"),
        "report_path": run.get("report_path"),
        "summary": run.get("summary"),
    }


def list_runs(project_root: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    runs_root = root / ".hermes" / "autoresearch" / "runs"
    rows = []
    if runs_root.exists():
        for entry in sorted(runs_root.iterdir(), reverse=True):
            run_json = entry / "run.json"
            if not run_json.exists():
                continue
            payload = _read_json(run_json)
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "created_at": payload["created_at"],
                    "project_id": payload["project_id"],
                    "family_id": payload["family_id"],
                    "status": payload["status"],
                    "report_path": payload.get("report_path"),
                }
            )
    return {"count": len(rows[:limit]), "runs": rows[:limit]}


def inspect_run(*, run_id: str, project_root: Optional[str] = None) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    run = _load_run(root, run_id)
    report_preview = None
    if run.get("report_path"):
        report_path = Path(run["report_path"])
        if report_path.exists():
            report_preview = report_path.read_text(encoding="utf-8")[:4000]
    return {"run": run, "report_preview": report_preview}


def publish_summary(
    *,
    run_id: str,
    project_root: Optional[str] = None,
    target: Optional[str] = None,
    send: bool = False,
) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    run = _load_run(root, run_id)
    summary = run.get("summary") or build_publish_summary(run)
    resolved_target = target or run.get("project", {}).get("publish_target")

    payload = {
        "run_id": run_id,
        "summary": summary,
        "target": resolved_target,
        "sent": False,
    }

    if send:
        from tools.send_message_tool import send_message_tool

        if not resolved_target:
            raise AutoResearchRuntimeError(
                "publish_summary(send=True) requires an explicit target or project.publish.default_target"
            )
        send_result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": resolved_target,
                    "message": summary,
                }
            )
        )
        payload["sent"] = bool(send_result.get("success"))
        payload["send_result"] = send_result

    return payload


