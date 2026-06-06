"""Downstream-only project watchdog bootstrap/backfill helpers.

This module intentionally emits project-specific watchdog artifacts from a
sanitized project index. It is not part of upstream NousResearch bootstrap; it
captures Yunuen's downstream Hermes Maintenance policy for deterministic project
watchdogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_TEMPLATE_DIR = REPO_ROOT / "docs" / "templates" / "project_watchdog"
ACTIVE_STATUSES = {"active", "working", "running"}
REQUIRED_GATES = (
    "kanban_board_bound",
    "matrix_route_bound",
    "github_repository_bound",
    "exactly_one_dispatch_owner",
    "coordinator_owned_scheduler",
)


class ProjectWatchdogValidationError(ValueError):
    """Raised when project watchdog metadata/artifacts are incomplete."""


@dataclass(frozen=True)
class BackfilledProject:
    project_tag: str
    root: Path
    policy_path: Path
    runtime_config_path: Path
    state_dir: Path
    latest_report_dir: Path
    recipe_paths: dict[str, Path]
    validation_gates_path: Path

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class BackfillResult:
    projects: list[BackfilledProject]


def _read_index(index_path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        projects = raw
    else:
        projects = raw.get("projects", [])
    if not isinstance(projects, list):
        raise ProjectWatchdogValidationError("project index must contain a projects list")
    return [project for project in projects if isinstance(project, dict)]


def _active_projects(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        project
        for project in projects
        if str(project.get("status", "active")).lower() in ACTIVE_STATUSES
    ]


def _require_string(project: dict[str, Any], dotted_key: str) -> str:
    value: Any = project
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ProjectWatchdogValidationError(
                f"{project.get('project_tag', '<unknown>')}: missing {dotted_key}"
            )
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise ProjectWatchdogValidationError(
            f"{project.get('project_tag', '<unknown>')}: {dotted_key} must be a non-empty string"
        )
    return value.strip()


def _owner(project: dict[str, Any], key: str) -> str:
    metadata = project.get("metadata")
    if not isinstance(metadata, dict):
        raise ProjectWatchdogValidationError(
            f"{project.get('project_tag', '<unknown>')}: missing metadata"
        )
    value = metadata.get(key)
    if key == "dispatch_owner" and isinstance(value, list):
        raise ProjectWatchdogValidationError(
            f"{project.get('project_tag', '<unknown>')}: expected exactly one dispatch_owner"
        )
    if not isinstance(value, str) or not value.strip():
        article = "exactly one " if key == "dispatch_owner" else ""
        raise ProjectWatchdogValidationError(
            f"{project.get('project_tag', '<unknown>')}: missing {article}{key}"
        )
    return value.strip()


def _relative_watchdog_path(tag: str, leaf: str) -> str:
    return f"projects/{tag}/watchdog/{leaf}"


def _validate_project_metadata(project: dict[str, Any]) -> dict[str, str]:
    tag = _require_string(project, "project_tag")
    board_slug = _require_string(project, "board_slug")
    coordinator_profile = _require_string(project, "coordinator_profile")
    room_alias = _require_string(project, "matrix.room_alias")
    space_alias = _require_string(project, "matrix.space_alias")
    repository = _require_string(project, "github.repository")
    dispatch_owner = _owner(project, "dispatch_owner")
    watchdog_owner = _owner(project, "watchdog_owner")
    pa_audit_owner = _owner(project, "pa_audit_owner")

    if dispatch_owner != coordinator_profile:
        raise ProjectWatchdogValidationError(
            f"{tag}: dispatch_owner must be the coordinator profile ({coordinator_profile})"
        )
    if watchdog_owner != coordinator_profile:
        raise ProjectWatchdogValidationError(
            f"{tag}: watchdog_owner must be the coordinator profile ({coordinator_profile})"
        )

    return {
        "tag": tag,
        "board_slug": board_slug,
        "coordinator_profile": coordinator_profile,
        "room_alias": room_alias,
        "space_alias": space_alias,
        "repository": repository,
        "dispatch_owner": dispatch_owner,
        "watchdog_owner": watchdog_owner,
        "pa_audit_owner": pa_audit_owner,
    }


def _policy_document(project: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    tag = values["tag"]
    return {
        "schema_version": 1,
        "downstream_only": True,
        "project_tag": tag,
        "board_slug": values["board_slug"],
        "coordinator_profile": values["coordinator_profile"],
        "owners": {
            "dispatch_owner": values["dispatch_owner"],
            "watchdog_owner": values["watchdog_owner"],
            "pa_audit_owner": values["pa_audit_owner"],
        },
        "bindings": {
            "matrix": {
                "room_alias": values["room_alias"],
                "space_alias": values["space_alias"],
                "encryption": project.get("matrix", {}).get("encryption", "explicit-required"),
            },
            "github": project.get("github", {}),
            "repository": project.get("repository", {}),
            "worker_profiles": project.get("worker_profiles", []),
        },
        "runtime_artifacts": {
            "runtime_config": _relative_watchdog_path(tag, "runtime_config.yaml"),
            "state_dir": _relative_watchdog_path(tag, "state"),
            "latest_report_dir": _relative_watchdog_path(tag, "latest-report"),
            "recipes_dir": _relative_watchdog_path(tag, "recipes"),
            "validation_gates": _relative_watchdog_path(tag, "validation_gates.yaml"),
        },
        "quality_boundary": {
            "project_specific_audit_profiles": "forbidden",
            "shadow_review_normal_worker_outputs": "forbidden",
            "review_model": "profile self-test/review plus explicit reviewer cards",
        },
    }


def _runtime_document(values: dict[str, str]) -> dict[str, Any]:
    tag = values["tag"]
    return {
        "schema_version": 1,
        "project_tag": tag,
        "board_slug": values["board_slug"],
        "coordinator_profile": values["coordinator_profile"],
        "state_dir": _relative_watchdog_path(tag, "state"),
        "latest_report_dir": _relative_watchdog_path(tag, "latest-report"),
        "report_status_lines": ["Self status", "Lineage status"],
        "scheduler": {
            "owner_profile": values["coordinator_profile"],
            "recipes": {
                "systemd": _relative_watchdog_path(tag, "recipes/project-watchdog.service"),
                "cron": _relative_watchdog_path(tag, "recipes/project-watchdog.crontab"),
                "hermes_cron": _relative_watchdog_path(tag, "recipes/hermes-cron.yaml"),
            },
        },
    }


def _validation_document(values: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project_tag": values["tag"],
        "gates": [
            {"name": "kanban_board_bound", "required": values["board_slug"]},
            {"name": "matrix_route_bound", "required": values["room_alias"]},
            {"name": "github_repository_bound", "required": values["repository"]},
            {"name": "exactly_one_dispatch_owner", "required": values["dispatch_owner"]},
            {"name": "coordinator_owned_scheduler", "required": values["coordinator_profile"]},
        ],
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _write_recipe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _recipe_texts(values: dict[str, str]) -> dict[str, str]:
    profile = values["coordinator_profile"]
    tag = values["tag"]
    prompt = (
        f"Run the deterministic project watchdog for project {tag}; inspect Kanban, Matrix, "
        "and GitHub bindings; report using Self status and Lineage status."
    )
    return {
        "systemd": "\n".join(
            [
                "[Unit]",
                f"Description=Hermes project watchdog for {tag}",
                "",
                "[Service]",
                "Type=oneshot",
                f"ExecStart=hermes -p {profile} chat -q '{prompt}' --toolsets safe",
                "",
            ]
        ),
        "cron": f"*/15 * * * * hermes -p {profile} chat -q '{prompt}' --toolsets safe\n",
        "hermes_cron": yaml.safe_dump(
            {
                "name": f"project-watchdog-{tag}",
                "schedule": "every 15m",
                "profile": profile,
                "enabled_toolsets": ["safe", "kanban"],
                "prompt": prompt,
            },
            sort_keys=True,
        ),
    }


def _ensure_templates_sanitized() -> None:
    forbidden = (
        "@yunuen.montelongo:matrix.org",
        "#h-yunuen",
        "/home/engs2272",
        "sk-",
        "ghp_",
        "github_pat_",
    )
    for template_path in LIVE_TEMPLATE_DIR.glob("*.yaml"):
        text = template_path.read_text(encoding="utf-8")
        if any(fragment in text for fragment in forbidden):
            raise ProjectWatchdogValidationError(
                f"template contains live identifier or secret-like fragment: {template_path}"
            )


def backfill_project_watchdogs(index_path: Path | str, output_root: Path | str) -> BackfillResult:
    index_path = Path(index_path)
    output_root = Path(output_root)
    _ensure_templates_sanitized()
    projects = []
    for project in _active_projects(_read_index(index_path)):
        values = _validate_project_metadata(project)
        tag = values["tag"]
        root = output_root / "projects" / tag / "watchdog"
        state_dir = root / "state"
        latest_report_dir = root / "latest-report"
        state_dir.mkdir(parents=True, exist_ok=True)
        latest_report_dir.mkdir(parents=True, exist_ok=True)

        policy_path = root / "policy.yaml"
        runtime_config_path = root / "runtime_config.yaml"
        gates_path = root / "validation_gates.yaml"
        _write_yaml(policy_path, _policy_document(project, values))
        _write_yaml(runtime_config_path, _runtime_document(values))
        _write_yaml(gates_path, _validation_document(values))

        recipe_texts = _recipe_texts(values)
        recipe_paths = {
            "systemd": root / "recipes" / "project-watchdog.service",
            "cron": root / "recipes" / "project-watchdog.crontab",
            "hermes_cron": root / "recipes" / "hermes-cron.yaml",
        }
        for name, path in recipe_paths.items():
            _write_recipe(path, recipe_texts[name])

        projects.append(
            BackfilledProject(
                project_tag=tag,
                root=root,
                policy_path=policy_path,
                runtime_config_path=runtime_config_path,
                state_dir=state_dir,
                latest_report_dir=latest_report_dir,
                recipe_paths=recipe_paths,
                validation_gates_path=gates_path,
            )
        )
    return BackfillResult(projects=projects)


def validate_project_watchdog_index(index_path: Path | str, output_root: Path | str) -> None:
    result = backfill_project_watchdogs(index_path=index_path, output_root=output_root)
    for project in result.projects:
        required_paths = [
            project.policy_path,
            project.runtime_config_path,
            project.state_dir,
            project.latest_report_dir,
            project.recipe_paths["systemd"],
            project.recipe_paths["cron"],
            project.recipe_paths["hermes_cron"],
            project.validation_gates_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise ProjectWatchdogValidationError(
                f"{project.project_tag}: missing watchdog artifacts: {missing}"
            )
        gates = yaml.safe_load(project.validation_gates_path.read_text(encoding="utf-8")) or {}
        names = {gate.get("name") for gate in gates.get("gates", []) if isinstance(gate, dict)}
        missing_gates = set(REQUIRED_GATES) - names
        if missing_gates:
            raise ProjectWatchdogValidationError(
                f"{project.project_tag}: missing validation gates: {sorted(missing_gates)}"
            )
