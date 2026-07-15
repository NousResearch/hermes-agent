"""Profile-aware StudyOS workspace and active-project boundary.

This module is the single internal entry point for locating a StudyOS Vault
and resolving its active project.  Model tools, HTTP surfaces, and desktop UI
all depend on this boundary instead of maintaining separate selection rules.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from plugins.study_os.schemas import (
    PROJECT_ID_RE,
    validate_study_project,
    validate_study_schedule,
)


def _configured_vault(config: Mapping[str, Any]) -> str | None:
    section = config.get("study_os")
    if not isinstance(section, Mapping):
        return None
    value = section.get("vault_path")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _legacy_vault_from_environment() -> str | None:
    """Read the pre-config.yaml Vault setting for backward compatibility."""

    try:
        from hermes_cli.config import get_env_value

        value = get_env_value("OBSIDIAN_VAULT_PATH")
    except Exception:
        value = os.environ.get("OBSIDIAN_VAULT_PATH")
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_vault(
    requested_path: str | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve a Vault using explicit → config → legacy → discovery precedence."""

    explicit = requested_path.strip() if isinstance(requested_path, str) else ""
    if explicit:
        candidate, source = explicit, "explicit"
    else:
        if config is None:
            try:
                from hermes_cli.config import load_config_readonly

                config = load_config_readonly()
            except Exception:
                config = {}
        configured = _configured_vault(config)
        legacy = _legacy_vault_from_environment()
        if configured:
            candidate, source = configured, "config"
        elif legacy:
            candidate, source = legacy, "legacy_env"
        else:
            current = Path(cwd or os.getcwd()).expanduser()
            if (current / ".obsidian").exists() or (current / "Box").exists():
                candidate, source = str(current), "cwd"
            else:
                candidate, source = "~/Documents/Obsidian Vault", "fallback"

    path = Path(candidate).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"StudyOS Vault not found: {path}. Configure study_os.vault_path "
            "in config.yaml or pass vault_path explicitly."
        )
    return path, source


def _validate_project_id(project_id: Any) -> str:
    value = str(project_id or "").strip()
    if not PROJECT_ID_RE.match(value):
        raise ValueError("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def study_state_dir(vault: Path) -> Path:
    """Return the Vault-owned StudyOS state directory."""

    root = (vault / ".StudyOS").resolve()
    try:
        root.relative_to(vault)
    except ValueError as exc:
        raise ValueError("StudyOS state path escapes Vault") from exc
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class StudyScheduleArtifact:
    """One canonical, validated schedule discovered in a workspace."""

    schedule: dict[str, Any]
    path: str


@dataclass(frozen=True)
class InvalidStudySchedule:
    """A schedule file that exists but cannot enter StudyOS projections."""

    schedule_id: str
    path: str
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "path": self.path,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class StudyScheduleCatalog:
    """Validated schedules and explicit issues from one project directory."""

    project_id: str
    schedules: tuple[StudyScheduleArtifact, ...]
    invalid_schedules: tuple[InvalidStudySchedule, ...]


@dataclass(frozen=True)
class StudyWorkspace:
    """A resolved Vault with one authoritative active-project pointer."""

    vault: Path
    source: str

    @classmethod
    def open(
        cls,
        requested_path: str | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        cwd: str | Path | None = None,
    ) -> "StudyWorkspace":
        vault, source = resolve_vault(requested_path, config=config, cwd=cwd)
        return cls(vault=vault, source=source)

    @property
    def projects_root(self) -> Path:
        root = (self.vault / ".StudyOS" / "projects").resolve()
        try:
            root.relative_to(self.vault)
        except ValueError as exc:
            raise ValueError("StudyOS projects path escapes Vault") from exc
        return root

    @property
    def study_dir(self) -> Path:
        return study_state_dir(self.vault)

    @property
    def active_project_path(self) -> Path:
        return self.projects_root / "active.json"

    def project(self, project_id: str | None = None) -> dict[str, Any]:
        selected = project_id or self.active_project_id()
        if selected is None:
            raise FileNotFoundError("No active StudyOS project selected")
        resolved_id = _validate_project_id(selected)
        manifest_path = (self.projects_root / resolved_id / "manifest.json").resolve()
        try:
            manifest_path.relative_to(self.projects_root)
        except ValueError as exc:
            raise ValueError("Project manifest path escapes Vault") from exc
        if not manifest_path.exists():
            raise FileNotFoundError(f"Project not found: {resolved_id}")
        raw = _read_json_object(manifest_path)
        ok, validated = validate_study_project(raw)
        if not ok:
            errors = validated if isinstance(validated, list) else ["Invalid project manifest"]
            raise ValueError("; ".join(errors))
        if not isinstance(validated, dict):
            raise ValueError("Project validator returned invalid data")
        return validated

    def list_projects(self) -> list[dict[str, Any]]:
        if not self.projects_root.exists():
            return []
        projects: list[dict[str, Any]] = []
        for manifest_path in sorted(self.projects_root.glob("*/manifest.json")):
            try:
                projects.append(self.project(manifest_path.parent.name))
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
        return projects

    def discover_schedules(
        self,
        project_id: str | None = None,
    ) -> StudyScheduleCatalog:
        """Discover valid schedules without silently dropping invalid files."""

        project = self.project(project_id)
        resolved_id = str(project["project_id"])
        schedule_root = (self.projects_root / resolved_id / "schedules").resolve()
        try:
            schedule_root.relative_to(self.projects_root)
        except ValueError as exc:
            raise ValueError("Schedule path escapes StudyOS projects root") from exc
        if not schedule_root.exists():
            return StudyScheduleCatalog(
                project_id=resolved_id,
                schedules=(),
                invalid_schedules=(),
            )

        schedules: list[StudyScheduleArtifact] = []
        invalid_schedules: list[InvalidStudySchedule] = []
        for schedule_path in sorted(schedule_root.glob("*.json")):
            schedule_id = schedule_path.stem
            relative_path = schedule_path.relative_to(self.vault).as_posix()
            errors: tuple[str, ...] = ()
            try:
                raw = _read_json_object(schedule_path)
            except json.JSONDecodeError:
                errors = (f"Invalid JSON: {schedule_path.name}",)
            except (OSError, ValueError) as exc:
                errors = (str(exc),)
            else:
                ok, validated = validate_study_schedule(raw, project=project)
                if not ok:
                    errors = tuple(
                        validated
                        if isinstance(validated, list)
                        else ["Schedule validator returned invalid errors"]
                    )
                elif not isinstance(validated, dict):
                    errors = ("Schedule validator returned invalid data",)
                elif validated.get("schedule_id") != schedule_id:
                    errors = ("schedule_id must match its canonical filename",)
                else:
                    schedules.append(
                        StudyScheduleArtifact(
                            schedule=validated,
                            path=relative_path,
                        )
                    )
            if errors:
                invalid_schedules.append(
                    InvalidStudySchedule(
                        schedule_id=schedule_id,
                        path=relative_path,
                        errors=errors,
                    )
                )

        return StudyScheduleCatalog(
            project_id=resolved_id,
            schedules=tuple(schedules),
            invalid_schedules=tuple(invalid_schedules),
        )

    def active_project_id(self) -> str | None:
        if not self.active_project_path.exists():
            return None
        try:
            active = _read_json_object(self.active_project_path)
            project_id = _validate_project_id(active.get("project_id"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not (self.projects_root / project_id / "manifest.json").exists():
            return None
        return project_id

    def select_project(self, project_id: str) -> dict[str, Any]:
        project = self.project(project_id)
        self.projects_root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"project_id": project["project_id"]},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=".active-",
            suffix=".json",
            dir=self.projects_root,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.active_project_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return project
