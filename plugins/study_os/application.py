"""Transport-independent StudyOS application module.

HTTP adapters and non-model clients cross this seam instead of reaching into
note parsing, workspace layout, or projection helpers directly. The existing
model tools retain their stable two-tool interface. Callers issue a named
query or command and receive plain data, while this module owns validation,
persistence ordering, and stable failure semantics.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from plugins.study_os.learning import (
    handle_study_activity,
    handle_study_review_detail,
    handle_study_review_submission,
)
from plugins.study_os.overview import build_study_overview
from plugins.study_os.reviews import StudyReviewReadModel
from plugins.study_os.workspace import StudyWorkspace


class StudyQuery(str, Enum):
    PROJECTS = "projects"
    SETTINGS = "settings"
    OVERVIEW = "overview"
    PROJECT = "project"
    SCHEDULES = "schedules"
    SCHEDULE = "schedule"
    REVIEW_DUE = "review_due"
    REVIEW_DETAIL = "review_detail"
    REVIEW_STATS = "review_stats"
    REVIEW_QUEUE = "review_queue"
    REVIEW_CONCEPTS = "review_concepts"
    PROFILE = "profile"


class StudyCommand(str, Enum):
    UPDATE_SETTINGS = "update_settings"
    SELECT_PROJECT = "select_project"
    DECIDE_PLAN_PROPOSAL = "decide_plan_proposal"
    SUBMIT_REVIEW = "submit_review"
    UPDATE_PROFILE = "update_profile"


class StudyApplicationError(RuntimeError):
    """Stable failure returned by the StudyOS application seam."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


_PROFILE_DEFAULTS: dict[str, Any] = {
    "daily_review_limit": 20,
    "review_level_filter": None,
    "subject_filter": None,
}


class StudyOSApplication:
    """Execute StudyOS queries and commands against the active profile."""

    def query(self, query: StudyQuery | str, /, **params: Any) -> dict[str, Any]:
        try:
            operation = StudyQuery(query)
            if operation is StudyQuery.PROJECTS:
                return self._projects()
            if operation is StudyQuery.SETTINGS:
                return self._settings()
            if operation is StudyQuery.OVERVIEW:
                return self._overview(**params)
            if operation is StudyQuery.PROJECT:
                return self._project(str(params.get("project_id") or ""))
            if operation is StudyQuery.SCHEDULES:
                return self._schedules(str(params.get("project_id") or ""))
            if operation is StudyQuery.SCHEDULE:
                return self._schedule(
                    str(params.get("project_id") or ""),
                    str(params.get("schedule_id") or ""),
                )
            if operation is StudyQuery.REVIEW_DUE:
                return self._review_due(**params)
            if operation is StudyQuery.REVIEW_DETAIL:
                return self._review_detail(str(params.get("note") or ""))
            if operation is StudyQuery.REVIEW_STATS:
                return self._review_stats(bool(params.get("rebuild", False)))
            if operation is StudyQuery.REVIEW_QUEUE:
                return self._review_queue(**params)
            if operation is StudyQuery.REVIEW_CONCEPTS:
                return self._review_concepts()
            if operation is StudyQuery.PROFILE:
                return self._profile()
        except StudyApplicationError:
            raise
        except FileNotFoundError as exc:
            raise StudyApplicationError(404, str(exc)) from exc
        except ValueError as exc:
            raise StudyApplicationError(400, str(exc)) from exc
        except OSError as exc:
            raise StudyApplicationError(400, str(exc)) from exc
        raise StudyApplicationError(400, f"Unsupported StudyOS query: {query}")

    def execute(self, command: StudyCommand | str, /, **params: Any) -> dict[str, Any]:
        try:
            operation = StudyCommand(command)
            if operation is StudyCommand.UPDATE_SETTINGS:
                return self._update_settings(str(params.get("vault_path") or ""))
            if operation is StudyCommand.SELECT_PROJECT:
                return self._select_project(str(params.get("project_id") or ""))
            if operation is StudyCommand.DECIDE_PLAN_PROPOSAL:
                return self._decide_plan_proposal(**params)
            if operation is StudyCommand.SUBMIT_REVIEW:
                return self._submit_review(dict(params))
            if operation is StudyCommand.UPDATE_PROFILE:
                return self._update_profile(dict(params))
        except StudyApplicationError:
            raise
        except FileNotFoundError as exc:
            raise StudyApplicationError(404, str(exc)) from exc
        except ValueError as exc:
            raise StudyApplicationError(400, str(exc)) from exc
        except OSError as exc:
            raise StudyApplicationError(400, str(exc)) from exc
        raise StudyApplicationError(400, f"Unsupported StudyOS command: {command}")

    @staticmethod
    def _workspace(required: bool = True) -> StudyWorkspace | None:
        try:
            return StudyWorkspace.open()
        except FileNotFoundError:
            if required:
                raise StudyApplicationError(404, "StudyOS vault not configured")
            return None

    def _projects(self) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "projects": [],
                "configured": False,
                "message": "StudyOS vault not configured",
            }
        return {
            "projects": workspace.list_projects(),
            "configured": True,
            "vault_path": str(workspace.vault),
            "active_project_id": workspace.active_project_id(),
        }

    def _settings(self) -> dict[str, Any]:
        from hermes_cli.config import load_config

        config = load_config()
        study_config = config.get("study_os")
        configured_value = (
            str(study_config.get("vault_path") or "").strip()
            if isinstance(study_config, dict)
            else ""
        )
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "configured": False,
                "vault_path": configured_value or None,
                "active_project_id": None,
            }
        return {
            "configured": True,
            "vault_path": str(workspace.vault),
            "source": workspace.source,
            "active_project_id": workspace.active_project_id(),
        }

    def _update_settings(self, raw_path: str) -> dict[str, Any]:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools

        requested = raw_path.strip()
        if not requested:
            raise StudyApplicationError(400, "vault_path must be a non-empty directory")
        try:
            workspace = StudyWorkspace.open(requested, config={})
        except FileNotFoundError as exc:
            raise StudyApplicationError(400, str(exc)) from exc

        config = load_config()
        study_config = config.setdefault("study_os", {})
        if not isinstance(study_config, dict):
            study_config = {}
            config["study_os"] = study_config
        study_config["vault_path"] = str(workspace.vault)

        enabled = _get_platform_tools(config, "cli")
        was_enabled = "study" in enabled
        enabled.add("study")
        _save_platform_tools(config, "cli", enabled)
        save_config(config)
        return {
            "configured": True,
            "vault_path": str(workspace.vault),
            "active_project_id": workspace.active_project_id(),
            "study_toolset_enabled": True,
            "requires_new_session": not was_enabled,
        }

    def _select_project(self, project_id: str) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        project = workspace.select_project(project_id)
        return {"active_project_id": project["project_id"], "project": project}

    def _overview(self, **params: Any) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        return build_study_overview(
            workspace,
            project_id=params.get("project_id"),
            as_of=params.get("as_of"),
            review_limit=int(params.get("review_limit", 10)),
            intervention_limit=int(params.get("intervention_limit", 5)),
        )

    def _decide_plan_proposal(self, **params: Any) -> dict[str, Any]:
        action = str(params.get("action") or "")
        if action not in {"accept", "reject"}:
            raise StudyApplicationError(400, "action must be accept or reject")
        workspace = self._workspace()
        assert workspace is not None
        return self._tool_data(
            handle_study_activity(
                {
                    "resource": "plan_proposal",
                    "action": action,
                    "vault_path": str(workspace.vault),
                    "project_id": str(params.get("project_id") or ""),
                    "data": {
                        "proposal_id": str(params.get("proposal_id") or ""),
                        "decision_note": params.get("decision_note"),
                    },
                }
            )
        )

    def _project(self, project_id: str) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        return workspace.project(project_id)

    def _schedules(self, project_id: str) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        catalog = workspace.discover_schedules(project_id)
        schedules = [
            {
                "schedule_id": artifact.schedule["schedule_id"],
                "project_id": artifact.schedule["project_id"],
                "title": artifact.schedule["title"],
                "timezone": artifact.schedule["timezone"],
                "range": artifact.schedule["range"],
                "phase_count": len(artifact.schedule.get("phases", [])),
                "event_count": len(artifact.schedule.get("events", [])),
            }
            for artifact in catalog.schedules
        ]
        return {
            "project_id": catalog.project_id,
            "schedules": schedules,
            "invalid_schedules": [
                issue.as_dict() for issue in catalog.invalid_schedules
            ],
        }

    def _schedule(self, project_id: str, schedule_id: str) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        catalog = workspace.discover_schedules(project_id)
        for artifact in catalog.schedules:
            if artifact.schedule.get("schedule_id") == schedule_id:
                return artifact.schedule
        for issue in catalog.invalid_schedules:
            if issue.schedule_id == schedule_id:
                raise StudyApplicationError(
                    400,
                    {"code": "VALIDATION_FAILED", "errors": list(issue.errors)},
                )
        raise StudyApplicationError(404, "Schedule not found")

    def _review_due(self, **params: Any) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "vault_path": None,
                "configured": False,
                "date": "",
                "count": 0,
                "subjects": [],
                "due": [],
            }
        return {
            "configured": True,
            **StudyReviewReadModel(workspace.vault).due(
                subject=str(params.get("subject") or ""),
                level=params.get("level"),
                limit=int(params.get("limit", 20)),
            ),
        }

    def _review_detail(self, note: str) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        return self._tool_data(
            handle_study_review_detail(
                {"vault_path": str(workspace.vault), "note": note}
            )
        )

    def _submit_review(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        params["vault_path"] = str(workspace.vault)
        params["diagnoses"] = params.get("diagnoses", [])
        return self._tool_data(handle_study_review_submission(params))

    def _review_stats(self, rebuild: bool) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "vault_path": None,
                "configured": False,
                "total": 0,
                "by_level": {},
                "spacing_coverage": 0.0,
                "reviewed_count": 0,
                "progress": 0.0,
                "concept_stats": {},
                "review_streak": 0,
                "due_count": 0,
                "cached": False,
            }
        return {
            "configured": True,
            **StudyReviewReadModel(workspace.vault).stats(rebuild=rebuild),
        }

    def _review_queue(self, **params: Any) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "vault_path": None,
                "configured": False,
                "new_concepts": [],
                "new_concepts_total": 0,
                "new_examples": [],
                "new_examples_total": 0,
            }
        return {
            "configured": True,
            **StudyReviewReadModel(workspace.vault).queue(
                state=str(params.get("state") or ""),
                limit=int(params.get("limit", 30)),
            ),
        }

    def _review_concepts(self) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {"vault_path": None, "configured": False, "concepts": []}
        return {
            "configured": True,
            **StudyReviewReadModel(workspace.vault).concepts(),
        }

    def _profile(self) -> dict[str, Any]:
        workspace = self._workspace(required=False)
        if workspace is None:
            return {
                "vault_path": None,
                "configured": False,
                **_PROFILE_DEFAULTS,
            }
        vault = workspace.vault
        config_path = workspace.study_dir / "config.json"
        data: Any = {}
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, OSError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        return {
            "vault_path": str(vault),
            "configured": True,
            "daily_review_limit": data.get(
                "daily_review_limit", _PROFILE_DEFAULTS["daily_review_limit"]
            ),
            "review_level_filter": data.get(
                "review_level_filter", _PROFILE_DEFAULTS["review_level_filter"]
            ),
            "subject_filter": data.get(
                "subject_filter", _PROFILE_DEFAULTS["subject_filter"]
            ),
        }

    def _update_profile(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace()
        assert workspace is not None
        vault = workspace.vault
        config_path = workspace.study_dir / "config.json"
        existing: Any = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, OSError):
                existing = {}
        if not isinstance(existing, dict):
            existing = {}
        merged = {**_PROFILE_DEFAULTS, **existing}
        daily_limit = params.get("daily_review_limit")
        if daily_limit is not None:
            if not isinstance(daily_limit, int) or isinstance(daily_limit, bool) or daily_limit < 1:
                raise StudyApplicationError(400, "daily_review_limit must be >= 1")
            merged["daily_review_limit"] = daily_limit
        review_level = params.get("review_level_filter")
        if review_level is not None:
            if not isinstance(review_level, int) or isinstance(review_level, bool) or not 0 <= review_level <= 5:
                raise StudyApplicationError(400, "review_level_filter must be 0–5")
            merged["review_level_filter"] = review_level
        subject_filter = params.get("subject_filter")
        if subject_filter is not None:
            merged["subject_filter"] = str(subject_filter) or None
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"vault_path": str(vault), "configured": True, **merged}

    @staticmethod
    def _tool_data(raw: str) -> dict[str, Any]:
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StudyApplicationError(500, "StudyOS returned invalid JSON") from exc
        if not isinstance(envelope, dict):
            raise StudyApplicationError(500, "StudyOS returned an invalid result")
        if envelope.get("ok"):
            data = envelope.get("data")
            if isinstance(data, dict):
                return data
            raise StudyApplicationError(500, "StudyOS returned invalid data")
        error = envelope.get("error", {})
        code = str(error.get("code") or "STUDY_OPERATION_FAILED") if isinstance(error, dict) else "STUDY_OPERATION_FAILED"
        status = 404 if code in {"NOTE_NOT_FOUND", "PROJECT_NOT_FOUND", "NOT_FOUND"} else 400
        raise StudyApplicationError(status, error)
