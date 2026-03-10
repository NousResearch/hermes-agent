"""Filesystem-backed research state."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import yaml

from research.config import get_research_config, get_workspace_dir
from research.models import DatasetRecord, InboxItem, ResearchProject, RunSpec, make_id, now_utc, slugify


@contextmanager
def _file_lock(path: Path):
    """Acquire an exclusive lock on a .lock file adjacent to *path*.

    Uses ``fcntl.flock`` so the lock is automatically released when the
    process exits or crashes.  The lock file is created if it doesn't exist.
    """
    lock_path = path.parent / f".{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class ResearchStateStore:
    """Manage persistent research project state on disk."""

    def __init__(self, cwd: str | Path | None = None):
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.config = get_research_config()
        self.workspace_root = get_workspace_dir(self.cwd)
        self.projects_root = self.workspace_root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def approvals_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "approvals"

    def layout(self, project_id: str) -> dict[str, Path]:
        root = self.project_dir(project_id)
        return {
            "root": root,
            "project": root / "project.yaml",
            "loop_state": root / "loop_state.json",
            "ideas": root / "ideas.jsonl",
            "literature": root / "literature.jsonl",
            "hypotheses": root / "hypotheses.jsonl",
            "experiments": root / "experiments",
            "runs": root / "runs",
            "reports": root / "reports",
            "runtime": root / "runtime",
            "inbox": root / "inbox",
            "inbox_unread": root / "inbox" / "unread",
            "approvals": root / "approvals",
            "evaluations": root / "evaluations.jsonl",
            "datasets": root / "datasets",
            "graph": root / "graph.jsonl",
        }

    def ensure_layout(self, project_id: str) -> dict[str, Path]:
        layout = self.layout(project_id)
        for key in ("root", "experiments", "runs", "reports", "runtime", "inbox", "inbox_unread", "approvals", "datasets"):
            layout[key].mkdir(parents=True, exist_ok=True)
        for key in ("ideas", "literature", "hypotheses", "evaluations", "graph"):
            layout[key].parent.mkdir(parents=True, exist_ok=True)
            layout[key].touch(exist_ok=True)
        if not layout["loop_state"].exists():
            _atomic_write_text(layout["loop_state"], json.dumps({"project_id": project_id, "checkpoints": []}, indent=2))
        return layout

    def init_project(
        self,
        name: str | None,
        brief: str = "",
        project_id: str | None = None,
        objective: str | None = None,
        spec_level: str = "partial_spec",
        metadata: dict[str, Any] | None = None,
    ) -> ResearchProject:
        project_name = (name or brief or "Autonomous Research Project").strip()
        project_slug = project_id or slugify(project_name, default="project")
        final_id = project_slug if project_slug.startswith("proj_") else f"proj_{project_slug}"
        layout = self.ensure_layout(final_id)
        project = ResearchProject(
            project_id=final_id,
            name=project_name,
            cwd=str(self.cwd),
            workspace_dir=str(self.workspace_root),
            objective=(objective or brief).strip(),
            user_brief=brief,
            spec_level=spec_level,  # type: ignore[arg-type]
            metadata=metadata or {},
        )
        self.save_project(project)
        if not layout["loop_state"].exists():
            self.save_loop_state(final_id, {"project_id": final_id, "status": "active", "checkpoints": []})
        else:
            current = self.load_loop_state(final_id)
            current.setdefault("project_id", final_id)
            current.setdefault("status", "active")
            current.setdefault("checkpoints", [])
            self.save_loop_state(final_id, current)
        return project

    def save_project(self, project: ResearchProject) -> None:
        layout = self.ensure_layout(project.project_id)
        payload = project.model_dump(mode="json")
        payload["updated_at"] = now_utc()
        _atomic_write_text(layout["project"], yaml.safe_dump(payload, sort_keys=False))

    def load_project(self, project_id: str) -> ResearchProject:
        layout = self.ensure_layout(project_id)
        data = yaml.safe_load(layout["project"].read_text(encoding="utf-8")) or {}
        return ResearchProject.model_validate(data)

    def list_projects(self) -> list[ResearchProject]:
        projects: list[ResearchProject] = []
        for project_file in sorted(self.projects_root.glob("*/project.yaml")):
            try:
                data = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
                projects.append(ResearchProject.model_validate(data))
            except Exception:
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def latest_project_id(self) -> str | None:
        projects = self.list_projects()
        return projects[0].project_id if projects else None

    def append_jsonl(self, project_id: str, name: str, record: dict[str, Any]) -> dict[str, Any]:
        layout = self.ensure_layout(project_id)
        if name not in {"ideas", "literature", "hypotheses", "evaluations", "graph"}:
            raise ValueError(f"Unsupported JSONL store: {name}")
        target = layout[name]
        with _file_lock(target):
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record

    def load_jsonl(self, project_id: str, name: str) -> list[dict[str, Any]]:
        layout = self.ensure_layout(project_id)
        target = layout[name]
        rows: list[dict[str, Any]] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    def save_experiment(self, project_id: str, experiment_id: str, payload: dict[str, Any]) -> Path:
        layout = self.ensure_layout(project_id)
        target = layout["experiments"] / f"{experiment_id}.yaml"
        _atomic_write_text(target, yaml.safe_dump(payload, sort_keys=False))
        return target

    def load_experiment(self, project_id: str, experiment_id: str) -> dict[str, Any]:
        path = self.layout(project_id)["experiments"] / f"{experiment_id}.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def list_experiments(self, project_id: str) -> list[dict[str, Any]]:
        folder = self.layout(project_id)["experiments"]
        experiments: list[dict[str, Any]] = []
        for path in sorted(folder.glob("*.yaml")):
            experiments.append(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        return experiments

    def save_run(self, run_spec: RunSpec | dict[str, Any]) -> Path:
        payload = run_spec.model_dump(mode="json") if isinstance(run_spec, RunSpec) else dict(run_spec)
        project_id = payload["project_id"]
        run_id = payload["run_id"]
        layout = self.ensure_layout(project_id)
        target = layout["runs"] / f"{run_id}.json"
        payload["updated_at"] = now_utc()
        _atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False))
        return target

    def load_run(self, project_id: str, run_id: str) -> dict[str, Any]:
        target = self.layout(project_id)["runs"] / f"{run_id}.json"
        return json.loads(target.read_text(encoding="utf-8"))

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        folder = self.layout(project_id)["runs"]
        runs: list[dict[str, Any]] = []
        for path in sorted(folder.glob("*.json")):
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(runs, key=lambda item: item.get("updated_at", ""), reverse=True)

    def save_loop_state(self, project_id: str, payload: dict[str, Any]) -> None:
        layout = self.ensure_layout(project_id)
        payload = dict(payload)
        payload["project_id"] = project_id
        payload["updated_at"] = now_utc()
        _atomic_write_text(layout["loop_state"], json.dumps(payload, indent=2, ensure_ascii=False))

    def load_loop_state(self, project_id: str) -> dict[str, Any]:
        layout = self.ensure_layout(project_id)
        return json.loads(layout["loop_state"].read_text(encoding="utf-8"))

    def add_inbox_item(self, project_id: str, item: InboxItem | dict[str, Any]) -> Path:
        payload = item.model_dump(mode="json") if isinstance(item, InboxItem) else dict(item)
        layout = self.ensure_layout(project_id)
        timestamp = payload.get("created_at", now_utc()).replace(":", "-")
        payload.setdefault("item_id", make_id("inbox"))
        target = layout["inbox_unread"] / f"{timestamp}_{payload['item_id']}.md"
        body = [
            f"# {payload['title']}",
            "",
            payload.get("summary", "").strip(),
        ]
        details = payload.get("details", "").strip()
        if details:
            body.extend(["", "## Details", "", details])
        _atomic_write_text(target, "\n".join(body).strip() + "\n")
        return target

    def unread_inbox(self, project_id: str) -> list[Path]:
        folder = self.layout(project_id)["inbox_unread"]
        return sorted(folder.glob("*.md"))

    def write_report(self, project_id: str, content: str, iteration: int | None = None) -> Path:
        layout = self.ensure_layout(project_id)
        if iteration is None:
            existing = sorted(layout["reports"].glob("iteration-*.md"))
            iteration = len(existing) + 1
        target = layout["reports"] / f"iteration-{iteration}.md"
        _atomic_write_text(target, content.rstrip() + "\n")
        return target

    def write_named_report(self, project_id: str, filename: str, content: str) -> Path:
        layout = self.ensure_layout(project_id)
        target = layout["reports"] / filename
        _atomic_write_text(target, content.rstrip() + "\n")
        return target

    def load_results(self, project_id: str) -> list[dict[str, Any]]:
        loop_state = self.load_loop_state(project_id)
        return list(loop_state.get("results") or [])

    def append_result(self, project_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        loop_state = self.load_loop_state(project_id)
        results = list(loop_state.get("results") or [])
        results.append(dict(payload))
        loop_state["results"] = results
        self.save_loop_state(project_id, loop_state)
        return results

    def status_summary(self, project_id: str) -> dict[str, Any]:
        project = self.load_project(project_id)
        loop_state = self.load_loop_state(project_id)
        runs = self.list_runs(project_id)
        unread = self.unread_inbox(project_id)
        evaluations = self.load_evaluations(project_id)
        datasets = self.list_dataset_records(project_id)
        graph_edges = self.load_graph_edges(project_id)
        return {
            "project": project.model_dump(mode="json"),
            "loop_state": loop_state,
            "run_count": len(runs),
            "active_run_ids": [run["run_id"] for run in runs if run.get("status") in {"pending", "queued", "running", "stopping"}],
            "unread_inbox": [str(path) for path in unread],
            "ideas": len(self.load_jsonl(project_id, "ideas")),
            "literature_notes": len(self.load_jsonl(project_id, "literature")),
            "hypotheses": len(self.load_jsonl(project_id, "hypotheses")),
            "experiments": len(self.list_experiments(project_id)),
            "evaluations": len(evaluations),
            "datasets": len(datasets),
            "results": len(loop_state.get("results") or []),
            "graph_edges": len(graph_edges),
        }

    def save_dataset_record(self, project_id: str, record: DatasetRecord | dict[str, Any]) -> Path:
        payload = record.model_dump(mode="json") if isinstance(record, DatasetRecord) else dict(record)
        layout = self.ensure_layout(project_id)
        dataset_id = payload.get("dataset_id") or f"ds_{now_utc().replace(':', '').replace('-', '')}"
        target = layout["datasets"] / f"{dataset_id}.json"
        _atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False))
        return target

    def load_dataset_record(self, project_id: str, dataset_id: str) -> dict[str, Any]:
        target = self.layout(project_id)["datasets"] / f"{dataset_id}.json"
        return json.loads(target.read_text(encoding="utf-8"))

    def list_dataset_records(self, project_id: str) -> list[dict[str, Any]]:
        folder = self.layout(project_id)["datasets"]
        records: list[dict[str, Any]] = []
        for path in sorted(folder.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

    def load_evaluations(self, project_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.load_jsonl(project_id, "evaluations")
        if run_id:
            rows = [r for r in rows if r.get("run_id") == run_id]
        return rows

    def load_graph_edges(self, project_id: str) -> list[dict[str, Any]]:
        return self.load_jsonl(project_id, "graph")

    def update_project_runs(self, project_id: str, run_ids: Iterable[str]) -> None:
        project = self.load_project(project_id)
        project.active_run_ids = list(dict.fromkeys(run_ids))
        project.updated_at = now_utc()
        self.save_project(project)
