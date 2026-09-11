"""Agent-guided Install flow with resumable step state.

The flow reads the package's declared requirements, detects prerequisites,
guides setup, runs the verification step, and reports success only after
verification passes. Step state is persisted under HERMES_HOME so a
half-finished install can be resumed on the next turn.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .history import history_path

STEPS = ("plan", "prerequisites", "setup", "apply", "verify", "done")


def _flows_dir() -> Path:
    return history_path().parent / "install_flows"


def detect_prerequisites(requirements: list[dict[str, Any]], *, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Check commands and env vars from a package's requirement list."""
    environment = env if env is not None else dict(os.environ)
    report: list[dict[str, Any]] = []
    for item in requirements:
        kind = str(item.get("kind") or "")
        name = str(item.get("name") or "")
        status: str
        if kind == "command":
            status = "present" if shutil.which(name) else "missing"
        elif kind == "env_var":
            status = "present" if environment.get(name) else "missing"
        else:
            status = "manual"  # accounts, services, permissions need the user
        report.append({**item, "status": status})
    return report


class InstallFlow:
    def __init__(self, flow_id: str | None = None, *, root: Path | None = None) -> None:
        self.root = root or _flows_dir()
        self.flow_id = flow_id or uuid.uuid4().hex
        self.path = self.root / f"{self.flow_id}.json"
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def step(self) -> str:
        return str(self.state.get("step") or "plan")

    @property
    def installed(self) -> bool:
        return bool(self.state.get("installed"))

    @property
    def files_installed(self) -> bool:
        return bool((self.state.get("apply_result") or {}).get("installed"))

    def start(self, *, skill_id: str, version: int | None, package: dict[str, Any]) -> dict[str, Any]:
        self.state = {
            "flow_id": self.flow_id,
            "skill_id": skill_id,
            "version": version,
            "package": package,
            "step": "prerequisites",
            "installed": False,
            "history": [{"step": "plan", "at": datetime.now(timezone.utc).isoformat()}],
        }
        self._save()
        return self.summary()

    def check_prerequisites(self, *, env: dict[str, str] | None = None) -> dict[str, Any]:
        requirements = list((self.state.get("package") or {}).get("requirements") or [])
        report = detect_prerequisites(requirements, env=env)
        self.state["prerequisites"] = report
        missing = [r for r in report if r["status"] != "present"]
        self.state["step"] = "setup" if missing else "apply"
        self._record()
        return self.summary()

    def mark_setup_complete(self, name: str) -> dict[str, Any]:
        for item in self.state.get("prerequisites") or []:
            if item.get("name") == name:
                item["status"] = "present"
        if not any(r["status"] != "present" for r in self.state.get("prerequisites") or []):
            self.state["step"] = "apply"
        self._record()
        return self.summary()

    def apply(self, applier: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        """Run the existing managed install path (receipt plan + apply)."""
        if self.step != "apply":
            raise RuntimeError(f"cannot apply from step {self.step}")
        result = applier(dict(self.state))
        self.state["apply_result"] = result
        self.state["step"] = "verify" if result.get("installed") is True else "apply"
        self._record()
        return self.summary()

    def verify(self, runner: Callable[[str], tuple[bool, str]] | None = None) -> dict[str, Any]:
        if self.step != "verify":
            raise RuntimeError(f"cannot verify from step {self.step}")
        step_text = str((self.state.get("package") or {}).get("verification_step") or "").strip()
        if runner is None:
            self.state["verification"] = {
                "ok": False,
                "approval_required": True,
                "detail": "Approve verification separately through the existing tool permissions.",
            }
            self._record()
            return self.summary()
        ok, detail = runner(step_text) if step_text else (False, "package has no verification step")
        self.state["verification"] = {"ok": ok, "detail": detail[:2000]}
        if ok:
            self.state["step"] = "done"
            self.state["installed"] = True
        self._record()
        return self.summary()

    def _record(self) -> None:
        self.state.setdefault("history", []).append(
            {"step": self.step, "at": datetime.now(timezone.utc).isoformat()}
        )
        self._save()

    def summary(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "skill_id": self.state.get("skill_id"),
            "version": self.state.get("version"),
            "step": self.step,
            "installed": self.installed,
            "files_installed": self.files_installed,
            "prerequisites": self.state.get("prerequisites", []),
            "missing": [r for r in self.state.get("prerequisites", []) if r.get("status") != "present"],
            "setup_instructions": (self.state.get("package") or {}).get("setup_instructions", []),
            "credential_handoff": (self.state.get("package") or {}).get("credential_handoff", []),
            "verification": self.state.get("verification"),
            "message": (
                "Installed and verified."
                if self.installed
                else "Files installed; setup verification is incomplete."
                if self.files_installed
                else "Not installed yet: setup and installation are still in progress."
            ),
        }
