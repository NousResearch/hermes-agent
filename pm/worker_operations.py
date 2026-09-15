"""Import-light routing and prerequisite policy for the private PM worker.

PM's own runtime is deliberately absent: its public staging entry point must
run directly, before the worker's dependencies exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal


@dataclass(frozen=True)
class Operation:
    module: str
    packages: tuple[str, ...] | None  # None selects the request's named package.
    bootstrap: Literal["always", "policy", "state", "never"]

    def resolve(self, name: str):
        return getattr(import_module(self.module), name)


OPERATIONS = {
    "ensure": Operation("pm.ensure", None, "state"),
    "stage_only": Operation("pm.ensure", None, "always"),
    "stage_tools": Operation("pm.build_operations", None, "never"),
    "prepare_tools": Operation("pm.build_operations", None, "always"),
    "sync_venv": Operation("pm.ensure", ("venv",), "policy"),
    "venv_is_current": Operation("pm.ensure", ("venv",), "never"),
    "build_environment": Operation("pm.operations", ("uv",), "policy"),
    "lock_project": Operation("pm.operations", ("uv",), "policy"),
    "ensure_environment": Operation("pm.operations", ("uv",), "policy"),
    "ensure_python_tool": Operation("pm.operations", ("uv",), "policy"),
    "check_project_lock": Operation("pm.build_operations", ("uv",), "policy"),
    "export_requirements": Operation("pm.build_operations", ("uv",), "policy"),
    "build_requirements_environment": Operation("pm.build_operations", ("uv",), "policy"),
    "prune_cache": Operation("pm.build_operations", ("uv",), "always"),
}
