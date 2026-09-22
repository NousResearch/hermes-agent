"""Compatibility runner that delegates execution to a local stock Hermes CLI."""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from hermes_cli.fleet_protocol import RunnerCapability


class FleetRunner:
    """Claim coordinator work only while the owning local Hermes is live.

    ``coordinator`` is intentionally duck-typed so the runner can use the
    normal HTTP client in production and a deterministic fake in tests.
    """

    def __init__(
        self,
        node_id: str,
        coordinator: Any,
        hermes_executable: str,
        profiles: Iterable[str],
        projects: Iterable[str],
        *,
        capabilities: Iterable[RunnerCapability] | None = None,
        models: Iterable[str] = (),
        tools: Iterable[str] = ("terminal", "git"),
        liveness_check: Callable[[], bool] | None = None,
        executor: Callable[[list[str]], Any] | None = None,
    ):
        self.node_id = node_id
        self.coordinator = coordinator
        self.hermes_executable = hermes_executable
        self.profiles = tuple(profiles)
        self.projects = tuple(projects)
        self.capabilities = tuple(capabilities or self._default_capabilities(models, tools))
        self._liveness_check = liveness_check or (lambda: True)
        self._executor = executor or self._execute
        self._stop = threading.Event()
        self._delivered: dict[str, str] = {}
        self._results: dict[str, str] = {}

    def _default_capabilities(self, models: Iterable[str], tools: Iterable[str]):
        current_platform = "windows" if platform.system().lower() == "windows" else platform.system().lower()
        return tuple(
            RunnerCapability(
                node_id=self.node_id,
                profile=profile,
                models=tuple(models),
                tools=tuple(tools),
                projects=self.projects,
                platform=current_platform,
            )
            for profile in self.profiles
        )

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def register(self) -> bool:
        if not self._is_live():
            return False
        for capability in self.capabilities:
            self.coordinator.register_runner(capability)
        return True

    def run_once(self, *, now: float | None = None) -> bool:
        if self.stopped or not self._is_live():
            return False
        claimed = False
        for capability in self.capabilities:
            self.coordinator.heartbeat_runner(capability, now=now)
            claim = self.coordinator.claim_task(capability, now=now)
            if claim is None:
                continue
            claimed = True
            task = self._find_task(claim.task_id)
            if task is None:
                self.coordinator.fail_task(
                    claim.task_id, claim.claim_id, error="coordinator returned an unknown task", now=now
                )
                continue
            if claim.task_id in self._delivered:
                self.coordinator.complete_task(
                    claim.task_id,
                    claim.claim_id,
                    result=self._results.get(claim.task_id, "duplicate delivery"),
                    now=now,
                )
                continue
            self._delivered[claim.task_id] = claim.claim_id
            result = self._execute_task(task, claim.runner_profile)
            if result.returncode == 0:
                text = (result.stdout or "").strip()
                self._results[claim.task_id] = text
                self.coordinator.complete_task(claim.task_id, claim.claim_id, result=text, now=now)
            else:
                error = (result.stderr or result.stdout or "Hermes exited with a failure").strip()
                self.coordinator.fail_task(claim.task_id, claim.claim_id, error=error, now=now)
        return claimed

    def run_forever(self, *, interval: float = 5.0) -> None:
        while not self.stopped:
            if self._is_live():
                self.register()
                self.run_once()
            self._stop.wait(interval)

    def _is_live(self) -> bool:
        return bool(self._liveness_check())

    def _find_task(self, task_id: str):
        for task in self.coordinator.list_tasks():
            candidate_id = task.get("task_id") if isinstance(task, dict) else task.task_id
            if candidate_id == task_id:
                return task
        return None

    def _execute_task(self, task: Any, profile: str):
        body = task.get("body") if isinstance(task, dict) else task.body
        return self._executor([self.hermes_executable, "-p", profile, "-z", body])

    @staticmethod
    def _execute(argv: list[str]):
        return subprocess.run(argv, check=False, capture_output=True, text=True)
