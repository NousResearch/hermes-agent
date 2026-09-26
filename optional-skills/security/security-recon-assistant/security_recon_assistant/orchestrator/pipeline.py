from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..core.executor import Executor
from ..core.guardian import Guardian, ViolationError
from ..scanners import discover
from ..scanners.base import ScanResult


@dataclass
class PipelineConfig:
    sequential: bool = True
    retry_failed: bool = False
    max_retries: int = 1
    stop_on_critical: bool = True


class Pipeline:
    def __init__(self, scanners=None, guardian: Guardian | None = None, executor: Executor | None = None, config: PipelineConfig | None = None):
        self.scanners = scanners if scanners is not None else [cls() for cls in discover().values()]
        self.guardian = guardian
        self.executor = executor or Executor()
        self.config = config or PipelineConfig()

    @staticmethod
    def _is_critical(result: ScanResult) -> bool:
        return any((getattr(finding, "severity", "").lower() == "critical") for finding in result.findings)

    def _scanner_command_for_guardian(self, scanner, target: str) -> str:
        if hasattr(scanner, "build_command"):
            try:
                return scanner.build_command(target)
            except Exception:
                return f"{scanner.name} {target}"
        return f"{scanner.name} {target}"

    def run(self, target: str) -> List[ScanResult]:
        results: list[ScanResult] = []

        for scanner in self.scanners:
            if self.guardian is not None:
                command_preview = self._scanner_command_for_guardian(scanner, target)
                try:
                    self.guardian.check_command(command_preview, {"target": target})
                except ViolationError:
                    break

            try:
                result = scanner.scan(target, self.executor)
            except Exception as exc:
                result = ScanResult(
                    scanner_name=getattr(scanner, "name", scanner.__class__.__name__.lower()),
                    success=False,
                    stderr=str(exc),
                    findings=[],
                )

            results.append(result)

            if self.config.retry_failed and not result.success:
                for _ in range(self.config.max_retries):
                    retry_result = scanner.scan(target, self.executor)
                    results.append(retry_result)
                    if retry_result.success:
                        break
                break

            if self.config.stop_on_critical and self._is_critical(result):
                break

        return results


class Orchestrator:
    def __init__(self, targets: Iterable[str], scope_path: str = "scope.yaml"):
        self.targets = list(targets)
        self.guardian = Guardian(scope_path)
        self.pipeline = Pipeline(guardian=self.guardian)

    def run(self) -> List[ScanResult]:
        all_results: list[ScanResult] = []
        for target in self.targets:
            all_results.extend(self.pipeline.run(target))
        return all_results
