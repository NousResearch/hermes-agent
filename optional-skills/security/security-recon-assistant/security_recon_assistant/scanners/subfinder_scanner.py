from __future__ import annotations

import json
from typing import Iterable, Optional

from ..core.dependencies import check
from ..core.executor import ExecutionResult, Executor
from .base import BaseScanner, ScanFinding, ScanResult


class SubfinderScanner(BaseScanner):
    name = "subfinder"
    description = "Enumerate subdomains with subfinder"

    def build_command(
        self,
        target: str,
        rate_limit: int = 50,
        excluded_sources: Optional[Iterable[str]] = None,
        recursive: bool = False,
    ) -> str:
        parts = ["subfinder", f"-d {target}", "-silent", "-oJ", f"-rate-limit {int(rate_limit)}"]
        if excluded_sources:
            parts.append(f"-exclude-sources {','.join(excluded_sources)}")
        if recursive:
            parts.append("-recursive")
        return " ".join(parts)

    def parse_output(self, output: str, target: str, command: str | None = None) -> ScanResult:
        if not output.strip():
            return ScanResult(scanner_name=self.name, success=True, command=command, findings=[])

        findings: list[ScanFinding] = []
        try:
            raw = output.strip()
            parsed = json.loads(raw)
            records = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            records = []
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    return ScanResult(
                        scanner_name=self.name,
                        success=False,
                        command=command,
                        stderr=f"Failed to parse subfinder output: {exc}",
                        findings=[],
                    )

        for row in records:
            if not isinstance(row, dict):
                continue
            host = str(row.get("host") or "").strip()
            if not host:
                continue
            findings.append(
                ScanFinding(
                    target=host,
                    severity="low",
                    title="Subdomain discovered",
                    description=f"Subdomain discovered for {target}",
                    evidence=json.dumps(row, ensure_ascii=False),
                    remediation="Inventory the asset and validate exposure.",
                )
            )

        return ScanResult(scanner_name=self.name, success=True, command=command, findings=findings)

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        executor = executor or Executor()
        command = self.build_command(target, **{k: v for k, v in kwargs.items() if k in {"rate_limit", "excluded_sources", "recursive"}})

        if executor is None and not check("subfinder"):
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                stderr="subfinder not installed",
                findings=[],
            )

        if isinstance(executor, Executor) and not check("subfinder"):
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                stderr="subfinder not installed",
                findings=[],
            )

        execution: ExecutionResult = executor.run(command)
        if not execution.success:
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                execution_time=execution.duration,
                stdout=execution.stdout,
                stderr=execution.stderr,
                exit_code=execution.exit_code,
                timeout=execution.timeout,
                findings=[],
            )

        parsed = self.parse_output(execution.stdout, target, command=command)
        parsed.execution_time = execution.duration
        parsed.stdout = execution.stdout
        parsed.stderr = execution.stderr
        parsed.exit_code = execution.exit_code
        return parsed


Scanner = SubfinderScanner
