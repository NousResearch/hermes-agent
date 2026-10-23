from ..core.dependencies import check
from ..core.executor import Executor
from .base import BaseScanner, ScanResult

class NucleiScanner(BaseScanner):
    name = "nuclei"
    description = "Template-based vulnerability scanning"

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        executor = executor or Executor()
        command = f"nuclei -u {target} -silent"
        if not check("nuclei"):
            return ScanResult(scanner_name=self.name, success=False, command=command, stderr="nuclei not installed")
        result = executor.run(command)
        return ScanResult(
            scanner_name=self.name,
            success=result.success,
            command=command,
            execution_time=result.duration,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

Scanner = NucleiScanner
