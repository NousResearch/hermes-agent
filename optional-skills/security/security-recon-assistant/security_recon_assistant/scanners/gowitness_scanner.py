from ..core.dependencies import check
from ..core.executor import Executor
from .base import BaseScanner, ScanResult

class GowitnessScanner(BaseScanner):
    name = "gowitness"
    description = "Website screenshot and visual recon"

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        command = f"gowitness scan single --url {target}"
        if not check("gowitness"):
            return ScanResult(scanner_name=self.name, success=False, command=command, stderr="gowitness not installed")
        executor = executor or Executor()
        result = executor.run(command)
        return ScanResult(scanner_name=self.name, success=result.success, command=command, execution_time=result.duration, stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code)

Scanner = GowitnessScanner
