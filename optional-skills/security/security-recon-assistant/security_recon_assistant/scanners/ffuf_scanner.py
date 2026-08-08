from ..core.dependencies import check
from ..core.executor import Executor
from .base import BaseScanner, ScanResult

class FfufScanner(BaseScanner):
    name = "ffuf"
    description = "Web content discovery with ffuf"

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        command = f"ffuf -u {target.rstrip('/')}/FUZZ -w /usr/share/wordlists/dirb/common.txt -of json"
        if not check("ffuf"):
            return ScanResult(scanner_name=self.name, success=False, command=command, stderr="ffuf not installed")
        executor = executor or Executor()
        result = executor.run(command)
        return ScanResult(scanner_name=self.name, success=result.success, command=command, execution_time=result.duration, stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code)

Scanner = FfufScanner
