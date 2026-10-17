from ..core.dependencies import check
from ..core.executor import Executor
from .base import BaseScanner, ScanResult

class WhatwebScanner(BaseScanner):
    name = "whatweb"
    description = "Web technology fingerprinting"

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        command = f"whatweb {target}"
        if not check("whatweb"):
            return ScanResult(scanner_name=self.name, success=False, command=command, stderr="whatweb not installed")
        executor = executor or Executor()
        result = executor.run(command)
        return ScanResult(scanner_name=self.name, success=result.success, command=command, execution_time=result.duration, stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code)

Scanner = WhatwebScanner
