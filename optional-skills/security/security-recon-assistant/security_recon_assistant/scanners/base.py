from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.executor import Executor
from ..core.models import ScanFinding, ScanResult


class BaseScanner(ABC):
    name: str = "base"
    description: str = "Base scanner interface"

    @abstractmethod
    def scan(self, target: str, executor: Executor | None = None, **kwargs: Any) -> ScanResult:
        raise NotImplementedError


__all__ = ["BaseScanner", "ScanResult", "ScanFinding"]
