"""
Hermes Core Model Provider Interface Package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ModelProvider(ABC):
    @abstractmethod
    def authenticate(self) -> bool:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass


__all__ = ["ModelProvider"]
