"""Base class for audit backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AuditBackend(ABC):
    """
    Abstract base class for audit backends.

    Audit backends receive audit events and persist them
    in some way (file, webhook, etc.).
    """

    @abstractmethod
    def emit(self, event: Dict[str, Any]) -> None:
        """
        Emit an audit event to this backend.

        This method should be non-blocking and return immediately.
        Backends should queue events internally and batch writes.

        Args:
            event: Audit event dict
        """
        pass

    def flush(self) -> None:
        """
        Flush any pending events.

        Called during shutdown or when forcing a flush.
        Default implementation is no-op.
        """
        pass

    def close(self) -> None:
        """
        Close the backend and flush pending events.

        Called during application shutdown.
        Default implementation calls flush().
        """
        self.flush()