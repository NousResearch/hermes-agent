"""QMD Integration — Native Hermes Anticipatory Memory Layer.

QMD (Queryable Memory Database) provides local vector-based memory
management as a strict alternative to Honcho. When enabled in config,
QMD takes precedence over Honcho settings.

Architecture:
  - QMDClientConfig: Configuration dataclass (mirrors HonchoClientConfig pattern)
  - QMDClient: HTTP client for the QMD MCP server
  - QMDSessionManager: Session-scoped memory management with anticipatory context

Usage:
  Set `memory_backend: qmd` in config.yaml to enable. This will override
  any honcho configuration. QMD uses local FAISS indexing with optional
  NousResearch embedding model support.
"""

from qmd_integration.client import QMDClientConfig, get_qmd_client, reset_qmd_client
from qmd_integration.session import QMDSessionManager

__all__ = [
    "QMDClientConfig",
    "get_qmd_client",
    "reset_qmd_client",
    "QMDSessionManager",
]
