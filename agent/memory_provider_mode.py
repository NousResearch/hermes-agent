"""Normalize external memory provider modes for AIAgent.

Modes:
  off   — no provider init (cron default; also default when skip_memory=True)
  tools — provider loaded + tools injected; no auto prompt/prefetch/turn-sync/session retain
  full  — normal interactive provider lifecycle (default when skip_memory=False)

Legacy ``skip_memory_provider``:
  True  → off
  False → tools  (enable provider without implying full auto-sync; safer than historical global-on)
  None  → derive from memory_provider_mode / skip_memory
"""

from __future__ import annotations

from typing import Any, Optional

VALID_MEMORY_PROVIDER_MODES = frozenset({"off", "tools", "full"})


def resolve_memory_provider_mode(
    *,
    skip_memory: bool = False,
    memory_provider_mode: Optional[str] = None,
    skip_memory_provider: Optional[bool] = None,
) -> str:
    """Return a canonical mode string.

    Priority:
      1. Explicit memory_provider_mode (if non-empty)
      2. Legacy skip_memory_provider bool
      3. Default: off when skip_memory else full
    """
    if memory_provider_mode is not None and str(memory_provider_mode).strip() != "":
        mode = str(memory_provider_mode).strip().lower()
        if mode not in VALID_MEMORY_PROVIDER_MODES:
            raise ValueError(
                f"Invalid memory_provider_mode {memory_provider_mode!r}; "
                f"expected one of {sorted(VALID_MEMORY_PROVIDER_MODES)}"
            )
        return mode

    if skip_memory_provider is not None:
        return "off" if skip_memory_provider else "tools"

    return "off" if skip_memory else "full"


def normalize_job_memory_provider(value: Any) -> Optional[str]:
    """Normalize a cron job field. None/empty → None (scheduler treats as off)."""
    if value is None:
        return None
    if isinstance(value, bool):
        # Defensive: bools are not valid job values
        raise ValueError("memory_provider must be 'off', 'tools', or 'full'")
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in VALID_MEMORY_PROVIDER_MODES:
        raise ValueError(
            f"Invalid memory_provider {value!r}; "
            f"expected one of {sorted(VALID_MEMORY_PROVIDER_MODES)}"
        )
    return text


def provider_lifecycle_enabled(mode: str) -> bool:
    return mode == "full"


def provider_tools_enabled(mode: str) -> bool:
    return mode in ("tools", "full")
