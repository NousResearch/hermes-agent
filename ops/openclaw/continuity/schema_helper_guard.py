from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable


def load_contract_schema_validator(helper_path: Path) -> tuple[Callable[..., Any] | None, Exception | None]:
    """Load schema_contract_validation sidecar from an explicit file path.

    Returns a (callable, error) tuple so callers can fail-close with a
    deterministic contract-specific error shape.
    """

    if not helper_path.exists():
        return None, FileNotFoundError(str(helper_path))

    try:
        spec = importlib.util.spec_from_file_location(
            "openclaw_schema_contract_validation_sidecar",
            helper_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("schema_helper_spec_unavailable")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validator = getattr(module, "validate_contract_payload_schema", None)
        if not callable(validator):
            raise AttributeError("validate_contract_payload_schema_missing")

        return validator, None
    except Exception as exc:  # pragma: no cover - import wiring verified in caller tests
        return None, exc


def format_schema_helper_unavailable_error(
    *,
    contract_prefix: str,
    helper_path: Path,
    import_error: Exception | None,
) -> str:
    import_error_name = type(import_error).__name__ if import_error is not None else "unknown"
    import_failure_reason = "missing_sidecar" if isinstance(import_error, FileNotFoundError) else "import_failed"
    return (
        f"{contract_prefix}_schema_helper_unavailable:"
        f"helper_path={helper_path}:"
        f"reason={import_failure_reason}:"
        f"error={import_error_name}"
    )
