"""Read-only status calculation for memory-integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

from .config import DEFAULT_MEMORY_SUBDIR, load_memory_integration_config

_MAX_FIELD_CHARS = 256


def _bounded_text(value: str | None, diagnostics: list[str], *, max_chars: int = _MAX_FIELD_CHARS) -> str | None:
    if value is None or len(value) <= max_chars:
        return value
    if "output_truncated" not in diagnostics:
        diagnostics.append("output_truncated")
    return value[: max_chars - 3] + "..."


def _display_path(path: Path | None, include_absolute: bool, diagnostics: list[str]) -> str | None:
    if path is None:
        return None
    if not include_absolute:
        return "<redacted>"
    return _bounded_text(str(path), diagnostics)


def _resolve_hermes_home(hermes_home: str | Path | None) -> Path:
    return Path(hermes_home) if hermes_home is not None else get_hermes_home()


def build_status(
    *,
    config: Mapping[str, Any] | None = None,
    hermes_home: str | Path | None = None,
    initialized: bool = False,
) -> dict[str, Any]:
    cfg = load_memory_integration_config(config)
    diagnostics: list[str] = []
    error_type: str | None = None
    configured = True

    if cfg.mode not in {"shared", "dedicated"}:
        diagnostics.append("mode_required")
        configured = False

    if cfg.vault_path is not None and not cfg.vault_path.is_absolute():
        diagnostics.append("vault_path_invalid")
        configured = False

    if cfg.mode == "dedicated" and cfg.memory_subdir != DEFAULT_MEMORY_SUBDIR:
        diagnostics.append("memory_subdir_not_allowed")
        configured = False

    vault_root: Path | None = None
    if configured:
        try:
            from vault_adapter import resolve_vault_path

            vault_root = resolve_vault_path(explicit_path=cfg.vault_path, use_env=True)
        except ImportError:
            diagnostics.append("adapter_unavailable")
            configured = False
        except Exception as exc:
            error_type = exc.__class__.__name__
            if "Path" in error_type:
                diagnostics.append("vault_path_error")
            else:
                diagnostics.append("vault_not_configured")
            configured = False

    memory_root = None
    if vault_root is not None:
        memory_root = vault_root / cfg.memory_subdir if cfg.mode == "shared" else vault_root

    home = _resolve_hermes_home(hermes_home)
    sidecar_path = home / "memory-integration" / "memory_integration.db"
    if not initialized:
        diagnostics.append("not_initialized")

    ok = configured and initialized and cfg.mode in {"shared", "dedicated"}
    result: dict[str, Any] = {
        "provider": "memory-integration",
        "ok": ok,
        "configured": configured,
        "initialized": initialized,
        "vault": {
            "mode": _bounded_text(cfg.mode, diagnostics),
            "root": _display_path(vault_root, cfg.include_absolute_paths, diagnostics),
            "memory_subdir": _bounded_text(cfg.memory_subdir, diagnostics) if cfg.mode == "shared" else None,
            "memory_root": _display_path(memory_root, cfg.include_absolute_paths, diagnostics),
            "exists": vault_root.exists() if vault_root is not None else False,
        },
        "sidecar": {
            "path": _display_path(sidecar_path, cfg.include_absolute_paths, diagnostics),
            "exists": sidecar_path.exists(),
        },
        "diagnostics": diagnostics,
    }
    if error_type is not None:
        result["error_type"] = error_type[:80]
    return result
