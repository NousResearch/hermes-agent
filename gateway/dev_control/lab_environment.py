"""Safety checks for the isolated Hermes Lab runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_LAB_HOME = Path("~/.oryn-lab").expanduser()
DEFAULT_LAB_HERMES_HOME = DEFAULT_LAB_HOME / "hermes-home"
DEFAULT_LAB_DB_PATH = DEFAULT_LAB_HERMES_HOME / "state.db"
STABLE_PORTS = {8642, 8643, 8644, 8646, 8647}
PRODUCTION_ROOTS = (
    Path("~/Projects/Oryn").expanduser(),
    Path("~/projects/Oryn").expanduser(),
    Path("~/Documents/Oryn.ai").expanduser(),
    Path("~/.hermes").expanduser(),
)
TRUTHY = {"1", "true", "yes", "on"}


def lab_paths_from_env() -> dict[str, str]:
    lab_home = Path(os.getenv("ORYN_LAB_HOME") or DEFAULT_LAB_HOME).expanduser()
    hermes_home = Path(os.getenv("HERMES_HOME") or os.getenv("ORYN_LAB_HERMES_HOME") or lab_home / "hermes-home").expanduser()
    return {
        "lab_home": str(lab_home),
        "hermes_home": str(hermes_home),
        "db_path": str(hermes_home / "state.db"),
        "run_dir": str(lab_home / "run"),
        "repos_dir": str(lab_home / "repos"),
        "worktrees_dir": str(lab_home / "worktrees"),
    }


def validate_lab_environment(
    *,
    hermes_home: Optional[Path | str] = None,
    gateway_port: Optional[int | str] = None,
    repo_roots: Optional[Iterable[Path | str]] = None,
    env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Return a validation report, refusing known production-footgun settings."""

    env = env or dict(os.environ)
    errors: list[str] = []
    warnings: list[str] = []
    paths = lab_paths_from_env()
    home = Path(hermes_home or env.get("HERMES_HOME") or paths["hermes_home"]).expanduser()
    port = _coerce_port(gateway_port or env.get("API_SERVER_PORT") or 8662)
    roots = [Path(root).expanduser() for root in (repo_roots or [])]

    if _is_same_or_child(home, Path("~/.hermes").expanduser()):
        errors.append(f"HERMES_HOME must not point at production ~/.hermes: {home}")
    if not _is_same_or_child(home, Path(paths["lab_home"]).expanduser()):
        errors.append(f"HERMES_HOME must stay under the lab root {paths['lab_home']}: {home}")
    if port in STABLE_PORTS:
        errors.append(f"Gateway port {port} is reserved for stable services.")
    for key in ("HERMES_DEV_MERGE_EXECUTOR_ENABLED", "HERMES_DEV_BRANCH_PROTECTION_CONFIRMED"):
        if str(env.get(key) or "").strip().lower() in TRUTHY:
            errors.append(f"{key} must be disabled in Hermes Lab.")
    for root in roots:
        for production_root in PRODUCTION_ROOTS:
            if _is_same_or_child(root, production_root):
                errors.append(f"Lab repo root must not point at production path {production_root}: {root}")

    return {
        "ok": not errors,
        "object": "hermes.dev_lab_safety",
        "lab_home": paths["lab_home"],
        "hermes_home": str(home),
        "gateway_port": port,
        "repo_roots": [str(root) for root in roots],
        "errors": errors,
        "warnings": warnings,
    }


def validate_lab_or_raise(**kwargs: Any) -> dict[str, Any]:
    result = validate_lab_environment(**kwargs)
    if not result["ok"]:
        raise RuntimeError("; ".join(result["errors"]))
    return result


def cli_report(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _coerce_port(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_same_or_child(path: Path, root: Path) -> bool:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents
