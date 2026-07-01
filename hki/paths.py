"""Path resolution for project-scoped HKI artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HKI_RELATIVE_DIR = Path(".hermes") / "hki"


@dataclass(frozen=True)
class HkiScope:
    """Resolved workspace scope for HKI commands."""

    cwd: Path
    root: Path
    output_dir: Path

    @property
    def inventory_path(self) -> Path:
        return self.output_dir / "inventory.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def sources_report_path(self) -> Path:
        return self.reports_dir / "sources.md"


def resolve_scope(cwd: str | Path | None = None) -> HkiScope:
    """Resolve the HKI workspace root from ``cwd``.

    The first slice is intentionally cwd-rooted. Project DB aware resolution can
    be added here later without changing the CLI or the HKI domain modules.
    """

    raw = Path.cwd() if cwd is None else Path(cwd).expanduser()
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"cwd does not exist: {raw}") from exc

    if not resolved.is_dir():
        raise ValueError(f"cwd is not a directory: {resolved}")

    root = resolved
    return HkiScope(cwd=resolved, root=root, output_dir=root / HKI_RELATIVE_DIR)


def ensure_output_dir(scope: HkiScope) -> Path:
    """Create and return the HKI output directory for ``scope``."""

    scope.output_dir.mkdir(parents=True, exist_ok=True)
    return scope.output_dir


def as_workspace_relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
