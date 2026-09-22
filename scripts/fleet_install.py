"""Build and validate the standalone stock-Hermes fleet runner bundle."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


_BUNDLE_MODULES = (
    "fleet_protocol.py",
    "fleet_store.py",
    "fleet_client.py",
    "fleet_runner.py",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_runner_bundle(output: str | Path, source_root: str | Path | None = None) -> Path:
    """Build a zipapp containing only the runner and its stdlib modules."""
    root = Path(source_root) if source_root is not None else _root()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("__main__.py", "from fleet_runner import main\nraise SystemExit(main())\n")
        archive.write(root / "scripts" / "fleet_runner.py", "fleet_runner.py")
        archive.write(root / "hermes_cli" / "__init__.py", "hermes_cli/__init__.py")
        for module in _BUNDLE_MODULES:
            archive.write(root / "hermes_cli" / module, f"hermes_cli/{module}")
    return output_path


def resolve_hermes_executable(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"Hermes executable does not exist: {candidate}")
    return str(candidate)


def liveness_marker_is_active(path: str | Path) -> bool:
    return Path(path).is_file()


def runner_command(python_executable: str | Path, bundle: str | Path, node_id: str) -> list[str]:
    return [str(python_executable), str(bundle), "--node-id", node_id]

