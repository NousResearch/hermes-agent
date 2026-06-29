from __future__ import annotations

from pathlib import Path

from .utils import read_text, strip_inline_yaml_comment, unquote_scalar

DEFAULT_OUTPUT_FOLDER = "_bmad-output"
IMPLEMENTATION_ARTIFACTS = "implementation-artifacts"
DOCS_BMAD_ARTIFACTS = Path("docs") / "bmad" / IMPLEMENTATION_ARTIFACTS


def implementation_artifacts_dir(project_root: str | Path) -> Path:
    root = Path(project_root)
    configured = _configured_artifacts_dir(root)
    if configured is not None:
        return configured
    legacy = root / DEFAULT_OUTPUT_FOLDER / IMPLEMENTATION_ARTIFACTS
    docs_bmad = root / DOCS_BMAD_ARTIFACTS
    if not _legacy_artifacts_present(root, legacy):
        if docs_bmad.is_dir() or (docs_bmad / "sprint-status.yaml").is_file():
            return docs_bmad
    return legacy


def sprint_status_path(project_root: str | Path) -> Path:
    artifacts = implementation_artifacts_dir(project_root)
    preferred = artifacts / "sprint-status.yaml"
    if preferred.is_file():
        return preferred
    legacy = Path(project_root) / DEFAULT_OUTPUT_FOLDER / "sprint-status.yaml"
    if legacy.is_file():
        return legacy
    return preferred


def implementation_artifacts_relpath(project_root: str | Path) -> str:
    root = Path(project_root)
    artifacts = implementation_artifacts_dir(root)
    try:
        return artifacts.relative_to(root).as_posix()
    except ValueError:
        return artifacts.resolve().relative_to(root.resolve()).as_posix()


def implementation_artifacts_glob(project_root: str | Path, pattern: str) -> str:
    return f"{implementation_artifacts_relpath(project_root)}/{pattern}"


def resolve_artifact_glob(project_root: str | Path, pattern: str) -> tuple[Path, str]:
    root = Path(project_root)
    root_resolved = root.resolve()
    artifacts_root = implementation_artifacts_dir(root)
    legacy_artifacts_root = root / DEFAULT_OUTPUT_FOLDER / IMPLEMENTATION_ARTIFACTS
    raw = Path(pattern)
    if raw.is_absolute():
        raise ValueError("success.config.glob must be relative to implementation artifacts")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("success.config.glob escapes project root") from exc
    for allowed_root in (artifacts_root, legacy_artifacts_root):
        try:
            relative = resolved.relative_to(allowed_root.resolve())
        except ValueError:
            continue
        return allowed_root, str(relative)
    raise ValueError("success.config.glob must stay within _bmad-output/implementation-artifacts or resolved implementation artifacts")


def _configured_artifacts_dir(project_root: Path) -> Path | None:
    config = _read_bmad_config(project_root / "_bmad" / "bmm" / "config.yaml")
    output_folder = config.get("output_folder") or DEFAULT_OUTPUT_FOLDER
    artifacts = config.get("implementation_artifacts")
    if not artifacts and "output_folder" in config:
        artifacts = f"{{output_folder}}/{IMPLEMENTATION_ARTIFACTS}"
    if not artifacts:
        return None
    artifacts = artifacts.replace("{project-root}", ".")
    output_folder = output_folder.replace("{project-root}", ".")
    artifacts = artifacts.replace("{output_folder}", output_folder)
    path = _project_relative_path(project_root, artifacts)
    if path is None:
        raise ValueError("BMAD config implementation_artifacts must stay within project root")
    return path


def _legacy_artifacts_present(project_root: Path, legacy: Path) -> bool:
    if (project_root / DEFAULT_OUTPUT_FOLDER / "sprint-status.yaml").is_file():
        return True
    if not legacy.is_dir():
        return False
    return any(legacy.glob("*.md")) or (legacy / "sprint-status.yaml").is_file()


def _read_bmad_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in read_text(path).splitlines():
        line = strip_inline_yaml_comment(raw_line)
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in {"output_folder", "implementation_artifacts"}:
            continue
        value = unquote_scalar(value.strip())
        if value:
            values[key] = value
    return values


def _project_relative_path(project_root: Path, value: str) -> Path | None:
    raw = Path(value)
    if raw.is_absolute():
        return None
    resolved = (project_root / raw).resolve()
    project_root_resolved = project_root.resolve()
    try:
        relative = resolved.relative_to(project_root_resolved)
    except ValueError:
        return None
    return project_root / relative
