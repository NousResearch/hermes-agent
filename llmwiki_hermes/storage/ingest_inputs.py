"""Input loading and file conversion for ingest."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from llmwiki_hermes.errors import IngestInputError

NATIVE_SOURCE_TYPES: dict[str, str] = {
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
}

MARKITDOWN_SOURCE_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".xml": "xml",
}

SUPPORTED_SOURCE_TYPES = NATIVE_SOURCE_TYPES | MARKITDOWN_SOURCE_TYPES


def flatten_json_text(raw: str) -> str:
    """Render JSON deterministically for downstream note generation."""

    payload = json.loads(raw)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class LoadedIngestInput:
    """A single ingest payload that has already been converted to text."""

    path: Path | None
    raw_content: str
    detected_source_type: str


@dataclass(frozen=True)
class FailedIngestInput:
    """A single ingest payload that could not be loaded."""

    path: str
    source_type: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "source_type": self.source_type,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ResolvedIngestInputs:
    """A batch of ingest inputs with partial-success reporting."""

    loaded_inputs: list[LoadedIngestInput]
    failed_inputs: list[FailedIngestInput]

    @property
    def processed_inputs(self) -> int:
        return len(self.loaded_inputs) + len(self.failed_inputs)


class IngestInputLoader:
    """Resolve files or stdin into normalized text inputs for the compiler."""

    def resolve(
        self,
        *,
        path: Path | None,
        stdin: bool,
        recursive: bool,
    ) -> ResolvedIngestInputs:
        if stdin:
            return ResolvedIngestInputs(
                loaded_inputs=[
                    LoadedIngestInput(
                        path=None,
                        raw_content=sys.stdin.read(),
                        detected_source_type="text",
                    )
                ],
                failed_inputs=[],
            )
        if path is None:
            raise IngestInputError("Pass a path or use --stdin.")

        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            pattern = "**/*" if recursive else "*"
            paths = [item for item in sorted(resolved.glob(pattern)) if item.is_file()]
            return self._resolve_paths(paths)
        return self._resolve_paths([resolved])

    def _resolve_paths(self, paths: list[Path]) -> ResolvedIngestInputs:
        loaded_inputs: list[LoadedIngestInput] = []
        failed_inputs: list[FailedIngestInput] = []

        for input_path in paths:
            loaded, failed = self._load_path(input_path)
            if loaded is not None:
                loaded_inputs.append(loaded)
            if failed is not None:
                failed_inputs.append(failed)

        return ResolvedIngestInputs(loaded_inputs=loaded_inputs, failed_inputs=failed_inputs)

    def _load_path(self, path: Path) -> tuple[LoadedIngestInput | None, FailedIngestInput | None]:
        suffix = path.suffix.lower()
        source_type = SUPPORTED_SOURCE_TYPES.get(suffix)
        if source_type is None:
            return None, FailedIngestInput(
                path=str(path),
                source_type="unknown",
                code="unsupported_input_type",
                message=f"Unsupported ingest input type: {suffix or '<no suffix>'}.",
            )

        if suffix in NATIVE_SOURCE_TYPES:
            raw_content = path.read_text(encoding="utf-8")
            if suffix == ".json":
                raw_content = flatten_json_text(raw_content)
            return (
                LoadedIngestInput(
                    path=path,
                    raw_content=raw_content,
                    detected_source_type=source_type,
                ),
                None,
            )

        try:
            raw_content = self._convert_with_markitdown(path)
        except _MarkItDownImportError as exc:
            return None, FailedIngestInput(
                path=str(path),
                source_type=source_type,
                code="markitdown_not_installed",
                message=str(exc),
            )
        except _MarkItDownConversionError as exc:
            return None, FailedIngestInput(
                path=str(path),
                source_type=source_type,
                code=exc.code,
                message=str(exc),
            )

        return (
            LoadedIngestInput(
                path=path,
                raw_content=raw_content,
                detected_source_type=source_type,
            ),
            None,
        )

    def _convert_with_markitdown(self, path: Path) -> str:
        try:
            from markitdown import MarkItDown, MissingDependencyException
        except ImportError as exc:  # pragma: no cover - covered via monkeypatch in tests
            raise _MarkItDownImportError(
                "MarkItDown support is not installed. Install llmwiki-hermes[markitdown]."
            ) from exc

        try:
            result = MarkItDown(enable_plugins=False).convert(path)
        except MissingDependencyException as exc:
            raise _MarkItDownConversionError(
                "markitdown_not_installed",
                (
                    f"MarkItDown is installed but missing converters for {path.suffix.lower()}. "
                    "Install llmwiki-hermes[markitdown]."
                ),
            ) from exc
        except Exception as exc:
            raise _MarkItDownConversionError(
                "markitdown_conversion_failed",
                f"MarkItDown failed to convert {path}: {exc}",
            ) from exc
        return result.text_content


class _MarkItDownImportError(Exception):
    """Raised when the MarkItDown dependency is unavailable."""


class _MarkItDownConversionError(Exception):
    """Raised when MarkItDown cannot convert a supported file."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
