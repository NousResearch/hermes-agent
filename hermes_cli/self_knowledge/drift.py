"""Conservative drift checks for hand-written self-knowledge sections."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.self_knowledge.parser import parse_auto_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = PROJECT_ROOT / "context" / "self" / "hermes-agent.md"
ALLOWLIST_PATH = PROJECT_ROOT / "context" / "self" / ".hermes-agent-allowlist.txt"

BACKTICK_RE = re.compile(r"`([^`]+)`")
FILE_PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+")
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}$")


@dataclass(frozen=True)
class DriftFinding:
    kind: str
    reference: str
    location_in_doc: str
    reason: str


def _strip_auto_blocks(text: str) -> str:
    blocks = parse_auto_blocks(text)
    stripped = text
    for block in sorted(blocks.values(), key=lambda b: b.start, reverse=True):
        stripped = stripped[: block.start] + "\n" * stripped[block.start : block.end].count("\n") + stripped[block.end :]
    return stripped


def _load_allowlist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _symbol_exists(reference: str, project_root: Path) -> bool:
    parts = reference.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        module_parts = parts[:split_at]
        attrs = parts[split_at:]
        module_path = project_root.joinpath(*module_parts).with_suffix(".py")
        if not module_path.exists():
            init_path = project_root.joinpath(*module_parts, "__init__.py")
            module_path = init_path if init_path.exists() else module_path
        if not module_path.exists():
            continue
        try:
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return False
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        names.update(
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        )
        return bool(attrs and attrs[0] in names)
    return False


def check_drift(
    doc_path: Path = DOC_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    allowlist_path: Path | None = ALLOWLIST_PATH,
) -> list[DriftFinding]:
    """Return drift findings for hand-written self-knowledge references."""
    path = Path(doc_path)
    text = path.read_text(encoding="utf-8")
    hand_text = _strip_auto_blocks(text)
    allowlist = _load_allowlist(allowlist_path)
    findings: list[DriftFinding] = []

    references: list[tuple[str, int]] = []
    for match in BACKTICK_RE.finditer(hand_text):
        references.append((match.group(1).strip(), match.start(1)))
    for match in FILE_PATH_RE.finditer(hand_text):
        references.append((match.group(0).strip(), match.start(0)))

    seen: set[tuple[str, int]] = set()
    for reference, offset in references:
        if (reference, offset) in seen or reference in allowlist:
            continue
        seen.add((reference, offset))
        location = f"{path}:{_line_for_offset(hand_text, offset)}"
        if "/" in reference and not reference.startswith(("http://", "https://")):
            if not (Path(project_root) / reference).exists():
                findings.append(
                    DriftFinding("file_path", reference, location, "referenced path does not exist")
                )
        elif SYMBOL_RE.match(reference):
            if not _symbol_exists(reference, Path(project_root)):
                findings.append(
                    DriftFinding("symbol", reference, location, "referenced symbol was not found")
                )
    return findings
