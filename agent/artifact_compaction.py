"""Artifact-aware context compaction for large scientific/tool outputs.

Large OpenQP logs, Molden files, JSON dumps, and parser outputs are too
expensive to keep verbatim in the active LLM context.  This module replaces
those raw payloads with compact, reproducible summary cards and stores the
original bytes under ``$HERMES_HOME/artifacts`` using a content hash.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home
from agent.model_metadata import estimate_tokens_rough

PARSER_VERSION = "artifact-aware-compaction-v1"
DEFAULT_MIN_CHARS = 8_000
DEFAULT_MIN_TOKENS = 2_000
DEFAULT_MAX_SUMMARY_TOKENS = 800
DEFAULT_RETRIEVE_MAX_CHARS = 12_000
SUMMARY_HEAD_LINES = 18
SUMMARY_TAIL_LINES = 12
MAX_SECTIONS_IN_CARD = 18

ARTIFACT_CARD_PREFIX = "[HERMES ARTIFACT SUMMARY CARD]"
_ARTIFACT_CARD_PREFIX = ARTIFACT_CARD_PREFIX


class ArtifactCompactionResult(dict):
    """Dict report with attribute aliases for compaction statistics."""

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "compacted_count": "artifact_count",
            "tokens_before": "before_total_tokens",
            "tokens_after": "after_total_tokens",
            "before_profile": "largest_before",
            "after_profile": "largest_after",
        }
        key = aliases.get(name, name)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class DetectedSection:
    name: str
    start_line: int
    end_line: int
    kind: str


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    sha256: str
    path: str
    metadata_path: str
    parser_version: str
    timestamp: str
    artifact_type: str
    byte_count: int
    char_count: int
    line_count: int
    token_estimate: int
    detected_sections: List[Dict[str, Any]]


def artifact_root() -> Path:
    root = Path(get_hermes_home()) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _line_count(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


def _extension_for_type(kind: str) -> str:
    return {
        "openqp_log": ".openqp.log",
        "molden": ".molden",
        "cube": ".cube",
        "xyz": ".xyz",
        "pdb": ".pdb",
        "cif": ".cif",
        "json": ".json",
        "source_code": ".source.txt",
        "documentation": ".md",
        "compiler_output": ".build.log",
        "parser_output": ".txt",
        "large_text": ".txt",
    }.get(kind, ".txt")


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    if len(stripped) < 400:
        return False
    try:
        json.loads(stripped)
        return True
    except Exception:
        # Very large tool dumps are sometimes truncated/non-strict but still
        # recognizable enough to compact as JSON-ish output.
        return bool(re.search(r'"[A-Za-z0-9_ -]{2,}"\s*:', stripped[:20_000]))



SOURCE_EXTENSIONS = (".f90", ".F90", ".f", ".F", ".for", ".c", ".h", ".cpp", ".cxx", ".hpp", ".py", ".sh", ".bash", ".yaml", ".yml", ".json")
_SOURCE_LANG_HINTS = {
    "fortran": [r"^\s*(module|subroutine|function|program)\s+\w+", r"^\s*use\s+\w+", r"!\$omp"],
    "python": [r"^\s*(def|class)\s+\w+", r"^\s*import\s+\w+", r"^\s*from\s+\w+\s+import"],
    "c_cpp": [r"^\s*#\s*include\b", r"^\s*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w:<>\s\*&]+\s+[A-Za-z_]\w*\s*\([^;]*\)\s*\{"],
    "shell": [r"^\s*#!/.*\b(?:bash|sh)\b", r"^\s*[A-Za-z_]\w*\s*\(\)\s*\{"],
}


def _unwrap_tool_json_text(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return the human payload inside common tool JSON wrappers, if any."""
    try:
        obj = json.loads(text)
    except Exception:
        return text, None
    if isinstance(obj, dict):
        for key in ("content", "output", "stdout", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value, obj
    return text, obj if isinstance(obj, dict) else None


def _strip_numbered_lines(text: str) -> str:
    """Remove read_file-style ``123|`` gutters without touching normal code."""
    lines = []
    stripped_count = 0
    for line in text.splitlines():
        new = re.sub(r"^\s*\d+\|", "", line)
        if new != line:
            stripped_count += 1
        lines.append(new)
    if stripped_count >= max(3, len(lines) // 3):
        return "\n".join(lines)
    return text


def _looks_like_source(text: str) -> bool:
    payload, wrapper = _unwrap_tool_json_text(text)
    if wrapper:
        path_hint = str(wrapper.get("path") or wrapper.get("file") or wrapper.get("filename") or "")
        if path_hint.endswith(SOURCE_EXTENSIONS):
            return True
        if payload is text and not any(isinstance(wrapper.get(k), str) for k in ("content", "output", "stdout", "text")):
            return False
    src = _strip_numbered_lines(payload)
    if "diff --git" in src and re.search(r"(?m)^\+\+\+ b/.*\.(?:py|f90|F90|c|cpp|h|hpp|sh|ya?ml|json)$", src):
        return True
    lines = src.splitlines()[:400]
    if not lines:
        return False
    score = 0
    joined = "\n".join(lines)
    for patterns in _SOURCE_LANG_HINTS.values():
        if any(re.search(pat, joined, re.MULTILINE) for pat in patterns):
            score += 2
    codey = sum(1 for line in lines if re.search(r"(::|;\s*$|\{\s*$|\}\s*$|^\s*(if|do|end|call|use|def|class|return|import|from)\b)", line, re.I))
    if codey >= 8:
        score += 2
    prose = sum(1 for line in lines if len(line.split()) > 14 and not line.lstrip().startswith(("#", "!", "//", "*")))
    return score >= 2 and prose < max(25, len(lines) // 2)


def _looks_like_documentation(text: str) -> bool:
    payload, _wrapper = _unwrap_tool_json_text(text)
    head = payload[:20_000]
    headings = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", head))
    return headings >= 3 or ("```" in head and headings >= 1)


def _looks_like_scientific_artifact(text: str) -> Optional[str]:
    payload, _wrapper = _unwrap_tool_json_text(text)
    low = payload[:40_000].lower()
    if "[molden format]" in low or ("[atoms]" in low and "[gto]" in low and "[mo]" in low):
        return "molden"
    if re.search(r"(?m)^\s*\d+\s*$", payload[:2000]) and len(re.findall(r"(?m)^\s*[A-Z][a-z]?\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+", payload[:20_000])) >= 3:
        return "xyz"
    if "data_file" in low and ("loop_" in low or "_atom_site" in low):
        return "cif"
    if re.search(r"(?m)^(ATOM  |HETATM)", payload[:20_000]):
        return "pdb"
    if "cube" in low[:1000] and len(re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", payload[:20_000])) > 200:
        return "cube"
    return None


def classify_artifact_content(text: str, *, min_chars: int = DEFAULT_MIN_CHARS) -> Optional[str]:
    """Classify large incoming content by source-aware context policy type."""
    if not isinstance(text, str) or len(text) < min_chars:
        return None
    if _looks_like_source(text):
        return "source_code"
    scientific = _looks_like_scientific_artifact(text)
    if scientific:
        return scientific
    if _looks_like_json(text):
        return "json"
    if _looks_like_documentation(text):
        return "documentation"
    return None


def _source_payload(text: str) -> str:
    payload, _wrapper = _unwrap_tool_json_text(text)
    return _strip_numbered_lines(payload)


def _source_language(src: str) -> str:
    low = src[:20_000].lower()
    if re.search(r"(?m)^\s*(module|subroutine|function|program)\s+\w+", src, re.I) or "!$omp" in low:
        return "fortran"
    if re.search(r"(?m)^\s*(def|class)\s+\w+", src):
        return "python"
    if "#include" in src[:5000] or re.search(r"(?m)^\s*[A-Za-z_][\w:<>\s\*&]+\s+[A-Za-z_]\w*\s*\([^;]*\)", src):
        return "c_cpp"
    if src.startswith("#!") or re.search(r"(?m)^\s*[A-Za-z_]\w*\s*\(\)\s*\{", src):
        return "shell"
    return "source"


def _detect_source_sections(text: str) -> List[DetectedSection]:
    src = _source_payload(text)
    lines = src.splitlines()
    starts: List[Tuple[str, int]] = []
    lang = _source_language(src)
    if lang == "fortran":
        pat = re.compile(r"^\s*(module(?!\s+procedure)|subroutine|function|program|type|interface)\s+(?:,\s*[^:]+::\s*)?(\w+)?", re.I)
        for idx, line in enumerate(lines, 1):
            m = pat.match(line)
            if m:
                kind = m.group(1).lower()
                name = m.group(2) or kind
                starts.append((f"{kind} {name}", idx))
    elif lang == "python":
        for idx, line in enumerate(lines, 1):
            m = re.match(r"^\s*(def|class)\s+(\w+)", line)
            if m:
                starts.append((f"{m.group(1)} {m.group(2)}", idx))
    else:
        func = re.compile(r"^\s*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w:<>\s\*&]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$")
        for idx, line in enumerate(lines, 1):
            m = func.match(line)
            if m and m.group(1) not in {"if", "for", "while", "switch"}:
                starts.append((f"function {m.group(1)}", idx))
    return _sections_from_starts(starts[:MAX_SECTIONS_IN_CARD * 4], len(lines), "source")


def _source_index(text: str) -> Dict[str, Any]:
    src = _source_payload(text)
    lines = src.splitlines()
    modules: List[str] = []
    routines: List[str] = []
    types: List[str] = []
    interfaces: List[str] = []
    used: List[str] = []
    omp: List[str] = []
    arrays: List[str] = []
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        m = re.match(r"module\s+(?!procedure\b)(\w+)", stripped, re.I)
        if m: modules.append(m.group(1))
        m = re.match(r"(?:public\s*(?:::)?\s*)?(subroutine|function)\s+(\w+)", stripped, re.I)
        if m: routines.append(m.group(2))
        m = re.match(r"type\s*(?:,\s*[^:]+)?(?:::)?\s*(\w+)?", stripped, re.I)
        if m and m.group(1): types.append(m.group(1))
        m = re.match(r"interface\s*(\w+)?", stripped, re.I)
        if m: interfaces.append(m.group(1) or f"interface@{idx}")
        m = re.match(r"use\s*(?:,\s*[^:]+)?(?:::)?\s*(\w+)", stripped, re.I)
        if m: used.append(m.group(1))
        if stripped.lower().startswith("!$omp"):
            omp.append(f"line {idx}: {stripped[:100]}")
        if "::" in stripped and "(" in stripped:
            left, right = stripped.split("::", 1)
            if re.search(r"\b(real|integer|logical|complex|character|type)\b", left, re.I):
                for name, dims in re.findall(r"\b(\w+)\s*\(([^)]{1,80})\)", right):
                    arrays.append(f"{name}({dims})")
    def uniq(values: List[str], n: int = 40) -> List[str]:
        return list(dict.fromkeys(values))[:n]
    return {
        "language": _source_language(src),
        "line_count": len(lines),
        "modules": uniq(modules),
        "public_routines": uniq(routines),
        "contained_procedures": uniq(routines),
        "derived_types": uniq(types),
        "interfaces": uniq(interfaces),
        "used_modules": uniq(used),
        "openmp_directives": omp[:30],
        "major_arrays": uniq(arrays, 50),
    }


def _scientific_metadata(text: str, kind: str) -> Dict[str, Any]:
    payload, _wrapper = _unwrap_tool_json_text(text)
    nums = len(re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", payload[:200_000]))
    atoms = len(re.findall(r"(?m)^\s*(?:ATOM  |HETATM|[A-Z][a-z]?\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+)", payload[:200_000]))
    return {"kind": kind, "line_count": _line_count(payload), "numeric_values_sampled": nums, "atom_like_records_sampled": atoms}

def detect_artifact_type(text: str, *, min_chars: int = DEFAULT_MIN_CHARS) -> Optional[str]:
    """Return artifact type if *text* should be externalized."""
    classified = classify_artifact_content(text, min_chars=min_chars)
    if classified:
        return classified
    if not isinstance(text, str) or len(text) < min_chars:
        return None
    head = text[:40_000]
    low = head.lower()
    openqp_markers = [
        "openqp",
        "pyoqp",
        "open quantum platform",
        "mrsf-dft",
        "mrsf-dft energy gradient calculation",
        "gradient (hartree/bohr)",
        "scf",
        "scf converg",
        "total energy",
        "geometry optimization",
        "z-vector",
        "tddft",
        "alternative_scf",
    ]
    if sum(1 for marker in openqp_markers if marker in low) >= 2:
        return "openqp_log"
    parser_markers = [
        "traceback",
        "parse summary",
        "parsed records",
        "validation_controls",
        "abs_diff_ha_per_bohr",
        "parser output",
        "records processed",
    ]
    if sum(1 for marker in parser_markers if marker in low) >= 2:
        return "parser_output"
    compiler_markers = ["undefined reference", "compilation terminated", "error:", "warning:", "make:", "cmake error", "ninja:", "ld:"]
    if sum(1 for marker in compiler_markers if marker in low) >= 2:
        return "compiler_output"
    # Large line-oriented output with many repeated structured rows is still
    # worth externalizing, but avoid compacting normal prose just because it is
    # long.
    lines = text.splitlines()
    if len(lines) > 250 and (sum(1 for line in lines[:300] if len(line) > 120) > 40):
        return "large_text"
    return None


def _detect_molden_sections(lines: List[str]) -> List[DetectedSection]:
    starts: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        if re.match(r"^\s*\[[^\]]+\]", line):
            starts.append((line.strip(), idx))
    return _sections_from_starts(starts, len(lines), "molden")


def _detect_json_sections(text: str, lines: List[str]) -> List[DetectedSection]:
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    sections: List[DetectedSection] = []
    if isinstance(obj, dict):
        for key in list(obj.keys())[:MAX_SECTIONS_IN_CARD * 2]:
            pattern = re.compile(rf'^\s*{re.escape(json.dumps(str(key)))}\s*:')
            start = next((i for i, line in enumerate(lines, start=1) if pattern.search(line)), 1)
            sections.append(DetectedSection(str(key), start, start, "json_key"))
    elif isinstance(obj, list):
        sections.append(DetectedSection("json_array", 1, len(lines), "json_array"))
    else:
        starts = []
        for idx, line in enumerate(lines, start=1):
            m = re.match(r'^\s*"([^"\\]{1,80})"\s*:', line)
            if m:
                starts.append((m.group(1), idx))
        sections = _sections_from_starts(starts[:MAX_SECTIONS_IN_CARD * 2], len(lines), "json_key")
    return sections


def _detect_openqp_sections(lines: List[str]) -> List[DetectedSection]:
    patterns = [
        ("input", re.compile(r"input|coordinates|basis", re.I)),
        ("scf", re.compile(r"\bscf\b|self-consistent|alternative_scf|trah", re.I)),
        ("mrsf_tddft", re.compile(r"mrsf|tddft|excitation|target state", re.I)),
        ("z_vector", re.compile(r"z-vector|z vector|gmres|cg solver", re.I)),
        ("gradient", re.compile(r"gradient \(hartree/bohr\)|energy gradient", re.I)),
        ("timing", re.compile(r"timing|wall time|total time|normal termination", re.I)),
        ("errors", re.compile(r"error|traceback|failed|abort|segmentation", re.I)),
    ]
    starts: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines, start=1):
        for name, pat in patterns:
            if name not in seen and pat.search(line):
                starts.append((name, idx))
                seen.add(name)
    return _sections_from_starts(starts, len(lines), "openqp")


def _detect_parser_sections(lines: List[str]) -> List[DetectedSection]:
    starts: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(#{1,6}\s+|[-=]{5,}\s*$|[A-Z][A-Za-z0-9 _/-]{3,}:$)", stripped):
            starts.append((stripped[:80].strip("#:-= "), idx))
    if not starts:
        starts = [("head", 1)]
    return _sections_from_starts(starts[:MAX_SECTIONS_IN_CARD * 2], len(lines), "parser")


def _sections_from_starts(starts: Iterable[Tuple[str, int]], total_lines: int, kind: str) -> List[DetectedSection]:
    clean: List[Tuple[str, int]] = []
    seen: set[Tuple[str, int]] = set()
    for name, start in starts:
        key = (name, start)
        if start < 1 or key in seen:
            continue
        seen.add(key)
        clean.append((name or f"section_{start}", start))
    sections: List[DetectedSection] = []
    for idx, (name, start) in enumerate(clean):
        end = (clean[idx + 1][1] - 1) if idx + 1 < len(clean) else total_lines
        sections.append(DetectedSection(name=name, start_line=start, end_line=max(start, end), kind=kind))
    return sections


def detect_sections(text: str, artifact_type: str) -> List[DetectedSection]:
    lines = text.splitlines()
    if not lines:
        return []
    if artifact_type == "source_code":
        sections = _detect_source_sections(text)
    elif artifact_type == "molden":
        sections = _detect_molden_sections(lines)
    elif artifact_type in {"cube", "xyz", "pdb", "cif"}:
        sections = _sections_from_starts([("metadata", 1), ("tail", max(1, len(lines) - 40))], len(lines), artifact_type)
    elif artifact_type == "json":
        sections = _detect_json_sections(text, lines)
    elif artifact_type == "documentation":
        starts = []
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^\s{0,3}#{1,6}\s+", line):
                starts.append((line.strip(" #")[:80], idx))
        sections = _sections_from_starts(starts[:MAX_SECTIONS_IN_CARD * 2] or [("document", 1)], len(lines), "documentation")
    elif artifact_type == "openqp_log":
        sections = _detect_openqp_sections(lines)
    elif artifact_type in {"parser_output", "compiler_output"}:
        sections = _detect_parser_sections(lines)
    else:
        sections = _sections_from_starts([("head", 1), ("tail", max(1, len(lines) - 80))], len(lines), artifact_type)
    if not sections:
        sections = [DetectedSection("full", 1, len(lines), artifact_type)]
    return sections[:MAX_SECTIONS_IN_CARD]


def _sample_lines(text: str) -> Tuple[List[str], List[str]]:
    lines = text.splitlines()
    return lines[:SUMMARY_HEAD_LINES], lines[-SUMMARY_TAIL_LINES:] if len(lines) > SUMMARY_HEAD_LINES else []


def _compact_json_preview(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return ""
    if isinstance(obj, dict):
        keys = list(obj.keys())[:20]
        return "top-level keys: " + ", ".join(map(str, keys))
    if isinstance(obj, list):
        return f"array length: {len(obj)}"
    return f"json scalar: {type(obj).__name__}"


def _enforce_summary_token_limit(card: str, max_summary_tokens: Optional[int]) -> str:
    if not max_summary_tokens or max_summary_tokens <= 0:
        return card
    if estimate_tokens_rough(card) <= max_summary_tokens:
        return card
    retrieval = "Retrieval: use artifact_retrieve with artifact_id or path plus section/line_range/regex to load only the needed raw slice."
    truncation = "\n[...summary card truncated to configured artifact_max_summary_tokens...]\n"
    # estimate_tokens_rough is roughly chars/4, so use a conservative char cap.
    max_chars = max(200, int(max_summary_tokens * 3.6))
    budget = max(0, max_chars - len(retrieval) - len(truncation))
    return card[:budget].rstrip() + truncation + retrieval


def _summary_card(meta: ArtifactMetadata, text: str, *, max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS) -> str:
    head, tail = _sample_lines(text)
    sections = meta.detected_sections[:MAX_SECTIONS_IN_CARD]
    section_lines = [
        f"- {s['name']} [{s['kind']}] lines {s['start_line']}-{s['end_line']}"
        for s in sections
    ]
    preview = _compact_json_preview(text) if meta.artifact_type == "json" else ""
    if meta.artifact_type == "source_code":
        idx = _source_index(text)
        def fmt(values: List[str]) -> str:
            return ", ".join(values) if values else "(none detected)"
        section_lines = [
            f"- {s['name']} [{s['kind']}] lines {s['start_line']}-{s['end_line']}"
            for s in sections
        ]
        card = (
            f"{_ARTIFACT_CARD_PREFIX}\n"
            f"artifact_label: Source code summary\n"
            f"artifact_type: {meta.artifact_type}\n"
            f"artifact_id: {meta.artifact_id}\n"
            f"path: {meta.path}\n"
            f"sha256: {meta.sha256}\n"
            f"parser_version: {meta.parser_version}\n"
            f"timestamp: {meta.timestamp}\n"
            f"file path: {meta.path}\n"
            f"size: {meta.byte_count} bytes, {idx['line_count']} source lines\n"
            f"token_estimate: {meta.token_estimate}\n"
            f"language: {idx['language']}\n"
            "source_index:\n"
            f"- modules: {fmt(idx['modules'])}\n"
            f"- public routines: {fmt(idx['public_routines'])}\n"
            f"- contained procedures: {fmt(idx['contained_procedures'])}\n"
            f"- derived types: {fmt(idx['derived_types'])}\n"
            f"- interfaces: {fmt(idx['interfaces'])}\n"
            f"- used modules: {fmt(idx['used_modules'])}\n"
            f"- OpenMP directives: {fmt(idx['openmp_directives'])}\n"
            f"- major arrays/dimensions: {fmt(idx['major_arrays'])}\n"
            "detected_symbols:\n"
            + ("\n".join(section_lines) if section_lines else "- full")
            + "\nRetrieval: use artifact_retrieve with artifact_id/path plus section=symbol name, regex, or line range. For source files over 5,000 lines, retrieve only relevant routines."
        )
        return _enforce_summary_token_limit(card, max_summary_tokens)
    if meta.artifact_type in {"molden", "cube", "xyz", "pdb", "cif"}:
        sci = _scientific_metadata(text, meta.artifact_type)
        card = (
            f"{_ARTIFACT_CARD_PREFIX}\n"
            f"artifact_label: Scientific artifact metadata\n"
            f"artifact_type: {meta.artifact_type}\n"
            f"artifact_id: {meta.artifact_id}\n"
            f"path: {meta.path}\n"
            f"sha256: {meta.sha256}\n"
            f"parser_version: {meta.parser_version}\n"
            f"timestamp: {meta.timestamp}\n"
            f"size: {meta.byte_count} bytes, {meta.line_count} lines\n"
            f"token_estimate: {meta.token_estimate}\n"
            "metadata_summary:\n"
            f"- kind: {sci['kind']}\n"
            f"- line_count: {sci['line_count']}\n"
            f"- atom_like_records_sampled: {sci['atom_like_records_sampled']}\n"
            f"- numeric_values_sampled: {sci['numeric_values_sampled']}\n"
            "policy: raw coordinates, grids, orbital coefficients, and trajectory values are stored but not injected.\n"
            "Retrieval: use artifact_retrieve only for focused metadata/line ranges when explicitly needed."
        )
        return _enforce_summary_token_limit(card, max_summary_tokens)
    head_preview = "\n".join(line[:240] for line in head[:10])
    tail_preview = "\n".join(line[:240] for line in tail[-6:])
    artifact_label = {
        "openqp_log": "OpenQP log",
        "molden": "Molden file",
        "cube": "Cube file",
        "xyz": "XYZ coordinates",
        "pdb": "PDB structure",
        "cif": "CIF structure",
        "json": "JSON dump",
        "source_code": "Source code summary",
        "documentation": "documentation",
        "compiler_output": "compiler output",
        "parser_output": "parser output",
        "large_text": "large text artifact",
    }.get(meta.artifact_type, meta.artifact_type)
    header = (
        f"{_ARTIFACT_CARD_PREFIX}\n"
        f"artifact_label: {artifact_label}\n"
        f"artifact_type: {meta.artifact_type}\n"
        f"artifact_id: {meta.artifact_id}\n"
        f"path: {meta.path}\n"
        f"sha256: {meta.sha256}\n"
        f"parser_version: {meta.parser_version}\n"
        f"timestamp: {meta.timestamp}\n"
        f"size: {meta.byte_count} bytes, {meta.line_count} lines\n"
        f"token_estimate: {meta.token_estimate}\n"
        "detected_sections:\n"
        "Detected sections:\n"
        + ("\n".join(section_lines) if section_lines else "- full")
        + "\n"
    )
    if preview:
        header += f"Preview: {preview}\n"
    card = header + (
        "Head sample:\n"
        f"```text\n{head_preview}\n```\n"
        "Tail sample:\n"
        f"```text\n{tail_preview}\n```\n"
        "Retrieval: use artifact_retrieve with artifact_id or path plus section/line_range/regex to load only the needed raw slice."
    )
    return _enforce_summary_token_limit(card, max_summary_tokens)


def externalize_artifact(
    text: str,
    artifact_type: Optional[str] = None,
    *,
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> Tuple[str, ArtifactMetadata]:
    """Persist *text* as an artifact and return ``(summary_card, metadata)``."""
    artifact_type = artifact_type or detect_artifact_type(text) or "large_text"
    stored_text = _source_payload(text) if artifact_type == "source_code" else text
    raw = stored_text.encode("utf-8", errors="replace")
    digest = sha256(raw).hexdigest()
    root = artifact_root()
    artifact_id = digest[:16]
    data_path = root / f"{digest}{_extension_for_type(artifact_type)}"
    meta_path = root / f"{digest}.metadata.json"
    if not data_path.exists():
        data_path.write_bytes(raw)
    sections = detect_sections(stored_text, artifact_type)
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        sha256=digest,
        path=str(data_path),
        metadata_path=str(meta_path),
        parser_version=PARSER_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        artifact_type=artifact_type,
        byte_count=len(raw),
        char_count=len(stored_text),
        line_count=_line_count(stored_text),
        token_estimate=estimate_tokens_rough(stored_text),
        detected_sections=[asdict(s) for s in sections],
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _summary_card(meta, stored_text, max_summary_tokens=max_summary_tokens), meta


def _message_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    return None


def _message_token_estimate(message: Dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        tokens = estimate_tokens_rough(content)
    else:
        tokens = estimate_tokens_rough(json.dumps(content, ensure_ascii=False, default=str))
    for tc in message.get("tool_calls") or []:
        try:
            tokens += estimate_tokens_rough(json.dumps(tc, ensure_ascii=False, default=str))
        except Exception:
            tokens += estimate_tokens_rough(str(tc))
    return tokens + 8


def profile_largest_messages(messages: List[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for idx, msg in enumerate(messages):
        text = _message_text(msg.get("content"))
        token_estimate = _message_token_estimate(msg)
        rows.append({
            "index": idx,
            "role": msg.get("role", ""),
            "token_estimate": token_estimate,
            "tokens_estimate": token_estimate,
            "char_count": len(text) if text is not None else len(str(msg.get("content") or "")),
            "artifact_card": bool(text and text.startswith(_ARTIFACT_CARD_PREFIX)),
            "preview": (text or str(msg.get("content") or ""))[:120].replace("\n", " "),
        })
    rows.sort(key=lambda r: r["token_estimate"], reverse=True)
    return rows[:limit]




def _message_category(message: Dict[str, Any]) -> str:
    text = _message_text(message.get("content")) or ""
    if text.startswith(_ARTIFACT_CARD_PREFIX):
        m = re.search(r"^artifact_type:\s*(\S+)", text, re.MULTILINE)
        kind = m.group(1) if m else "artifact_card"
    else:
        kind = detect_artifact_type(text, min_chars=1000) or "conversation"
    if kind in {"openqp_log", "compiler_output", "parser_output", "large_text"}:
        return "logs"
    if kind in {"molden", "cube", "xyz", "pdb", "cif"}:
        return "scientific_artifacts"
    if kind == "json":
        return "structured_data"
    if kind == "source_code":
        return "source_code"
    if kind == "documentation":
        return "documentation"
    return kind


def context_budget_profile(messages: List[Dict[str, Any]], *, top_n: int = 20) -> Dict[str, Any]:
    total = sum(_message_token_estimate(m) for m in messages)
    by_category: Dict[str, int] = {}
    raw_large_artifact_tokens = 0
    for m in messages:
        tokens = _message_token_estimate(m)
        cat = _message_category(m)
        by_category[cat] = by_category.get(cat, 0) + tokens
        text = _message_text(m.get("content")) or ""
        if cat in {"logs", "scientific_artifacts", "structured_data"} and not text.startswith(_ARTIFACT_CARD_PREFIX):
            raw_large_artifact_tokens += tokens
    status = "ok"
    if total > 120_000:
        status = "emergency_compaction"
    elif total > 80_000:
        status = "warning"
    expected_after = total - raw_large_artifact_tokens + min(raw_large_artifact_tokens, max(0, int(total * 0.02)))
    return {
        "total_estimated_tokens": total,
        "status": status,
        "target_active_context_tokens": 60_000,
        "warning_tokens": 80_000,
        "emergency_compaction_tokens": 120_000,
        "top_20_largest_messages": profile_largest_messages(messages, limit=top_n),
        "token_contribution_by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
        "raw_logs_generated_artifacts_token_cap": int(60_000 * 0.20),
        "raw_large_artifact_tokens": raw_large_artifact_tokens,
        "expected_reduction_after_compaction": max(0, total - expected_after),
        "expected_tokens_after_compaction": expected_after,
    }

def compact_artifacts_in_messages(
    messages: List[Dict[str, Any]],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Replace large artifact-like message contents with summary cards.

    Returns ``(new_messages, profile)``.  Input messages are not mutated.
    """
    before_context_profile = context_budget_profile(messages, top_n=20)
    before = before_context_profile["top_20_largest_messages"]
    new_messages: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        text = _message_text(content)
        text_tokens = estimate_tokens_rough(text) if text else 0
        kind = (
            detect_artifact_type(text or "", min_chars=min_chars)
            if text and len(text) >= min_chars and text_tokens >= min_tokens
            else None
        )
        if text and kind and not text.startswith(_ARTIFACT_CARD_PREFIX):
            card, meta = externalize_artifact(
                text,
                kind,
                max_summary_tokens=max_summary_tokens,
            )
            new_msg = {**msg, "content": card}
            new_messages.append(new_msg)
            artifacts.append({"message_index": idx, **asdict(meta)})
        else:
            new_messages.append(msg.copy())
    after_context_profile = context_budget_profile(new_messages, top_n=20)
    after = after_context_profile["top_20_largest_messages"]
    before_tokens = sum(_message_token_estimate(m) for m in messages)
    after_tokens = sum(_message_token_estimate(m) for m in new_messages)
    profile = ArtifactCompactionResult({
        "parser_version": PARSER_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "min_chars": min_chars,
        "min_tokens": min_tokens,
        "max_summary_tokens": max_summary_tokens,
        "before_total_tokens": before_tokens,
        "after_total_tokens": after_tokens,
        "saved_tokens": max(0, before_tokens - after_tokens),
        "largest_before": before,
        "largest_after": after,
        "context_budget_before": before_context_profile,
        "context_budget_after": after_context_profile,
        "token_contribution_by_category": after_context_profile["token_contribution_by_category"],
        "top_20_largest_messages": after_context_profile["top_20_largest_messages"],
        "expected_reduction_after_compaction": before_context_profile["expected_reduction_after_compaction"],
        "artifacts": artifacts,
    })
    return new_messages, profile


def profile_message_sizes(messages: List[Dict[str, Any]], *, top_n: int = 8) -> List[Dict[str, Any]]:
    """Public profiler for largest messages, ordered by estimated tokens."""
    rows = profile_largest_messages(messages, limit=top_n)
    for row in rows:
        if "token_estimate" in row:
            row.setdefault("tokens_estimate", row["token_estimate"])
    return rows


def load_artifact_metadata(identifier: str) -> ArtifactMetadata:
    root = artifact_root()
    candidate = Path(identifier).expanduser()
    if candidate.exists() and candidate.name.endswith(".metadata.json"):
        data = json.loads(candidate.read_text(encoding="utf-8"))
    elif candidate.exists():
        digest = sha256(candidate.read_bytes()).hexdigest()
        data = json.loads((root / f"{digest}.metadata.json").read_text(encoding="utf-8"))
    else:
        matches = list(root.glob(f"{identifier}*.metadata.json"))
        if not matches:
            raise FileNotFoundError(f"No artifact metadata found for {identifier!r}")
        data = json.loads(matches[0].read_text(encoding="utf-8"))
    return ArtifactMetadata(**data)


def retrieve_artifact_section(
    identifier: Optional[str] = None,
    *,
    sha256: Optional[str] = None,
    section: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    regex: Optional[str] = None,
    context_lines: int = 20,
    max_chars: int = DEFAULT_RETRIEVE_MAX_CHARS,
) -> Dict[str, Any]:
    """Load a focused slice of a stored artifact."""
    identifier = identifier or sha256
    if not identifier:
        return {"success": False, "error": "identifier, sha256, or path is required"}
    if line_start is not None and start_line is None:
        start_line = line_start
    if line_end is not None and end_line is None:
        end_line = line_end
    meta = load_artifact_metadata(identifier)
    path = Path(meta.path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    selected_name = section
    if section:
        lowered = section.lower()
        hit = next((s for s in meta.detected_sections if s.get("name", "").lower() == lowered), None)
        if hit is None:
            hit = next((s for s in meta.detected_sections if lowered in s.get("name", "").lower()), None)
        if hit:
            start_line = int(hit["start_line"])
            end_line = int(hit["end_line"])
            selected_name = hit["name"]
    if regex:
        pat = re.compile(regex, re.MULTILINE)
        match = pat.search(text)
        if match:
            prefix = text[:match.start()]
            line_no = prefix.count("\n") + 1
            start_line = max(1, line_no - context_lines)
            end_line = min(len(lines), line_no + context_lines)
            selected_name = f"regex:{regex}"
        else:
            return {"success": False, "error": f"regex not found: {regex}", "artifact": asdict(meta)}
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = min(len(lines), start_line + 200 - 1)
    start_line = max(1, int(start_line))
    end_line = min(len(lines), int(end_line))
    snippet_lines = lines[start_line - 1:end_line]
    snippet = "\n".join(snippet_lines)
    truncated = False
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "\n[...artifact slice truncated...]"
        truncated = True
    return {
        "success": True,
        "artifact": asdict(meta),
        "artifact_id": meta.artifact_id,
        "sha256": meta.sha256,
        "path": meta.path,
        "section": selected_name or "line_range",
        "line_start": start_line,
        "line_end": end_line,
        "start_line": start_line,
        "end_line": end_line,
        "truncated": truncated,
        "content": snippet,
    }
