"""Lightweight graph schema validation for the UA Flywheel code-scan module.

Provides NodeType/EdgeType enums, alias maps for normalizing external input,
and validation functions that return structured issue/warning lists.

Warning severity taxonomy (UA-002):
- INFO: orphan documentation files (docs/, README, CHANGELOG, .md, .txt, .rst)
- MINOR: orphan asset or fixture files (tests/fixtures, assets/, data/, .json, .yaml)
- MODERATE: orphan source files not matched by INFO/MINOR heuristics
- MAJOR: only when deterministic heuristics support suspicious isolated source
  (not assigned by LLM intuition)

Python stdlib only — no external dependencies.
"""
from enum import Enum
from typing import Dict, List, Set


class NodeType(str, Enum):
    """Valid node types in the dependency/structure graph."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    # Reserved for Phase 3+:
    # INTERFACE = "interface"
    # CONFIG = "config"


class EdgeType(str, Enum):
    """Valid edge types in the dependency/structure graph."""
    IMPORTS = "imports"
    CONTAINS = "contains"
    CALLS = "calls"
    TESTED_BY = "tested_by"
    CONFIGURES = "configures"
    DOCUMENTS = "documents"


# Alias map for normalizing LLM or external input
NODE_TYPE_ALIASES: Dict[str, NodeType] = {
    "func": NodeType.FUNCTION, "fn": NodeType.FUNCTION,
    "method": NodeType.FUNCTION,
    "file": NodeType.FILE, "src": NodeType.FILE,
    "class": NodeType.CLASS, "type": NodeType.CLASS,
    "module": NodeType.MODULE, "pkg": NodeType.MODULE, "package": NodeType.MODULE,
}

EDGE_TYPE_ALIASES: Dict[str, EdgeType] = {
    "import": EdgeType.IMPORTS, "imports_from": EdgeType.IMPORTS,
    "contains": EdgeType.CONTAINS, "has": EdgeType.CONTAINS,
    "calls": EdgeType.CALLS, "invoke": EdgeType.CALLS, "invokes": EdgeType.CALLS,
    "tested_by": EdgeType.TESTED_BY, "tested_by_file": EdgeType.TESTED_BY,
    "configures": EdgeType.CONFIGURES, "configured_by": EdgeType.CONFIGURES,
    "documents": EdgeType.DOCUMENTS, "doc": EdgeType.DOCUMENTS,
}


class WarningSeverity(str, Enum):
    """Deterministic severity levels for validation warnings.

    Values: INFO, MINOR, MODERATE, MAJOR.
    Severity comes from deterministic script output — never LLM intuition.
    """
    INFO = "info"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


def _resolve_node_type(raw: str):
    """Resolve a raw string to a NodeType via direct value or alias.

    Returns the NodeType or None if unresolvable.
    """
    # Try direct enum value match
    try:
        return NodeType(raw)
    except (ValueError, TypeError):
        pass

    # Try alias lookup
    return NODE_TYPE_ALIASES.get(raw)


def _resolve_edge_type(raw: str):
    """Resolve a raw string to an EdgeType via direct value or alias.

    Returns the EdgeType or None if unresolvable.
    """
    try:
        return EdgeType(raw)
    except (ValueError, TypeError):
        pass

    return EDGE_TYPE_ALIASES.get(raw)


def validate_node(node: dict) -> List[str]:
    """Validate a node dict. Returns list of issue strings (empty = valid).

    Checks:
    - node_type is present and resolves to a valid NodeType
    - node_id is present and is a string
    - filePath is present
    """
    issues: List[str] = []

    # Check node_type
    raw_type = node.get("node_type")
    if raw_type is None:
        issues.append("Missing required field: node_type")
    elif _resolve_node_type(raw_type) is None:
        issues.append(f"Unknown node_type: '{raw_type}'")

    # Check node_id
    node_id = node.get("node_id")
    if node_id is None:
        issues.append("Missing required field: node_id")
    elif not isinstance(node_id, str):
        issues.append(f"node_id must be a string, got {type(node_id).__name__}")

    # Check filePath
    if "filePath" not in node:
        issues.append("Missing required field: filePath")

    return issues


def validate_edge(edge: dict, known_node_ids: Set[str]) -> List[str]:
    """Validate an edge dict against known node IDs. Returns issue list.

    Checks:
    - edge_type is present and resolves to a valid EdgeType
    - source and target reference existing node IDs
    - No self-referencing edges
    """
    issues: List[str] = []

    # Check edge_type
    raw_type = edge.get("edge_type")
    if raw_type is None:
        issues.append("Missing required field: edge_type")
    elif _resolve_edge_type(raw_type) is None:
        issues.append(f"Unknown edge_type: '{raw_type}'")

    # Check source
    source = edge.get("source")
    if source is None:
        issues.append("Missing required field: source")
    elif source not in known_node_ids:
        issues.append(f"Edge source '{source}' does not match any known node")

    # Check target
    target = edge.get("target")
    if target is None:
        issues.append("Missing required field: target")
    elif target not in known_node_ids:
        issues.append(f"Edge target '{target}' does not match any known node")

    # Self-referencing check
    if source is not None and source == target:
        issues.append(f"Self-referencing edge: source and target are both '{source}'")

    return issues


def _classify_orphan_severity(node_id: str, node_data: dict) -> WarningSeverity:
    """Classify the severity of an orphan node warning using deterministic heuristics.

    Heuristics (in priority order):
    1. INFO — documentation paths: docs/, README, CHANGELOG, .md, .txt, .rst files
    2. MINOR — fixture/asset paths: tests/fixtures, assets/, data/, .json, .yaml files
    3. MODERATE — isolated source files not matched above
    4. MAJOR — only when specific suspicious patterns detected (none enabled yet)
    5. Default for unmatched paths here: MODERATE; warnings without file paths
       fall back to INFO in classify_warning_severity().
    """
    file_path = node_data.get("filePath", node_id)

    # INFO: documentation files
    info_patterns = [
        "/docs/", "docs/",  # in docs/ directory
        "readme", "license", "contributing",  # root docs (not CHANGELOG)
        ".md", ".txt", ".rst",  # doc extensions
    ]
    file_lower = file_path.lower()
    for pattern in info_patterns:
        if pattern in file_lower:
            return WarningSeverity.INFO

    # MINOR: fixture/asset/test-data files
    minor_patterns = [
        "/fixtures/", "/tests/fixtures",  # test fixtures
        "/assets/", "assets/",  # asset directories
        "/data/", "data/",  # data directories
        ".json", ".yaml", ".yml", ".xml", ".csv", ".toml",  # data extensions
        ".png", ".jpg", ".svg", ".gif",  # image assets
        "changelog",  # orphan changelog files (without .md extension)
    ]
    for pattern in minor_patterns:
        if pattern in file_lower:
            return WarningSeverity.MINOR

    return WarningSeverity.MODERATE


def classify_warning_severity(warning_message: str, node_data: dict) -> WarningSeverity:
    """Classify the severity of a warning message deterministically.

    This is the public classification function. It never uses LLM intuition —
    severity comes entirely from deterministic heuristics applied to node metadata.
    """
    # Only handle orphan node warnings via node data heuristics
    if node_data.get("filePath"):
        return _classify_orphan_severity(
            node_data.get("node_id", ""), node_data
        )
    # Default for unmatched warnings
    return WarningSeverity.INFO


def build_warning_summary(classified_warnings: List[dict]) -> Dict[str, int]:
    """Build a severity summary from classified warning entries.

    Args:
        classified_warnings: List of dicts with 'severity' key containing
            a WarningSeverity value.

    Returns:
        Dict mapping severity string to count, e.g.
        {"INFO": 3, "MINOR": 1, "MODERATE": 0, "MAJOR": 0}
    """
    summary = {s.value: 0 for s in WarningSeverity}
    for entry in classified_warnings:
        sev = entry.get("severity", WarningSeverity.INFO.value)
        if sev in summary:
            summary[sev] += 1
    return summary


def validate_graph(graph: dict) -> dict:
    """Validate an entire graph dict. Returns {
        "issues": [...], "warnings": [...],
        "severity_summary": {"info": N, "minor": N, "moderate": N, "major": N},
        "severity_classified_warnings": [{"severity": ..., "message": ...}, ...]
    }.

    Validates all nodes and edges, then reports orphan nodes (not referenced
    by any edge) as warnings with deterministic severity classification.

    Backward compatibility: 'issues' and 'warnings' are always present as
    lists of strings, matching the pre-UA-002 contract.
    """
    result = {
        "issues": [],
        "warnings": [],
        "severity_summary": {s.value: 0 for s in WarningSeverity},
        "severity_classified_warnings": [],
    }

    # Require nodes and edges keys
    if "nodes" not in graph:
        result["issues"].append("Missing required key: nodes")
        return result
    if "edges" not in graph:
        result["issues"].append("Missing required key: edges")
        return result

    nodes = graph["nodes"]
    edges = graph["edges"]

    # Validate each node
    node_ids: Set[str] = set()
    _node_id_to_data: Dict[str, dict] = {}
    for node in nodes:
        node_id = node.get("node_id")
        if node_id is not None:
            node_ids.add(node_id)
            _node_id_to_data[node_id] = node
        node_issues = validate_node(node)
        result["issues"].extend(node_issues)

    # Validate each edge
    referenced_nodes: Set[str] = set()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source:
            referenced_nodes.add(source)
        if target:
            referenced_nodes.add(target)
        edge_issues = validate_edge(edge, node_ids)
        result["issues"].extend(edge_issues)

    # Report orphan nodes (not referenced by any edge) as warnings with severity
    orphan_ids = node_ids - referenced_nodes
    for oid in sorted(orphan_ids):
        warning_msg = f"Orphan node: '{oid}' is not referenced by any edge"
        result["warnings"].append(warning_msg)

        # Classify severity deterministically using node data
        node_data = _node_id_to_data.get(oid, {})
        severity = classify_warning_severity(warning_msg, node_data)
        result["severity_classified_warnings"].append({
            "severity": severity.value,
            "message": warning_msg,
        })
        result["severity_summary"][severity.value] += 1

    return result
