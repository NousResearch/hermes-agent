"""Boring Skill Graphs v0 helpers.

The graph is read-only runtime metadata built from SKILL.md frontmatter. Runtime
callers can rank skills and suggest one-hop explicit dependencies, but this module
never auto-loads skills and never walks dependencies recursively.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.skill_utils import parse_frontmatter
from utils import atomic_json_write

SCHEMA_VERSION = 1
VALID_LAYERS = {"atom", "molecule", "compound"}
DEFAULT_LAYER = "atom"
RISKY_WRITES = {
    "public_social",
    "social_post",
    "email_send",
    "send_email",
    "message_send",
    "sms_send",
    "qbo",
    "quickbooks",
    "production",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        return [text]
    return [value]


def _as_str_list(value: Any) -> list[str]:
    items: list[str] = []
    for item in _as_list(value):
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _skill_category(skill_md: Path, skills_root: Path) -> str:
    rel = skill_md.relative_to(skills_root)
    if len(rel.parts) <= 2:
        return "general"
    return "/".join(rel.parts[:-2])


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<yaml>.*?)(?:\n---\s*\n|\n---\s*\Z)", re.DOTALL)
_HERMES_KEY_RE = re.compile(r"^\s{4,}([A-Za-z_][\w-]*):\s*(.*)$")
_HERMES_CONTINUATION_RE = re.compile(r"^\s{6,}(layer|when|mode):\s*(.*)$")
_HERMES_LIST_ITEM_RE = re.compile(r"^\s{6,}-\s*(.*)$")
_INLINE_DEP_FIELD_RE = re.compile(r"(?:^|\s{2,})(id|skill|layer|when|mode):\s*")


def _extract_frontmatter_yaml(content: str) -> str:
    match = _FRONTMATTER_RE.search(content)
    return match.group("yaml") if match else ""


def _parse_bracket_list(value: str) -> list[str]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return [text] if text else []
    inner = text[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]


def _parse_inline_dependency_text(text: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    matches = list(_INLINE_DEP_FIELD_RE.finditer(text))
    if matches:
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            fields[match.group(1)] = text[start:end].strip()
    elif text.strip():
        fields["skill"] = text.strip()
    return _normalize_dependency(fields)


def _parse_raw_hermes_metadata(content: str) -> dict[str, Any]:
    """Best-effort parser for malformed legacy frontmatter.

    Some generated skills have YAML that falls back to flat key parsing. The
    graph should still recover boring metadata like layer, writes and explicit
    one-hop dependencies instead of silently dropping edges.
    """
    yaml_block = _extract_frontmatter_yaml(content)
    result: dict[str, Any] = {}
    current_key: str | None = None
    last_dependency: dict[str, Any] | None = None
    list_keys = {"depends_on", "suggests", "reads", "writes", "verifies", "tags", "related_skills"}

    for raw_line in yaml_block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped in {"metadata:", "hermes:"}:
            continue

        item_match = _HERMES_LIST_ITEM_RE.match(line)
        if item_match and current_key:
            item_text = item_match.group(1).strip()
            bucket = result.setdefault(current_key, [])
            if not isinstance(bucket, list):
                bucket = []
                result[current_key] = bucket
            if current_key == "depends_on":
                dependency = _parse_inline_dependency_text(item_text)
                if dependency.get("skill"):
                    bucket.append(dependency)
                    last_dependency = dependency
            elif item_text:
                bucket.append(item_text.strip().strip("'\""))
            continue

        continuation_match = _HERMES_CONTINUATION_RE.match(line)
        if continuation_match and current_key == "depends_on" and last_dependency is not None:
            last_dependency[continuation_match.group(1)] = continuation_match.group(2).strip().strip("'\"")
            continue

        key_match = _HERMES_KEY_RE.match(line)
        if not key_match:
            continue
        key, value = key_match.group(1), key_match.group(2).strip()
        if key in list_keys:
            current_key = key
            last_dependency = None
            if value and value != "[]":
                if key == "depends_on":
                    result[key] = [_parse_inline_dependency_text(item) for item in _parse_bracket_list(value)]
                else:
                    result[key] = _parse_bracket_list(value)
            else:
                result[key] = []
            continue

        current_key = None
        last_dependency = None
        result[key] = value.strip().strip("'\"") if value else None

    return result


def _hermes_metadata(frontmatter: dict[str, Any], content: str) -> dict[str, Any]:
    metadata = frontmatter.get("metadata") if isinstance(frontmatter.get("metadata"), dict) else {}
    hermes_block = metadata.get("hermes") if isinstance(metadata.get("hermes"), dict) else {}
    graph_block = hermes_block.get("graph") if isinstance(hermes_block.get("graph"), dict) else {}
    parsed = {key: value for key, value in hermes_block.items() if key != "graph"}
    parsed.update(graph_block)
    raw = _parse_raw_hermes_metadata(content)
    merged = dict(raw)
    for key, value in parsed.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    for key in (
        "layer",
        "depends_on",
        "suggests",
        "related_skills",
        "reads",
        "writes",
        "verifies",
        "approval_gate",
        "approval_required",
        "side_effects",
        "tags",
        "max_dependency_depth",
    ):
        value = frontmatter.get(key)
        if key not in merged and value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _canonical_id(source_kind: str, category: str, skill_name: str) -> str:
    return f"{source_kind}:{category}/{skill_name}"


def _normalize_dependency(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        skill = str(raw.get("skill") or raw.get("name") or raw.get("id") or "").strip()
        return {
            "skill": skill,
            "mode": str(raw.get("mode") or "explicit").strip() or "explicit",
            "when": str(raw.get("when") or "").strip(),
            "layer": str(raw.get("layer") or "").strip(),
        }
    skill = str(raw).strip()
    return {"skill": skill, "mode": "explicit", "when": "", "layer": ""}


def _normalize_suggestion(raw: Any) -> dict[str, Any]:
    suggestion = _normalize_dependency(raw)
    if not (isinstance(raw, dict) and raw.get("mode")):
        suggestion["mode"] = "advisory"
    return suggestion


def _resolve_dependency_id(dep_skill: str, name_to_id: dict[str, str], current_category: str) -> str:
    if ":" in dep_skill and "/" in dep_skill:
        return dep_skill
    if dep_skill in name_to_id:
        return name_to_id[dep_skill]
    return _canonical_id("local", current_category, dep_skill)


def build_skill_graph(skills_root: str | Path, *, generated_at: str | None = None) -> dict[str, Any]:
    """Build a deterministic read-only skill graph from a skills directory."""
    root = Path(skills_root)
    nodes: list[dict[str, Any]] = []

    for skill_md in sorted(root.rglob("SKILL.md"), key=lambda p: str(p.relative_to(root))):
        if any(part in {".graph", ".git", "__pycache__"} for part in skill_md.parts):
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _body = parse_frontmatter(content)
        except Exception:
            frontmatter = {}

        skill_name = str(frontmatter.get("name") or skill_md.parent.name).strip() or skill_md.parent.name
        description = str(frontmatter.get("description") or "").strip()
        category = _skill_category(skill_md, root)
        hermes = _hermes_metadata(frontmatter, content)
        layer = str(hermes.get("layer") or DEFAULT_LAYER).strip() or DEFAULT_LAYER
        depends_on = [_normalize_dependency(item) for item in _as_list(hermes.get("depends_on"))]
        depends_on = [item for item in depends_on if item.get("skill")]
        suggestion_source = _as_list(hermes.get("suggests")) + _as_list(hermes.get("related_skills"))
        suggests = [_normalize_suggestion(item) for item in suggestion_source]
        suggests = [item for item in suggests if item.get("skill")]
        rel_path = str(skill_md.relative_to(root))

        nodes.append(
            {
                "id": _canonical_id("local", category, skill_name),
                "name": skill_name,
                "description": description,
                "category": category,
                "layer": layer,
                "source_kind": "local",
                "path": rel_path,
                "depends_on": depends_on,
                "suggests": suggests,
                "max_dependency_depth": int(hermes.get("max_dependency_depth") or 1),
                "reads": _as_str_list(hermes.get("reads")),
                "writes": _as_str_list(hermes.get("writes")),
                "verifies": _as_str_list(hermes.get("verifies")),
                "side_effects": hermes.get("side_effects"),
                "approval_required": hermes.get("approval_required"),
                "approval_gate": hermes.get("approval_gate"),
                "tags": _as_str_list(hermes.get("tags")),
            }
        )

    nodes.sort(key=lambda node: node["id"])
    name_to_id: dict[str, str] = {}
    for node in nodes:
        name_to_id.setdefault(node["name"], node["id"])

    edges: list[dict[str, Any]] = []
    for node in nodes:
        graph_links = [("depends_on", dep) for dep in node.get("depends_on", [])]
        graph_links.extend(("suggests", dep) for dep in node.get("suggests", []))
        seen_targets: set[tuple[str, str]] = set()
        for link_kind, dep in graph_links:
            target_id = _resolve_dependency_id(dep["skill"], name_to_id, node["category"])
            mode = dep.get("mode") or ("advisory" if link_kind == "suggests" else "explicit")
            edge_key = (target_id, mode)
            if edge_key in seen_targets:
                continue
            seen_targets.add(edge_key)
            edge = {
                "from": node["id"],
                "to": target_id,
                "mode": mode,
                "when": dep.get("when") or "",
            }
            if dep.get("layer"):
                edge["layer"] = dep["layer"]
            edges.append(edge)
    edges.sort(key=lambda edge: (edge["from"], edge["to"], edge.get("mode", ""), edge.get("when", "")))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso(),
        "skills_root": str(root),
        "counts": {"nodes": len(nodes), "edges": len(edges)},
        "nodes": nodes,
        "edges": edges,
    }


def render_skill_graph_markdown(graph: dict[str, Any]) -> str:
    """Render a stable Markdown view of the graph."""
    lines = ["# Hermes Skill Graph", "", "## Counts"]
    counts = graph.get("counts") or {}
    lines.append(f"- Nodes: {counts.get('nodes', len(graph.get('nodes', [])))}")
    lines.append(f"- Edges: {counts.get('edges', len(graph.get('edges', [])))}")
    lines.append("")
    lines.append("## Nodes")
    for node in sorted(graph.get("nodes", []), key=lambda n: n.get("id", "")):
        desc = node.get("description") or ""
        line = f"- `{node.get('id')}` [{node.get('layer', '')}]"
        if desc:
            line += f": {desc}"
        lines.append(line)
    lines.append("")
    lines.append("## Edges")
    edges = sorted(graph.get("edges", []), key=lambda e: (e.get("from", ""), e.get("to", "")))
    if not edges:
        lines.append("- None")
    for edge in edges:
        when = f" - {edge.get('when')}" if edge.get("when") else ""
        lines.append(f"- `{edge.get('from')}` -> `{edge.get('to')}` ({edge.get('mode', 'explicit')}){when}")
    lines.append("")
    return "\n".join(lines)


def write_skill_graph(graph: dict[str, Any], json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    """Write deterministic JSON and optional Markdown manifests."""
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(json_target, graph, sort_keys=True)
    if markdown_path is not None:
        md_target = Path(markdown_path)
        md_target.parent.mkdir(parents=True, exist_ok=True)
        md_target.write_text(render_skill_graph_markdown(graph), encoding="utf-8")


def load_skill_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _warning(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    payload.update(extra)
    return payload


def _has_cycle(edges: list[dict[str, Any]], node_ids: set[str]) -> bool:
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        if source in node_ids and target in node_ids:
            graph[source].append(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for target in graph.get(node_id, []):
            if visit(target):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in sorted(node_ids) if node_id not in visited)


def lint_skill_graph(graph: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Lint graph shape and risky metadata. v0 is warning-first by default."""
    warnings: list[dict[str, Any]] = []
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_ids = [node.get("id") for node in nodes if node.get("id")]
    node_id_set = set(node_ids)

    for node_id, count in Counter(node_ids).items():
        if count > 1:
            warnings.append(_warning("duplicate_node_id", f"Duplicate node id: {node_id}", node_id=node_id))

    for node in nodes:
        node_id = node.get("id") or "<missing>"
        layer = node.get("layer")
        if layer not in VALID_LAYERS:
            warnings.append(_warning("invalid_layer", f"Invalid layer for {node_id}: {layer}", node_id=node_id))
        writes = {str(item) for item in _as_list(node.get("writes"))}
        if writes & RISKY_WRITES and not node.get("approval_gate"):
            warnings.append(
                _warning(
                    "approval_gate_missing",
                    f"Risky writes require approval metadata for {node_id}",
                    node_id=node_id,
                    writes=sorted(writes & RISKY_WRITES),
                )
            )

    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        if source not in node_id_set:
            warnings.append(_warning("missing_source", f"Missing edge source: {source}", node_id=source))
        if target not in node_id_set:
            warnings.append(_warning("missing_dependency", f"Missing dependency: {target}", node_id=target))

    if _has_cycle(edges, node_id_set):
        warnings.append(_warning("cycle", "Dependency cycle detected"))

    errors = warnings if strict else []
    active_warnings = [] if strict else warnings
    return {"ok": not errors and not active_warnings, "warnings": active_warnings, "errors": errors}


_TOKEN_RE = re.compile(r"[a-z0-9]+")

_RECOVERY_QUERY_TOKENS = {
    "blocked",
    "cap",
    "continue",
    "continuation",
    "failed",
    "failure",
    "interrupted",
    "interruption",
    "limit",
    "limits",
    "recover",
    "recovery",
    "resume",
    "stuck",
}
_RECOVERY_NODE_RE = re.compile(
    r"\b(recovery|continuation|interrupted|interruption|tool[- ]call limits?|iteration[- ]cap|rate[- ]limit|resume|fallback)\b"
)
_CANONICAL_NODE_RE = re.compile(r"\b(canonical|primary|authoritative)\b")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _score_node(node: dict[str, Any], query_tokens: set[str], query_text: str) -> float:
    haystack_parts = [
        node.get("name", ""),
        node.get("description", ""),
        node.get("category", ""),
        node.get("layer", ""),
        " ".join(_as_str_list(node.get("tags"))),
        " ".join(_as_str_list(node.get("reads"))),
        " ".join(_as_str_list(node.get("writes"))),
        " ".join(_as_str_list(node.get("verifies"))),
    ]
    haystack = " ".join(str(part) for part in haystack_parts).lower()
    node_tokens = _tokens(haystack)
    overlap = len(query_tokens & node_tokens)
    score = float(overlap)
    name = str(node.get("name", "")).lower()
    if name and name in query_text:
        score += 5.0
    for token in query_tokens:
        if token and token in name:
            score += 0.35
    if node.get("layer") == "compound":
        score += 0.1

    asks_for_recovery = bool(query_tokens & _RECOVERY_QUERY_TOKENS)
    if _CANONICAL_NODE_RE.search(haystack) and not asks_for_recovery:
        score += 0.6
    if _RECOVERY_NODE_RE.search(haystack) and not asks_for_recovery:
        score -= 2.0
    return score


def query_skill_graph(graph: dict[str, Any], query: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Return ranked skills with one-hop dependency suggestions only."""
    query_text = query.lower()
    query_tokens = _tokens(query)
    nodes = list(graph.get("nodes") or [])
    node_by_id = {node.get("id"): node for node in nodes}
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges") or []:
        edges_by_source[edge.get("from")].append(edge)

    ranked: list[dict[str, Any]] = []
    for node in nodes:
        score = _score_node(node, query_tokens, query_text)
        suggestions: list[dict[str, Any]] = []
        for edge in sorted(edges_by_source.get(node.get("id"), []), key=lambda item: item.get("to", "")):
            target = node_by_id.get(edge.get("to"))
            if not target:
                continue
            suggestions.append(
                {
                    "skill": target.get("name"),
                    "id": target.get("id"),
                    "mode": edge.get("mode") or "explicit",
                    "when": edge.get("when") or "",
                }
            )
        ranked.append(
            {
                "id": node.get("id"),
                "name": node.get("name"),
                "description": node.get("description") or "",
                "category": node.get("category") or "general",
                "layer": node.get("layer") or DEFAULT_LAYER,
                "score": round(score, 4),
                "dependency_suggestions": suggestions,
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item.get("category", ""), item.get("name", "")))
    if limit is not None and limit > 0:
        return ranked[:limit]
    return ranked


def get_skill_graph_canary_fixtures() -> list[dict[str, Any]]:
    """Default internal-only canaries for the v0 graph query layer."""
    return [
        {
            "id": "opik-hourly-watch",
            "query": "run Hermes Opik hourly watch",
            "expected_top_skill": "opik-hourly-watch-canonical-runbook",
            "side_effects": "internal_only",
        }
    ]


def run_skill_graph_canaries(
    graph: dict[str, Any], fixtures: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    fixtures = fixtures or get_skill_graph_canary_fixtures()
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        ranked = query_skill_graph(graph, fixture["query"], limit=1)
        top_skill = ranked[0]["name"] if ranked else None
        matched = top_skill == fixture.get("expected_top_skill")
        results.append(
            {
                "id": fixture.get("id"),
                "matched": matched,
                "top_skill": top_skill,
                "expected_top_skill": fixture.get("expected_top_skill"),
                "side_effects": fixture.get("side_effects"),
            }
        )
    return {"ok": all(item["matched"] for item in results), "results": results}
