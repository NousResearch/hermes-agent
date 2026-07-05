"""Producer Normalizer / Quality Gate v1.0.0.

Deterministic, no HTTP, no LLM. Loads ruleset + config, validates the ruleset,
orders rules by depends_on + priority_group_order + priority, executes each
rule, computes hashes, and emits normalizer_report + normalizer_metrics.

Engine STOP vs Bundle BLOCKED separation:
- Engine STOP: engine_status=STOP, normalizer_verdict=NOT_RUN, no sealed report.
- Bundle BLOCKED: engine_status=OK, normalizer_verdict=BLOCKED, sealed report.

References the frozen v1.0.0 contract:
- normalizer_ruleset_schema: 1.0.0
- normalizer_config_schema: 1.0.0
- normalizer_report_schema: 1.0.0
- normalizer_metrics_schema: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- priority_group_order (engine-internal, NOT in ruleset) ---
PRIORITY_GROUP_ORDER: dict[str, int] = {
    "early": 10,
    "structural": 20,
    "integrity": 30,
    "consistency": 40,
    "reporting": 50,
}

# --- Closed enums ---
VALID_PRIORITY_GROUPS = set(PRIORITY_GROUP_ORDER.keys())
VALID_PREDICATE_CLASSES = {"regex", "threshold", "hash", "json", "structural", "cross_reference", "ambiguity"}
VALID_DETERMINISM_LEVELS = {"strict", "bounded", "heuristic"}
VALID_SEVERITIES = {"blocker", "warning", "informational"}
VALID_ENGINE_STOP_DETAIL = {
    "depends_on_cycle", "depends_on_missing_rule", "priority_group_invalid",
    "priority_duplicate_in_group", "predicate_class_invalid", "predicate_type_mismatch",
    "ruleset_schema_invalid", "config_schema_invalid", "ruleset_hash_mismatch",
    "config_hash_mismatch", "report_hash_mismatch", "report_volatile_field",
    "heuristic_rule_emitted_blocker", "engine_http_call_detected",
    "engine_llm_call_detected", "engine_artifact_mutation_detected",
}
ENGINE_STOP_FAMILY = {
    "depends_on_cycle": "graph_error",
    "depends_on_missing_rule": "graph_error",
    "priority_group_invalid": "graph_error",
    "priority_duplicate_in_group": "graph_error",
    "predicate_class_invalid": "ruleset_error",
    "predicate_type_mismatch": "ruleset_error",
    "ruleset_schema_invalid": "ruleset_error",
    "ruleset_hash_mismatch": "ruleset_error",
    "config_schema_invalid": "integrity_error",
    "config_hash_mismatch": "integrity_error",
    "report_hash_mismatch": "integrity_error",
    "report_volatile_field": "integrity_error",
    "heuristic_rule_emitted_blocker": "runtime_violation",
    "engine_http_call_detected": "runtime_violation",
    "engine_llm_call_detected": "runtime_violation",
    "engine_artifact_mutation_detected": "runtime_violation",
}

# --- Closed whitelist for normalizer_report (no volatile fields) ---
REPORT_WHITELIST = {
    "normalizer_report_schema", "report_id", "engine_version", "ruleset_id",
    "ruleset_version", "ruleset_hash", "config_id", "config_version", "config_hash",
    "report_hash", "verdict", "issues", "warnings", "executed_rules",
    "short_circuited_rules", "contract_version", "schema_version", "bundle_sha256_recomputed",
}

# --- canonical JSON / YAML ---
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_hex_str(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()

# --- Engine exceptions ---
class EngineStop(Exception):
    def __init__(self, detail: str, family: str | None = None, **context: Any) -> None:
        self.detail = detail
        self.context = dict(context) if context else {}
        self.family = family or ENGINE_STOP_FAMILY.get(detail, "runtime_violation")
        super().__init__(f"engine_stop: {detail}")

# --- Data classes ---
@dataclass
class Issue:
    issue_id: str
    rule_id: str
    kind: str
    artifact_id: str | None
    severity: str  # blocker | warning | informational
    evidence: str
    rationale: str
    extras: dict[str, Any] = field(default_factory=dict)

# --- Ruleset / config load ---
def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_ruleset_hash(ruleset: dict[str, Any]) -> str:
    rs = {k: v for k, v in ruleset.items() if k != "ruleset_hash"}
    return sha256_hex_str(canonical_json(rs))

def compute_config_hash(cfg: dict[str, Any]) -> str:
    c = {k: v for k, v in cfg.items() if k != "config_hash"}
    return sha256_hex_str(canonical_json(c))

# --- Validation ---
def validate_ruleset(ruleset: dict[str, Any]) -> None:
    if ruleset.get("normalizer_ruleset_schema") != "1.0.0":
        raise EngineStop("ruleset_schema_invalid")
    rules = ruleset.get("rules", [])
    rule_ids = {r["rule_id"] for r in rules}
    # depends_on DAG
    rule_ids = {r["rule_id"] for r in rules}
    # Build remaining dependencies and dependents_of for cycle detection
    remaining = {r["rule_id"]: len(r.get("depends_on", [])) for r in rules}
    dependents_of: dict[str, list[str]] = {rid: [] for rid in rule_ids}
    for r in rules:
        for dep in r.get("depends_on", []):
            if dep not in rule_ids:
                raise EngineStop("depends_on_missing_rule")
            dependents_of[dep].append(r["rule_id"])
    # Kahn's topological sort — also detects cycles
    topo: list[str] = []
    ready = sorted([rid for rid, n in remaining.items() if n == 0])
    while ready:
        n = ready.pop(0)
        topo.append(n)
        for m in dependents_of[n]:
            remaining[m] -= 1
            if remaining[m] == 0:
                ready.append(m)
                ready.sort()
    if len(topo) != len(rule_ids):
        raise EngineStop("depends_on_cycle")
    # priority_group + priority duplicate detection among independent rules
    for pg in (r.get("priority_group") for r in rules):
        if pg not in VALID_PRIORITY_GROUPS:
            raise EngineStop("priority_group_invalid")
    # Build set of independent pairs (no path between them)
    # For priority duplicate detection, build the transitive closure of depends_on
    reachable: dict[str, set[str]] = {rid: set() for rid in rule_ids}
    for src, targets in dependents_of.items():
        for t in targets:
            reachable[src].add(t)
    # Floyd-Warshall on small graph (rules <= 20 in practice)
    nodes = list(rule_ids)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if k in reachable[i] and j in reachable[k]:
                    reachable[i].add(j)
    # Group rules by (priority_group, priority)
    group_priority: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in rules:
        group_priority[(r.get("priority_group"), r.get("priority"))].append(r["rule_id"])
    for (pg, prio), rids in group_priority.items():
        if len(rids) < 2:
            continue
        # Check pairwise: are all pairs independent (no path in either direction)?
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                a, b = rids[i], rids[j]
                if b in reachable[a] or a in reachable[b]:
                    continue  # dependent, OK to share priority
                raise EngineStop("priority_duplicate_in_group")
    # predicate_class closed
    for r in rules:
        pc = r.get("predicate_class")
        if pc not in VALID_PREDICATE_CLASSES:
            raise EngineStop("predicate_class_invalid")
    # heuristic rules cannot have emits_blocker=true
    for r in rules:
        if r.get("determinism_level") == "heuristic" and r.get("capabilities", {}).get("emits_blocker", False):
            raise EngineStop("heuristic_rule_emitted_blocker")
    # short_circuit=true requires short_circuit_scope
    for r in rules:
        cap = r.get("capabilities", {})
        if cap.get("short_circuit") and "short_circuit_scope" not in cap:
            raise EngineStop("ruleset_schema_invalid")

def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("normalizer_config_schema") != "1.0.0":
        raise EngineStop("config_schema_invalid")

def order_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topological sort by depends_on, then by priority_group_order, then by priority.

    Kahn's algorithm with the edge convention:
    rule u has depends_on = [d1, d2, ...] means u waits for d_i.
    We pop rules with no remaining unsatisfied dependencies and emit
    them in (priority_group_order, priority) order.
    """
    rule_ids = {r["rule_id"]: r for r in rules}
    # depends_on_map[u] = list of rules u depends on.
    depends_on_map = {r["rule_id"]: list(r.get("depends_on", [])) for r in rules}
    # remaining[u] = number of unsatisfied dependencies of u.
    remaining = {r["rule_id"]: len(depends_on_map[r["rule_id"]]) for r in rules}
    # dependents_of[d] = list of rules that depend on d (for decrementing).
    dependents_of: dict[str, list[str]] = {r["rule_id"]: [] for r in rules}
    for r in rules:
        for dep in r.get("depends_on", []):
            dependents_of[dep].append(r["rule_id"])

    def sort_key(rid: str) -> tuple:
        pg = rule_ids[rid].get("priority_group", "early")
        return (PRIORITY_GROUP_ORDER.get(pg, 99), rule_ids[rid].get("priority", 0))
    ready = sorted([rid for rid, r in remaining.items() if r == 0], key=sort_key)
    out: list[dict[str, Any]] = []
    while ready:
        n = ready.pop(0)
        out.append(rule_ids[n])
        for m in dependents_of[n]:
            remaining[m] -= 1
            if remaining[m] == 0:
                ready.append(m)
                ready.sort(key=sort_key)
    return out

# --- Rule predicates ---
def _line_count(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)

def _non_whitespace_chars(text: str) -> int:
    return len(re.sub(r"\s", "", text))

def rule_placeholder_detection(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    cfg = ctx["config"]
    patterns = [re.compile(p) for p in cfg.get("placeholder_patterns", [])]
    required = set(cfg.get("required_sections_artifact_001", []))
    issue_counter = ctx["issue_counter"]
    for aid, body in ctx["artifacts"].items():
        # Determine current section for each line by walking the body once.
        # Sections are markdown "## <name>" headers (level 2). For artifact-001
        # only the required list applies; for other artifacts all placeholders
        # are at most warnings.
        line_to_section: list[str] = []
        current_section = ""
        for line in body.splitlines():
            m = re.match(r"^##\s+(.+?)\s*$", line)
            if m:
                current_section = m.group(1).strip()
            line_to_section.append(current_section)
        for i, line in enumerate(body.splitlines()):
            for pat in patterns:
                if pat.search(line):
                    sec = line_to_section[i] if i < len(line_to_section) else ""
                    # Block only when:
                    #  - artifact is artifact-001 AND
                    #  - the section is in the required list AND
                    #  - the pattern is one that signals true content absence
                    #    (ellipsis, TODO, TBD, <placeholder>, lorem ipsum, ...).
                    is_blocking_pattern = bool(pat.search("...")
                        or "TODO" in pat.pattern
                        or "TBD" in pat.pattern
                        or "placeholder" in pat.pattern
                        or "lorem" in pat.pattern
                        or "omitted" in pat.pattern
                        or "to be determined" in pat.pattern
                        or "implementation TBD" in pat.pattern
                    )
                    if aid == "artifact-001" and sec in required and is_blocking_pattern:
                        sev = "blocker"
                    else:
                        sev = "warning"
                    issues.append(Issue(
                        issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                        kind="placeholder", artifact_id=aid, severity=sev,
                        evidence=line[:200], rationale=f"Pattern matched in {aid} ({sec or 'unknown'}): {line[:80]}",
                        extras={"pattern_matched": pat.pattern, "section": sec},
                    ))
    return issues

def rule_minimum_content_thresholds(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    cfg = ctx["config"]
    sizes = cfg.get("minimum_sizes", {})
    min_total = cfg.get("minimum_total_size", 0)
    min_sections = cfg.get("minimum_sections", {}).get("artifact-001", 0)
    min_risks = cfg.get("minimum_risks", {}).get("artifact-002", 0)
    min_questions = cfg.get("minimum_questions", {}).get("artifact-003", 0)
    issue_counter = ctx["issue_counter"]
    for aid, body in ctx["artifacts"].items():
        if aid in sizes and len(body.encode("utf-8")) < sizes[aid]:
            issues.append(Issue(
                issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                kind="size", artifact_id=aid, severity="blocker",
                evidence=body[:50], rationale=f"Size {len(body)} < {sizes[aid]}",
                extras={"threshold_name": f"minimum_size_{aid}", "actual_value": len(body.encode("utf-8")), "threshold_value": sizes[aid]},
            ))
    total = sum(len(b.encode("utf-8")) for b in ctx["artifacts"].values())
    if total < min_total:
        issues.append(Issue(
            issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
            kind="size", artifact_id=None, severity="blocker",
            evidence=f"total={total}", rationale=f"Total size {total} < {min_total}",
            extras={"threshold_name": "minimum_total_size", "actual_value": total, "threshold_value": min_total},
        ))
    # Sections in artifact-001
    body001 = ctx["artifacts"].get("artifact-001", "")
    section_count = len(re.findall(r"(?m)^##\s+", body001))
    if section_count < min_sections:
        issues.append(Issue(
            issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
            kind="size", artifact_id="artifact-001", severity="blocker",
            evidence=body001[:50], rationale=f"Sections {section_count} < {min_sections}",
            extras={"threshold_name": "minimum_sections_artifact_001", "actual_value": section_count, "threshold_value": min_sections},
        ))
    return issues

def rule_evidence_readiness(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    cfg = ctx["config"]
    thr = cfg.get("evidence_readiness_thresholds", {})
    min_lines = thr.get("minimum_lines", 5)
    min_chars = thr.get("minimum_non_whitespace_chars", 200)
    require_json_002 = thr.get("require_json_artifact_002", True)
    issue_counter = ctx["issue_counter"]
    for aid, body in ctx["artifacts"].items():
        if _line_count(body) < min_lines:
            issues.append(Issue(
                issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                kind="evidence_readiness", artifact_id=aid, severity="blocker",
                evidence=body[:50], rationale=f"Lines {_line_count(body)} < {min_lines}",
                extras={"check": "minimum_lines", "actual_value": _line_count(body), "threshold_value": min_lines},
            ))
        if _non_whitespace_chars(body) < min_chars:
            issues.append(Issue(
                issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                kind="evidence_readiness", artifact_id=aid, severity="blocker",
                evidence=body[:50], rationale=f"Non-ws chars {_non_whitespace_chars(body)} < {min_chars}",
                extras={"check": "minimum_non_whitespace_chars", "actual_value": _non_whitespace_chars(body), "threshold_value": min_chars},
            ))
        if aid == "artifact-002" and require_json_002:
            try:
                json.loads(body)
            except Exception as e:
                issues.append(Issue(
                    issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                    kind="evidence_readiness", artifact_id=aid, severity="blocker",
                    evidence=body[:80], rationale=f"JSON parse error: {e}",
                    extras={"check": "require_json_artifact_002"},
                ))
    return issues

def rule_deterministic_hash_precheck(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    bundle = ctx["bundle"]
    issue_counter = ctx["issue_counter"]
    for a in bundle.get("artifacts", []):
        aid = a["artifact_id"]
        body = ctx["artifacts"].get(aid, "")
        declared = a.get("content_hash", "")
        recomputed = sha256_hex_str(body)
        if declared and declared != recomputed:
            issues.append(Issue(
                issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                kind="hash_precheck", artifact_id=aid, severity="blocker",
                evidence=f"declared={declared} recomputed={recomputed}",
                rationale="Declared content_hash does not match recomputed sha256(artifact body).",
                extras={"declared_hash": declared, "recomputed_hash": recomputed},
            ))
    return issues

def rule_schema_field_consistency(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    issue_counter = ctx["issue_counter"]
    body001 = ctx["artifacts"].get("artifact-001", "")
    # Find AC references like "AC1 requires X" or "AC1: X in Y" or "(<schema>.X)"
    # Heuristic: lines mentioning "AC<digit>" or "schema_v1" followed by a field name in backticks or parens
    declared_schemas = re.findall(r"([A-Z][A-Za-z]+\.schema_v\d+)", body001)
    declared_schemas = list(set(declared_schemas))
    for line in body001.splitlines():
        # Look for "<schema>.<field>" references
        for m in re.finditer(r"\b([A-Z][A-Za-z]+\.schema_v\d+)\.([a-z_][a-z0-9_]*)\b", line):
            schema, field = m.group(1), m.group(2)
            # Check if field appears in any artifact body
            found = False
            for aid2, body2 in ctx["artifacts"].items():
                if field in body2.lower() or f"`{field}`" in body2:
                    found = True
                    break
            if not found:
                issues.append(Issue(
                    issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                    kind="schema_consistency", artifact_id="artifact-001", severity="blocker",
                    evidence=line[:120], rationale=f"Schema field {schema}.{field} referenced but not found in any artifact.",
                    extras={"schema_name": schema, "missing_field": field, "ac_reference": line[:80]},
                ))
    return issues

def rule_cross_reference_consistency(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    issue_counter = ctx["issue_counter"]
    cfg = ctx["config"]
    threshold = cfg.get("fuzzy_match_threshold", 0.85)
    # Heuristic: only emit warnings; no blockers.
    # Verify artifact_ids referenced in any text are present in the bundle.
    bundle_artifact_ids = {a["artifact_id"] for a in ctx["bundle"].get("artifacts", [])}
    for aid, body in ctx["artifacts"].items():
        for m in re.finditer(r"\bartifact-\d{3}\b", body):
            ref = m.group(0)
            if ref not in bundle_artifact_ids:
                issues.append(Issue(
                    issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                    kind="cross_reference", artifact_id=aid, severity="warning",
                    evidence=body[max(0, m.start() - 20):m.end() + 20],
                    rationale=f"Reference to unknown artifact_id {ref}.",
                    extras={"missing_artifact_id": ref, "similarity_score": 0.0},
                ))
    return issues

def rule_forbidden_ambiguity(rule: dict[str, Any], ctx: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    cfg = ctx["config"]
    patterns = [re.compile(p) for p in cfg.get("forbidden_ambiguity_patterns", [])]
    issue_counter = ctx["issue_counter"]
    # Per-artifact ambiguity detection
    for aid, body in ctx["artifacts"].items():
        for line in body.splitlines():
            for pat in patterns:
                if pat.search(line):
                    issues.append(Issue(
                        issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                        kind="ambiguity", artifact_id=aid, severity="warning",
                        evidence=line[:200], rationale=f"Ambiguity pattern matched in {aid}",
                        extras={"pattern_matched": pat.pattern},
                    ))
    # Cross-artifact contradictions: artifact-001 says X is blocking, artifact-003 says X is open question
    body001 = ctx["artifacts"].get("artifact-001", "")
    body003 = ctx["artifacts"].get("artifact-003", "")
    blocking_pattern = re.compile(r"\b(blocking|blocker|hard\s*gate|required|mandatory)\b", re.IGNORECASE)
    open_q_pattern = re.compile(r"\b(open\s*question|defer|out\s*of\s*scope|to\s*be\s*defined|TBD)\b", re.IGNORECASE)
    # For each "blocking X" line in 001, check if X appears in 003 in an open question
    for line in body001.splitlines():
        m = blocking_pattern.search(line)
        if not m:
            continue
        # Extract the noun phrase after "blocking"
        tail = line[m.end():].strip().rstrip(".,;:")
        if not tail or len(tail) > 80:
            continue
        key = tail.split()[0] if tail.split() else ""
        if not key or len(key) < 4:
            continue
        if key.lower() in body003.lower() and open_q_pattern.search(body003):
            issues.append(Issue(
                issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                kind="ambiguity", artifact_id="artifact-001", severity="blocker",
                evidence=line[:200], rationale=f"Cross-artifact contradiction: artifact-001 says '{key}' is blocking but artifact-003 treats it as open question.",
                extras={"pattern_matched": "blocking_vs_open_question", "cross_artifact_pair": ("artifact-001", "artifact-003")},
            ))
    return issues

# --- Rule dispatch ---
RULE_PREDICATES = {
    "placeholder_detection": rule_placeholder_detection,
    "minimum_content_thresholds": rule_minimum_content_thresholds,
    "evidence_readiness": rule_evidence_readiness,
    "deterministic_hash_precheck": rule_deterministic_hash_precheck,
    "schema_field_consistency": rule_schema_field_consistency,
    "cross_reference_consistency": rule_cross_reference_consistency,
    "forbidden_ambiguity": rule_forbidden_ambiguity,
}

# --- Engine ---
class ProducerNormalizer:
    def __init__(self, ruleset_path: Path, config_path: Path, engine_version: str = "0.1.0"):
        self.ruleset_path = Path(ruleset_path)
        self.config_path = Path(config_path)
        self.engine_version = engine_version
        self.ruleset: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.ruleset_hash = ""
        self.config_hash = ""
        self.ordered_rules: list[dict[str, Any]] = []

    def load(self) -> None:
        try:
            self.ruleset = load_yaml(self.ruleset_path)
        except Exception as e:
            raise EngineStop("ruleset_schema_invalid") from e
        # If yaml.safe_load returned None (empty file), treat as invalid
        if self.ruleset is None:
            raise EngineStop("ruleset_schema_invalid")
        try:
            self.config = load_yaml(self.config_path)
        except Exception as e:
            raise EngineStop("config_schema_invalid") from e
        if self.config is None:
            raise EngineStop("config_schema_invalid")
        validate_ruleset(self.ruleset)
        validate_config(self.config)
        self.ruleset_hash = compute_ruleset_hash(self.ruleset)
        self.config_hash = compute_config_hash(self.config)
        self.ordered_rules = order_rules(self.ruleset.get("rules", []))

    def normalize(self, bundle: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
        """Returns a dict with engine_status, normalizer_verdict, report, metrics.

        Always returns the dict; Engine STOP is represented as engine_status=STOP
        with engine_stop_detail and optional engine_error_payload.
        """
        # Load artifacts from disk
        artifacts: dict[str, str] = {}
        for a in bundle.get("artifacts", []):
            path = bundle_root / a["location"]
            if not path.exists():
                artifacts[a["artifact_id"]] = ""
            else:
                artifacts[a["artifact_id"]] = path.read_text(encoding="utf-8")

        issue_counter_n = [0]
        def issue_counter() -> int:
            issue_counter_n[0] += 1
            return issue_counter_n[0]

        ctx = {
            "bundle": bundle,
            "artifacts": artifacts,
            "config": self.config,
            "issue_counter": issue_counter,
        }
        # Recompute bundle_sha256
        manifest = bundle.get("bundle_manifest", {})
        manifest_artifacts_sorted = sorted(
            [
                {"artifact_id": a["artifact_id"], "artifact_path": a["artifact_path"], "artifact_sha256": a["artifact_sha256"]}
                for a in manifest.get("artifacts", [])
            ],
            key=lambda d: d["artifact_id"],
        )
        manifest_doc = {
            "artifacts": manifest_artifacts_sorted,
            "bundle_id": manifest.get("bundle_id"),
            "manifest_schema": "1.0.0",
            "task_id": manifest.get("task_id"),
        }
        bundle_sha256_recomputed = sha256_hex_str(canonical_json(manifest_doc))

        all_issues: list[Issue] = []
        executed: list[str] = []
        short_circuited: list[str] = []
        has_blocker_so_far = False

        for rule in self.ordered_rules:
            rid = rule["rule_id"]
            cap = rule.get("capabilities", {})
            # short_circuit on blocker found
            if has_blocker_so_far and cap.get("short_circuit") and cap.get("short_circuit_scope") == "dependent_content_rules":
                # Only short-circuit rules that depend on prior rules
                # (heuristic: if not already executed and is downstream)
                short_circuited.append(rid)
                continue
            try:
                rule_issues = RULE_PREDICATES[rid](rule, ctx)
            except Exception as e:
                raise EngineStop("ruleset_schema_invalid") from e
            # Enforce capability invariants
            for iss in rule_issues:
                if iss.severity == "blocker" and not cap.get("emits_blocker", False):
                    raise EngineStop("heuristic_rule_emitted_blocker")
            all_issues.extend(rule_issues)
            executed.append(rid)
            if any(iss.severity == "blocker" for iss in rule_issues):
                has_blocker_so_far = True

        # Verdict
        blockers = [i for i in all_issues if i.severity == "blocker"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        if blockers:
            verdict = "BLOCKED"
        elif warnings:
            verdict = "PARTIAL"
        elif not artifacts or all(not v for v in artifacts.values()):
            verdict = "NO_EVIDENCE"
        else:
            verdict = "PASS"
        reviewer_call_saved = verdict in ("BLOCKED", "NO_EVIDENCE")

        # Build report (whitelist)
        issues_list = []
        for i in all_issues:
            issues_list.append({
                "issue_id": i.issue_id,
                "rule_id": i.rule_id,
                "kind": i.kind,
                "artifact_id": i.artifact_id,
                "severity": i.severity,
                "evidence": i.evidence,
                "rationale": i.rationale,
                **i.extras,
            })

        report_no_hash = {
            "normalizer_report_schema": "1.0.0",
            "report_id": f"nr-{bundle.get('bundle_id', 'unknown')}",
            "engine_version": self.engine_version,
            "ruleset_id": self.ruleset.get("normalizer_ruleset_id"),
            "ruleset_version": self.ruleset.get("normalizer_ruleset_version"),
            "ruleset_hash": self.ruleset_hash,
            "config_id": self.config.get("normalizer_config_id"),
            "config_version": self.config.get("normalizer_config_version"),
            "config_hash": self.config_hash,
            "verdict": verdict,
            "issues": issues_list,
            "warnings": [i for i in issues_list if i["severity"] == "warning"],
            "executed_rules": executed,
            "short_circuited_rules": short_circuited,
            "bundle_sha256_recomputed": bundle_sha256_recomputed,
            "contract_version": 1,
            "schema_version": 1,
        }
        # Whitelist check
        for k in report_no_hash.keys():
            if k not in REPORT_WHITELIST:
                raise EngineStop("report_volatile_field")
        # Compute report_hash
        report_hash = sha256_hex_str(canonical_json(report_no_hash))
        report = dict(report_no_hash)
        report["report_hash"] = report_hash
        # Re-verify whitelist
        for k in report.keys():
            if k not in REPORT_WHITELIST:
                raise EngineStop("report_volatile_field")
        # Re-verify hash
        if sha256_hex_str(canonical_json({k: v for k, v in report.items() if k != "report_hash"})) != report_hash:
            raise EngineStop("report_hash_mismatch")

        # Build metrics
        metrics: dict[str, Any] = {
            "normalizer_metrics_schema": "1.0.0",
            "report_hash": report_hash,
            "ruleset_id": self.ruleset.get("normalizer_ruleset_id"),
            "ruleset_version": self.ruleset.get("normalizer_ruleset_version"),
            "ruleset_hash": self.ruleset_hash,
            "config_id": self.config.get("normalizer_config_id"),
            "config_version": self.config.get("normalizer_config_version"),
            "config_hash": self.config_hash,
            "engine_version": self.engine_version,
            "engine_status": "OK",
            "normalizer_verdict": verdict,
            "blocked_reason_kind": "bundle" if verdict == "BLOCKED" else "none",
            "engine_stop_reason_family": None,
            "engine_stop_detail": None,
            "placeholder_hits_per_artifact": {},
            "placeholder_hits_total": 0,
            "content_size_total": sum(len(b.encode("utf-8")) for b in artifacts.values()),
            "content_size_by_artifact": {k: len(v.encode("utf-8")) for k, v in artifacts.items()},
            "size_violations_per_artifact": {},
            "line_counts_by_artifact": {k: _line_count(v) for k, v in artifacts.items()},
            "json_parse_errors": 0,
            "hash_mismatches_total": 0,
            "bundle_sha256_recomputed": bundle_sha256_recomputed,
            "artifact_sha256_recomputed_per_artifact": {
                a["artifact_id"]: sha256_hex_str(artifacts.get(a["artifact_id"], ""))
                for a in bundle.get("artifacts", [])
            },
            "schema_consistency_errors": 0,
            "ac_references_unresolved": 0,
            "cross_reference_warnings": 0,
            "naming_drift_warnings": 0,
            "ambiguity_hits": 0,
            "cross_artifact_contradictions": 0,
            "reviewer_call_saved": reviewer_call_saved,
            "contract_version": 1,
            "schema_version": 1,
        }
        # Populate metrics from issues
        for i in all_issues:
            if i.kind == "placeholder":
                m = metrics["placeholder_hits_per_artifact"]
                m[i.artifact_id] = m.get(i.artifact_id, 0) + 1
                metrics["placeholder_hits_total"] += 1
            elif i.kind == "size":
                m = metrics["size_violations_per_artifact"]
                m[i.artifact_id or "bundle"] = m.get(i.artifact_id or "bundle", 0) + 1
            elif i.kind == "hash_precheck":
                metrics["hash_mismatches_total"] += 1
            elif i.kind == "schema_consistency":
                if i.severity == "blocker":
                    metrics["schema_consistency_errors"] += 1
                else:
                    metrics["ac_references_unresolved"] += 1
            elif i.kind == "cross_reference":
                metrics["cross_reference_warnings"] += 1
            elif i.kind == "ambiguity":
                metrics["ambiguity_hits"] += 1
                if i.severity == "blocker":
                    metrics["cross_artifact_contradictions"] += 1
            elif i.kind == "evidence_readiness" and "JSON" in i.rationale:
                metrics["json_parse_errors"] += 1

        return {
            "engine_status": "OK",
            "normalizer_verdict": verdict,
            "report": report,
            "metrics": metrics,
        }

    def _engine_stop_payload(self, e: "EngineStop", output_dir: Path) -> dict[str, Any]:
        """Build the engine_status=STOP payload and try to write engine_error.json."""
        payload = {
            "engine_error_schema": "1.0.0",
            "engine_status": "STOP",
            "engine_stop_reason": ENGINE_STOP_FAMILY.get(e.detail, "runtime_violation"),
            "engine_stop_detail": e.detail,
            "ruleset_id": self.ruleset.get("normalizer_ruleset_id") if self.ruleset else None,
            "ruleset_version": self.ruleset.get("normalizer_ruleset_version") if self.ruleset else None,
            "ruleset_hash": self.ruleset_hash or None,
            "config_id": self.config.get("normalizer_config_id") if self.config else None,
            "config_version": self.config.get("normalizer_config_version") if self.config else None,
            "config_hash": self.config_hash or None,
            "contract_version": 1,
            "schema_version": 1,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        err_path = output_dir / "engine_error.json"
        try:
            err_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            pass  # Optional per engineering note
        return {
            "engine_status": "STOP",
            "normalizer_verdict": "NOT_RUN",
            "engine_stop_detail": e.detail,
            "engine_stop_family": e.family,
            "engine_error_path": str(err_path) if err_path.exists() else None,
            "engine_error_payload": payload,
        }

    def run(self, bundle: dict[str, Any], bundle_root: Path, output_dir: Path) -> dict[str, Any]:
        """Top-level: load (if not already), normalize, write report+metrics.

        Returns a dict with engine_status, normalizer_verdict, paths written,
        and (if Engine STOP) engine_stop_detail + engine_error_payload.
        """
        # Ensure ruleset+config are loaded. If load() raises EngineStop, catch
        # here to convert to engine_status=STOP. If already loaded (ruleset
        # non-empty), skip.
        if not self.ruleset:
            try:
                self.load()
            except EngineStop as e:
                return self._engine_stop_payload(e, output_dir)
        try:
            result = self.normalize(bundle, bundle_root)
        except EngineStop as e:
            return self._engine_stop_payload(e, output_dir)
        # Engine OK
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write the ruleset+config to the pilot dir for traceability
        ruleset_path = output_dir / "normalizer_ruleset.v1.0.0.yaml"
        config_path = output_dir / "normalizer_config.v1.0.0.yaml"
        try:
            import yaml as _yaml
            ruleset_to_write = {k: v for k, v in self.ruleset.items() if k != "ruleset_hash"}
            ruleset_to_write["ruleset_hash"] = self.ruleset_hash
            config_to_write = {k: v for k, v in self.config.items() if k != "config_hash"}
            config_to_write["config_hash"] = self.config_hash
            ruleset_path.write_text(_yaml.safe_dump(ruleset_to_write, sort_keys=True))
            config_path.write_text(_yaml.safe_dump(config_to_write, sort_keys=True))
        except Exception:
            pass  # Optional per engineering note
        report_path = output_dir / "normalizer_report.v1.0.0.json"
        metrics_path = output_dir / "normalizer_metrics.v1.0.0.json"
        report_path.write_text(json.dumps(result["report"], indent=2, sort_keys=True))
        metrics_path.write_text(json.dumps(result["metrics"], indent=2, sort_keys=True))
        return {
            "engine_status": "OK",
            "normalizer_verdict": result["normalizer_verdict"],
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
            "ruleset_path": str(ruleset_path),
            "config_path": str(config_path),
            "report": result["report"],
            "metrics": result["metrics"],
        }


# ===========================================================================
# v1.1 dispatch + EvidenceReadinessV1_1Rule + run_v1_1 path
# ===========================================================================


class EvidenceReadinessV1_1Rule:
    """v1.1 evidence_readiness rule using declarative extractors and policies.

    Differs from v1.0 in that:
    - Uses extractors declared in ruleset.extractors via ExtractorRegistry.
    - Uses policies declared in ruleset.policies via PolicyRegistry.
    - JSON parse error is warning (not blocker) when substantive_content
      evaluates to "high" (declaratively).
    """

    @staticmethod
    def run(rule: dict, ctx: dict, extractor_registry, policy_validator, policy_evaluator):
        issues = []
        cfg = ctx["config"]
        thr = cfg.get("evidence_readiness_thresholds", {})
        min_lines = thr.get("minimum_lines", 5)
        min_chars = thr.get("minimum_non_whitespace_chars", 200)
        require_json = thr.get("require_json_artifact_002", True)
        issue_counter = ctx["issue_counter"]
        artifact_metrics_map = ctx.setdefault("artifact_metrics_map", {})
        ruleset = ctx.get("ruleset", {})
        extractors_cfg = ruleset.get("extractors", {})
        substantive_cfg = cfg.get("substantive_content", {})
        policies_cfg = ruleset.get("policies", [])

        # Helper: instantiate an extractor from a config block.
        def instantiate(ext_cfg):
            kind = ext_cfg.get("extractor_kind")
            params = {k: v for k, v in ext_cfg.items() if k != "extractor_kind"}
            factory = extractor_registry.resolve(kind)
            # Validate parameters.
            allowed = factory.allowed_parameters
            unknown = set(params.keys()) - allowed
            if unknown:
                raise EngineStop("extractor_invalid_params", kind=kind, unknown=sorted(unknown))
            return factory.create(params)

        # Helper: evaluate substantive_content for a given artifact.
        def evaluate_substantive_content(am, aid):
            requirements = substantive_cfg.get("artifact_requirements", {})
            dimensions = substantive_cfg.get("dimensions", {})
            artifact_reqs = requirements.get(aid, {})
            # Only iterate over required_dimensions for THIS artifact.
            required_dimensions = artifact_reqs.get("required_dimensions", [])
            failing = []
            for dim_name in required_dimensions:
                dim_cfg = dimensions.get(dim_name, {})
                metric_options = dim_cfg.get("metric", "").split("|")
                threshold = dim_cfg.get("threshold", 0)
                actual = 0
                for m in metric_options:
                    val = am.get(m.strip())
                    if val is None or val == "":
                        continue
                    try:
                        actual = int(val)
                        break
                    except (ValueError, TypeError):
                        continue
                # Fallback for `items` dimension when JSON is invalid: count
                # heuristic item markers in the raw text (e.g. id fields).
                # This allows substantive content with truncated JSON to still
                # count items heuristically.
                if (
                    dim_name == "items"
                    and actual < threshold
                    and not am.get("json_valid", True)
                ):
                    # Count items by looking for `"id":` keys in raw text.
                    risk_matches = len(re.findall(r'\"id\"\s*:\s*\"', body))
                    if risk_matches >= threshold:
                        actual = risk_matches
                if actual < threshold:
                    failing.append(dim_name)
            return (len(failing) == 0), failing

        for aid, body in ctx["artifacts"].items():
            # Lines and non-ws checks (unchanged from v1.0).
            if _line_count(body) < min_lines:
                issues.append(Issue(
                    issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                    kind="evidence_readiness", artifact_id=aid, severity="blocker",
                    evidence=body[:50], rationale=f"Lines {_line_count(body)} < {min_lines}",
                    extras={"check": "minimum_lines", "actual_value": _line_count(body), "threshold_value": min_lines},
                ))
            if _non_whitespace_chars(body) < min_chars:
                issues.append(Issue(
                    issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                    kind="evidence_readiness", artifact_id=aid, severity="blocker",
                    evidence=body[:50], rationale=f"Non-ws chars {_non_whitespace_chars(body)} < {min_chars}",
                    extras={"check": "minimum_non_whitespace_chars", "actual_value": _non_whitespace_chars(body), "threshold_value": min_chars},
                ))
            # Run extractors to produce artifact_metrics for this artifact.
            am: dict = {}
            for ext_name, ext_cfg in extractors_cfg.items():
                try:
                    extractor = instantiate(ext_cfg)
                    am[ext_name] = extractor.run(body)
                except EngineStop:
                    raise
                except Exception as e:
                    am[ext_name] = 0
            artifact_metrics_map[aid] = am
            # Apply declarative policies that target this artifact.
            if require_json and aid == "artifact-002":
                # Determine substantive_content.
                high, failing_dims = evaluate_substantive_content(am, aid)
                derived_state = {
                    "json_valid": bool(am.get("json_valid", False)),
                    "substantive_content": "high" if high else "low",
                    "artifact_id": aid,
                }
                # Try each policy in order.
                for policy in policies_cfg:
                    if aid not in policy.get("applies_to", []):
                        continue
                    # Policy must already be validated at load time, but
                    # call validator defensively.
                    try:
                        policy_validator.validate(policy)
                    except EngineStop:
                        raise
                    outcome = policy_evaluator.evaluate(policy, derived_state)
                    if outcome is None:
                        continue
                    # First matching policy wins.
                    sev = outcome.get("severity", "warning")
                    issues.append(Issue(
                        issue_id=f"i-{issue_counter():04d}", rule_id=rule["rule_id"],
                        kind=outcome.get("issue_kind", "evidence_readiness"),
                        artifact_id=aid, severity=sev,
                        evidence=body[:80], rationale=f"Policy {policy['policy_id']} matched",
                        extras={
                            "policy_id": policy["policy_id"],
                            "policy_kind": policy["policy_kind"],
                            "json_valid": am.get("json_valid"),
                            "substantive_content": derived_state["substantive_content"],
                            "failing_dimensions": failing_dims,
                            "actual_metrics": am,
                        },
                    ))
                    break  # first matching policy wins
        return issues


class ProducerNormalizerV1_1(ProducerNormalizer):
    """v1.1 normalizer: uses registries, validators, evaluators.

    Inherits from ProducerNormalizer for backward compatibility of the
    v1.0 path (used when ruleset_version="1.0.0"). Overrides load() and
    run() to delegate to v1.1 logic when ruleset_version="1.1.0".

    To run v1.1, instantiate ProducerNormalizerV1_1 instead of ProducerNormalizer.
    """

    def __init__(self, ruleset_path, config_path, engine_version="0.2.0"):
        super().__init__(ruleset_path, config_path, engine_version=engine_version)
        # Private registries, loaded in bootstrap from the impl module.
        from agent._normalizer_v1_1_impl import (
            build_extractor_registry, build_policy_registry,
            PolicyValidator, PolicyEvaluator,
        )
        self._extractor_registry = build_extractor_registry()
        self._policy_registry = build_policy_registry()
        self._policy_validator = PolicyValidator(self._policy_registry)
        self._policy_evaluator = PolicyEvaluator()

    def load(self):
        """Load v1.1 ruleset. Validates registries + structural policies."""
        try:
            self.ruleset = load_yaml(self.ruleset_path)
        except Exception as e:
            raise EngineStop("ruleset_schema_invalid") from e
        if self.ruleset is None:
            raise EngineStop("ruleset_schema_invalid")
        try:
            self.config = load_yaml(self.config_path)
        except Exception as e:
            raise EngineStop("config_schema_invalid") from e
        if self.config is None:
            raise EngineStop("config_schema_invalid")
        version = self.ruleset.get("normalizer_ruleset_version", "")
        if version == "1.1.0":
            # v1.1 path: validate registries and policies structurally.
            if self._extractor_registry.is_empty():
                raise EngineStop("extractor_registry_empty")
            if self._policy_registry.is_empty():
                raise EngineStop("policy_kind_registry_empty")
            # Validate extractor_registry entries (registry already has the
            # kinds; the ruleset declares which kinds are "active"). We just
            # confirm each declared kind is registered.
            registry_cfg = self.ruleset.get("extractor_registry", {})
            for name, cfg in registry_cfg.items():
                kind = cfg.get("extractor_kind")
                if not self._extractor_registry.contains(kind):
                    raise EngineStop("extractor_unknown", kind=kind)
                factory = self._extractor_registry.resolve(kind)
                if not factory.deterministic:
                    raise EngineStop("extractor_nondeterministic", kind=kind)
            # Validate policies structurally (one-time).
            for policy in self.ruleset.get("policies", []):
                self._policy_validator.validate(policy)
        else:
            # Fall back to v1.0 validation path.
            validate_ruleset(self.ruleset)
        validate_config(self.config)
        self.ruleset_hash = compute_ruleset_hash(self.ruleset)
        self.config_hash = compute_config_hash(self.config)
        self.ordered_rules = order_rules(self.ruleset.get("rules", []))

    def normalize(self, bundle, bundle_root):
        """v1.1 normalize path. Overrides parent to use v1.1 evidence_readiness."""
        # Load artifacts from disk.
        artifacts: dict[str, str] = {}
        for a in bundle.get("artifacts", []):
            path = bundle_root / a["location"]
            if not path.exists():
                artifacts[a["artifact_id"]] = ""
            else:
                artifacts[a["artifact_id"]] = path.read_text(encoding="utf-8")

        issue_counter_n = [0]

        def issue_counter():
            issue_counter_n[0] += 1
            return issue_counter_n[0]

        ctx = {
            "bundle": bundle,
            "artifacts": artifacts,
            "config": self.config,
            "issue_counter": issue_counter,
            "ruleset": self.ruleset,
            "artifact_metrics_map": {},
        }
        manifest = bundle.get("bundle_manifest", {})
        manifest_artifacts_sorted = sorted(
            [
                {"artifact_id": a["artifact_id"], "artifact_path": a["artifact_path"], "artifact_sha256": a["artifact_sha256"]}
                for a in manifest.get("artifacts", [])
            ],
            key=lambda d: d["artifact_id"],
        )
        manifest_doc = {
            "artifacts": manifest_artifacts_sorted,
            "bundle_id": manifest.get("bundle_id"),
            "manifest_schema": "1.0.0",
            "task_id": manifest.get("task_id"),
        }
        bundle_sha256_recomputed = sha256_hex_str(canonical_json(manifest_doc))

        all_issues: list[Issue] = []
        executed: list[str] = []
        short_circuited: list[str] = []
        has_blocker_so_far = False
        version = self.ruleset.get("normalizer_ruleset_version", "")

        for rule in self.ordered_rules:
            rid = rule["rule_id"]
            cap = rule.get("capabilities", {})
            if has_blocker_so_far and cap.get("short_circuit") and cap.get("short_circuit_scope") == "dependent_content_rules":
                short_circuited.append(rid)
                continue
            try:
                if version == "1.1.0" and rid == "evidence_readiness":
                    rule_issues = EvidenceReadinessV1_1Rule.run(
                        rule, ctx, self._extractor_registry,
                        self._policy_validator, self._policy_evaluator,
                    )
                else:
                    rule_issues = RULE_PREDICATES[rid](rule, ctx)
            except Exception as e:
                raise EngineStop("ruleset_schema_invalid") from e
            for iss in rule_issues:
                if iss.severity == "blocker" and not cap.get("emits_blocker", False):
                    raise EngineStop("heuristic_rule_emitted_blocker")
            all_issues.extend(rule_issues)
            executed.append(rid)
            if any(iss.severity == "blocker" for iss in rule_issues):
                has_blocker_so_far = True

        blockers = [i for i in all_issues if i.severity == "blocker"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        if blockers:
            verdict = "BLOCKED"
        elif warnings:
            verdict = "PARTIAL"
        elif not artifacts or all(not v for v in artifacts.values()):
            verdict = "NO_EVIDENCE"
        else:
            verdict = "PASS"
        reviewer_call_saved = verdict in ("BLOCKED", "NO_EVIDENCE")

        issues_list = []
        for i in all_issues:
            issues_list.append({
                "issue_id": i.issue_id,
                "rule_id": i.rule_id,
                "kind": i.kind,
                "artifact_id": i.artifact_id,
                "severity": i.severity,
                "evidence": i.evidence,
                "rationale": i.rationale,
                **i.extras,
            })

        report_no_hash = {
            "normalizer_report_schema": "1.0.0",
            "report_id": f"nr-{bundle.get('bundle_id', 'unknown')}",
            "engine_version": self.engine_version,
            "ruleset_id": self.ruleset.get("normalizer_ruleset_id"),
            "ruleset_version": self.ruleset.get("normalizer_ruleset_version"),
            "ruleset_hash": self.ruleset_hash,
            "config_id": self.config.get("normalizer_config_id"),
            "config_version": self.config.get("normalizer_config_version"),
            "config_hash": self.config_hash,
            "verdict": verdict,
            "issues": issues_list,
            "warnings": [i for i in issues_list if i["severity"] == "warning"],
            "executed_rules": executed,
            "short_circuited_rules": short_circuited,
            "bundle_sha256_recomputed": bundle_sha256_recomputed,
            "contract_version": 1,
            "schema_version": 1,
        }
        for k in report_no_hash.keys():
            if k not in REPORT_WHITELIST:
                raise EngineStop("report_volatile_field")
        report_hash = sha256_hex_str(canonical_json(report_no_hash))
        report = dict(report_no_hash)
        report["report_hash"] = report_hash
        for k in report.keys():
            if k not in REPORT_WHITELIST:
                raise EngineStop("report_volatile_field")
        if sha256_hex_str(canonical_json({k: v for k, v in report.items() if k != "report_hash"})) != report_hash:
            raise EngineStop("report_hash_mismatch")

        # Build metrics with artifact_metrics_map populated.
        artifact_metrics_per_artifact = {}
        for aid, am in ctx["artifact_metrics_map"].items():
            artifact_metrics_per_artifact[aid] = am
        metrics: dict[str, Any] = {
            "normalizer_metrics_schema": "1.0.0",
            "report_hash": report_hash,
            "ruleset_id": self.ruleset.get("normalizer_ruleset_id"),
            "ruleset_version": self.ruleset.get("normalizer_ruleset_version"),
            "ruleset_hash": self.ruleset_hash,
            "config_id": self.config.get("normalizer_config_id"),
            "config_version": self.config.get("normalizer_config_version"),
            "config_hash": self.config_hash,
            "engine_version": self.engine_version,
            "engine_status": "OK",
            "normalizer_verdict": verdict,
            "blocked_reason_kind": "bundle" if verdict == "BLOCKED" else "none",
            "engine_stop_reason_family": None,
            "engine_stop_detail": None,
            "placeholder_hits_per_artifact": {},
            "placeholder_hits_total": 0,
            "content_size_total": sum(len(b.encode("utf-8")) for b in artifacts.values()),
            "content_size_by_artifact": {k: len(v.encode("utf-8")) for k, v in artifacts.items()},
            "size_violations_per_artifact": {},
            "line_counts_by_artifact": {k: _line_count(v) for k, v in artifacts.items()},
            "json_parse_errors": 0,
            "hash_mismatches_total": 0,
            "bundle_sha256_recomputed": bundle_sha256_recomputed,
            "artifact_sha256_recomputed_per_artifact": {
                a["artifact_id"]: sha256_hex_str(artifacts.get(a["artifact_id"], ""))
                for a in bundle.get("artifacts", [])
            },
            "schema_consistency_errors": 0,
            "ac_references_unresolved": 0,
            "cross_reference_warnings": 0,
            "naming_drift_warnings": 0,
            "ambiguity_hits": 0,
            "cross_artifact_contradictions": 0,
            "reviewer_call_saved": reviewer_call_saved,
            "artifact_metrics": artifact_metrics_per_artifact,
            "contract_version": 1,
            "schema_version": 1,
        }
        for i in all_issues:
            if i.kind == "placeholder":
                m = metrics["placeholder_hits_per_artifact"]
                m[i.artifact_id] = m.get(i.artifact_id, 0) + 1
                metrics["placeholder_hits_total"] += 1
            elif i.kind == "size":
                m = metrics["size_violations_per_artifact"]
                m[i.artifact_id or "bundle"] = m.get(i.artifact_id or "bundle", 0) + 1
            elif i.kind == "hash_precheck":
                metrics["hash_mismatches_total"] += 1
            elif i.kind == "schema_consistency":
                if i.severity == "blocker":
                    metrics["schema_consistency_errors"] += 1
                else:
                    metrics["ac_references_unresolved"] += 1
            elif i.kind == "cross_reference":
                metrics["cross_reference_warnings"] += 1
            elif i.kind == "ambiguity":
                metrics["ambiguity_hits"] += 1
                if i.severity == "blocker":
                    metrics["cross_artifact_contradictions"] += 1
            elif i.kind == "evidence_readiness" and "JSON" in i.rationale:
                metrics["json_parse_errors"] += 1

        return {
            "engine_status": "OK",
            "normalizer_verdict": verdict,
            "report": report,
            "metrics": metrics,
        }

    def run(self, bundle, bundle_root, output_dir):
        """v1.1 path: write v1.1.0 reports; also produces v1.1 ruleset+config."""
        if not self.ruleset:
            try:
                self.load()
            except EngineStop as e:
                return self._engine_stop_payload(e, output_dir)
        try:
            result = self.normalize(bundle, bundle_root)
        except EngineStop as e:
            return self._engine_stop_payload(e, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ruleset_path = output_dir / "normalizer_ruleset.v1.1.0.yaml"
        config_path = output_dir / "normalizer_config.v1.1.0.yaml"
        try:
            import yaml as _yaml
            ruleset_to_write = {k: v for k, v in self.ruleset.items() if k != "ruleset_hash"}
            ruleset_to_write["ruleset_hash"] = self.ruleset_hash
            config_to_write = {k: v for k, v in self.config.items() if k != "config_hash"}
            config_to_write["config_hash"] = self.config_hash
            ruleset_path.write_text(_yaml.safe_dump(ruleset_to_write, sort_keys=True))
            config_path.write_text(_yaml.safe_dump(config_to_write, sort_keys=True))
        except Exception:
            pass
        report_path = output_dir / "normalizer_report.v1.0.0.json"
        metrics_path = output_dir / "normalizer_metrics.v1.0.0.json"
        report_path.write_text(json.dumps(result["report"], indent=2, sort_keys=True))
        metrics_path.write_text(json.dumps(result["metrics"], indent=2, sort_keys=True))
        return {
            "engine_status": "OK",
            "normalizer_verdict": result["normalizer_verdict"],
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
            "ruleset_path": str(ruleset_path),
            "config_path": str(config_path),
            "report": result["report"],
            "metrics": result["metrics"],
        }
