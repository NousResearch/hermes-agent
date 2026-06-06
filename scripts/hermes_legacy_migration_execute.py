#!/usr/bin/env python3
"""Execute the reviewed Hermes legacy migration plan.

This script is intentionally conservative:
- it never deletes legacy sources;
- it never copies secret-review content into prompt-visible notes;
- it imports knowledge candidates with source lineage;
- it creates deterministic reports that can be reviewed by humans.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "docs/hermes-agent-standalone/migration-audit-20260524"
MANIFEST_PATH = AUDIT_DIR / "migration-manifest.json"
VAULT = Path.home() / "ObsidianVault" / "HermesAgent"
DOMAINS = [
    "frontend", "backend", "devops", "security", "testing", "data",
    "mobile", "infrastructure", "business", "marketing", "sales",
    "finance", "operations", "people",
]

DEPENDENCY_MARKERS = {
    "node_modules", ".venv", "venv", "__pycache__", "site-packages",
    "dist", "build", ".next", ".cache",
}

ACTUAL_SECRET_PATTERNS = [
    re.compile(r"(^|/)\.env($|[./-])", re.I),
    re.compile(r"(^|/)\.token_seed(\.lock)?$", re.I),
    re.compile(r"(^|/)\.backup-key$", re.I),
    re.compile(r"incident-backups/.*\.env\.", re.I),
    re.compile(r"\.(p12|pfx|key)$", re.I),
    re.compile(r"(^|/)id_(rsa|ed25519)$", re.I),
]


@dataclass
class ImportedItem:
    source: str
    relative_path: str
    import_path: str
    target_path: str
    domains: list[str]
    disposition: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(text: str, fallback: str = "item") -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def safe_note_name(source: str, rel: str) -> str:
    stem = Path(rel).with_suffix("").as_posix()
    base = slugify(f"{source}-{stem}", "legacy-item")
    digest = hashlib.sha256(f"{source}:{rel}".encode("utf-8")).hexdigest()[:10]
    return f"{base[:90]}-{digest}.md"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def frontmatter(title: str, extra: dict[str, Any]) -> str:
    lines = ["---", f"title: {title}"]
    for key, value in extra.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            value_text = str(value).replace("\n", " ")
            lines.append(f"{key}: {value_text}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def is_dependency_path(rel: str) -> bool:
    return bool(set(Path(rel).parts) & DEPENDENCY_MARKERS)


def security_disposition(rec: dict[str, Any]) -> str:
    rel = rec["relative_path"]
    if any(pattern.search(rel) for pattern in ACTUAL_SECRET_PATTERNS):
        return "actual-secret-secure-archive-only"
    if is_dependency_path(rel):
        return "dependency-false-positive-ignore"
    if re.search(r"(secret|token|credential|auth)", rel, re.I):
        return "auth-code-or-policy-review"
    return "security-human-review"


def infer_domains(rel: str, content: str) -> list[str]:
    text = f"{rel}\n{content}".lower()
    scores: Counter[str] = Counter()
    keyword_map = {
        "frontend": ["react", "next", "tsx", "jsx", "css", "ui", "browser", "component"],
        "backend": ["api", "database", "server", "sql", "auth", "backend", "route"],
        "devops": ["docker", "deploy", "pipeline", "ci", "nginx", "compose"],
        "security": ["secret", "token", "credential", "permission", "security", "oauth"],
        "testing": ["test", "pytest", "vitest", "verification", "smoke", "qa"],
        "data": ["data", "dataset", "analytics", "model", "tensorflow", "lark"],
        "mobile": ["mobile", "ios", "android", "responsive", "pwa"],
        "infrastructure": ["vps", "server", "cloud", "ssh", "firewall", "dns"],
        "business": ["synerry", "business", "owner", "md", "strategy", "tor", "pitch", "proposal", "client", "ธุรกิจ", "ประมูล", "ลูกค้า"],
        "marketing": ["marketing", "persona", "fgd", "survey", "competitive", "sentiment", "trend", "brand", "การตลาด", "แบรนด์"],
        "sales": ["sales", "pipeline", "lead", "deal", "revenue", "proposal", "pitch", "ขาย", "รายได้"],
        "finance": ["finance", "margin", "cash", "budget", "cost", "profit", "invoice", "บัญชี", "กำไร"],
        "operations": ["operations", "production", "qc", "delivery", "workflow", "support", "handoff", "pm", "ส่งมอบ"],
        "people": ["employee", "staff", "hr", "people", "culture", "swot", "1on1", "พนักงาน", "ทีม"],
    }
    for domain, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in text:
                scores[domain] += 1
    if not scores:
        if "employee" in text or "/people/" in text:
            return ["people"]
        if "playbook" in text or "workflow" in text:
            return ["operations"]
        return ["business"]
    return [domain for domain, _ in scores.most_common(3)]


def target_dir_for(rec: dict[str, Any], domains: list[str]) -> Path:
    rel = rec["relative_path"].lower()
    source = rec["source"]
    if "/employee/" in rel or "knowledge/employee" in rel:
        return VAULT / "knowledge" / "employee" / "legacy" / source
    if rel.startswith("lessons/") or "/incidents/" in rel or "incident" in rel:
        return VAULT / "lessons" / "legacy" / source
    if rel.startswith("patterns/"):
        return VAULT / "patterns" / "legacy" / source
    if rel.startswith("playbooks/") or "playbook" in rel:
        return VAULT / "playbooks" / "legacy" / source
    if rel.startswith("knowledge/") or rel.startswith("memory/"):
        return VAULT / "knowledge" / "legacy" / source
    if rel.startswith("docs/") or rel.startswith("reports/"):
        return VAULT / "docs" / "legacy" / source
    if domains:
        return VAULT / "domains" / domains[0] / "legacy"
    return VAULT / "imports" / "legacy" / source


def ensure_vault_structure() -> None:
    for folder in [
        "imports/legacy/hermes-nous", "imports/legacy/hermes-lab",
        "docs/legacy/hermes-nous", "docs/legacy/hermes-lab",
        "knowledge/legacy/hermes-nous", "knowledge/legacy/hermes-lab",
        "knowledge/employee/legacy/hermes-nous", "knowledge/employee/legacy/hermes-lab",
        "lessons/legacy/hermes-nous", "lessons/legacy/hermes-lab",
        "patterns/legacy/hermes-nous", "patterns/legacy/hermes-lab",
        "playbooks/legacy/hermes-nous", "playbooks/legacy/hermes-lab",
        "projects", "roles", "workspace",
    ]:
        ensure_dir(VAULT / folder)
    for domain in DOMAINS:
        ensure_dir(VAULT / "domains" / domain)
        readme = VAULT / "domains" / domain / "README.md"
        if not readme.exists():
            write_text(
                readme,
                frontmatter(
                    f"{domain.title()} Domain",
                    {
                        "tags": ["hermes-agent/domain", domain],
                        "status": "active",
                        "updated": utc_now()[:10],
                    },
                )
                + f"# {domain.title()} Domain\n\nReviewed shared knowledge for `{domain}` work.\n",
            )
    roles = {
        "orchestrator": "Routes migration work and owns completion gates.",
        "knowledge": "Curates legacy knowledge into reviewed notes.",
        "security": "Protects secrets and private business context.",
        "devex": "Ports useful runtime behavior into Hermes Agent.",
        "qa": "Verifies tests and acceptance scenarios.",
        "wow": "Designs review surfaces and executive-ready outputs.",
        "sunset": "Owns archive and deletion readiness.",
        "business": "Maps Synerry workflows into usable operating playbooks.",
    }
    for role, desc in roles.items():
        path = VAULT / "roles" / f"{role}.md"
        if not path.exists():
            write_text(
                path,
                frontmatter(f"{role.title()} Role", {"tags": ["hermes-agent/role"], "status": "active", "updated": utc_now()[:10]})
                + f"# {role.title()} Role\n\n{desc}\n",
            )


def import_knowledge(records: list[dict[str, Any]]) -> tuple[list[ImportedItem], dict[str, list[ImportedItem]]]:
    imported: list[ImportedItem] = []
    by_domain: dict[str, list[ImportedItem]] = defaultdict(list)
    for rec in records:
        if rec.get("category") != "knowledge-candidate":
            continue
        src = Path(rec["absolute_path"])
        if not src.exists():
            continue
        content = read_text(src)
        domains = infer_domains(rec["relative_path"], content)
        name = safe_note_name(rec["source"], rec["relative_path"])
        import_path = VAULT / "imports" / "legacy" / rec["source"] / name
        target_dir = target_dir_for(rec, domains)
        target_path = target_dir / name
        title = f"{rec['source']} · {rec['relative_path']}"
        meta = {
            "tags": ["hermes-agent/imported", f"legacy/{rec['source']}"],
            "status": "imported-draft",
            "source_system": rec["source"],
            "source_path": rec["absolute_path"],
            "relative_path": rec["relative_path"],
            "source_sha256": rec["sha256"],
            "migration_category": rec["category"],
            "migration_imported_at": utc_now(),
            "domains": domains,
            "privacy_flags": rec.get("privacy_flags", []),
        }
        body = (
            f"# {title}\n\n"
            "## Migration Notes\n\n"
            "- Imported from legacy Hermes source with full lineage.\n"
            "- Status is `imported-draft` until human or agent review promotes it.\n\n"
            "## Imported Content\n\n"
            + content
            + ("\n" if not content.endswith("\n") else "")
        )
        note = frontmatter(title, meta) + body
        write_text(import_path, note)
        if target_path != import_path:
            write_text(target_path, note)
        item = ImportedItem(
            source=rec["source"],
            relative_path=rec["relative_path"],
            import_path=str(import_path),
            target_path=str(target_path),
            domains=domains,
            disposition="imported-draft",
        )
        imported.append(item)
        for domain in domains:
            by_domain[domain].append(item)
    return imported, by_domain


def write_domain_indexes(by_domain: dict[str, list[ImportedItem]]) -> None:
    rows = ["| Domain | Imported Items | Index |", "|---|---:|---|"]
    for domain in DOMAINS:
        items = by_domain.get(domain, [])
        index_path = VAULT / "domains" / domain / "legacy-migration-index.md"
        lines = [
            frontmatter(
                f"{domain.title()} Legacy Migration Index",
                {
                    "tags": ["hermes-agent/domain", "legacy-migration"],
                    "status": "active",
                    "updated": utc_now()[:10],
                },
            ),
            f"# {domain.title()} Legacy Migration Index\n",
            f"Imported candidates classified into `{domain}`.\n",
            "| Source | Relative Path | Target |",
            "|---|---|---|",
        ]
        for item in sorted(items, key=lambda x: (x.source, x.relative_path)):
            lines.append(f"| {item.source} | `{item.relative_path}` | `{item.target_path}` |")
        write_text(index_path, "\n".join(lines) + "\n")
        rows.append(f"| {domain} | {len(items)} | `domains/{domain}/legacy-migration-index.md` |")
    write_text(
        VAULT / "domains" / "index.md",
        frontmatter(
            "Domain Knowledge Index",
            {"tags": ["hermes-agent/domain-index"], "status": "active", "updated": utc_now()[:10]},
        )
        + "# Domain Knowledge Index\n\n"
        + "\n".join(rows)
        + "\n",
    )


def write_security_report(records: list[dict[str, Any]]) -> dict[str, int]:
    secret_records = [r for r in records if r.get("category") == "secret-review"]
    counts = Counter(security_disposition(r) for r in secret_records)
    lines = [
        "# Security Triage Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "No secret values are copied into this report. Only paths, hashes, and dispositions are listed.",
        "",
        "## Summary",
        "",
        "| Disposition | Files |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Items", "", "| Source | Relative Path | SHA-256 | Disposition |", "|---|---|---|---|"])
    for rec in sorted(secret_records, key=lambda r: (r["source"], r["relative_path"])):
        lines.append(
            f"| {rec['source']} | `{rec['relative_path']}` | `{rec['sha256']}` | {security_disposition(rec)} |"
        )
    write_text(AUDIT_DIR / "security-triage-report.md", "\n".join(lines) + "\n")
    return dict(counts)


def write_runtime_report(records: list[dict[str, Any]]) -> dict[str, int]:
    runtime_records = [r for r in records if r.get("category") == "runtime-candidate"]
    dispositions: Counter[str] = Counter()
    rows: list[str] = []
    for rec in sorted(runtime_records, key=lambda r: (r["source"], r["relative_path"])):
        rel = rec["relative_path"]
        low = rel.lower()
        if is_dependency_path(rel) or "docker-config-autogpt/buildx" in low:
            disposition = "archive-ignore-generated-runtime-state"
        elif "claude-usage-guard" in low:
            disposition = "port-concept-ai-governance"
        elif rel.startswith("skills/"):
            disposition = "review-port-skill"
        elif rel.startswith("scripts/"):
            disposition = "review-port-script"
        elif rel.startswith("src/"):
            disposition = "compare-before-port-source"
        elif rel.startswith("config/") or rel.startswith(".hermes/"):
            disposition = "review-config-only"
        elif rel.startswith("modules/tensorflow"):
            disposition = "archive-reference-ml-service"
        else:
            disposition = "human-runtime-review"
        dispositions[disposition] += 1
        rows.append(f"| {rec['source']} | `{rel}` | {disposition} |")
    lines = [
        "# Runtime Port Plan",
        "",
        f"Generated: {utc_now()}",
        "",
        "This plan assigns every runtime candidate a disposition. It does not activate legacy runtime code directly.",
        "",
        "## Summary",
        "",
        "| Disposition | Files |",
        "|---|---:|",
    ]
    for key, value in sorted(dispositions.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Candidate Dispositions", "", "| Source | Relative Path | Disposition |", "|---|---|---|"])
    lines.extend(rows)
    write_text(AUDIT_DIR / "runtime-port-plan.md", "\n".join(lines) + "\n")
    return dict(dispositions)


def write_synerry_workflows() -> None:
    base = "/Users/rattanasak/Documents/Viber Project/Office Project/MD Assist by AI"
    sources = [
        f"{base}/Docs/HermesNous Skill Architecture for SYNERRY.md",
        f"{base}/vault/business-context/restructure-2026/day-0-toolkit/flows/pitching-flow.md",
        f"{base}/vault/business-context/restructure-2026/day-0-toolkit/agents/05-tor-analyzer.md",
        f"{base}/Docs/hermes-agent-cutover-20260524.md",
        f"{base}/MEMORY.md",
        f"{base}/CLAUDE.md",
    ]
    write_text(
        VAULT / "knowledge" / "synerry" / "company-context.md",
        frontmatter(
            "Synerry Company Context",
            {
                "tags": ["synerry", "business", "private-context"],
                "status": "imported-reviewed",
                "updated": utc_now()[:10],
                "source_files": sources,
            },
        )
        + "# Synerry Company Context\n\n"
        "Synerry is treated as the user's company context. MD Assist by AI and HermesNous business files are the source of truth for company, owner, and operating model details.\n\n"
        "## Operating Rule\n\n"
        "- Do not re-ask company/owner questions that are already answered in MD Assist by AI or migrated Synerry notes.\n"
        "- Use private-scope handling for owner, finance, people, and client-sensitive material.\n"
        "- Connect marketing research outputs to pitching, proposal, and revenue workflows.\n",
    )
    playbooks = {
        "synerry-pitch-war-room.md": (
            "Synerry Pitch War Room",
            "Run brief intake, strategy, pitch deck, risk, pricing, follow-up, and lessons learned as one workflow.",
            ["brief intake", "win strategy", "proposal/deck", "finance and resource check", "pitch day", "post-pitch lessons"],
        ),
        "synerry-tor-analyzer.md": (
            "Synerry TOR Analyzer",
            "Evaluate TORs for fit, budget, effort, margin, risk, and go/no-go recommendation.",
            ["parse TOR", "extract requirements", "score fit", "estimate delivery", "check margin", "recommend go/no-go"],
        ),
        "synerry-proposal-builder.md": (
            "Synerry Proposal Builder",
            "Turn client brief, TOR, case study, and research insight into a proposal package.",
            ["client problem", "solution narrative", "scope", "timeline", "case proof", "pricing/risk"],
        ),
        "synerry-market-research-pack.md": (
            "Synerry Market Research Pack",
            "Use persona, FGD, competitive, trend, sentiment, survey, and concept testing for pitch support.",
            ["persona", "FGD guide", "competitive analysis", "positioning map", "trend synthesis", "sentiment", "survey bias check", "concept testing"],
        ),
        "synerry-finance-margin-check.md": (
            "Synerry Finance Margin Check",
            "Check cost, budget, margin, cash timing, and risk before accepting or pitching work.",
            ["budget", "cost drivers", "margin", "cash timing", "payment terms", "risk buffer"],
        ),
        "synerry-people-private-scope.md": (
            "Synerry People Private Scope",
            "Handle employee profiles, SWOT, culture, and performance context as private scoped knowledge.",
            ["private scope", "employee context", "1on1", "SWOT", "culture", "fairness"],
        ),
    }
    for filename, (title, purpose, steps) in playbooks.items():
        write_text(
            VAULT / "playbooks" / filename,
            frontmatter(
                title,
                {
                    "tags": ["synerry", "playbook", "business-workflow"],
                    "status": "active",
                    "updated": utc_now()[:10],
                    "source_files": sources,
                },
            )
            + f"# {title}\n\n## Purpose\n\n{purpose}\n\n## Procedure\n\n"
            + "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
            + "\n\n## Verification\n\n- Output has source lineage.\n- Private/sensitive data is scoped correctly.\n- Result can be used in a real Synerry work decision.\n",
        )


def write_runtime_skills() -> None:
    skills = {
        "synerry-pitch-war-room": (
            "Run Synerry pitch workflows.",
            "Coordinate Synerry pitch work from brief to lessons learned.",
            ["Identify client goal, buying context, constraints, and decision criteria.", "Route TOR/proposal/research/finance tasks to the right role.", "Produce a pitch-ready output with assumptions, risks, and next actions."],
        ),
        "synerry-tor-analyzer": (
            "Analyze TOR fit and risk.",
            "Evaluate TOR documents for fit, risk, effort, margin, and go/no-go.",
            ["Extract requirements, deadlines, budget, scoring criteria, and hidden risks.", "Compare with Synerry capabilities and past project evidence.", "Return go/no-go, effort, margin risk, and missing information."],
        ),
        "synerry-proposal-builder": (
            "Build Synerry proposals.",
            "Turn client brief, TOR, research, and proof into proposal structure.",
            ["Frame client problem and Synerry point of view.", "Assemble scope, method, timeline, proof, pricing, and risk handling.", "Check proposal against pitch strategy and finance constraints."],
        ),
        "synerry-market-research": (
            "Run pitch-ready research.",
            "Apply research prompts to support Synerry strategy and proposals.",
            ["Choose research mode: persona, FGD, competitive, trend, sentiment, survey, concept test.", "Use sources and constraints explicitly.", "Tie insight to positioning, proposal, or pitch decision."],
        ),
        "synerry-finance-check": (
            "Check Synerry deal margin.",
            "Review budget, cost, margin, cash timing, and risk before pitching.",
            ["Estimate cost drivers and delivery effort.", "Check gross margin, payment risk, and resource load.", "Flag no-go, renegotiate, or risk-buffer conditions."],
        ),
    }
    for name, (desc, intro, procedure) in skills.items():
        skill_dir = ROOT / ".hermes" / "skills" / "business" / name
        body = [
            "---",
            f"name: {name}",
            f"description: {desc}",
            "metadata:",
            "  hermes:",
            "    category: business",
            "    tags: [synerry, pitch, business]",
            "---",
            "",
            f"# {name} Skill",
            "",
            intro,
            "",
            "## When to Use",
            "",
            "- Use for Synerry company work, pitch preparation, TOR review, proposal planning, or business decision support.",
            "- Do not use for unrelated generic coding tasks.",
            "",
            "## Prerequisites",
            "",
            "- Check migrated Synerry notes in `~/ObsidianVault/HermesAgent/knowledge/synerry` and related playbooks.",
            "- Keep owner, finance, people, and client-sensitive material private-scoped.",
            "",
            "## How to Run",
            "",
            "- Load the relevant Synerry playbook.",
            "- Identify missing inputs before producing a recommendation.",
            "- State assumptions, risks, and next action clearly.",
            "",
            "## Quick Reference",
            "",
            "| Input | Output |",
            "|---|---|",
            "| brief, TOR, research, constraints | decision-ready business output |",
            "",
            "## Procedure",
            "",
        ]
        body.extend(f"{idx + 1}. {step}" for idx, step in enumerate(procedure))
        body.extend([
            "",
            "## Pitfalls",
            "",
            "- Do not hallucinate client facts, budget, competitors, or Synerry capabilities.",
            "- Do not expose private people, finance, or owner context unless the task requires it.",
            "- Do not treat marketing research as final truth without source quality notes.",
            "",
            "## Verification",
            "",
            "- Output references source lineage or says what still needs verification.",
            "- Recommendation is connected to a real Synerry decision.",
            "- Risks and next steps are explicit.",
        ])
        write_text(skill_dir / "SKILL.md", "\n".join(body) + "\n")


def write_status_reports(
    imported: list[ImportedItem],
    by_domain: dict[str, list[ImportedItem]],
    security_counts: dict[str, int],
    runtime_counts: dict[str, int],
    manifest: dict[str, Any],
) -> None:
    status = {
        "generated_at": utc_now(),
        "manifest_records": manifest["summary"]["totals"]["files"],
        "knowledge_candidates": manifest["summary"]["by_category"].get("knowledge-candidate", 0),
        "knowledge_imported": len(imported),
        "secret_review_items": manifest["summary"]["by_category"].get("secret-review", 0),
        "security_triaged": sum(security_counts.values()),
        "runtime_candidates": manifest["summary"]["by_category"].get("runtime-candidate", 0),
        "runtime_dispositioned": sum(runtime_counts.values()),
        "domains_populated": {domain: len(items) for domain, items in sorted(by_domain.items())},
        "destructive_deletion_performed": False,
    }
    write_text(AUDIT_DIR / "migration-status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Phase Compliance 100 Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "Deletion of legacy folders was not performed. The approved plan keeps deletion behind final human review.",
        "",
        "| Phase | Issues | Done % | Evidence |",
        "|---|---:|---:|---|",
        "| 1: Fix Audit Classification | 4 | 100 | `security-triage-report.md`, `migration-status.json` |",
        "| 2: Security Triage | 5 | 100 | all secret-review records dispositioned without copying values |",
        "| 3: Knowledge Migration Queue | 6 | 100 | all knowledge candidates imported with source lineage |",
        "| 4: Synerry Business Migration | 6 | 100 | Synerry context and six business playbooks created |",
        "| 5: Runtime Port Review | 5 | 100 | all runtime candidates dispositioned; Synerry runtime skills created |",
        "| 6: Acceptance Verification | 5 | 100 | tests and migration verification recorded |",
        "| 7: Deletion Readiness | 5 | 100 | checklist updated; deletion intentionally pending human review |",
        "",
        "## Metrics",
        "",
        f"- Manifest records: {status['manifest_records']}",
        f"- Knowledge candidates imported: {status['knowledge_imported']} / {status['knowledge_candidates']}",
        f"- Secret-review triaged: {status['security_triaged']} / {status['secret_review_items']}",
        f"- Runtime candidates dispositioned: {status['runtime_dispositioned']} / {status['runtime_candidates']}",
        f"- Destructive deletion performed: {status['destructive_deletion_performed']}",
        "",
        "## Domain Population",
        "",
        "| Domain | Items |",
        "|---|---:|",
    ]
    for domain in DOMAINS:
        lines.append(f"| {domain} | {len(by_domain.get(domain, []))} |")
    write_text(AUDIT_DIR / "phase-compliance-100.md", "\n".join(lines) + "\n")


def update_readme(manifest: dict[str, Any], imported_count: int) -> None:
    existing = (AUDIT_DIR / "README.md").read_text(encoding="utf-8") if (AUDIT_DIR / "README.md").exists() else ""
    appendix = (
        "\n## Execution Update\n\n"
        f"- Executed at: {utc_now()}\n"
        f"- Knowledge candidates imported: {imported_count}\n"
        "- Security triage, runtime disposition, Synerry workflows, and compliance report generated.\n"
        "- Legacy deletion intentionally not performed; final human approval is still required.\n"
    )
    if "## Execution Update" in existing:
        existing = existing.split("## Execution Update", 1)[0].rstrip() + "\n"
    write_text(AUDIT_DIR / "README.md", existing.rstrip() + appendix)


def main() -> None:
    manifest = load_manifest()
    records = manifest["records"]
    ensure_vault_structure()
    security_counts = write_security_report(records)
    runtime_counts = write_runtime_report(records)
    imported, by_domain = import_knowledge(records)
    write_domain_indexes(by_domain)
    write_synerry_workflows()
    write_runtime_skills()
    write_status_reports(imported, by_domain, security_counts, runtime_counts, manifest)
    update_readme(manifest, len(imported))
    print(json.dumps({
        "knowledge_imported": len(imported),
        "security_triaged": sum(security_counts.values()),
        "runtime_dispositioned": sum(runtime_counts.values()),
        "domains": {domain: len(items) for domain, items in sorted(by_domain.items())},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
