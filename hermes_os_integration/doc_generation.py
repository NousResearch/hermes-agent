"""Generate architecture documents from review output."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from .architecture_first import ArchitectureReviewReport, project_document_templates, render_review_report


@dataclass(frozen=True)
class DocumentWrite:
    path: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class DocumentGenerationResult:
    writes: List[DocumentWrite] = field(default_factory=list)
    artifact_ref: str = ""


def generate_missing_docs(project_path: str, report: ArchitectureReviewReport, overwrite: bool = False):
    templates = project_document_templates()
    docs_dir = os.path.join(project_path, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    writes = []
    for doc in report.missing_documents:
        path = os.path.join(docs_dir, doc)
        if os.path.exists(path) and not overwrite:
            writes.append(DocumentWrite(path=path, status="skipped", reason="exists"))
            continue
        content = _document_content(doc, templates.get(doc, ""), report)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        writes.append(DocumentWrite(path=path, status="written"))
    return DocumentGenerationResult(writes=writes)


def write_review_artifact(project_path: str, report: ArchitectureReviewReport, overwrite: bool = False):
    artifact_dir = os.path.join(project_path, ".hermes", "architecture-reviews")
    os.makedirs(artifact_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(artifact_dir, stamp + "-" + report.project_id + ".md")
    if os.path.exists(path) and not overwrite:
        return DocumentWrite(path=path, status="skipped", reason="exists")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_review_report(report))
    return DocumentWrite(path=path, status="written")


def roadmap_from_review(report: ArchitectureReviewReport):
    items = []
    for index, gap in enumerate(report.critical_gaps + report.priority_roadmap, start=1):
        items.append({
            "id": "roadmap-%03d" % index,
            "title": gap,
            "priority": "high" if index <= 3 else "medium",
            "source": "architecture-review:" + report.project_id,
        })
    return items


def _document_content(doc: str, template: str, report: ArchitectureReviewReport):
    lines = [
        template.rstrip(),
        "",
        "## Architecture Review Trace",
        "",
        "- Project: " + report.project_id,
        "- Score: " + str(report.architecture_score),
        "- Blocked: " + str(report.blocked).lower(),
        "",
        "## Review Findings",
    ]
    lines.extend("- " + item for item in (report.critical_gaps or ["No critical gaps"]))
    lines.append("")
    lines.append("## Next Steps")
    lines.extend("- " + item for item in (report.priority_roadmap or ["Keep this document current"]))
    return "\n".join(lines) + "\n"
