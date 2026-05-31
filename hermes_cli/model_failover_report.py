"""Model-failover traceability report helpers.

These helpers render the final HTML/dashboard sections promised by the model
failover + Linear traceability goal. They intentionally accept sparse dicts so
closeout code can produce an honest report even when some evidence is missing.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

REQUIRED_TRACEABILITY_SECTIONS = (
    "Model Routing and Failures",
    "Attribution Reconciliation",
    "Helper/Code Reliability",
    "Opus Usage Audit",
    "Offline Receipts and Residual Risks",
)

_SECTION_KEYS = {
    "Model Routing and Failures": "model_routing_and_failures",
    "Attribution Reconciliation": "attribution_reconciliation",
    "Helper/Code Reliability": "helper_code_reliability",
    "Opus Usage Audit": "opus_usage_audit",
    "Offline Receipts and Residual Risks": "offline_receipts_and_residual_risks",
}

_DEFAULTS = {
    "model_routing_and_failures": "None recorded.",
    "attribution_reconciliation": "None recorded.",
    "helper_code_reliability": "None recorded.",
    "opus_usage_audit": "None recorded.",
    "offline_receipts_and_residual_risks": "None recorded.",
}


def _coerce_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        lines: list[str] = []
        for item in value:
            lines.extend(_coerce_lines(item))
        return lines
    if isinstance(value, Mapping):
        return [f"{key}: {val}" for key, val in value.items()]
    return [str(value)]


def _section_body(value: Any, default: str) -> str:
    lines = _coerce_lines(value)
    if not lines:
        lines = [default]
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    return f"<ul>{items}</ul>"


def render_model_failover_traceability_sections(evidence: Mapping[str, Any] | None = None) -> str:
    """Return escaped HTML for the five required traceability sections."""

    evidence = evidence or {}
    chunks: list[str] = []
    for title in REQUIRED_TRACEABILITY_SECTIONS:
        key = _SECTION_KEYS[title]
        value = evidence.get(key, evidence.get(title))
        chunks.append(f"<section><h2>{escape(title)}</h2>{_section_body(value, _DEFAULTS[key])}</section>")
    return "\n".join(chunks)


def missing_model_failover_traceability_sections(html: str | None) -> list[str]:
    """Return any required section headings missing from rendered HTML/Markdown."""

    rendered = html or ""
    return [title for title in REQUIRED_TRACEABILITY_SECTIONS if title not in rendered]


def build_model_failover_dashboard_markdown(evidence: Mapping[str, Any] | None = None) -> str:
    """Return a Linear-dashboard-friendly Markdown traceability summary."""

    evidence = evidence or {}
    sections: list[str] = ["# Model Failover Traceability Dashboard"]
    for title in REQUIRED_TRACEABILITY_SECTIONS:
        key = _SECTION_KEYS[title]
        lines = _coerce_lines(evidence.get(key, evidence.get(title))) or [_DEFAULTS[key]]
        sections.append(f"## {title}")
        sections.extend(f"- {line}" for line in lines)
    return "\n\n".join(sections) + "\n"
