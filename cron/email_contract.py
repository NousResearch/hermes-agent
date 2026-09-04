"""Central generation contract for cron jobs delivered by email.

The platform adapter owns MIME construction and transport normalization.  This
module owns the boundary immediately before delivery: report jobs that promise
HTML must actually produce a complete HTML document, otherwise the scheduler
repairs the response once and fails closed.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Mapping

EMAIL_CONTRACT_FAILED_MARKER = "[email_contract_failed]"

CANONICAL_PURPLE_PALETTE = {
    "header_start": "#6C27D7",
    "header_end": "#4F1E9C",
    "heading": "#4C1D95",
    "text": "#1F2430",
    "callout": "#F7F5FC",
    "callout_border": "#E0D7F5",
    "table_header": "#F3F0FB",
    "badge": "#EDE9FE",
    "border": "#E5E7EB",
    "body": "#FFFFFF",
    "outer": "#F4F5F7",
    "footer": "#FAFAFA",
}
CANONICAL_PURPLE_PALETTE_ID = "purple_v1"


@dataclass(frozen=True)
class EmailContract:
    format: str
    retries: int = 0
    palette: str | None = None


@dataclass(frozen=True)
class ContractResult:
    content: str
    errors: tuple[str, ...]
    contract: EmailContract | None

    @property
    def valid(self) -> bool:
        return not self.errors


def _deliver_tokens(deliver: Any) -> list[str]:
    if isinstance(deliver, str):
        raw = deliver
    elif isinstance(deliver, (list, tuple, set)):
        raw = ",".join(str(item) for item in deliver)
    else:
        raw = str(deliver or "")
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def job_delivers_to_email(job: Mapping[str, Any]) -> bool:
    """Return whether the stored routing value explicitly targets email."""
    return any(token == "email" or token.startswith("email:") for token in _deliver_tokens(job.get("deliver")))


def resolve_email_contract(job: Mapping[str, Any]) -> EmailContract | None:
    """Resolve an explicit contract or infer one for existing email jobs.

    Existing jobs predate ``email_contract``.  An email job whose prompt asks
    for a doctype/title/branded HTML document is therefore protected
    automatically.  Plain operational alerts keep their historical behavior.
    """
    if not job_delivers_to_email(job):
        return None

    configured = job.get("email_contract")
    if configured is False:
        return None
    if isinstance(configured, str):
        configured = {"format": configured}
    if isinstance(configured, Mapping):
        fmt = str(configured.get("format") or "").strip().lower().replace("-", "_")
        if fmt in {"html", "branded_html"}:
            retries = configured.get("retries", configured.get("retry_invalid_output", 1))
            try:
                retries = max(0, min(1, int(retries)))
            except (TypeError, ValueError):
                retries = 1
            palette = str(configured.get("palette") or "").strip().lower() or None
            return EmailContract("branded_html", retries, palette)
        if fmt in {"text", "plain", "plain_text"}:
            return EmailContract("plain_text", 0)

    prompt = str(job.get("prompt") or "").lower()
    html_signals = (
        "<!doctype html>" in prompt
        or "bare-html document" in prompt
        or "bare html document" in prompt
        or "branded html" in prompt
    )
    if html_signals and ("<title>" in prompt or "email subject" in prompt or "subject line" in prompt):
        return EmailContract("branded_html", 1)
    return EmailContract("plain_text", 0)


def _normalize_html_document(content: str) -> str:
    """Extract exactly one complete doctype HTML envelope when present."""
    text = str(content or "")
    start = re.search(r"(?is)<!doctype\s+html[^>]*>\s*<html\b", text)
    if not start:
        return text.strip()
    closing = list(re.finditer(r"(?is)</html\s*>", text[start.start():]))
    if not closing:
        return text.strip()
    end = start.start() + closing[-1].end()
    return text[start.start():end].strip()


def validate_email_output(job: Mapping[str, Any], content: str) -> ContractResult:
    """Normalize and validate one generated cron response."""
    contract = resolve_email_contract(job)
    text = str(content or "")
    if contract is None:
        return ContractResult(text, (), None)

    if contract.format == "plain_text":
        plain_errors: tuple[str, ...] = () if text.strip() else ("response is empty",)
        return ContractResult(text.strip(), plain_errors, contract)

    normalized = _normalize_html_document(text)
    errors: list[str] = []
    lowered = normalized.lower()
    if not re.match(r"(?is)^<!doctype\s+html[^>]*>\s*<html\b", normalized):
        errors.append("response does not start with a complete <!DOCTYPE html> document")
    if not re.search(r"(?is)</html\s*>$", normalized):
        errors.append("response does not end with </html>")
    if len(re.findall(r"(?is)<!doctype\s+html", normalized)) != 1:
        errors.append("response must contain exactly one HTML doctype")
    if len(re.findall(r"(?is)</html\s*>", normalized)) != 1:
        errors.append("response must contain exactly one closing </html> tag")

    title_match = re.search(r"(?is)<title\b[^>]*>(.*?)</title\s*>", normalized)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    if not title:
        errors.append("HTML document has no non-empty <title>")
    elif title.lower() in {"hermes agent", "re: hermes agent"} or title.lower().startswith("re:"):
        errors.append("HTML title is generic or reply-prefixed")

    if "```" in normalized:
        errors.append("HTML document contains a Markdown code fence")
    if "[truncated]" in lowered:
        errors.append("HTML document contains a truncation marker")

    if contract.palette == CANONICAL_PURPLE_PALETTE_ID:
        upper = normalized.upper()
        missing = [
            value
            for value in CANONICAL_PURPLE_PALETTE.values()
            if value not in upper
        ]
        gradient = re.search(
            r"(?is)linear-gradient\([^)]*#6c27d7[^)]*#4f1e9c[^)]*\)",
            normalized,
        )
        if missing:
            errors.append(
                "HTML document is missing canonical palette colors: "
                + ", ".join(missing)
            )
        if not gradient:
            errors.append(
                "HTML document has no #6C27D7 to #4F1E9C header gradient"
            )

    return ContractResult(normalized, tuple(errors), contract)


def build_repair_prompt(job: Mapping[str, Any], errors: tuple[str, ...]) -> str:
    """Build the single same-session correction turn for invalid HTML."""
    job_name = str(job.get("name") or job.get("id") or "cron job")
    reasons = "; ".join(errors) or "unknown HTML contract violation"
    contract = resolve_email_contract(job)
    palette_instruction = ""
    if contract and contract.palette == CANONICAL_PURPLE_PALETTE_ID:
        palette_instruction = (
            " Use the canonical palette exactly: header linear-gradient from #6C27D7 "
            "to #4F1E9C; headings #4C1D95; text #1F2430; callout #F7F5FC "
            "with #E0D7F5 border; table headers #F3F0FB; badges #EDE9FE; "
            "borders #E5E7EB; body #FFFFFF; outer background #F4F5F7; "
            "footer #FAFAFA. Include every palette color in the document CSS."
        )
    return (
        "EMAIL OUTPUT CONTRACT REPAIR (mandatory).\n\n"
        f"The final response for {job_name!r} was rejected before delivery: {reasons}.\n"
        "Using the research and tool results already present in this conversation, "
        "rebuild the complete report now. Do not call tools or repeat side effects. "
        "Return exactly one bare HTML document beginning with <!DOCTYPE html> and "
        "ending with </html>. Include a non-empty, human-readable <title>. Do not "
        "include narration, Markdown fences, preamble, postamble, [SILENT], or "
        "[PIPELINE_FAILED]."
        f"{palette_instruction} This is the one permitted repair attempt."
    )


def format_contract_error(errors: tuple[str, ...]) -> str:
    return f"{EMAIL_CONTRACT_FAILED_MARKER} " + "; ".join(errors)


def render_contract_failure_email(
    job: Mapping[str, Any], error: str, *, generated: str, run_id: str
) -> str:
    """Render a deterministic branded alert after the one repair fails."""
    job_name = html.escape(str(job.get("name") or job.get("id") or "Cron job"))
    detail = html.escape(str(error).replace(EMAIL_CONTRACT_FAILED_MARKER, "").strip())
    title = f"Email Report Generation Failed — {job_name}"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0;padding:0;background-color:#F4F5F7;color:#1F2430;font-family:Arial,sans-serif;">
<table role="presentation" style="width:100%;max-width:640px;border-collapse:collapse;background-color:#FFFFFF;border:1px solid #E5E7EB;">
<tr><td style="background:linear-gradient(135deg,#6C27D7 0%,#4F1E9C 100%);color:#FFFFFF;padding:20px;"><h1 style="margin:0;font-size:22px;color:#FFFFFF;">{title}</h1></td></tr>
<tr><td style="padding:20px;"><h2 style="margin:0 0 12px 0;font-size:18px;color:#4C1D95;">Report delivery blocked</h2>
<p style="margin:0 0 12px 0;">The generated response failed the centralized email-output contract. The malformed response was not delivered.</p>
<p style="margin:0 0 18px 0;padding:12px;background-color:#F7F5FC;border:1px solid #E0D7F5;"><strong>Validation error:</strong> {detail}</p>
<table role="presentation" style="display:none;background-color:#F3F0FB;border-color:#E5E7EB;"><tr><td style="background-color:#EDE9FE;">Canonical palette</td></tr></table>
<p style="background-color:#FAFAFA;color:#1F2430;font-size:12px;margin:0;padding:12px;">Generated automatically by Hermes Agent<br>Cron Job: {job_name}<br>Generated: {html.escape(generated)}<br>Status: Failed before report delivery<br>Run ID: {html.escape(run_id)}</p>
</td></tr></table></body></html>"""
