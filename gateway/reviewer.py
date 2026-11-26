"""
gateway/reviewer.py — automated code-review via the h2reviewer profile.

Public API
----------
review(artifact_kind, *, paths, diff, branch, context, profile) -> ReviewVerdict

The function:
1. Renders a prompt asking the reviewer to assess the artifact and output a
   fenced ``reviewer-verdict`` JSON block.
2. Shells out via ``hermes -p <profile> chat -q "<prompt>"`` (same pattern as
   ``_default_spawn`` in kanban_db.py), capturing stdout.
3. Parses the LAST fenced ``reviewer-verdict`` block in the response.
4. Returns a structured ``ReviewVerdict``.

Fail-safe: on any parse error the verdict is BLOCKED and ``parsed_ok=False``
so a human can inspect ``raw_response`` and decide.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    severity: Literal["critical", "warning", "info"]
    file: str         # "path:line" or plain "path"
    issue: str


@dataclass
class ReviewVerdict:
    verdict: Literal["APPROVED", "BLOCKED", "NEEDS_INFO"]
    findings: list[Finding]
    needs_info: str | None
    summary: str
    raw_response: str
    parsed_ok: bool   # False when the fenced block was missing or malformed


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a code reviewer. Evaluate the following artifact carefully.

Artifact kind: {kind}
{extra_sections}
{context_section}
Instructions
------------
* Identify issues by severity: critical (must-fix before merge), warning
  (should-fix), or info (nice-to-have / stylistic).
* After your review prose, output **exactly one** fenced block tagged
  ``reviewer-verdict`` containing a JSON object with keys:
    - "verdict": one of "APPROVED", "BLOCKED", or "NEEDS_INFO"
    - "findings": list of {{"severity": ..., "file": ..., "issue": ...}}
    - "needs_info": null or a string describing what clarification is needed
    - "summary": one-sentence summary of the overall assessment

Example fence:

```reviewer-verdict
{{"verdict": "APPROVED", "findings": [], "needs_info": null, "summary": "Looks good."}}
```

Now write your review:
"""

_VALID_VERDICTS: frozenset[str] = frozenset({"APPROVED", "BLOCKED", "NEEDS_INFO"})
_VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "warning", "info"})


def _render_prompt(
    artifact_kind: Literal["pr", "branch", "files"],
    *,
    paths: list[str] | None,
    diff: str | None,
    branch: str | None,
    context: str | None,
) -> str:
    sections: list[str] = []

    if branch:
        sections.append(f"Branch: {branch}")

    if paths:
        sections.append("Files:\n" + "\n".join(f"  - {p}" for p in paths))

    if diff:
        # Truncate diffs that are very large to avoid prompt bloat; reviewers
        # can always ask for more via NEEDS_INFO.
        max_diff_chars = 40_000
        if len(diff) > max_diff_chars:
            diff = diff[:max_diff_chars] + "\n... [diff truncated] ..."
        sections.append(f"Diff:\n```diff\n{diff}\n```")

    extra = "\n\n".join(sections)
    context_section = (
        f"\nAdditional context:\n{context}" if context else ""
    )

    return _PROMPT_TEMPLATE.format(
        kind=artifact_kind,
        extra_sections=extra,
        context_section=context_section,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches the LAST ```reviewer-verdict ... ``` block in the response.
_FENCE_RE = re.compile(
    r"```reviewer-verdict\s*\n(.*?)```",
    re.DOTALL,
)


def _parse_response(raw: str) -> ReviewVerdict:
    """Extract and validate the last ``reviewer-verdict`` fence in *raw*.

    Returns a BLOCKED / parsed_ok=False verdict on any failure.
    """

    def _blocked(msg: str = "") -> ReviewVerdict:
        return ReviewVerdict(
            verdict="BLOCKED",
            findings=[],
            needs_info=None,
            summary=msg or "Parse failure — human review required.",
            raw_response=raw,
            parsed_ok=False,
        )

    matches = _FENCE_RE.findall(raw)
    if not matches:
        return _blocked("No reviewer-verdict block found in response.")

    json_text = matches[-1].strip()  # use the LAST match

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return _blocked(f"JSON parse error: {exc}")

    if not isinstance(data, dict):
        return _blocked("reviewer-verdict block is not a JSON object.")

    verdict_raw = data.get("verdict", "")
    if verdict_raw not in _VALID_VERDICTS:
        return _blocked(
            f"Invalid verdict value {verdict_raw!r}; expected one of "
            f"{sorted(_VALID_VERDICTS)}."
        )

    verdict: Literal["APPROVED", "BLOCKED", "NEEDS_INFO"] = verdict_raw  # type: ignore[assignment]

    raw_findings = data.get("findings", [])
    if not isinstance(raw_findings, list):
        return _blocked("'findings' must be a JSON array.")

    findings: list[Finding] = []
    for i, f in enumerate(raw_findings):
        if not isinstance(f, dict):
            return _blocked(f"findings[{i}] is not an object.")
        sev = f.get("severity", "")
        if sev not in _VALID_SEVERITIES:
            return _blocked(
                f"findings[{i}].severity {sev!r} is not one of "
                f"{sorted(_VALID_SEVERITIES)}."
            )
        file_loc = f.get("file", "")
        issue = f.get("issue", "")
        if not isinstance(file_loc, str) or not isinstance(issue, str):
            return _blocked(f"findings[{i}] 'file' and 'issue' must be strings.")
        findings.append(
            Finding(
                severity=sev,  # type: ignore[arg-type]
                file=file_loc,
                issue=issue,
            )
        )

    needs_info = data.get("needs_info")
    if needs_info is not None and not isinstance(needs_info, str):
        needs_info = str(needs_info)

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    return ReviewVerdict(
        verdict=verdict,
        findings=findings,
        needs_info=needs_info,
        summary=summary,
        raw_response=raw,
        parsed_ok=True,
    )


# ---------------------------------------------------------------------------
# Profile invocation
# ---------------------------------------------------------------------------

def _resolve_hermes_argv() -> list[str]:
    """Return the argv prefix that invokes hermes (same logic as kanban_db.py)."""
    return [sys.executable, "-m", "hermes_cli.main"]


def _invoke_profile(profile: str, prompt: str) -> str:
    """Run ``hermes -p <profile> chat -q <prompt>`` and return stdout.

    Raises ``subprocess.CalledProcessError`` on non-zero exit,
    ``FileNotFoundError`` if the hermes executable is missing.
    """
    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile,
        "chat",
        "-q", prompt,
    ]
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def review(
    artifact_kind: Literal["pr", "branch", "files"],
    *,
    paths: list[str] | None = None,
    diff: str | None = None,
    branch: str | None = None,
    context: str | None = None,
    profile: str = "h2reviewer",
) -> ReviewVerdict:
    """Spawn the reviewer profile, capture stdout, parse the verdict block.

    Parameters
    ----------
    artifact_kind:
        One of ``"pr"``, ``"branch"``, or ``"files"``.
    paths:
        List of file paths (or "path:line" strings) that are in scope.
    diff:
        Unified diff to include in the prompt (e.g. from ``git diff``).
    branch:
        Branch name to mention in the prompt.
    context:
        Free-form additional context appended to the prompt.
    profile:
        Hermes profile name to invoke.  Defaults to ``"h2reviewer"``.

    Returns
    -------
    ReviewVerdict
        Structured verdict.  On any parse failure the verdict is ``"BLOCKED"``
        and ``parsed_ok`` is ``False``; ``raw_response`` always contains the
        full subprocess output so a human can inspect it.
    """
    prompt = _render_prompt(
        artifact_kind,
        paths=paths,
        diff=diff,
        branch=branch,
        context=context,
    )

    try:
        raw = _invoke_profile(profile, prompt)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raw = f"[reviewer invocation failed: {exc}]"
        return ReviewVerdict(
            verdict="BLOCKED",
            findings=[],
            needs_info=None,
            summary=f"Reviewer process error: {exc}",
            raw_response=raw,
            parsed_ok=False,
        )

    return _parse_response(raw)
