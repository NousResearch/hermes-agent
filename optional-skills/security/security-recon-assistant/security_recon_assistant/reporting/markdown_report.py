from typing import Any, Dict, List
from datetime import datetime

class MarkdownReportGenerator:
    """Generates an AI-friendly Markdown report from scan results."""
    def __init__(self, **kwargs: Any) -> None: pass

    def save(self, filepath: str, target: str, results: List[Any], scope: Any, **kwargs: Any) -> None:
        md = self._generate_markdown(target, results, scope, **kwargs)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

    def _generate_markdown(self, target: str, results: List[Any], scope: Any, **kwargs: Any) -> str:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Security Recon Report", f"**Target:** `{target}`", f"**Date:** `{date}`",
            "", "## Summary", f"- **Scanners:** {len(results)}",
            f"- **Findings:** {sum(len(r.findings) for r in results)}",
            "", "## Details", ""
        ]
        for r in results:
            lines.extend([
                f"### {r.scanner_name}", f"- **Success:** {'✅' if r.success else '❌'}",
                f"- **Command:** `{r.command}`", f"- **Findings:** {len(r.findings)}", ""
            ])
            if r.findings:
                lines.append("#### Findings:")
                lines.extend([f"- **{f.type}**: `{f.data}` (Confidence: {f.confidence})" for f in r.findings])
            if not r.success and r.error_message:
                lines.extend(["#### Error:", "```text", r.error_message, "```"])
            lines.append("")
        return "\n".join(lines)
