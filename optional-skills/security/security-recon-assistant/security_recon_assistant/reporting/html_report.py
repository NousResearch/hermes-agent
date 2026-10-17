from __future__ import annotations

import json

from jinja2 import Template

from ..core.models import ScopeConfig
from ..scanners.base import ScanResult


class HTMLReportGenerator:
        def __init__(self):
                self.template = Template(
                        """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Security Recon Report</title>
    <style>
        body { font-family: Inter, Arial, sans-serif; margin: 2rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }
        th { background: #f5f5f5; text-align: left; }
        .ok { color: #137333; }
        .ko { color: #a50e0e; }
    </style>
</head>
<body>
    <h1>Security Recon Report</h1>
    <p><strong>Target:</strong> {{ target }}</p>
    <p><strong>Scanners:</strong> {{ results|length }}</p>
    <h2>Results</h2>
    <table>
        <tr><th>Scanner</th><th>Status</th><th>Execution Time</th><th>Findings</th></tr>
        {% for r in results %}
        <tr>
            <td>{{ r.scanner_name }}</td>
            <td class="{{ 'ok' if r.success else 'ko' }}">{{ 'SUCCESS' if r.success else 'FAILED' }}</td>
            <td>{{ r.execution_time if r.execution_time is not none else '-' }}</td>
            <td>
                {% for f in r.findings %}
                    <div><strong>[{{ f.severity|upper }}]</strong> {{ f.title }} — {{ f.target }}</div>
                {% else %}
                    <div>No findings</div>
                {% endfor %}
            </td>
        </tr>
        {% endfor %}
    </table>
    <h2>Scope</h2>
    <pre>{{ scope_json }}</pre>
</body>
</html>
"""
                )

        def generate(self, target: str, results: list[ScanResult], scope: ScopeConfig) -> str:
                return self.template.render(
                        target=target,
                        results=results,
                        scope_json=json.dumps(
                                {
                                        "allowed_domains": sorted(scope.allowed_domains),
                                        "excluded_domains": sorted(scope.excluded_domains),
                                        "max_depth": scope.max_depth,
                                        "rate_limit": scope.rate_limit,
                                        "check_ssl": scope.check_ssl,
                                },
                                indent=2,
                                ensure_ascii=False,
                        ),
                )

        def save(self, filepath: str, target: str, results: list[ScanResult], scope: ScopeConfig) -> None:
                with open(filepath, "w", encoding="utf-8") as f:
                        f.write(self.generate(target=target, results=results, scope=scope))


class HtmlReport:
        def __init__(self, findings):
                self.findings = findings

        def write(self, path):
                template = Template(
                        """
<!doctype html>
<html><head><title>Security Recon Report</title></head>
<body>
<h1>Security Recon Report</h1>
<pre>{{ findings }}</pre>
</body></html>
"""
                )
                with open(path, "w", encoding="utf-8") as f:
                        f.write(template.render(findings=self.findings))
