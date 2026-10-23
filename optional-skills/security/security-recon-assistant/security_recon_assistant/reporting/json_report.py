from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..core.models import ScopeConfig
from ..scanners.base import ScanResult


class JSONReportGenerator:
    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def _flatten_findings(self, results: Iterable[ScanResult]) -> list[dict]:
        flattened: list[dict] = []
        seen = set()
        for result in results:
            for finding in result.findings:
                key = (result.scanner_name, finding.severity, finding.title)
                if key in seen:
                    continue
                seen.add(key)
                row = finding.to_dict()
                row["scanner"] = result.scanner_name
                flattened.append(row)
        return flattened

    def generate(
        self,
        target: str,
        results: Iterable[ScanResult],
        scope: ScopeConfig,
        total_duration: float,
        custom_metadata: Optional[dict] = None,
    ) -> dict:
        results = list(results)
        findings = self._flatten_findings(results)

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in findings:
            sev = (finding.get("severity") or "").lower()
            if sev in severity_counts:
                severity_counts[sev] += 1

        metadata = {
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_scanners": len(results),
            "total_duration": total_duration,
        }
        if custom_metadata:
            metadata.update(custom_metadata)

        return {
            "report": "security_recon_report",
            "metadata": metadata,
            "scope": {
                "allowed_domains": sorted(scope.allowed_domains),
                "excluded_domains": sorted(scope.excluded_domains),
                "max_depth": scope.max_depth,
                "rate_limit": scope.rate_limit,
                "check_ssl": scope.check_ssl,
            },
            "summary": {
                "total_findings": len(findings),
                "critical_findings": severity_counts["critical"],
                "high_findings": severity_counts["high"],
                "medium_findings": severity_counts["medium"],
                "low_findings": severity_counts["low"],
                "info_findings": severity_counts["info"],
                "successful_scans": sum(1 for result in results if result.success),
                "failed_scans": sum(1 for result in results if not result.success),
            },
            "findings": findings,
        }

    def save(self, filepath: str, target: str, results: Iterable[ScanResult], scope: ScopeConfig, total_duration: float, custom_metadata: Optional[dict] = None) -> None:
        payload = self.generate(
            target=target,
            results=results,
            scope=scope,
            total_duration=total_duration,
            custom_metadata=custom_metadata,
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2 if self.pretty else None, ensure_ascii=False)


class JsonReport:
    def __init__(self, findings):
        self.findings = findings

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.findings, f, indent=2, ensure_ascii=False)
