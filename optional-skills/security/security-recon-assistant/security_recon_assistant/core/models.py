from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator


ALLOWED_SEVERITIES = {"info", "low", "medium", "high", "critical"}


class ScanFinding(BaseModel):
    target: str
    severity: str
    title: str
    description: str
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    cve: Optional[str] = None
    port: Optional[int] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in ALLOWED_SEVERITIES:
            raise ValueError(f"Unsupported severity: {value}")
        return normalized

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class ScanResult(BaseModel):
    scanner_name: str
    success: bool
    command: Optional[str] = None
    execution_time: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timeout: bool = False
    findings: List[Any] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def deep_copy_findings(self) -> "ScanResult":
        copied = []
        for finding in self.findings:
            if isinstance(finding, ScanFinding):
                copied.append(ScanFinding.model_validate(finding.model_dump()))
            elif isinstance(finding, dict):
                copied.append(ScanFinding.model_validate(finding))
            else:
                copied.append(finding)
        self.findings = copied
        return self

    def to_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        data["findings"] = [
            finding.to_dict() if isinstance(finding, ScanFinding) else finding
            for finding in self.findings
        ]
        return data


class Finding(BaseModel):
    scanner: str
    target: str
    severity: str
    description: str
    evidence: Optional[str] = None


class ScanConfig(BaseModel):
    targets: List[str]
    scanners: List[str] = Field(default_factory=lambda: ["subfinder", "nmap"])
    max_depth: int = Field(default=1, ge=1, le=3)


class ScopeConfig(BaseModel):
    allowed_domains: Set[str] = Field(default_factory=set)
    excluded_domains: Set[str] = Field(default_factory=set)
    allowed_ips: Set[str] = Field(default_factory=set)
    max_depth: int = 3
    rate_limit: int = 50
    check_ssl: bool = False

    @field_validator("allowed_domains", "excluded_domains", "allowed_ips", mode="before")
    @classmethod
    def normalize_collection(cls, value):
        if value is None:
            return set()
        if isinstance(value, str):
            value = [value]
        return {str(item).strip().lower() for item in value if str(item).strip()}

    @property
    def excluded(self) -> Set[str]:
        return self.excluded_domains
