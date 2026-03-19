"""
cosmos Drift Detector

"My infrastructure is drifting! Quick, someone anchor it!"

Detect configuration drift between desired and actual infrastructure state.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger


class DriftType(Enum):
    """Types of drift."""
    ADDED = "added"           # Resource exists but not in config
    REMOVED = "removed"       # Resource in config but doesn't exist
    MODIFIED = "modified"     # Resource exists but differs from config
    NO_DRIFT = "no_drift"     # Resource matches config


class DriftSeverity(Enum):
    """Drift severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DriftItem:
    """A single drift detection."""
    resource_type: str
    resource_name: str
    resource_id: str
    drift_type: DriftType
    severity: DriftSeverity
    expected: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    differences: List[Dict[str, Any]] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "resource_id": self.resource_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "expected": self.expected,
            "actual": self.actual,
            "differences": self.differences,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""
    id: str
    workspace: str
    items: List[DriftItem]
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_resources: int = 0
    drifted_resources: int = 0

    def summary(self) -> Dict[str, Any]:
        """Get drift summary."""
        by_type = {}
        by_severity = {}

        for item in self.items:
            by_type[item.drift_type.value] = by_type.get(item.drift_type.value, 0) + 1
            by_severity[item.severity.value] = by_severity.get(item.severity.value, 0) + 1

        return {
            "total_resources": self.total_resources,
            "drifted_resources": self.drifted_resources,
            "drift_percentage": (self.drifted_resources / self.total_resources * 100) if self.total_resources > 0 else 0,
            "by_type": by_type,
            "by_severity": by_severity,
        }

    def has_critical_drift(self) -> bool:
        """Check if any critical drift exists."""
        return any(item.severity == DriftSeverity.CRITICAL for item in self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace": self.workspace,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at.isoformat(),
            "summary": self.summary(),
        }


class DriftDetector:
    """
    Infrastructure drift detection for cosmos.

    Features:
    - Compare Terraform state with real infrastructure
    - Multi-cloud drift detection
    - Severity classification
    - Automated remediation suggestions
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("./data/drift")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.reports: Dict[str, DriftReport] = {}

        # Severity rules for different resource types
        self.severity_rules = {
            # Security-related changes are critical
            "aws_security_group": DriftSeverity.CRITICAL,
            "aws_iam_role": DriftSeverity.CRITICAL,
            "aws_iam_policy": DriftSeverity.CRITICAL,
            "azurerm_network_security_group": DriftSeverity.CRITICAL,
            "google_compute_firewall": DriftSeverity.CRITICAL,

            # Data resources are high severity
            "aws_rds_instance": DriftSeverity.HIGH,
            "aws_s3_bucket": DriftSeverity.HIGH,
            "azurerm_storage_account": DriftSeverity.HIGH,

            # Compute resources are medium
            "aws_instance": DriftSeverity.MEDIUM,
            "azurerm_virtual_machine": DriftSeverity.MEDIUM,
            "google_compute_instance": DriftSeverity.MEDIUM,

            # Network resources
            "aws_vpc": DriftSeverity.MEDIUM,
            "aws_subnet": DriftSeverity.LOW,
        }

    def _get_severity(self, resource_type: str, drift_type: DriftType) -> DriftSeverity:
        """Determine severity for a drift item."""
        # Removed resources are always at least high severity
        if drift_type == DriftType.REMOVED:
            return DriftSeverity.HIGH

        # Added (unmanaged) resources are medium
        if drift_type == DriftType.ADDED:
            return DriftSeverity.MEDIUM

        # Use configured severity or default to medium
        return self.severity_rules.get(resource_type, DriftSeverity.MEDIUM)

    def _compare_values(
        self,
        expected: Any,
        actual: Any,
        path: str = ""
    ) -> List[Dict[str, Any]]:
        """Compare two values and return differences."""
        differences = []

        if type(expected) != type(actual):
            differences.append({
                "path": path,
                "expected": expected,
                "actual": actual,
                "type": "type_mismatch",
            })
        elif isinstance(expected, dict):
            all_keys = set(expected.keys()) | set(actual.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                if key not in expected:
                    differences.append({
                        "path": key_path,
                        "expected": None,
                        "actual": actual[key],
                        "type": "added",
                    })
                elif key not in actual:
                    differences.append({
                        "path": key_path,
                        "expected": expected[key],
                        "actual": None,
                        "type": "removed",
                    })
                else:
                    differences.extend(self._compare_values(
                        expected[key], actual[key], key_path
                    ))
        elif isinstance(expected, list):
            if len(expected) != len(actual):
                differences.append({
                    "path": path,
                    "expected": f"list of {len(expected)} items",
                    "actual": f"list of {len(actual)} items",
                    "type": "length_mismatch",
                })
            else:
                for i, (e, a) in enumerate(zip(expected, actual)):
                    differences.extend(self._compare_values(
                        e, a, f"{path}[{i}]"
                    ))
        elif expected != actual:
            differences.append({
                "path": path,
                "expected": expected,
                "actual": actual,
                "type": "value_mismatch",
            })

        return differences

    async def detect_terraform_drift(
        self,
        terraform_manager,
        workspace,
    ) -> DriftReport:
        """Detect drift in a Terraform workspace."""
        import uuid

        # Refresh state to get current infrastructure
        await terraform_manager.refresh(workspace)

        # Get planned changes
        plan = await terraform_manager.plan(workspace, save_plan=False)

        # Build drift items from plan
        items = []
        for change in plan.changes:
            if change.action == TerraformAction.NO_OP:
                continue

            drift_type = DriftType.MODIFIED
            if change.action == TerraformAction.CREATE:
                drift_type = DriftType.REMOVED  # Config says create, so it's missing
            elif change.action == TerraformAction.DELETE:
                drift_type = DriftType.ADDED  # Config says delete, so it shouldn't exist

            differences = self._compare_values(change.before, change.after)

            item = DriftItem(
                resource_type=change.resource_type,
                resource_name=change.name,
                resource_id=change.address,
                drift_type=drift_type,
                severity=self._get_severity(change.resource_type, drift_type),
                expected=change.after,
                actual=change.before,
                differences=differences,
            )
            items.append(item)

        # Create report
        state = await terraform_manager.get_state(workspace)
        total_resources = len(state.resources) if state else 0

        report = DriftReport(
            id=str(uuid.uuid4()),
            workspace=workspace.name,
            items=items,
            total_resources=total_resources,
            drifted_resources=len(items),
        )

        self.reports[report.id] = report
        self._save_report(report)

        logger.info(f"Drift detection completed: {len(items)} drifted resources")
        return report

    async def detect_aws_drift(
        self,
        aws_manager,
        resource_types: List[str] = None,
    ) -> DriftReport:
        """Detect drift in AWS resources."""
        import uuid

        resource_types = resource_types or ["ec2", "s3", "iam"]
        items = []

        # This would integrate with AWS Config or CloudFormation drift detection
        # For now, return empty report
        logger.info("AWS drift detection requires AWS Config integration")

        report = DriftReport(
            id=str(uuid.uuid4()),
            workspace="aws",
            items=items,
            total_resources=0,
            drifted_resources=0,
        )

        return report

    async def detect_azure_drift(
        self,
        azure_manager,
        resource_groups: List[str] = None,
    ) -> DriftReport:
        """Detect drift in Azure resources."""
        import uuid

        # This would integrate with Azure Resource Manager
        logger.info("Azure drift detection requires ARM integration")

        report = DriftReport(
            id=str(uuid.uuid4()),
            workspace="azure",
            items=[],
            total_resources=0,
            drifted_resources=0,
        )

        return report

    # =========================================================================
    # REPORT MANAGEMENT
    # =========================================================================

    def _save_report(self, report: DriftReport):
        """Save drift report to storage."""
        report_file = self.storage_path / f"{report.id}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def get_report(self, report_id: str) -> Optional[DriftReport]:
        """Get a drift report by ID."""
        return self.reports.get(report_id)

    def list_reports(
        self,
        workspace: str = None,
        limit: int = 50,
    ) -> List[DriftReport]:
        """List drift reports."""
        reports = list(self.reports.values())

        if workspace:
            reports = [r for r in reports if r.workspace == workspace]

        return sorted(reports, key=lambda r: r.created_at, reverse=True)[:limit]

    # =========================================================================
    # REMEDIATION
    # =========================================================================

    def generate_remediation_plan(self, report: DriftReport) -> List[Dict[str, Any]]:
        """Generate remediation suggestions for drift items."""
        remediation = []

        for item in report.items:
            suggestion = {
                "resource": f"{item.resource_type}.{item.resource_name}",
                "drift_type": item.drift_type.value,
                "severity": item.severity.value,
                "action": None,
                "command": None,
                "description": "",
            }

            if item.drift_type == DriftType.MODIFIED:
                suggestion["action"] = "apply"
                suggestion["command"] = f"terraform apply -target={item.resource_id}"
                suggestion["description"] = "Apply configuration to restore expected state"

            elif item.drift_type == DriftType.ADDED:
                suggestion["action"] = "import_or_destroy"
                suggestion["command"] = f"terraform import {item.resource_id} <resource_id>"
                suggestion["description"] = "Either import the resource or destroy it"

            elif item.drift_type == DriftType.REMOVED:
                suggestion["action"] = "apply"
                suggestion["command"] = f"terraform apply -target={item.resource_id}"
                suggestion["description"] = "Apply to create the missing resource"

            remediation.append(suggestion)

        # Sort by severity
        severity_order = {
            DriftSeverity.CRITICAL.value: 0,
            DriftSeverity.HIGH.value: 1,
            DriftSeverity.MEDIUM.value: 2,
            DriftSeverity.LOW.value: 3,
            DriftSeverity.INFO.value: 4,
        }

        return sorted(remediation, key=lambda x: severity_order.get(x["severity"], 5))

    async def auto_remediate(
        self,
        report: DriftReport,
        terraform_manager,
        workspace,
        max_severity: DriftSeverity = DriftSeverity.MEDIUM,
    ) -> Dict[str, Any]:
        """Automatically remediate drift up to specified severity."""
        severity_order = {
            DriftSeverity.CRITICAL: 0,
            DriftSeverity.HIGH: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.LOW: 3,
            DriftSeverity.INFO: 4,
        }
        max_level = severity_order[max_severity]

        remediated = []
        skipped = []

        for item in report.items:
            if severity_order[item.severity] <= max_level:
                if item.drift_type in [DriftType.MODIFIED, DriftType.REMOVED]:
                    # Apply changes for this resource
                    result = await terraform_manager.apply(
                        workspace,
                        auto_approve=True,
                        target=item.resource_id,
                    )
                    if result["success"]:
                        remediated.append(item.resource_id)
                    else:
                        skipped.append({
                            "resource": item.resource_id,
                            "reason": result.get("stderr", "Unknown error"),
                        })
                else:
                    skipped.append({
                        "resource": item.resource_id,
                        "reason": "Manual intervention required for added resources",
                    })
            else:
                skipped.append({
                    "resource": item.resource_id,
                    "reason": f"Severity {item.severity.value} exceeds max {max_severity.value}",
                })

        return {
            "remediated": remediated,
            "skipped": skipped,
            "total": len(report.items),
        }


# Import TerraformAction for type hints
try:
    from Cosmos.infrastructure.terraform_manager import TerraformAction
except ImportError:
    pass

# Singleton instance
drift_detector = DriftDetector()
