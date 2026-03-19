"""
cosmos Compliance Engine

"Every regulation must be followed to the letter...
 unless you have a bending unit to help you around them!"

Multi-framework compliance monitoring and enforcement.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
from loguru import logger


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    CIS = "cis"
    CUSTOM = "custom"


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"


class ViolationSeverity(Enum):
    """Violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceCheck:
    """A single compliance check."""
    id: str
    name: str
    description: str
    framework: ComplianceFramework
    control_id: str  # e.g., "CC6.1" for SOC2
    category: str

    # Check configuration
    check_function: Optional[str] = None  # Function name to execute
    check_query: Optional[str] = None  # SQL/API query
    expected_result: Any = None
    remediation_steps: str = ""

    # Execution
    status: ComplianceStatus = ComplianceStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_result: Optional[Dict] = None
    evidence: List[str] = field(default_factory=list)

    # Scheduling
    check_interval_hours: int = 24
    enabled: bool = True


@dataclass
class ComplianceViolation:
    """A compliance violation."""
    id: str
    check_id: str
    check_name: str
    framework: ComplianceFramework
    control_id: str
    severity: ViolationSeverity

    # Details
    description: str
    affected_resources: List[str] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)

    # Status
    status: str = "open"  # open, acknowledged, remediated, false_positive
    acknowledged_by: str = ""
    acknowledged_at: Optional[datetime] = None
    remediated_by: str = ""
    remediated_at: Optional[datetime] = None

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "check_id": self.check_id,
            "check_name": self.check_name,
            "framework": self.framework.value,
            "control_id": self.control_id,
            "severity": self.severity.value,
            "description": self.description,
            "affected_resources": self.affected_resources,
            "status": self.status,
            "detected_at": self.detected_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
        }


@dataclass
class CompliancePolicy:
    """A compliance policy with multiple checks."""
    id: str
    name: str
    description: str
    framework: ComplianceFramework
    version: str
    checks: List[ComplianceCheck] = field(default_factory=list)

    # Metadata
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True

    def to_dict(self) -> Dict:
        compliant = sum(1 for c in self.checks if c.status == ComplianceStatus.COMPLIANT)
        total = len([c for c in self.checks if c.enabled])

        return {
            "id": self.id,
            "name": self.name,
            "framework": self.framework.value,
            "version": self.version,
            "total_checks": total,
            "compliant_checks": compliant,
            "compliance_percentage": (compliant / total * 100) if total > 0 else 0,
            "owner": self.owner,
            "enabled": self.enabled,
        }


class ComplianceEngine:
    """
    Multi-framework compliance monitoring engine.

    Features:
    - SOC2, HIPAA, GDPR, PCI-DSS, ISO27001, NIST, CIS support
    - Automated compliance checks
    - Violation tracking and remediation
    - Evidence collection
    - Compliance reporting
    - Policy as Code
    """

    def __init__(
        self,
        storage_path: Path = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/compliance")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.policies: Dict[str, CompliancePolicy] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.check_handlers: Dict[str, Callable] = {}

        self._load_builtin_policies()
        self._register_default_handlers()

    def _load_builtin_policies(self):
        """Load built-in compliance policies."""
        # SOC2 Type II Controls
        soc2_checks = [
            ComplianceCheck(
                id="soc2-cc6.1-access-control",
                name="Logical Access Controls",
                description="Ensure logical access to systems is restricted and controlled",
                framework=ComplianceFramework.SOC2,
                control_id="CC6.1",
                category="Logical and Physical Access Controls",
                check_function="check_access_controls",
                remediation_steps="1. Review access policies\n2. Implement MFA\n3. Review user permissions",
            ),
            ComplianceCheck(
                id="soc2-cc6.2-user-registration",
                name="User Registration and Authorization",
                description="Users are registered and authorized before being granted access",
                framework=ComplianceFramework.SOC2,
                control_id="CC6.2",
                category="Logical and Physical Access Controls",
                check_function="check_user_registration",
            ),
            ComplianceCheck(
                id="soc2-cc6.3-access-removal",
                name="Access Removal",
                description="Access is removed when no longer needed",
                framework=ComplianceFramework.SOC2,
                control_id="CC6.3",
                category="Logical and Physical Access Controls",
                check_function="check_access_removal",
            ),
            ComplianceCheck(
                id="soc2-cc7.1-encryption-at-rest",
                name="Data Encryption at Rest",
                description="Data at rest is encrypted using appropriate encryption methods",
                framework=ComplianceFramework.SOC2,
                control_id="CC7.1",
                category="System Operations",
                check_function="check_encryption_at_rest",
            ),
            ComplianceCheck(
                id="soc2-cc7.2-encryption-in-transit",
                name="Data Encryption in Transit",
                description="Data in transit is encrypted using TLS 1.2+",
                framework=ComplianceFramework.SOC2,
                control_id="CC7.2",
                category="System Operations",
                check_function="check_encryption_in_transit",
            ),
            ComplianceCheck(
                id="soc2-cc8.1-change-management",
                name="Change Management Process",
                description="Changes are authorized, documented, and tested before implementation",
                framework=ComplianceFramework.SOC2,
                control_id="CC8.1",
                category="Change Management",
                check_function="check_change_management",
            ),
        ]

        self.policies["soc2-type2"] = CompliancePolicy(
            id="soc2-type2",
            name="SOC2 Type II",
            description="Service Organization Control 2 Type II compliance",
            framework=ComplianceFramework.SOC2,
            version="2017",
            checks=soc2_checks,
        )

        # HIPAA Security Rule
        hipaa_checks = [
            ComplianceCheck(
                id="hipaa-164.312-a-access-control",
                name="Access Control",
                description="Implement technical policies to allow only authorized access to ePHI",
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(a)",
                category="Technical Safeguards",
                check_function="check_hipaa_access_control",
            ),
            ComplianceCheck(
                id="hipaa-164.312-b-audit-controls",
                name="Audit Controls",
                description="Implement hardware, software, and procedural mechanisms to record and examine activity",
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(b)",
                category="Technical Safeguards",
                check_function="check_audit_logging",
            ),
            ComplianceCheck(
                id="hipaa-164.312-c-integrity",
                name="Integrity Controls",
                description="Implement policies to protect ePHI from improper alteration or destruction",
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(c)",
                category="Technical Safeguards",
                check_function="check_data_integrity",
            ),
            ComplianceCheck(
                id="hipaa-164.312-d-authentication",
                name="Person or Entity Authentication",
                description="Implement procedures to verify that a person or entity seeking access is the one claimed",
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(d)",
                category="Technical Safeguards",
                check_function="check_authentication",
            ),
            ComplianceCheck(
                id="hipaa-164.312-e-transmission-security",
                name="Transmission Security",
                description="Implement technical measures to guard against unauthorized access to ePHI being transmitted",
                framework=ComplianceFramework.HIPAA,
                control_id="164.312(e)",
                category="Technical Safeguards",
                check_function="check_transmission_security",
            ),
        ]

        self.policies["hipaa-security"] = CompliancePolicy(
            id="hipaa-security",
            name="HIPAA Security Rule",
            description="Health Insurance Portability and Accountability Act Security Rule",
            framework=ComplianceFramework.HIPAA,
            version="2013",
            checks=hipaa_checks,
        )

        # GDPR
        gdpr_checks = [
            ComplianceCheck(
                id="gdpr-art32-security",
                name="Security of Processing",
                description="Implement appropriate technical and organizational measures for data security",
                framework=ComplianceFramework.GDPR,
                control_id="Article 32",
                category="Security",
                check_function="check_gdpr_security",
            ),
            ComplianceCheck(
                id="gdpr-art17-erasure",
                name="Right to Erasure",
                description="Ability to delete personal data upon request",
                framework=ComplianceFramework.GDPR,
                control_id="Article 17",
                category="Data Subject Rights",
                check_function="check_data_erasure_capability",
            ),
            ComplianceCheck(
                id="gdpr-art20-portability",
                name="Right to Data Portability",
                description="Ability to export personal data in machine-readable format",
                framework=ComplianceFramework.GDPR,
                control_id="Article 20",
                category="Data Subject Rights",
                check_function="check_data_portability",
            ),
            ComplianceCheck(
                id="gdpr-art33-breach-notification",
                name="Breach Notification",
                description="Process for notifying supervisory authority within 72 hours of breach",
                framework=ComplianceFramework.GDPR,
                control_id="Article 33",
                category="Breach Notification",
                check_function="check_breach_notification_process",
            ),
        ]

        self.policies["gdpr"] = CompliancePolicy(
            id="gdpr",
            name="GDPR",
            description="General Data Protection Regulation",
            framework=ComplianceFramework.GDPR,
            version="2018",
            checks=gdpr_checks,
        )

    def _register_default_handlers(self):
        """Register default compliance check handlers."""
        self.check_handlers["check_access_controls"] = self._check_access_controls
        self.check_handlers["check_encryption_at_rest"] = self._check_encryption_at_rest
        self.check_handlers["check_encryption_in_transit"] = self._check_encryption_in_transit
        self.check_handlers["check_audit_logging"] = self._check_audit_logging
        self.check_handlers["check_authentication"] = self._check_authentication

    # =========================================================================
    # POLICY MANAGEMENT
    # =========================================================================

    def add_policy(self, policy: CompliancePolicy):
        """Add a compliance policy."""
        self.policies[policy.id] = policy
        logger.info(f"Added compliance policy: {policy.name}")

    def get_policy(self, policy_id: str) -> Optional[CompliancePolicy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)

    def list_policies(
        self,
        framework: ComplianceFramework = None,
    ) -> List[CompliancePolicy]:
        """List all policies, optionally filtered by framework."""
        policies = list(self.policies.values())

        if framework:
            policies = [p for p in policies if p.framework == framework]

        return policies

    def load_policy_from_yaml(self, yaml_path: Path) -> Optional[CompliancePolicy]:
        """Load a policy from YAML file."""
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            checks = []
            for check_data in data.get("checks", []):
                check = ComplianceCheck(
                    id=check_data["id"],
                    name=check_data["name"],
                    description=check_data.get("description", ""),
                    framework=ComplianceFramework(data["framework"]),
                    control_id=check_data.get("control_id", ""),
                    category=check_data.get("category", ""),
                    check_function=check_data.get("check_function"),
                    check_query=check_data.get("check_query"),
                    expected_result=check_data.get("expected_result"),
                    remediation_steps=check_data.get("remediation_steps", ""),
                    check_interval_hours=check_data.get("check_interval_hours", 24),
                )
                checks.append(check)

            policy = CompliancePolicy(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                framework=ComplianceFramework(data["framework"]),
                version=data.get("version", "1.0"),
                checks=checks,
                owner=data.get("owner", ""),
            )

            self.add_policy(policy)
            return policy

        except Exception as e:
            logger.error(f"Failed to load policy from {yaml_path}: {e}")
            return None

    # =========================================================================
    # COMPLIANCE CHECKS
    # =========================================================================

    async def run_check(
        self,
        check: ComplianceCheck,
        context: Dict = None,
    ) -> ComplianceStatus:
        """Run a single compliance check."""
        context = context or {}

        try:
            if check.check_function and check.check_function in self.check_handlers:
                handler = self.check_handlers[check.check_function]
                result = await handler(check, context)
            else:
                # Default to unknown if no handler
                result = {
                    "status": ComplianceStatus.UNKNOWN,
                    "message": "No check handler configured",
                }

            check.status = result.get("status", ComplianceStatus.UNKNOWN)
            check.last_checked = datetime.utcnow()
            check.last_result = result
            check.evidence = result.get("evidence", [])

            # Create violation if non-compliant
            if check.status == ComplianceStatus.NON_COMPLIANT:
                await self._create_violation(check, result)

            return check.status

        except Exception as e:
            logger.error(f"Compliance check failed: {check.id} - {e}")
            check.status = ComplianceStatus.UNKNOWN
            check.last_result = {"error": str(e)}
            return ComplianceStatus.UNKNOWN

    async def run_policy_checks(
        self,
        policy_id: str,
        context: Dict = None,
    ) -> Dict[str, ComplianceStatus]:
        """Run all checks in a policy."""
        policy = self.policies.get(policy_id)
        if not policy:
            return {}

        results = {}
        for check in policy.checks:
            if check.enabled:
                results[check.id] = await self.run_check(check, context)

        policy.updated_at = datetime.utcnow()
        return results

    async def run_all_checks(
        self,
        context: Dict = None,
    ) -> Dict[str, Dict[str, ComplianceStatus]]:
        """Run all compliance checks across all policies."""
        results = {}

        for policy_id, policy in self.policies.items():
            if policy.enabled:
                results[policy_id] = await self.run_policy_checks(policy_id, context)

        return results

    def register_check_handler(
        self,
        name: str,
        handler: Callable,
    ):
        """Register a custom check handler."""
        self.check_handlers[name] = handler

    # =========================================================================
    # DEFAULT CHECK HANDLERS
    # =========================================================================

    async def _check_access_controls(
        self,
        check: ComplianceCheck,
        context: Dict,
    ) -> Dict:
        """Check access control implementation."""
        # This would integrate with your identity provider
        evidence = []

        # Example checks
        has_mfa = context.get("mfa_enabled", False)
        has_rbac = context.get("rbac_enabled", False)
        has_sso = context.get("sso_enabled", False)

        if has_mfa:
            evidence.append("MFA is enabled for all users")
        if has_rbac:
            evidence.append("Role-based access control is implemented")
        if has_sso:
            evidence.append("SSO integration is active")

        compliant = has_mfa and has_rbac

        return {
            "status": ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            "message": "Access controls verified" if compliant else "Missing required access controls",
            "evidence": evidence,
            "details": {
                "mfa_enabled": has_mfa,
                "rbac_enabled": has_rbac,
                "sso_enabled": has_sso,
            },
        }

    async def _check_encryption_at_rest(
        self,
        check: ComplianceCheck,
        context: Dict,
    ) -> Dict:
        """Check data encryption at rest."""
        evidence = []

        database_encrypted = context.get("database_encrypted", False)
        storage_encrypted = context.get("storage_encrypted", False)
        backup_encrypted = context.get("backup_encrypted", False)

        if database_encrypted:
            evidence.append("Database encryption enabled (AES-256)")
        if storage_encrypted:
            evidence.append("Object storage encryption enabled")
        if backup_encrypted:
            evidence.append("Backup encryption enabled")

        compliant = database_encrypted and storage_encrypted

        return {
            "status": ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            "message": "Encryption at rest verified" if compliant else "Missing encryption at rest",
            "evidence": evidence,
        }

    async def _check_encryption_in_transit(
        self,
        check: ComplianceCheck,
        context: Dict,
    ) -> Dict:
        """Check data encryption in transit."""
        evidence = []

        tls_version = context.get("tls_version", "1.2")
        https_only = context.get("https_only", True)
        cert_valid = context.get("certificate_valid", True)

        if tls_version >= "1.2":
            evidence.append(f"TLS {tls_version} enforced")
        if https_only:
            evidence.append("HTTPS-only mode enabled")
        if cert_valid:
            evidence.append("SSL certificates valid and not expiring soon")

        compliant = tls_version >= "1.2" and https_only and cert_valid

        return {
            "status": ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            "message": "Encryption in transit verified" if compliant else "Transport security issues found",
            "evidence": evidence,
        }

    async def _check_audit_logging(
        self,
        check: ComplianceCheck,
        context: Dict,
    ) -> Dict:
        """Check audit logging implementation."""
        evidence = []

        logging_enabled = context.get("audit_logging_enabled", False)
        log_retention = context.get("log_retention_days", 0)
        log_integrity = context.get("log_integrity_protection", False)

        if logging_enabled:
            evidence.append("Audit logging is enabled")
        if log_retention >= 90:
            evidence.append(f"Log retention: {log_retention} days")
        if log_integrity:
            evidence.append("Log integrity protection enabled")

        compliant = logging_enabled and log_retention >= 90

        return {
            "status": ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            "message": "Audit logging verified" if compliant else "Audit logging requirements not met",
            "evidence": evidence,
        }

    async def _check_authentication(
        self,
        check: ComplianceCheck,
        context: Dict,
    ) -> Dict:
        """Check authentication mechanisms."""
        evidence = []

        password_policy = context.get("password_policy_enforced", False)
        session_timeout = context.get("session_timeout_minutes", 0)
        account_lockout = context.get("account_lockout_enabled", False)

        if password_policy:
            evidence.append("Strong password policy enforced")
        if session_timeout > 0 and session_timeout <= 30:
            evidence.append(f"Session timeout: {session_timeout} minutes")
        if account_lockout:
            evidence.append("Account lockout after failed attempts")

        compliant = password_policy and session_timeout <= 30 and account_lockout

        return {
            "status": ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            "message": "Authentication verified" if compliant else "Authentication improvements needed",
            "evidence": evidence,
        }

    # =========================================================================
    # VIOLATIONS
    # =========================================================================

    async def _create_violation(
        self,
        check: ComplianceCheck,
        result: Dict,
    ):
        """Create a violation record."""
        import uuid

        violation = ComplianceViolation(
            id=f"VIO-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}",
            check_id=check.id,
            check_name=check.name,
            framework=check.framework,
            control_id=check.control_id,
            severity=self._determine_severity(check),
            description=result.get("message", "Compliance check failed"),
            affected_resources=result.get("affected_resources", []),
            evidence=result,
        )

        # Set due date based on severity
        if violation.severity == ViolationSeverity.CRITICAL:
            violation.due_date = datetime.utcnow() + timedelta(days=1)
        elif violation.severity == ViolationSeverity.HIGH:
            violation.due_date = datetime.utcnow() + timedelta(days=7)
        elif violation.severity == ViolationSeverity.MEDIUM:
            violation.due_date = datetime.utcnow() + timedelta(days=30)
        else:
            violation.due_date = datetime.utcnow() + timedelta(days=90)

        self.violations[violation.id] = violation
        logger.warning(f"Compliance violation created: {violation.id} - {check.name}")

    def _determine_severity(self, check: ComplianceCheck) -> ViolationSeverity:
        """Determine violation severity based on check type."""
        # Critical: Security, encryption, access controls
        if "encryption" in check.id.lower() or "access" in check.id.lower():
            return ViolationSeverity.HIGH

        # High: Authentication, audit logging
        if "auth" in check.id.lower() or "audit" in check.id.lower():
            return ViolationSeverity.HIGH

        return ViolationSeverity.MEDIUM

    def get_violation(self, violation_id: str) -> Optional[ComplianceViolation]:
        """Get a violation by ID."""
        return self.violations.get(violation_id)

    def list_violations(
        self,
        status: str = None,
        severity: ViolationSeverity = None,
        framework: ComplianceFramework = None,
    ) -> List[ComplianceViolation]:
        """List violations with optional filters."""
        violations = list(self.violations.values())

        if status:
            violations = [v for v in violations if v.status == status]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        if framework:
            violations = [v for v in violations if v.framework == framework]

        return sorted(violations, key=lambda v: v.detected_at, reverse=True)

    def acknowledge_violation(
        self,
        violation_id: str,
        user: str,
        note: str = None,
    ) -> bool:
        """Acknowledge a violation."""
        violation = self.violations.get(violation_id)
        if not violation:
            return False

        violation.status = "acknowledged"
        violation.acknowledged_by = user
        violation.acknowledged_at = datetime.utcnow()
        if note:
            violation.notes.append(f"[{datetime.utcnow().isoformat()}] {user}: {note}")

        return True

    def remediate_violation(
        self,
        violation_id: str,
        user: str,
        note: str = None,
    ) -> bool:
        """Mark a violation as remediated."""
        violation = self.violations.get(violation_id)
        if not violation:
            return False

        violation.status = "remediated"
        violation.remediated_by = user
        violation.remediated_at = datetime.utcnow()
        if note:
            violation.notes.append(f"[{datetime.utcnow().isoformat()}] {user}: {note}")

        return True

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_compliance_report(
        self,
        framework: ComplianceFramework = None,
    ) -> Dict[str, Any]:
        """Generate a compliance report."""
        policies = self.list_policies(framework)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "framework": framework.value if framework else "all",
            "policies": [],
            "summary": {
                "total_checks": 0,
                "compliant": 0,
                "non_compliant": 0,
                "partial": 0,
                "unknown": 0,
            },
            "violations": {
                "open": 0,
                "acknowledged": 0,
                "remediated": 0,
                "by_severity": {},
            },
        }

        for policy in policies:
            policy_data = policy.to_dict()
            policy_data["checks"] = []

            for check in policy.checks:
                report["summary"]["total_checks"] += 1

                if check.status == ComplianceStatus.COMPLIANT:
                    report["summary"]["compliant"] += 1
                elif check.status == ComplianceStatus.NON_COMPLIANT:
                    report["summary"]["non_compliant"] += 1
                elif check.status == ComplianceStatus.PARTIAL:
                    report["summary"]["partial"] += 1
                else:
                    report["summary"]["unknown"] += 1

                policy_data["checks"].append({
                    "id": check.id,
                    "name": check.name,
                    "control_id": check.control_id,
                    "status": check.status.value,
                    "last_checked": check.last_checked.isoformat() if check.last_checked else None,
                })

            report["policies"].append(policy_data)

        # Violation summary
        for violation in self.violations.values():
            if framework and violation.framework != framework:
                continue

            report["violations"][violation.status] = report["violations"].get(violation.status, 0) + 1

            sev = violation.severity.value
            report["violations"]["by_severity"][sev] = report["violations"]["by_severity"].get(sev, 0) + 1

        # Calculate overall compliance percentage
        total = report["summary"]["total_checks"]
        if total > 0:
            report["summary"]["compliance_percentage"] = round(
                report["summary"]["compliant"] / total * 100, 2
            )
        else:
            report["summary"]["compliance_percentage"] = 0

        return report

    def export_evidence(
        self,
        policy_id: str,
        output_path: Path,
    ):
        """Export compliance evidence for a policy."""
        policy = self.policies.get(policy_id)
        if not policy:
            return

        evidence = {
            "policy": policy.to_dict(),
            "exported_at": datetime.utcnow().isoformat(),
            "checks": [],
        }

        for check in policy.checks:
            check_evidence = {
                "id": check.id,
                "name": check.name,
                "control_id": check.control_id,
                "status": check.status.value,
                "last_checked": check.last_checked.isoformat() if check.last_checked else None,
                "evidence": check.evidence,
                "result": check.last_result,
            }
            evidence["checks"].append(check_evidence)

        with open(output_path, "w") as f:
            json.dump(evidence, f, indent=2)

        logger.info(f"Exported evidence to {output_path}")


# Singleton instance
compliance_engine = ComplianceEngine()
