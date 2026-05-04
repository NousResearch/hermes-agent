"""Credential metadata audit helpers.

This module audits Hermes credential *definitions*, not live secret values. The
purpose is to catch weak recovery assumptions before cron/tools depend on a key:
where to recover/rotate it, whether it is phishing-resistant OAuth vs long-lived
secret material, and whether a human-readable recovery note exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_STRONG_AUTH_METHODS = {"oauth_device", "oauth_pkce", "oauth", "sso", "webauthn"}
_LONG_LIVED_METHODS = {"api_key", "bot_token", "personal_access_token", "static_token"}


@dataclass(frozen=True)
class CredentialAuditFinding:
    key: str
    severity: str
    issue: str
    recommendation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "severity": self.severity,
            "issue": self.issue,
            "recommendation": self.recommendation,
        }


def _is_secret(meta: Mapping[str, Any]) -> bool:
    return bool(meta.get("password")) or any(
        marker in str(meta.get("prompt", "")).lower()
        for marker in ("token", "api key", "secret", "password")
    )


def audit_credential_metadata(
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    include_advanced: bool = False,
) -> list[CredentialAuditFinding]:
    """Audit credential definitions for auth/recovery assumptions.

    Expected optional metadata fields:
    - ``auth_method``: oauth_device/oauth_pkce/oauth/sso/webauthn/api_key/bot_token/...
    - ``phishing_resistant``: bool, explicit for OAuth/SAML/WebAuthn entries
    - ``recovery``: human-readable rotation/recovery path

    Missing metadata is a warning, not a hard failure, so existing credentials can
    be upgraded incrementally.
    """

    findings: list[CredentialAuditFinding] = []
    for key, raw_meta in sorted(definitions.items()):
        meta = dict(raw_meta or {})
        if meta.get("advanced") and not include_advanced:
            continue
        if not _is_secret(meta):
            continue

        auth_method = str(meta.get("auth_method") or "").strip().lower()
        if not auth_method:
            findings.append(
                CredentialAuditFinding(
                    key=key,
                    severity="warn",
                    issue="missing auth_method metadata",
                    recommendation=(
                        "Declare auth_method (oauth_device/oauth_pkce/oauth/sso/webauthn "
                        "preferred; api_key/bot_token/personal_access_token if unavoidable)."
                    ),
                )
            )
        elif auth_method in _LONG_LIVED_METHODS and meta.get("phishing_resistant") is not False:
            findings.append(
                CredentialAuditFinding(
                    key=key,
                    severity="warn",
                    issue="long-lived secret lacks explicit phishing_resistant=false marker",
                    recommendation="Mark phishing_resistant=false and document rotation/recovery assumptions.",
                )
            )
        elif auth_method in _STRONG_AUTH_METHODS and meta.get("phishing_resistant") is not True:
            findings.append(
                CredentialAuditFinding(
                    key=key,
                    severity="warn",
                    issue="strong auth method lacks explicit phishing_resistant=true marker",
                    recommendation="Mark phishing_resistant=true so audits can distinguish OAuth/WebAuthn from static secrets.",
                )
            )

        if not meta.get("url") and not meta.get("recovery"):
            findings.append(
                CredentialAuditFinding(
                    key=key,
                    severity="warn",
                    issue="missing recovery or rotation URL",
                    recommendation="Add url or recovery text that tells operators how to rotate/recover this credential.",
                )
            )
        if not meta.get("recovery"):
            findings.append(
                CredentialAuditFinding(
                    key=key,
                    severity="info",
                    issue="missing recovery note",
                    recommendation="Add concise recovery text for cron/tool outage handling.",
                )
            )
    return findings


def summarize_credential_audit(
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    include_advanced: bool = False,
) -> dict[str, Any]:
    findings = audit_credential_metadata(definitions, include_advanced=include_advanced)
    return {
        "status": "pass" if not any(f.severity == "warn" for f in findings) else "warn",
        "checked": sum(
            1
            for meta in definitions.values()
            if (include_advanced or not meta.get("advanced")) and _is_secret(meta)
        ),
        "warnings": sum(1 for f in findings if f.severity == "warn"),
        "infos": sum(1 for f in findings if f.severity == "info"),
        "findings": [f.as_dict() for f in findings],
    }
