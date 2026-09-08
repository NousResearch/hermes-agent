"""Local preflight facts, distinct from the Gateway's publication scan."""

from typing import Any

from .agent_led.share_flow import scan_credentials


def prepared_security_check(
    files: list[dict[str, str]], description: str, scan: dict[str, Any],
    *, include_gateway_pending: bool = True,
) -> dict[str, Any]:
    findings = scan_credentials(
        [{"path": item["path"], "content": item["content_utf8"]} for item in files]
        + [{"path": "Owner description", "content": description}]
    )
    rows = []
    for key, label, kinds in (
        ("private_keys", "Private keys", {"private_key"}),
        ("live_credentials", "Live credentials", {
            "aws_access_key", "github_token", "slack_token",
            "openai_style_key", "google_api_key",
        }),
        ("secret_assignments", "Secret-like assignments", {"assignment"}),
    ):
        count = sum(item["kind"] in kinds for item in findings)
        rows.append({
            "key": key, "label": label,
            "status": "blocked" if count else "pass",
            "finding_count": count,
            "details": ["Credential-shaped content detected; remove it before sharing."]
            if count else ["No known local matches detected."],
        })
    guard = scan.get("guard") or {}
    guard_status = (
        "blocked" if guard.get("allowed") is False
        else ("advisory" if guard.get("findings") else "pass")
        if guard.get("allowed") is True else "unavailable"
    )
    rows.append({
        "key": "skills_guard", "label": "Harmful instruction patterns",
        "status": guard_status,
        "finding_count": len(guard.get("findings") or []),
        # Do not surface raw matches, filenames, or scanner reasons containing secrets.
        "details": ["The local skills guard checks known harmful instruction patterns."],
    })
    for key, label in ((
        ("organization_policy", "Organization policy"),
        ("personal_information", "Personal information"),
    ) if include_gateway_pending else ()):
        rows.append({
            "key": key, "label": label, "status": "pending",
            "finding_count": 0,
            "details": ["Required Gateway check after you authorize upload; before publication."],
        })
    blocked = any(row["status"] == "blocked" for row in rows)
    advisory = any(row["status"] == "advisory" for row in rows)
    unavailable = guard_status == "unavailable"
    local_status = "blocked" if blocked else "unavailable" if unavailable else "advisory" if advisory else "pass"
    return {
        "schema_version": 1,
        "source": "local_preflight",
        "status": "pending" if include_gateway_pending and not blocked and not unavailable else local_status,
        "local_status": local_status,
        "upload_allowed": not blocked and not unavailable,
        "summary": (
            "Local security checks found content that must be removed before sharing."
            if blocked else "Local security scanning could not complete. Try preparation again."
            if unavailable else "Local security checks have findings to review."
            if advisory else "Local security checks found no known matches."
        ) + (" Required Gateway checks run after you authorize upload and before publication."
             if include_gateway_pending else ""),
        "checks": rows,
    }
