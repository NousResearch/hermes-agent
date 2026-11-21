"""check amazon ses account, domain, and dkim health."""

import json
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_client(region: str | None = None):
    """return a boto3 ses v2 client."""
    kwargs = {"service_name": "sesv2"}
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


def collect_status(client, domain: str | None) -> dict:
    """gather all ses account and identity info in one dict."""

    result: dict = {
        "production_access": None,
        "sending_enabled": None,
        "enforcement": None,
        "quota": {},
        "review": {},
        "domain_verification": None,
        "domain_sending": None,
        "dkim": {},
    }

    try:
        acct = client.get_account()
        result["production_access"] = acct.get("ProductionAccessEnabled", False)
        result["sending_enabled"] = acct.get("SendingEnabled", False)
        result["enforcement"] = acct.get("EnforcementStatus", "UNKNOWN")
        quota = acct.get("SendQuota", {})
        result["quota"] = {
            "Max24HourSend": quota.get("Max24HourSend", 0),
            "MaxSendRate": quota.get("MaxSendRate", 0),
            "SentLast24Hours": quota.get("SentLast24Hours", 0),
        }
        # mail type, website, and review status are nested under Details.
        details = acct.get("Details", {})
        review = details.get("ReviewDetails", {})
        result["review"] = {
            "Status": review.get("Status", "UNKNOWN"),
            "CaseId": review.get("CaseId", None),
        }
        result["mail_type"] = details.get("MailType", "UNKNOWN")
        result["website"] = details.get("WebsiteURL", "")
    except (BotoCoreError, ClientError) as exc:
        result["account_error"] = str(exc)
        return result  # can't continue without account info

    if not domain:
        return result

    try:
        identity = client.get_email_identity(EmailIdentity=domain)
        result["domain_verification"] = identity.get("VerificationStatus", "UNKNOWN")
        result["verified_for_sending"] = identity.get(
            "VerifiedForSendingStatus", False
        )

        dkim = identity.get("DkimAttributes", {})
        result["dkim"] = {
            "SigningEnabled": dkim.get("SigningEnabled", False),
            "Status": dkim.get("Status", "UNKNOWN"),
            "Tokens": dkim.get("Tokens", []),
            "SigningAttributesOrigin": dkim.get(
                "SigningAttributesOrigin", "UNKNOWN"
            ),
            "CurrentSigningKeyLength": dkim.get(
                "CurrentSigningKeyLength", "UNKNOWN"
            ),
        }

        mail_from = identity.get("MailFromAttributes", {})
        result["mail_from"] = {
            "MailFromDomain": mail_from.get("MailFromDomain"),
            "MailFromDomainStatus": mail_from.get(
                "MailFromDomainStatus", "UNKNOWN"
            ),
            "BehaviorOnMxFailure": mail_from.get(
                "BehaviorOnMxFailure", "UNKNOWN"
            ),
        }

    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NotFoundException":
            result["domain_error"] = (
                f"Domain '{domain}' is not a verified identity in this region."
            )
        else:
            result["domain_error"] = str(exc)

    return result


def main() -> None:
    domain = None
    if len(sys.argv) > 1:
        domain = sys.argv[1]
    else:
        domain = os.getenv("SES_DOMAIN")

    region = os.getenv("AWS_DEFAULT_REGION")
    client = get_client(region)
    status = collect_status(client, domain)
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
