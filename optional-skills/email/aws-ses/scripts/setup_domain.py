"""verify a domain identity in amazon ses and print dns records."""

import argparse
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


def create_identity(client, domain: str, mail_from: str | None = None) -> dict:
    """create a domain identity; reprints dns records if it already exists."""
    try:
        result = client.create_email_identity(EmailIdentity=domain)
    except ClientError as exc:
        # re-running on an existing domain is the common case — fetch it instead.
        if exc.response.get("Error", {}).get("Code") == "AlreadyExistsException":
            result = client.get_email_identity(EmailIdentity=domain)
        else:
            raise

    # aws-managed (easy) dkim tokens may be absent on an existing identity.
    if not result.get("DkimAttributes", {}).get("Tokens"):
        identity = client.get_email_identity(EmailIdentity=domain)
        result["DkimAttributes"] = identity.get("DkimAttributes", {})

    if mail_from:
        try:
            client.put_email_identity_mail_from_attributes(
                EmailIdentity=domain,
                MailFromDomain=mail_from,
                BehaviorOnMxFailure="USE_DEFAULT_VALUE",
            )
            result["_mail_from_configured"] = mail_from
        except (BotoCoreError, ClientError) as exc:
            result["_mail_from_error"] = str(exc)

    return result


def print_dns_records(domain: str, dkim_tokens: list[str]) -> None:
    """print the dns records the user must create."""
    print(f"\n{'='*60}")
    print(f"DNS records to add for: {domain}")
    print(f"{'='*60}\n")

    print("── DKIM CNAME records (create all three) ──\n")
    for i, token in enumerate(dkim_tokens, start=1):
        name = f"{token}._domainkey.{domain}"
        value = f"{token}.dkim.amazonses.com"
        print(f"  Record {i}:")
        print(f"    Name:  {name}")
        print("    Type:  CNAME")
        print(f"    Value: {value}")
        print()

    print("── SPF TXT record ──\n")
    print(f"    Name:  {domain}  (or '@' depending on DNS provider)")
    print("    Type:  TXT")
    print('    Value: "v=spf1 +a +mx include:amazonses.com ~all"')
    print()
    print("    If you already have an SPF record, add include:amazonses.com")
    print("    before the final ~all or -all mechanism.")
    print()

    print("── Provider-specific notes ──")
    print()
    print("  • The CNAME Value does NOT include a trailing dot.")
    print("    If your DNS provider auto-appends the zone name, leave the dot off.")
    print("    If it does NOT auto-append, you may need a trailing dot:")
    print(f"      {dkim_tokens[0] if dkim_tokens else '<token>'}.dkim.amazonses.com.")
    print()
    print("  • Hetzner DNS Console may warn:")
    print('    "Record points to a target inside this zone that does not exist"')
    print("    This is a false positive — the record is correct. Verify with dig.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a domain identity in SES and print DKIM/SPF DNS records."
    )
    parser.add_argument("domain", help="Domain name to verify (e.g. mydomain.com)")
    parser.add_argument(
        "--mail-from",
        dest="mail_from",
        help="Optional custom MAIL FROM domain (e.g. bounce.mydomain.com)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be created without making any AWS API calls.",
    )
    parser.add_argument(
        "--region",
        help="AWS region (default: $AWS_DEFAULT_REGION)",
    )
    args = parser.parse_args()

    domain: str = args.domain
    region: str = args.region or os.getenv("AWS_DEFAULT_REGION")

    if args.dry_run:
        print("── DRY RUN — no API calls will be made ──\n")
        print(f"Region:          {region or '(from boto3 config)'}")
        print(f"Domain:          {domain}")
        print("API call:        sesv2.create_email_identity()")
        print("Signing method:  AWS-managed Easy DKIM")
        if args.mail_from:
            print(f"MAIL FROM:       {args.mail_from}")
            print("API call:        sesv2.put_email_identity_mail_from_attributes()")
        print()
        print("After creation, you will receive three DKIM tokens.")
        print("Use them to build CNAME records (see examples below):")
        print()
        print(f"  <token1>._domainkey.{domain}  CNAME  <token1>.dkim.amazonses.com")
        print(f"  <token2>._domainkey.{domain}  CNAME  <token2>.dkim.amazonses.com")
        print(f"  <token3>._domainkey.{domain}  CNAME  <token3>.dkim.amazonses.com")
        print()
        print(f'  {domain}  TXT  "v=spf1 +a +mx include:amazonses.com ~all"')
        print()
        print("── End of dry run ──")
        return

    client = get_client(region)

    try:
        result = create_identity(client, domain, args.mail_from)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        msg = exc.response.get("Error", {}).get("Message", str(exc))
        print(f"✗ Failed to create identity: [{code}] {msg}", file=sys.stderr)
        sys.exit(1)
    except BotoCoreError as exc:
        print(f"✗ Network/credential error: {exc}", file=sys.stderr)
        sys.exit(1)

    dkim = result.get("DkimAttributes", {})
    dkim_tokens: list[str] = dkim.get("Tokens", [])

    print(f"✓ Domain identity created/confirmed: {domain}")
    print(f"  DKIM signing status: {dkim.get('Status', 'PENDING')}")
    print(f"  Verification status: {result.get('VerificationStatus', 'PENDING')}")

    if args.mail_from and "_mail_from_error" not in result:
        print(f"  MAIL FROM domain:    {args.mail_from} (configured)")

    if dkim_tokens:
        print_dns_records(domain, dkim_tokens)
    else:
        msg = (
            "\n⚠  No DKIM tokens returned yet. AWS may still be provisioning the "
            "identity.\n"
            "   Run this script again in a few seconds, or check the SES console."
        )
        print(msg, file=sys.stderr)

    if "_mail_from_error" in result:
        print(f"\n⚠  MAIL FROM configuration failed: {result['_mail_from_error']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
