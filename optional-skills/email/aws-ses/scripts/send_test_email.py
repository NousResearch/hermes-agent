"""send a test email through amazon ses v2."""

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def build_email_request(
    *,
    from_address: str,
    to_addresses: list[str],
    subject: str,
    body_text: str | None = None,
    body_html: str | None = None,
    reply_to: str | None = None,
    config_set: str | None = None,
    cc_addresses: list[str] | None = None,
) -> dict:
    """assemble the kwargs dict for sesv2.send_email()."""
    destination: dict[str, list[str]] = {"ToAddresses": to_addresses}
    if cc_addresses:
        destination["CcAddresses"] = cc_addresses

    content: dict = {}

    if body_html and body_text:
        # multipart: html + plain-text
        content["Simple"] = {
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": body_text, "Charset": "UTF-8"},
                "Html": {"Data": body_html, "Charset": "UTF-8"},
            },
        }
    elif body_html:
        content["Simple"] = {
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Html": {"Data": body_html, "Charset": "UTF-8"},
            },
        }
    else:
        # plain-text only
        content["Simple"] = {
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": body_text or "(empty body)", "Charset": "UTF-8"},
            },
        }

    kwargs: dict = {
        "FromEmailAddress": from_address,
        "Destination": destination,
        "Content": content,
    }

    if reply_to:
        kwargs["ReplyToAddresses"] = [reply_to]

    if config_set:
        kwargs["ConfigurationSetName"] = config_set

    return kwargs


def get_client(region: str | None = None):
    """return a boto3 ses v2 client."""
    kwargs = {"service_name": "sesv2"}
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


def check_sandbox(client) -> dict:
    """return production-access + sending flags (or error)."""
    try:
        acct = client.get_account()
        return {
            "production_access": acct.get("ProductionAccessEnabled", False),
            "sending_enabled": acct.get("SendingEnabled", False),
        }
    except (BotoCoreError, ClientError) as exc:
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test email through Amazon SES v2."
    )
    parser.add_argument("--from", dest="from_address", required=True,
                        help="Verified sender email address (e.g. hello@mydomain.com)")
    parser.add_argument("--to", dest="to_addresses", action="append", required=True,
                        help="Recipient email (repeatable, e.g. --to a@x.com --to b@x.com)")
    parser.add_argument("--cc", dest="cc_addresses", action="append",
                        help="CC recipient (repeatable)")
    parser.add_argument("--subject", required=True,
                        help="Email subject line")
    parser.add_argument("--body", dest="body_text",
                        help="Plain-text body (used if --body-html is not given)")
    parser.add_argument("--body-html", dest="body_html",
                        help="HTML body (optional; if both given, multipart is sent)")
    parser.add_argument("--reply-to", dest="reply_to",
                        help="Reply-To address")
    parser.add_argument("--config-set", dest="config_set",
                        help="SES configuration set name (optional)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the would-be API request and exit (NO network call)")
    parser.add_argument("--region",
                        help="AWS region (default: $AWS_DEFAULT_REGION or boto3 config)")
    args = parser.parse_args()

    region = args.region or os.getenv("AWS_DEFAULT_REGION")

    request_kwargs = build_email_request(
        from_address=args.from_address,
        to_addresses=args.to_addresses,
        subject=args.subject,
        body_text=args.body_text,
        body_html=args.body_html,
        reply_to=args.reply_to,
        config_set=args.config_set,
        cc_addresses=args.cc_addresses,
    )

    if args.dry_run:
        print("── DRY RUN — no email will be sent ──\n")
        print(f"Region:    {region}")
        print("API call:  sesv2.send_email()\n")
        print("Request payload:")
        print(json.dumps(request_kwargs, indent=2, default=str))
        print("\n── End of dry run ──")
        return

    client = get_client(region)

    sandbox = check_sandbox(client)
    if "error" in sandbox:
        print(f"WARNING: Could not check sandbox status: {sandbox['error']}",
              file=sys.stderr)
    elif not sandbox.get("production_access"):
        print(
            "WARNING: Account is in SANDBOX mode. SES will only deliver to "
            "verified email addresses or domains. If the recipient is not "
            "verified, the send will be rejected.",
            file=sys.stderr,
        )

    try:
        response = client.send_email(**request_kwargs)
        msg_id = response.get("MessageId", "UNKNOWN")
        print("✓ Email sent successfully.")
        print(f"  MessageId: {msg_id}")
        print(f"  Region:    {region}")
        print(f"  From:      {args.from_address}")
        print(f"  To:        {', '.join(args.to_addresses)}")
        if args.cc_addresses:
            print(f"  Cc:        {', '.join(args.cc_addresses)}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        msg = exc.response.get("Error", {}).get("Message", str(exc))
        print(f"✗ Send failed: [{code}] {msg}", file=sys.stderr)
        if "not verified" in msg.lower():
            print(
                "HINT: In sandbox, both sender AND recipient must be verified. "
                "Verify the recipient email in the SES console, or request "
                "production access.",
                file=sys.stderr,
            )
        sys.exit(1)
    except BotoCoreError as exc:
        print(f"✗ Network/credential error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
