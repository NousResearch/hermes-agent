"""derive an ses smtp password from an iam secret key.

the password is not the raw iam secret — it is a sigv4-style derivation:
  1. sign "11111111" with HMAC-SHA256 keyed by "AWS4" + secret
  2. chain through region, "ses", "aws4_request"
  3. sign the literal "SendRawEmail"
  4. base64-encode a 0x04 version byte prepended to that signature

the smtp username is the iam access key id. region is REQUIRED — the
derived password differs per region.
see: https://docs.aws.amazon.com/ses/latest/dg/smtp-credentials.html
"""

import argparse
import base64
import hashlib
import hmac
import os
import sys

import boto3


def derive_smtp_password(iam_secret: str, region: str) -> str:
    """derive the ses smtp password for an iam secret and region."""
    if not iam_secret:
        raise ValueError("IAM secret key must not be empty")
    if not region:
        raise ValueError("region is required for smtp password derivation")

    def _sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    signature = _sign(("AWS4" + iam_secret).encode("utf-8"), "11111111")
    signature = _sign(signature, region)
    signature = _sign(signature, "ses")
    signature = _sign(signature, "aws4_request")
    signature = _sign(signature, "SendRawEmail")

    return base64.b64encode(bytes([0x04]) + signature).decode("ascii")


def _resolve_credentials(secret_from_stdin: bool) -> tuple[str, str | None]:
    """resolve (iam_secret, iam_key_id) from stdin or the boto3 chain."""
    if secret_from_stdin:
        secret = sys.stdin.readline().strip()
        if not secret:
            raise ValueError("--secret-stdin given but nothing was read from stdin.")
        try:
            creds = boto3.Session().get_credentials()
            key_id = creds.access_key if creds else None
        except Exception:
            key_id = None
        return secret, key_id

    creds = boto3.Session().get_credentials()
    if not creds or not creds.secret_key:
        raise ValueError(
            "No IAM secret available. Configure the standard AWS credential "
            "chain (AWS_PROFILE, shared credentials file, SSO, or an IAM role) "
            "or pipe the secret in with --secret-stdin."
        )
    return creds.secret_key, creds.access_key


def write_credentials_file(path: str, username: str, password: str, region: str) -> None:
    """write smtp credentials to a new file readable only by the owner."""
    # O_EXCL: never clobber or follow a pre-existing file/symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"SMTP_USERNAME={username}\n")
        fh.write(f"SMTP_PASSWORD={password}\n")
        fh.write(f"SMTP_SERVER=email-smtp.{region}.amazonaws.com\n")
        fh.write("SMTP_PORT=587\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an SES SMTP password from an IAM secret key. The secret is "
            "read from the boto3 credential chain (or stdin), never from argv."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region for the SMTP endpoint (required; also read from AWS_DEFAULT_REGION)",
    )
    parser.add_argument(
        "--secret-stdin",
        action="store_true",
        help="Read the IAM secret from stdin instead of the boto3 credential chain.",
    )
    out = parser.add_mutually_exclusive_group()
    out.add_argument(
        "--output",
        metavar="PATH",
        help="Write credentials to a NEW file with 0600 permissions (refuses to overwrite).",
    )
    out.add_argument(
        "--show-password",
        action="store_true",
        help="Print the SMTP password to stdout. Avoid in logged or shared sessions.",
    )
    args = parser.parse_args()

    region = args.region or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        print("ERROR: region is required. Pass --region or set AWS_DEFAULT_REGION.", file=sys.stderr)
        sys.exit(1)

    try:
        iam_secret, iam_key_id = _resolve_credentials(args.secret_stdin)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    smtp_password = derive_smtp_password(iam_secret, region)
    username = iam_key_id or "<not resolved>"

    if args.output:
        try:
            write_credentials_file(args.output, username, smtp_password, region)
        except FileExistsError:
            print(f"ERROR: {args.output} already exists; refusing to overwrite.", file=sys.stderr)
            sys.exit(1)
        print(f"Wrote SMTP credentials to {args.output} (mode 0600).")
        return

    print(f"SMTP_USERNAME: {username}")
    if args.show_password:
        print("WARNING: printing a live SMTP password.", file=sys.stderr)
        print(f"SMTP_PASSWORD: {smtp_password}")
    else:
        print("SMTP_PASSWORD: <hidden — rerun with --output PATH or --show-password>")
    print(f"SMTP_SERVER:   email-smtp.{region}.amazonaws.com")
    print("SMTP_PORT:     587")


if __name__ == "__main__":
    main()
