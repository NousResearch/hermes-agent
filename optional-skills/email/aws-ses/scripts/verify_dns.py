"""check dkim cnames and spf txt records for an ses domain.

dkim tokens are fetched from the ses api (get_email_identity) — they are
generated per identity per region, not a fixed list. --selector overrides
the api and checks specific tokens manually.
"""

import argparse
import os
import re
import subprocess
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError


_LABEL = re.compile(r"^(?!-)[A-Za-z0-9_-]{1,63}(?<!-)$")
_RECORD_TYPES = {"CNAME", "TXT"}


def is_valid_dns_name(name: str) -> bool:
    """true for a plain dns name (letters, digits, '-', '_'; no leading '-')."""
    if not name or len(name) > 253:
        return False
    return all(_LABEL.match(label) for label in name.rstrip(".").split("."))


def get_client(region: str | None = None):
    """return a boto3 ses v2 client."""
    kwargs = {"service_name": "sesv2"}
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


def get_dkim_tokens(client, domain: str) -> list[str]:
    """fetch the domain's dkim tokens from the ses api."""
    identity = client.get_email_identity(EmailIdentity=domain)
    return identity.get("DkimAttributes", {}).get("Tokens", [])


def dig_short(name: str, record_type: str = "CNAME") -> list[str]:
    """run `dig +short -q <name> -t <type>` and return non-empty result lines."""
    if not is_valid_dns_name(name):
        raise ValueError(f"refusing to query invalid DNS name: {name!r}")
    if record_type not in _RECORD_TYPES:
        raise ValueError(f"unsupported record type: {record_type!r}")
    try:
        # -q/-t pin each value to its role so a name can never be read as a dig option.
        proc = subprocess.run(
            ["dig", "+short", "-q", name, "-t", record_type],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # dig returns non-zero on NXDOMAIN; empty output is handled below
        )
    except FileNotFoundError:
        print("ERROR: 'dig' not found. Install dnsutils / bind-tools.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return []

    lines = [line.strip().rstrip(".") for line in proc.stdout.splitlines() if line.strip()]
    return lines


def check_dkim(domain: str, selectors: list[str]) -> dict:
    """check each dkim selector for a cname pointing to *.dkim.amazonses.com."""
    results: dict[str, dict] = {}
    for selector in selectors:
        name = f"{selector}._domainkey.{domain}"
        answers = dig_short(name, "CNAME")
        passed = any(
            ans.endswith(".dkim.amazonses.com") for ans in answers
        )
        results[selector] = {
            "name": name,
            "answers": answers,
            "passed": passed,
        }
    return results


def check_spf(domain: str) -> dict:
    """check txt records for an spf entry that includes amazonses.com."""
    answers = dig_short(domain, "TXT")
    spf_records = [a for a in answers if "v=spf1" in a]
    has_ses = any("include:amazonses.com" in a for a in spf_records)
    return {
        "domain": domain,
        "spf_records": spf_records,
        "includes_amazonses": has_ses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify SES DKIM CNAMEs and SPF record for a domain."
    )
    parser.add_argument("domain", help="Domain name to check (e.g. mydomain.com)")
    parser.add_argument(
        "--selector",
        action="append",
        dest="selectors",
        help="DKIM selector to check manually (repeat for multiple). Defaults to the tokens from the SES API.",
    )
    parser.add_argument(
        "--region",
        help="AWS region (default: $AWS_DEFAULT_REGION)",
    )
    args = parser.parse_args()
    domain: str = args.domain
    region: str = args.region or os.getenv("AWS_DEFAULT_REGION")

    if not is_valid_dns_name(domain):
        print(f"ERROR: invalid domain name: {domain!r}", file=sys.stderr)
        sys.exit(1)

    if args.selectors:
        selectors = args.selectors
    else:
        try:
            client = get_client(region)
            selectors = get_dkim_tokens(client, domain)
        except (BotoCoreError, ClientError) as exc:
            print(
                f"ERROR: could not fetch DKIM tokens from SES: {exc}\n"
                "Pass --selector <token> to check records manually.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not selectors:
        print("ERROR: no DKIM tokens found for this domain.", file=sys.stderr)
        sys.exit(1)

    bad = [s for s in selectors if not _LABEL.match(s)]
    if bad:
        print(f"ERROR: invalid DKIM selector(s): {', '.join(map(repr, bad))}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking DNS for: {domain}\n")

    print("── DKIM CNAME records ──")
    dkim = check_dkim(domain, selectors)
    any_pass = False
    for selector, info in dkim.items():
        flag = "PASS" if info["passed"] else "FAIL"
        print(f"  {flag}  {info['name']}")
        if info["answers"]:
            for ans in info["answers"]:
                print(f"         → {ans}")
            if not info["passed"]:
                print("         (expected target ending in .dkim.amazonses.com)")
        else:
            print("         (no CNAME record found)")
        if info["passed"]:
            any_pass = True

    print()

    print("── SPF TXT record ──")
    spf = check_spf(domain)
    if spf["includes_amazonses"]:
        print("  PASS  SPF includes amazonses.com")
    else:
        print("  FAIL  SPF does NOT include amazonses.com")
        if spf["spf_records"]:
            for rec in spf["spf_records"]:
                print(f"         Found: {rec}")
            print("         Expected: v=spf1 ... include:amazonses.com ... ~all")
        else:
            print("         (no SPF TXT record found for this domain)")

    print()

    if any_pass and spf["includes_amazonses"] and all(
        info["passed"] for info in dkim.values()
    ):
        print("✓ All checks passed — domain is ready for SES sending.")
    else:
        if not any_pass:
            print("✗ DKIM: No valid DKIM CNAMEs found. Add the three records from the SES console.")
        elif not all(info["passed"] for info in dkim.values()):
            print("✗ DKIM: Some DKIM CNAMEs are missing. Add all records from the SES console.")
        if not spf["includes_amazonses"]:
            print("✗ SPF:  Add 'include:amazonses.com' to your SPF TXT record.")


if __name__ == "__main__":
    main()
