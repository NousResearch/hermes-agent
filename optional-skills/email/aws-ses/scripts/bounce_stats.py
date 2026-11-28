"""fetch ses bounce, complaint, and suppression-list summary."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone, UTC

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_client(region: str | None = None):
    """return a boto3 ses v2 client."""
    kwargs = {"service_name": "sesv2"}
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


def get_cloudwatch_client(region: str | None = None):
    """return a boto3 cloudwatch client."""
    kwargs = {"service_name": "cloudwatch"}
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


def mask_email(address: str) -> str:
    """redact the local part of an address: alice@x.com -> a***@x.com."""
    local, sep, domain = address.partition("@")
    if not sep:
        return "***"
    return f"{local[:1]}***@{domain}"


def list_suppressed(
    client,
    max_items: int = 1000,
    reason_filter: str | None = None,
) -> list[dict]:
    """paginate through list_suppressed_destinations (capped at max_items)."""
    items: list[dict] = []
    next_token: str | None = None
    seen_tokens: set = set()

    while len(items) < max_items:
        kwargs: dict = {"PageSize": min(100, max_items - len(items))}
        if next_token:
            kwargs["NextToken"] = next_token
        if reason_filter:
            # Reasons is a LIST of suppression reasons (BOUNCE / COMPLAINT).
            kwargs["Reasons"] = [reason_filter]

        resp = client.list_suppressed_destinations(**kwargs)
        for entry in resp.get("SuppressedDestinationSummaries", []):
            items.append({
                "email": entry.get("EmailAddress", ""),
                "reason": entry.get("Reason", "UNKNOWN"),
                "updated": str(entry.get("LastUpdateTime", "")),
            })

        next_token = resp.get("NextToken")
        if not next_token or next_token in seen_tokens:
            break
        seen_tokens.add(next_token)

    return items


def collect_stats(
    client,
    max_suppressed: int = 1000,
    reveal_addresses: bool = False,
) -> dict:
    """gather suppression + account health in one dict.

    sample addresses are masked unless reveal_addresses is set — they are
    recipient PII and this output often ends up in logs or chat transcripts.
    """

    result: dict = {
        "account": {},
        "suppression": {
            "total": 0,
            "bounces": 0,
            "complaints": 0,
            "samples": [],
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        acct = client.get_account()
        result["account"] = {
            "production_access": acct.get("ProductionAccessEnabled", False),
            "sending_enabled": acct.get("SendingEnabled", False),
            "enforcement": acct.get("EnforcementStatus", "UNKNOWN"),
            "quota": {
                "Max24HourSend": acct.get("SendQuota", {}).get("Max24HourSend", 0),
                "MaxSendRate": acct.get("SendQuota", {}).get("MaxSendRate", 0),
                "SentLast24Hours": acct.get("SendQuota", {}).get("SentLast24Hours", 0),
            },
        }
    except (BotoCoreError, ClientError) as exc:
        result["account_error"] = str(exc)
        return result

    try:
        suppressed = list_suppressed(client, max_items=max_suppressed)
        result["suppression"]["total"] = len(suppressed)
        result["suppression"]["bounces"] = sum(
            1 for s in suppressed if s["reason"] == "BOUNCE"
        )
        result["suppression"]["complaints"] = sum(
            1 for s in suppressed if s["reason"] == "COMPLAINT"
        )
        # most recent first
        suppressed.sort(key=lambda s: s.get("updated", ""), reverse=True)
        samples = suppressed[:10]
        if not reveal_addresses:
            samples = [{**s, "email": mask_email(s["email"])} for s in samples]
        result["suppression"]["samples"] = samples
    except (BotoCoreError, ClientError) as exc:
        result["suppression_error"] = str(exc)

    return result


def get_complaint_rate(region: str | None = None) -> float | None:
    """return the ses reputation complaint rate (ratio) from cloudwatch.

    aws publishes Reputation.ComplaintRate as a RATIO, not a percent —
    0.001 == 0.1%. callers must multiply by 100 to display a percentage.
    this is the metric ses actually scores suspension on, not the
    cumulative suppression-list count.
    """
    cw = get_cloudwatch_client(region)
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    resp = cw.get_metric_statistics(
        Namespace="AWS/SES",
        MetricName="Reputation.ComplaintRate",
        StartTime=start,
        EndTime=end,
        Period=86400,
        Statistics=["Average"],
    )
    datapoints = sorted(resp.get("Datapoints", []), key=lambda d: d.get("Timestamp", start))
    if datapoints:
        return float(datapoints[-1].get("Average", 0.0))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SES bounce/complaint/suppression summary."
    )
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of a summary table.")
    parser.add_argument("--max-suppressed", type=int, default=1000,
                        help="Max suppression-list entries to scan (default: 1000).")
    parser.add_argument("--region",
                        help="AWS region (default: $AWS_DEFAULT_REGION)")
    parser.add_argument("--show-addresses", action="store_true",
                        help="Include full recipient addresses in --json samples "
                             "(masked by default; they are personal data).")
    args = parser.parse_args()

    region = args.region or os.getenv("AWS_DEFAULT_REGION")
    client = get_client(region)
    stats = collect_stats(
        client,
        max_suppressed=args.max_suppressed,
        reveal_addresses=args.show_addresses,
    )

    if args.json:
        try:
            rate = get_complaint_rate(region)
            # ratio -> percent for the json consumer
            stats["complaint_rate_24h_pct"] = None if rate is None else rate * 100
        except (BotoCoreError, ClientError):
            stats["complaint_rate_24h_pct"] = None
        print(json.dumps(stats, indent=2, default=str))
        return

    acct = stats.get("account", {})
    supp = stats.get("suppression", {})

    print("═" * 50)
    print("  SES Bounce & Complaint Summary")
    print("═" * 50)
    print()

    if "account_error" in stats:
        print(f"✗ Account check failed: {stats['account_error']}")
        return

    print("── Account Health ──")
    print(f"  Production access:  {acct.get('production_access')}")
    print(f"  Sending enabled:    {acct.get('sending_enabled')}")
    print(f"  Enforcement:        {acct.get('enforcement')}")
    q = acct.get("quota", {})
    print(f"  Sent (24h):         {q.get('SentLast24Hours', 0):,.0f} of {q.get('Max24HourSend', 0):,.0f}")
    print(f"  Max send rate:      {q.get('MaxSendRate', 0):.0f} / sec")
    print()

    print("── Suppression List (cumulative, all-time) ──")
    if "suppression_error" in stats:
        print(f"  Error: {stats['suppression_error']}")
    else:
        total = supp.get("total", 0)
        bounces = supp.get("bounces", 0)
        complaints = supp.get("complaints", 0)
        other = total - bounces - complaints

        print(f"  Total suppressed:   {total}")
        print(f"    Bounces:          {bounces}")
        print(f"    Complaints:       {complaints}")
        if other:
            print(f"    Other:            {other}")

    print()

    print("── Reputation Complaint Rate (24h, CloudWatch) ──")
    try:
        rate = get_complaint_rate(region)  # ratio: 0.001 == 0.1%
        if rate is None:
            print("  (no data points in the last 24h)")
        else:
            print(f"  {rate * 100:.3f}%")
            if rate > 0.001:
                print("  ⚠  Rate above 0.1% — review your list hygiene and opt-in flow.")
            elif rate > 0.0005:
                print("  ⚡ Rate elevated — monitor closely.")
    except (BotoCoreError, ClientError) as exc:
        print(f"  (unavailable: {exc})")

    print()
    print(f"  Timestamp: {stats.get('timestamp', '')}")


if __name__ == "__main__":
    main()
