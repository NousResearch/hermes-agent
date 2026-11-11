#!/usr/bin/env python3
"""Hippius storage query helper — list buckets, objects, and usage stats.

Usage:
    python3 query_storage.py
    python3 query_storage.py --bucket <bucket-name>
    python3 query_storage.py --bucket <bucket-name> --prefix <prefix/>

Requires environment variables:
    HIPPIUS_S3_ACCESS_KEY  — Hippius access key (starts with hip_)
    HIPPIUS_S3_SECRET_KEY  — Hippius secret key
"""

import argparse
import os
import sys

ENDPOINT = "https://s3.hippius.com"
REGION = "decentralized"


def _client():
    try:
        import boto3
    except ImportError:
        print("Error: boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)

    access_key = os.environ.get("HIPPIUS_S3_ACCESS_KEY")
    secret_key = os.environ.get("HIPPIUS_S3_SECRET_KEY")

    if not access_key:
        print(
            "Error: HIPPIUS_S3_ACCESS_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not secret_key:
        print(
            "Error: HIPPIUS_S3_SECRET_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=REGION,
    )


def _human_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def list_buckets(s3) -> list[dict]:
    response = s3.list_buckets()
    buckets = response.get("Buckets", [])
    print(f"\n{'Bucket Name':<40} {'Created'}")
    print("-" * 70)
    for b in buckets:
        created = b["CreationDate"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"{b['Name']:<40} {created}")
    print(f"\nTotal: {len(buckets)} bucket(s)")
    return buckets


def list_objects(s3, bucket: str, prefix: str = "") -> None:
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix

    total_objects = 0
    total_bytes = 0

    print(f"\nObjects in s3://{bucket}/{prefix}")
    print(f"{'Key':<60} {'Size':>12}  {'Last Modified'}")
    print("-" * 100)

    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            size_str = _human_size(obj["Size"])
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M UTC")
            key_display = obj["Key"]
            if len(key_display) > 58:
                key_display = "…" + key_display[-57:]
            print(f"{key_display:<60} {size_str:>12}  {modified}")
            total_objects += 1
            total_bytes += obj["Size"]

    print("-" * 100)
    print(f"Total: {total_objects} object(s) — {_human_size(total_bytes)}")


def storage_stats(s3, buckets: list[dict]) -> None:
    """Aggregate storage usage across all buckets."""
    print("\nStorage usage summary:")
    print(f"{'Bucket':<40} {'Objects':>10} {'Total Size':>14}")
    print("-" * 70)

    grand_objects = 0
    grand_bytes = 0

    paginator = s3.get_paginator("list_objects_v2")
    for b in buckets:
        bucket_objects = 0
        bucket_bytes = 0
        try:
            for page in paginator.paginate(Bucket=b["Name"]):
                for obj in page.get("Contents", []):
                    bucket_objects += 1
                    bucket_bytes += obj["Size"]
        except Exception as exc:
            print(f"{b['Name']:<40} {'ERROR':>10}  {exc}")
            continue

        print(
            f"{b['Name']:<40} {bucket_objects:>10,} {_human_size(bucket_bytes):>14}"
        )
        grand_objects += bucket_objects
        grand_bytes += bucket_bytes

    print("-" * 70)
    print(f"{'TOTAL':<40} {grand_objects:>10,} {_human_size(grand_bytes):>14}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Hippius decentralized storage (Bittensor Subnet 75)."
    )
    parser.add_argument(
        "--bucket",
        metavar="BUCKET",
        help="Bucket to inspect. If omitted, lists all buckets and prints usage stats.",
    )
    parser.add_argument(
        "--prefix",
        metavar="PREFIX",
        default="",
        help="Key prefix to filter objects within a bucket (default: all objects).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-bucket storage usage stats across all buckets (slower).",
    )
    args = parser.parse_args()

    s3 = _client()

    if args.bucket:
        list_objects(s3, args.bucket, args.prefix)
    else:
        buckets = list_buckets(s3)
        if args.stats and buckets:
            storage_stats(s3, buckets)
        elif buckets:
            print(
                "\nTip: pass --stats to see per-bucket usage, or --bucket <name> to list objects."
            )


if __name__ == "__main__":
    main()
