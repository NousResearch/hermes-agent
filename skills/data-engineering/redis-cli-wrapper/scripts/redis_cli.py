#!/usr/bin/env python3
"""Redis CLI wrapper for Hermes Agent."""
import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

try:
    import redis
except ImportError:
    print("Error: redis not installed. Run: pip install redis", file=sys.stderr)
    sys.exit(1)

from tabulate import tabulate


def get_redis_client() -> redis.Redis:
    """Create Redis client from environment."""
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    password = os.environ.get("REDIS_PASSWORD")
    return redis.from_url(url, password=password, decode_responses=True)


def cmd_keys(args: argparse.Namespace) -> None:
    """Scan keys by pattern."""
    client = get_redis_client()
    pattern = args.pattern or "*"
    limit = args.limit
    
    keys = []
    cursor = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=1000)
        keys.extend(batch)
        if limit and len(keys) >= limit:
            keys = keys[:limit]
            break
        if cursor == 0:
            break
    
    if args.count_only:
        print(len(keys))
    else:
        if args.format == "json":
            json.dump(keys, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for k in keys:
                print(k)


def cmd_memory(args: argparse.Namespace) -> None:
    """Memory analysis."""
    client = get_redis_client()
    
    if args.stats:
        info = client.info("memory")
        if args.format == "json":
            json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for k, v in info.items():
                print(f"{k}: {v}")
        return
    
    # Find largest keys (using MEMORY USAGE)
    pattern = args.pattern or "*"
    top_n = args.top_keys or 20
    
    keys = []
    cursor = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=5000)
        keys.extend(batch)
        if cursor == 0:
            break
    
    # Sample if too many keys
    if len(keys) > 10000:
        import random
        keys = random.sample(keys, 10000)
        print(f"Sampled 10000 keys from {len(keys)} total", file=sys.stderr)
    
    key_sizes = []
    for key in keys:
        try:
            size = client.memory_usage(key)
            if size:
                key_sizes.append((key, int(size)))
        except Exception:
            pass
    
    key_sizes.sort(key=lambda x: x[1], reverse=True)
    top = key_sizes[:top_n]
    
    if args.format == "json":
        json.dump([{"key": k, "bytes": b, "human": human_size(b)} for k, b in top], sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(tabulate([(k, human_size(b)) for k, b in top], headers=["Key", "Size"], tablefmt="grid"))


def cmd_ttl(args: argparse.Namespace) -> None:
    """TTL analysis."""
    client = get_redis_client()
    pattern = args.pattern or "*"
    
    keys = []
    cursor = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=5000)
        keys.extend(batch)
        if cursor == 0:
            break
    
    if len(keys) > 10000:
        import random
        keys = random.sample(keys, 10000)
        print(f"Sampled 10000 keys from {len(keys)} total", file=sys.stderr)
    
    now = time.time()
    ttl_data = []
    no_expire = 0
    expiring_soon = 0
    
    for key in keys:
        try:
            ttl = client.ttl(key)
            if ttl == -1:
                no_expire += 1
            elif ttl == -2:
                continue  # key doesn't exist
            else:
                ttl_data.append(ttl)
                if args.expiring_within and ttl <= args.expiring_within:
                    expiring_soon += 1
        except Exception:
            pass
    
    if args.distribution:
        # Bucket TTLs
        buckets = {
            "<1m": 0, "1-5m": 0, "5-15m": 0, "15-30m": 0,
            "30-60m": 0, "1-6h": 0, "6-24h": 0, "1-7d": 0, ">7d": 0
        }
        for ttl in ttl_data:
            if ttl < 60:
                buckets["<1m"] += 1
            elif ttl < 300:
                buckets["1-5m"] += 1
            elif ttl < 900:
                buckets["5-15m"] += 1
            elif ttl < 1800:
                buckets["15-30m"] += 1
            elif ttl < 3600:
                buckets["30-60m"] += 1
            elif ttl < 21600:
                buckets["1-6h"] += 1
            elif ttl < 86400:
                buckets["6-24h"] += 1
            elif ttl < 604800:
                buckets["1-7d"] += 1
            else:
                buckets[">7d"] += 1
        
        if args.format == "json":
            json.dump(buckets, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(tabulate(buckets.items(), headers=["TTL Range", "Count"], tablefmt="grid"))
        
        if args.expiring_within:
            print(f"\nKeys expiring within {args.expiring_within}s: {expiring_soon}")
        print(f"Keys with no expiry: {no_expire}")
    else:
        # Just summary
        print(f"Total keys scanned: {len(keys)}")
        print(f"Keys with TTL: {len(ttl_data)}")
        print(f"Keys with no expiry: {no_expire}")
        if ttl_data:
            print(f"Min TTL: {min(ttl_data)}s, Max TTL: {max(ttl_data)}s, Avg TTL: {sum(ttl_data)/len(ttl_data):.0f}s")


def cmd_info(args: argparse.Namespace) -> None:
    """Parse and display Redis INFO."""
    client = get_redis_client()
    section = args.section
    
    info = client.info(section) if section else client.info()
    
    if args.format == "json":
        json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        if isinstance(info, dict):
            for k, v in info.items():
                print(f"{k}: {v}")
        else:
            print(info)


def human_size(bytes_val: int) -> str:
    """Convert bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"


def main():
    parser = argparse.ArgumentParser(description="Redis CLI wrapper for Hermes Agent")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # keys command
    p_keys = subparsers.add_parser("keys", help="Scan keys by pattern")
    p_keys.add_argument("--pattern", default="*", help="Key pattern (default: *)")
    p_keys.add_argument("--limit", type=int, help="Max keys to return")
    p_keys.add_argument("--count-only", action="store_true", help="Only output count")
    
    # memory command
    p_mem = subparsers.add_parser("memory", help="Memory analysis")
    p_mem.add_argument("--stats", action="store_true", help="Show INFO memory stats")
    p_mem.add_argument("--top-keys", type=int, help="Show top N largest keys")
    p_mem.add_argument("--pattern", default="*", help="Key pattern for sampling")
    
    # ttl command
    p_ttl = subparsers.add_parser("ttl", help="TTL analysis")
    p_ttl.add_argument("--distribution", action="store_true", help="Show TTL distribution")
    p_ttl.add_argument("--expiring-within", type=int, help="Count keys expiring within N seconds")
    p_ttl.add_argument("--pattern", default="*", help="Key pattern for sampling")
    
    # info command
    p_info = subparsers.add_parser("info", help="Parse Redis INFO")
    p_info.add_argument("--section", help="Specific INFO section (memory, stats, replication, etc.)")
    
    args = parser.parse_args()
    
    if args.command == "keys":
        cmd_keys(args)
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command == "ttl":
        cmd_ttl(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()