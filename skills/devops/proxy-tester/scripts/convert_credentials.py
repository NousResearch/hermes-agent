#!/usr/bin/env python3
"""Convert colon-separated proxy credentials to full proxy URIs.

Input format (one per line):
  host:port:user:pass
  host:port
  [scheme://]host:port:user:pass  (scheme optional, defaults to http)

Output:
  scheme://user:pass@host:port  (one per line)

Usage:
  convert_credentials.py < input.txt > output.txt
  convert_credentials.py --scheme socks5 < input.txt > output.txt
  cat proxies.txt | convert_credentials.py --scheme http > uris.txt
"""

import argparse
import re
import sys
from typing import Optional


def parse_line(line: str, default_scheme: str = "http") -> Optional[str]:
    """Parse a single line and return a proper proxy URI or None if malformed."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # If already a full URI (contains ://), pass through
    if "://" in line:
        return line

    parts = line.split(":")
    if len(parts) == 2:
        # host:port
        host, port = parts
        return f"{default_scheme}://{host}:{port}"
    elif len(parts) >= 4:
        # host:port:user:pass (extra cols belong in password)
        host = parts[0]
        port = parts[1]
        user = parts[2]
        passwd = ":".join(parts[3:])  # password may contain colons
        return f"{default_scheme}://{user}:{passwd}@{host}:{port}"
    else:
        # malformed
        return None


def main():
    ap = argparse.ArgumentParser(description="Convert colon-separated proxy credentials to full proxy URIs")
    ap.add_argument("--scheme", "-s", default="http", choices=["http", "https", "socks4", "socks5"],
                    help="Default protocol scheme (when not present in input)")
    ap.add_argument("file", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                    help="Input file (default: stdin)")
    args = ap.parse_args()

    out = []
    skipped = []
    for idx, raw in enumerate(args.file, 1):
        result = parse_line(raw, args.scheme)
        if result:
            out.append(result)
        else:
            skipped.append((idx, raw.rstrip("\n")))

    # Write URIs to stdout
    for uri in out:
        print(uri)

    # Report skipped lines to stderr
    if skipped:
        print(f"\n⚠ Skipped {len(skipped)} malformed line(s):", file=sys.stderr)
        for lineno, line in skipped:
            print(f"  L{lineno}: {line}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
