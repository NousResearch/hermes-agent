#!/usr/bin/env python3
"""Cloud storage setup — check rclone install and remotes."""

import argparse
import json
import os
import subprocess
import sys

def _run(args, check=False, timeout=30):
    try:
        result = subprocess.run(
            ["rclone"] + args,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )
        return result
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None

def check_setup():
    status = {}
    result = _run(["version"], timeout=5)
    if result is None:
        status["rclone_installed"] = False
        status["status"] = "rclone not installed. Install with: pkg install rclone (Termux) or apt install rclone (Linux)"
        print(json.dumps(status))
        return 1

    status["rclone_installed"] = True
    status["rclone_version"] = result.stdout.splitlines()[0] if result.stdout else "unknown"

    result = _run(["listremotes"], timeout=5)
    remotes = []
    if result and result.stdout:
        remotes = [r.rstrip(":") for r in result.stdout.strip().split("\n") if r.strip()]
    status["remotes_count"] = len(remotes)
    status["remotes"] = remotes

    if remotes:
        try:
            about = _run(["about", remotes[0] + ":"], timeout=10)
            status["sample_remote"] = remotes[0]
            status["sample_about"] = about.stdout.strip() if about and about.stdout else None
        except Exception:
            status["sample_about"] = None
        status["status"] = "ready"
    else:
        status["status"] = "rclone installed but no remotes configured. Run: rclone config"

    print(json.dumps(status, indent=2))
    return 0 if remotes else 1

def main():
    parser = argparse.ArgumentParser(description="cloud-storage setup")
    parser.add_argument("--check", action="store_true", help="Check installation")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    if args.check:
        sys.exit(check_setup())

    parser.print_help()

if __name__ == "__main__":
    main()
