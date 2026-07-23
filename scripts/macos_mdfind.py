#!/usr/bin/env python3
"""
macOS Spotlight (mdfind) wrapper for Hermes Agent.
Provides safe, limited, and structured access to macOS global search.
"""
import argparse
import subprocess
import sys
import platform

def main():
    if platform.system() != "Darwin":
        print("Error: This script is only supported on macOS.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Search files globally using macOS mdfind.")
    parser.add_argument("--name", help="Search by file name (case-insensitive)")
    parser.add_argument("--content", help="Search by file content (including rich documents)")
    parser.add_argument("--type", help="File extension/type (e.g., pdf, md)")
    parser.add_argument("--days", type=int, help="Modified within the last N days")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of results to return (prevent token overflow)")
    parser.add_argument("--onlyin", help="Restrict search to a specific absolute directory path")

    args = parser.parse_args()

    query_parts = []
    # Using 'cd' modifier for case-insensitive and diacritic-insensitive matching
    if args.name:
        query_parts.append(f'kMDItemFSName == "*{args.name}*"cd')
    if args.content:
        query_parts.append(f'kMDItemTextContent == "*{args.content}*"cd')
    if args.type:
        query_parts.append(f'kMDItemFSName == "*.{args.type}"')
    if args.days:
        query_parts.append(f'kMDItemFSContentChangeDate >= $time.today(-{args.days})')

    if not query_parts:
        print("Error: At least one search criteria (--name, --content, --type, --days) must be provided.", file=sys.stderr)
        sys.exit(1)

    query = " && ".join(query_parts)
    
    cmd = ["mdfind"]
    if args.onlyin:
        cmd.extend(["-onlyin", args.onlyin])
    cmd.append(query)

    try:
        # Secure execution: passing list of arguments prevents shell injection
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        
        if not lines:
            print("No results found.")
            return

        limited_lines = lines[:args.limit]
        for line in limited_lines:
            print(line)
            
        if len(lines) > args.limit:
            print(f"\n... and {len(lines) - args.limit} more results. Showing top {args.limit}.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing mdfind: {e.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
