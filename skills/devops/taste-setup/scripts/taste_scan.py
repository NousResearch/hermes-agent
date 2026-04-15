#!/usr/bin/env python3
"""
Main Taste scan entry point.
Runs email/calendar scan and Spotify sync via MCP.
"""
import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

def main():
    """Run all Taste scans."""
    print("=" * 60)
    print("Taste Scan - Starting")
    print("=" * 60)
    
    # Email and calendar scan
    print("\n[1/2] Email and Calendar Scan")
    print("-" * 60)
    try:
        from email_scan import main as email_main
        email_main()
    except Exception as e:
        print(f"Email scan failed: {e}")
    
    # Spotify sync via MCP
    print("\n[2/2] Spotify Sync (via MCP)")
    print("-" * 60)
    try:
        from spotify_sync_mcp import main as spotify_main
        spotify_main()
    except Exception as e:
        print(f"Spotify sync failed: {e}")
    
    print("\n" + "=" * 60)
    print("Taste Scan - Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()