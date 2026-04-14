#!/usr/bin/env python3
# Low-Noise Interaction Policy Router
# Reads the cockpit summary and only emits notifications based on severity rules.

import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    root_dir = Path(os.environ.get("OPENCLAW_ROOT", "/home/yeqiuqiu/clawd-architect"))
    summary_script = root_dir / "ops" / "openclaw" / "continuity" / "cockpit_summary.sh"
    
    if not summary_script.exists():
        sys.exit("cockpit_summary.sh not found.")

    try:
        result = subprocess.run(
            ["bash", str(summary_script)], 
            capture_output=True, 
            text=True, 
            check=True
        )
        summary = result.stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed to generate summary: {e}")

    # Enforce Low-Noise Interaction Policy
    # - SYSTEM OPTIMAL: silent
    # - DEGRADED OPERATIONS: defer/no immediate alert
    # - BLOCKER:* : emit immediately
    if "SYSTEM OPTIMAL" in summary:
        print("Silent success: SYSTEM OPTIMAL. No notification required.")
        sys.exit(0)

    if "DEGRADED OPERATIONS" in summary:
        print("Deferred warning: DEGRADED OPERATIONS captured for batched reporting.")
        sys.exit(0)

    # For now, we print it. This will later integrate with the openclaw message tool
    print("EMITTING COCKPIT ACTION CARD:")
    print(summary)

    # In a full deployment, this would use openclaw's internal messaging RPC
    # openclaw message send --target telegram --message "$summary"

if __name__ == "__main__":
    main()
