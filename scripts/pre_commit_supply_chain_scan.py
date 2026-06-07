#!/usr/bin/env python3
"""Pre-commit supply chain scan — mirrors .github/workflows/supply-chain-audit.yml.

Scans staged diff for critical patterns:
  - .pth files (auto-execute on Python startup)
  - base64.decode + exec/eval combos (the litellm attack pattern)
  - subprocess with encoded/obfuscated commands
  - install-hook files (setup.py, sitecustomize.py, usercustomize.py, __init__.pth)

Exit code 1 = critical finding, blocks commit.
"""
import subprocess
import sys


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def main() -> int:
    # Staged diff — only what's about to be committed
    diff = run(["git", "diff", "--cached"])
    if not diff.strip():
        return 0

    findings: list[str] = []

    # --- .pth files ---
    pth_files = run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    ).strip()
    pth_hits = [f for f in pth_files.split("\n") if f.endswith(".pth")]
    if pth_hits:
        findings.append(
            "CRITICAL: .pth file(s) staged: "
            + ", ".join(pth_hits)
            + "\n  .pth files auto-execute on Python startup — supply chain risk."
        )

    # --- base64 decode + exec/eval on the same line ---
    import re
    for line in diff.split("\n"):
        if not line.startswith("+"):
            continue
        if re.search(r"base64\.(b64decode|decodebytes|urlsafe_b64_decode)", line) and \
           re.search(r"\bexec\(|\beval\(", line):
            findings.append(
                f"CRITICAL: base64 decode + exec/eval combo:\n  {line.strip()}"
            )

    # --- subprocess with encoded/obfuscated command ---
    for line in diff.split("\n"):
        if not line.startswith("+"):
            continue
        if re.search(r"subprocess\.(Popen|call|run)\s*\(", line) and \
           re.search(r"base64|\\x[0-9a-fA-F]{2}|chr\(", line):
            findings.append(
                f"CRITICAL: subprocess with encoded/obfuscated command:\n  {line.strip()}"
            )

    # --- install-hook files (repo-root only) ---
    hook_files = {"setup.py", "setup.cfg", "sitecustomize.py", "usercustomize.py", "__init__.pth"}
    staged_files = run(
        ["git", "diff", "--cached", "--name-only"]
    ).strip().split("\n")
    hook_hits = [f for f in staged_files if f in hook_files]
    if hook_hits:
        findings.append(
            "CRITICAL: install-hook file(s) staged: "
            + ", ".join(hook_hits)
            + "\n  These can execute code during pip install or interpreter startup."
        )

    if findings:
        print("\n🚨 SUPPLY CHAIN RISK DETECTED\n")
        for f in findings:
            print(f)
            print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
