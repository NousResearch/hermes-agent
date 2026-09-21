#!/usr/bin/env python3
"""
Moss Repo Scanner — scans Sahil's public repos for fix opportunities.
Inspired by P0/P1 PR patterns from NousResearch/hermes-agent.

Runs as a no_agent cron script. Outputs findings for delivery.

P0/P1 patterns observed:
- Auth credential write-through (multi-profile rotation)
- Nix dependency fixes
- Gateway command-line matcher hardening
- SQLite trigram tokenizer fallback
- Model picker caps
- Session message preservation
- Custom provider persistence
- Skill rmtree scope guard
- CUA environment scrubbing
- Curator snapshot pruning
- Docker gateway takeover
- Stale lock eviction
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

STATE_FILE = os.path.expanduser("~/.hermes/data/moss-repo-scan.json")
CLONE_CACHE_DIR = os.path.expanduser("~/.hermes/data/moss-clones")
TARGET_REPOS = [
    "Sahil-SS9/hermes-agent",
    "Sahil-SS9/GitRadar-Self-Improvement",
    "Sahil-SS9/hermes-memlock",
    "Sahil-SS9/hermaguard",
    "Sahil-SS9/Toolaria",
    "Sahil-SS9/hermes-multichannel-prompt-optimizer",
    "Sahil-SS9/mnemosyne",
    "Sahil-SS9/hermes-simplify-swarm",
    "Sahil-SS9/MrHermagi-tutorbot",
    "Sahil-SS9/hermes-Custom-CLI-Themes",
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_scan": None, "reported_findings": {}}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def make_fingerprint(repo, filepath, pattern_type, context_lines):
    """Create a stable fingerprint that survives line shifts."""
    context = "".join(context_lines).strip()
    raw = f"{repo}:{filepath}:{pattern_type}:{context}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def ensure_clone(repo):
    """Clone or fetch a repo into persistent cache. Returns path or None."""
    clone_dir = os.path.join(CLONE_CACHE_DIR, repo.replace("/", "-"))
    os.makedirs(CLONE_CACHE_DIR, exist_ok=True)

    if not os.path.exists(clone_dir):
        result = subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", clone_dir],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None
    else:
        subprocess.run(
            ["git", "-C", clone_dir, "fetch", "origin"],
            capture_output=True, text=True, timeout=30
        )

    subprocess.run(
        ["git", "-C", clone_dir, "checkout", "main"],
        capture_output=True, text=True, timeout=10
    )
    return clone_dir

def grep_pattern(pattern, clone_dir, file_types=None):
    """Run grep and return matching lines with context."""
    includes = []
    for ext in (file_types or [".py", ".ts", ".tsx"]):
        includes.extend(["--include", f"*{ext}"])

    result = subprocess.run(
        ["grep", "-rn", "-B1", "-A1"] + includes + [pattern, clone_dir],
        capture_output=True, text=True, timeout=30
    )
    if not result.stdout.strip():
        return []

    matches = []
    for block in result.stdout.strip().split("\n--\n"):
        lines = block.strip().split("\n")
        if not lines:
            continue
        # First line is the match, surrounding lines are context
        match_line = lines[0] if lines else ""
        context = lines[1:4] if len(lines) > 1 else []
        matches.append({"match_line": match_line, "context": context})

    return matches

def scan_repo(repo, state):
    """Scan a single repo for fix opportunities. Returns new findings only."""
    clone_dir = ensure_clone(repo)
    if clone_dir is None:
        return [{"type": "clone_failed", "repo": repo, "items": ["Could not clone repo"]}]

    findings = []
    reported = state["reported_findings"].get(repo, {})

    try:
        # Pattern 1: Hardcoded constants that should be configurable
        matches = grep_pattern(
            r"max_models=50\|timeout=120\|max_iterations=25",
            clone_dir, [".py", ".ts", ".tsx"]
        )
        for m in matches[:10]:
            filepath = m["match_line"].split(":")[0].replace(clone_dir + "/", "")
            fp = make_fingerprint(repo, filepath, "hardcoded_constants", m["context"])
            if fp not in reported:
                reported[fp] = {"type": "hardcoded_constants", "seen_at": datetime.now(timezone.utc).isoformat()}
                findings.append({
                    "type": "hardcoded_constants",
                    "file": filepath,
                    "detail": m["match_line"].replace(clone_dir + "/", ""),
                })

        # Pattern 2: Bare except: pass (multi-line aware)
        matches = grep_pattern(
            r"except.*:\s*$",
            clone_dir, [".py"]
        )
        for m in matches[:10]:
            filepath = m["match_line"].split(":")[0].replace(clone_dir + "/", "")
            # Check if next line is pass
            if any("pass" in c for c in m["context"]):
                fp = make_fingerprint(repo, filepath, "bare_except_pass", m["context"])
                if fp not in reported:
                    reported[fp] = {"type": "bare_except_pass", "seen_at": datetime.now(timezone.utc).isoformat()}
                    findings.append({
                        "type": "bare_except_pass",
                        "file": filepath,
                        "detail": m["match_line"].replace(clone_dir + "/", ""),
                    })

        # Pattern 3: TODO/FIXME with security keywords (signal-to-noise filter)
        matches = grep_pattern(
            r"TODO\|FIXME\|HACK\|XXX",
            clone_dir, [".py", ".ts", ".tsx"]
        )
        security_keywords = ["security", "crash", "data loss", "regression", "auth", "leak", "race"]
        for m in matches[:20]:
            filepath = m["match_line"].split(":")[0].replace(clone_dir + "/", "")
            content_lower = m["match_line"].lower()
            # Only report TODOs with security-relevant keywords
            if any(kw in content_lower for kw in security_keywords):
                fp = make_fingerprint(repo, filepath, "todo_security", m["context"])
                if fp not in reported:
                    reported[fp] = {"type": "todo_security", "file": filepath, "seen_at": datetime.now(timezone.utc).isoformat()}
                    findings.append({
                        "type": "todo_security",
                        "file": filepath,
                        "detail": m["match_line"].replace(clone_dir + "/", ""),
                    })

        # Pattern 4: print() in production code (not in comments or test files)
        matches = grep_pattern(
            r"^[^#]*print\(",
            clone_dir, [".py"]
        )
        for m in matches[:10]:
            filepath = m["match_line"].split(":")[0].replace(clone_dir + "/", "")
            # Skip test files — print is fine there
            if "test" in filepath or "conftest" in filepath:
                continue
            fp = make_fingerprint(repo, filepath, "print_statements", m["context"])
            if fp not in reported:
                reported[fp] = {"type": "print_statements", "file": filepath, "seen_at": datetime.now(timezone.utc).isoformat()}
                findings.append({
                    "type": "print_statements",
                    "file": filepath,
                    "detail": m["match_line"].replace(clone_dir + "/", ""),
                })

    finally:
        # No cleanup — clones are persistent in CLONE_CACHE_DIR
        pass

    state["reported_findings"][repo] = reported
    return findings

def main():
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    all_findings = {}

    for repo in TARGET_REPOS:
        findings = scan_repo(repo, state)
        if findings:
            all_findings[repo] = findings

    state["last_scan"] = now
    save_state(state)

    if all_findings:
        total = sum(len(v) for v in all_findings.values())
        print(f"Moss repo scan: {total} new findings across {len(all_findings)} repos")
        for repo, findings in all_findings.items():
            print(f"\n{repo}:")
            for f in findings:
                print(f"  [{f['type']}] {f.get('file', '?')}")
                if f.get("detail"):
                    print(f"    {f['detail']}")
    else:
        # Silent — nothing new to report
        pass

if __name__ == "__main__":
    main()