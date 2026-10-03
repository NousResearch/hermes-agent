#!/usr/bin/env python3
"""E2E test for _run_skill_verification with a REAL LLM fork.

Usage:
    python tests/manual/e2e_skill_verification.py <skill_name> <action> [allow_repair]

Runs the actual verification pipeline: creates a disposable sandbox, forks a
real AIAgent, and reports VERIFIED/FAILED/UNABLE/repaired.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from types import SimpleNamespace


def load_env(path):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


env = load_env(os.path.expanduser("~/.hermes/.env"))
os.environ.setdefault("DEEPSEEK_API_KEY", env.get("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("DEEPSEEK_BASE_URL", env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

from agent.background_review import _run_skill_verification

# Minimal parent-agent stand-in: only the attrs _run_skill_verification reads.
agent = SimpleNamespace(
    model="deepseek-v4-pro",
    platform="cli",
    provider="deepseek",
    api_mode=None,
    base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=env.get("DEEPSEEK_API_KEY", ""),
    _credential_pool=None,
    request_overrides={},
    session_id="e2e-skill-verification",
    background_review_callback=None,
)

if __name__ == "__main__":
    skill_name = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "create"
    allow_repair = (len(sys.argv) < 4 or sys.argv[3].lower() in ("1", "true", "yes"))

    print(f"=== E2E: verifying skill '{skill_name}' (action={action}, allow_repair={allow_repair}) ===")
    print(f"model={agent.model} provider={agent.provider}")
    print("This will call the real LLM API — may take 1-3 min.\n")
    sys.stdout.flush()

    status, detail = _run_skill_verification(agent, skill_name, action, allow_repair=allow_repair)
    print(f"\n=== RESULT: {status} ===")
    print(detail)
    sys.exit(0 if status in ("verified", "repaired", "unable") else 1)
