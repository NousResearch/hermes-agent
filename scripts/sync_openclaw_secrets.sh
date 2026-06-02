#!/usr/bin/env bash
# Copy allowlisted secrets from OpenClaw into ~/.hermes/.env (no overwrite if already set).
set -euo pipefail

HERMES_ENV="${HOME}/.hermes/.env"
OPENCLAW_JSON="${HOME}/.openclaw/openclaw.json"
export HERMES_ENV OPENCLAW_JSON
mkdir -p "${HOME}/.hermes"
touch "$HERMES_ENV"

python3 <<'PY'
import json
import os
import re
from pathlib import Path

hermes_env = Path(os.environ["HOME"]) / ".hermes" / ".env"
openclaw_json = Path(os.environ["HOME"]) / ".openclaw" / "openclaw.json"

def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def write_env(path: Path, data: dict[str, str]) -> None:
    lines = []
    if path.is_file():
        lines = path.read_text().splitlines()
    keys = {ln.split("=", 1)[0].strip() for ln in lines if "=" in ln and not ln.strip().startswith("#")}
    added = []
    for k, v in sorted(data.items()):
        if k in keys:
            continue
        lines.append(f'{k}="{v}"')
        added.append(k)
    path.write_text("\n".join(lines).rstrip() + "\n")
    for k in added:
        print(f"  + {k}")

existing = read_env(hermes_env)
to_add: dict[str, str] = {}

if openclaw_json.is_file():
    cfg = json.loads(openclaw_json.read_text())
    entries = (cfg.get("plugins") or {}).get("entries") or {}
    tav = entries.get("tavily") or {}
    key = ((tav.get("config") or {}).get("webSearch") or {}).get("apiKey")
    if key and not existing.get("TAVILY_API_KEY"):
        to_add["TAVILY_API_KEY"] = key

if to_add:
    print("Syncing to ~/.hermes/.env:")
    write_env(hermes_env, to_add)
else:
    print("No new keys to sync (TAVILY_API_KEY already set or missing in OpenClaw).")
PY
