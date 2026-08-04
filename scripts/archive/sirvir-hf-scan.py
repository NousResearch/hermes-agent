#!/usr/bin/env python3
"""Sirvir — Daily HuggingFace Model Scan

Scans HuggingFace for new GGUF models matching fleet archetypes.
Filters by known creators, logs findings, alerts on potential upgrades.

Run as no_agent cron. Silent when no new models found.
Owner: Sirvir. Schedule: 0 8 * * * (daily 08:00)
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# Fleet archetypes — size ranges in billions of parameters
ARCHETYPES = {
    "27-28B_dense": {"size_min": 26, "size_max": 29, "current": "Darwin-28B-REASON.Q5_K_M"},
    "35B_MoE": {"size_min": 33, "size_max": 37, "current": None},
    "27B_hybrid": {"size_min": 25, "size_max": 28, "current": None},
    "9B_aux": {"size_min": 8, "size_max": 11, "current": "Qwythos-9B-Claude-Mythos-5-1M"},
}

KNOWN_CREATORS = {"unsloth", "bartowski", "Ex0bit", "I-Nano", "I-Compact", "Jackrong"}

HF_API = "https://huggingface.co/api/models"
HEADERS = {"User-Agent": "Sirvir-KenseiAgent/1.0"}
QUALITY_DB = Path("/home/kensei/.hermes/skills/turbofit/references/creator-quality-database.yaml")

def fetch_models():
    """Fetch recent GGUF models from HuggingFace."""
    params = "?filter=gguf&sort=lastModified&direction=-1&limit=50"
    url = f"{HF_API}{params}"
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (URLError, json.JSONDecodeError) as e:
        print(f"ERROR: HF API failed: {e}", file=sys.stderr)
        return []

def find_matches(models):
    """Find models matching fleet archetypes from known creators."""
    matches = []
    for m in models:
        author = m.get("author", "")
        if author not in KNOWN_CREATORS:
            continue
        
        # Rough size estimate from model name/description
        tags = " ".join(m.get("tags", []))
        name = m.get("modelId", "")
        desc = f"{name} {tags}".lower()
        
        for archetype, spec in ARCHETYPES.items():
            # Simple heuristic: check if model name contains size indicators
            if any(str(s) in desc for s in range(spec["size_min"], spec["size_max"] + 1)):
                matches.append({
                    "model_id": name,
                    "author": author,
                    "archetype": archetype,
                    "last_modified": m.get("lastModified", ""),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                })
                break
    return matches

def main():
    models = fetch_models()
    if not models:
        return
    
    matches = find_matches(models)
    if not matches:
        return  # Silent — no new matches
    
    # Output findings
    print(f"🔍 HF Scan — {datetime.now().strftime('%d/%m/%Y')}")
    print(f"{len(matches)} new GGUF model(s) from known creators")
    for m in matches:
        current = ARCHETYPES[m["archetype"]]["current"]
        status = "NEW" if not current else "POTENTIAL UPGRADE"
        print(f"  {status} | {m['author']}/{m['model_id']} | {m['archetype']} | ⬇{m['downloads']}")

if __name__ == "__main__":
    main()
