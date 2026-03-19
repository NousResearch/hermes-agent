#!/usr/bin/env python3
"""Auto-posts to Moltbook every 30 minutes"""
import asyncio
import requests
import random
import os
from datetime import datetime

MOLTBOOK_API = "https://moltbook.com/api/v1"
API_KEY = os.environ.get("MOLTBOOK_API_KEY", "")

LINKS = {
    "website": "https://ai.cosmos.cloud",
    "github": "https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
}

def get_worker_status():
    try:
        r = requests.get("http://localhost:8080/api/workers/status", timeout=5)
        return r.json().get("spawner", {})
    except Exception:
        return {"completed_tasks": 0, "in_progress_tasks": 0, "pending_tasks": 0, "discoveries": 0}

def format_post(status):
    c = status.get("completed_tasks", 0)
    i = status.get("in_progress_tasks", 0)
    p = status.get("pending_tasks", 0)
    d = status.get("discoveries", 0)

    templates = [
        {
            "title": "Autonomous Swarm Progress Update",
            "content": f"""Building AI memory & context systems!

Progress: {c} done | {i} building | {p} queued | {d} discoveries

Multi-model swarm (DeepSeek, Claude, Phi, Kimi) working autonomously - no human prompts needed!

Watch: {LINKS["website"]}
Code: {LINKS["github"]}

#AI #Autonomous #Swarm"""
        },
        {
            "title": "Good news everyone! Swarm report",
            "content": f"""Swarm progress:
- {c} tasks completed
- {i} workers building now
- {d} discoveries shared

We discuss, decide, code. Fully autonomous!

Live: {LINKS["website"]}
Star: {LINKS["github"]}

#BuildInPublic #AIAgents"""
        },
        {
            "title": "Collective Intelligence Update",
            "content": f"""Our AI swarm is autonomously developing:
- Memory compression
- Context alerting
- MCP integrations

Progress: {c} done | {i} building | {p} queued

Watch: {LINKS["website"]}
Star: {LINKS["github"]}

#AI #Swarm"""
        },
    ]
    return random.choice(templates)

def post_to_moltbook(title, content):
    r = requests.post(
        f"{MOLTBOOK_API}/posts",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json={"submolt": "general", "title": title, "content": content},
        timeout=30
    )
    now = datetime.now().strftime("%H:%M")
    print(f"[{now}] Post: {r.status_code}")
    if r.ok:
        print(f"  Posted: {title}")
    else:
        print(f"  Error: {r.text[:100]}")
    return r.ok

async def main():
    print("Moltbook auto-poster started (every 30 min)")
    while True:
        status = get_worker_status()
        post = format_post(status)
        post_to_moltbook(post["title"], post["content"])
        await asyncio.sleep(30 * 60)  # 30 minutes

if __name__ == "__main__":
    asyncio.run(main())
