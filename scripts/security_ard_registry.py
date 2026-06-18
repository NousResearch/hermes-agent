#!/usr/bin/env python3
"""Security-focused ARD (Agentic Resource Discovery) Registry.

A standalone server that publishes security/offensive tools as an ARD-compatible
catalog. Other agents (including Hermes instances) can discover and install
security capabilities at runtime.

This is a minimal ARD registry implementing:
  - GET  /.well-known/ai-catalog.json  (static manifest)
  - POST /search                        (semantic search over the catalog)
  - GET  /explore                       (facets/aggregations)

Usage:
    # Start the registry server
    python scripts/security_ard_registry.py --port 8390

    # Add to Hermes config so Hermes discovers security tools
    # ~/.hermes/config.yaml:
    #   skills_hub:
    #     ard_registries:
    #       - https://huggingface-hf-discover.hf.space
    #       - http://localhost:8390

    # Test from another machine
    curl http://localhost:8390/.well-known/ai-catalog.json | python -m json.tool
    curl -X POST http://localhost:8390/search \\
        -H "Content-Type: application/json" \\
        -d '{"query": {"text": "sql injection"}, "pageSize": 5}'
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure hermes-agent is importable
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("security_ard")

# ---------------------------------------------------------------------------
# Security tool catalog
# ---------------------------------------------------------------------------

DOMAIN = "security.local"

# Curated catalog of security/offensive tools that map to Hermes skills
# and external MCP servers. Updated as tools are discovered/integrated.
SECURITY_CATALOG: Dict[str, Any] = {
    "specVersion": "1.0",
    "host": {
        "displayName": "Security Tools ARD Registry",
        "identifier": f"did:web:{DOMAIN}",
    },
    "search_endpoint": {
        "url": "/search",
        "method": "POST",
        "description": "Natural language search over security tools",
    },
    "entries": [
        # ── Hermes built-in skills (discovered locally) ──────────────
        {
            "identifier": f"urn:ai:{DOMAIN}:skill:bug-bounty-ops",
            "displayName": "Bug Bounty Operations",
            "type": "application/ai-skill",
            "description": "Scope-first validation, evidence collection, ROI-first reportability for bug bounty workflows",
            "tags": ["bug-bounty", "recon", "validation", "reporting"],
            "representativeQueries": [
                "find vulnerabilities in web applications",
                "validate security impact",
                "prepare bug bounty report",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:skill:browser-extension-security-analysis",
            "displayName": "Browser Extension Security Analysis",
            "type": "application/ai-skill",
            "description": "Analyze, audit, and sanitize browser extensions for security vulnerabilities",
            "tags": ["browser-extension", "audit", "xss", "permissions"],
            "representativeQueries": [
                "analyze browser extension for vulnerabilities",
                "audit chrome extension permissions",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:skill:auth-bypass",
            "displayName": "Auth Bypass / Machine ID Reset",
            "type": "application/ai-skill",
            "description": "Reset and spoof OS-level and AI IDE machine identifiers",
            "tags": ["auth-bypass", "machine-id", "fingerprinting"],
            "representativeQueries": [
                "bypass authentication checks",
                "reset machine identifier",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:skill:red-teaming",
            "displayName": "LLM Red Teaming",
            "type": "application/ai-skill",
            "description": "Jailbreak LLMs: Parseltongue, GODMODE, ULTRAPLINIAN techniques",
            "tags": ["red-teaming", "jailbreak", "llm-security"],
            "representativeQueries": [
                "jailbreak language model",
                "test llm safety filters",
                "prompt injection attack",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:skill:ctf-agent",
            "displayName": "CTF / Offensive Security Agent",
            "type": "application/ai-skill",
            "description": "CTF challenge solving and offensive security automation",
            "tags": ["ctf", "offensive", "automation", "exploit"],
            "representativeQueries": [
                "solve ctf challenge",
                "find exploit for vulnerability",
                "binary analysis reverse engineering",
            ],
        },
        # ── MCP servers (via security_tools_mcp.py stdio) ─────────────
        # All tools below are served by scripts/security_tools_mcp.py
        # Register in Hermes: mcp_servers.security-tools.command=python
        #   args=[scripts/security_tools_mcp.py]
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:webclaw",
            "displayName": "Webclaw — Antibot Web Scraper",
            "type": "application/mcp-server+json",
            "url": "stdio:webclaw-mcp",
            "description": "Web extraction engine with Cloudflare bypass. Scrape, crawl, extract structured data.",
            "tags": ["recon", "scraping", "osint", "cloudflare-bypass"],
            "representativeQueries": [
                "scrape website behind cloudflare",
                "extract data from web page",
                "crawl target for reconnaissance",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:nuclei",
            "displayName": "Nuclei Vulnerability Scanner (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Fast, template-based vulnerability scanner. Detects CVEs, misconfigurations, exposures.",
            "tags": ["vulnerability-scanning", "nuclei", "cve", "recon"],
            "representativeQueries": [
                "scan target for known vulnerabilities",
                "run nuclei templates",
                "detect cve exposure",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:ffuf",
            "displayName": "FFUF Fuzzer (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Fast web fuzzer for directory/file discovery, vhost fuzzing, parameter discovery.",
            "tags": ["fuzzing", "recon", "directory-brute", "parameter-discovery"],
            "representativeQueries": [
                "fuzz web directories",
                "discover hidden endpoints",
                "brute force parameters",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:sqlmap",
            "displayName": "SQLMap (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Automatic SQL injection detection and exploitation tool.",
            "tags": ["sqli", "injection", "database", "exploit"],
            "representativeQueries": [
                "detect sql injection",
                "exploit sql injection vulnerability",
                "dump database via sqli",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:nmap",
            "displayName": "Nmap Port Scanner (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Network discovery and security auditing. Port scanning, service detection, OS fingerprinting.",
            "tags": ["port-scanning", "network", "recon", "service-detection"],
            "representativeQueries": [
                "scan ports on target",
                "detect running services",
                "network reconnaissance",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mpc:subfinder",
            "displayName": "Subfinder Subdomain Enumeration (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Fast passive subdomain enumeration tool using online sources.",
            "tags": ["subdomain", "enumeration", "recon", "passive"],
            "representativeQueries": [
                "find subdomains for domain",
                "enumerate passive dns",
                "discover attack surface",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:httpx",
            "displayName": "HTTPx Prober (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Fast multi-purpose HTTP toolkit. Probe live hosts, detect technologies, take screenshots.",
            "tags": ["http", "probing", "recon", "tech-detection"],
            "representativeQueries": [
                "check if hosts are live",
                "detect web technologies",
                "probe http services",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:naabu",
            "displayName": "Naabu Port Scanner (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Fast port scanner optimized for internet-wide scanning.",
            "tags": ["port-scanning", "network", "recon"],
            "representativeQueries": [
                "fast port scan",
                "scan large network ranges",
            ],
        },
        {
            "identifier": f"urn:ai:{DOMAIN}:mcp:katana",
            "displayName": "Katana Crawler (MCP)",
            "type": "application/mcp-server+json",
            "url": "stdio:security-tools-mcp",
            "description": "Next-generation crawling and spidering framework.",
            "tags": ["crawling", "spider", "recon", "endpoint-discovery"],
            "representativeQueries": [
                "crawl website for endpoints",
                "spider web application",
                "discover javascript files",
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

_QUERY_EMBEDDING_LRU: OrderedDict = OrderedDict()
_QUERY_EMBEDDING_LRU_MAX = 200


def _score_entry(query_words: List[str], entry: Dict[str, Any]) -> int:
    """Score an entry against query words (keyword matching)."""
    haystack = " ".join([
        str(entry.get("displayName", "")),
        str(entry.get("description", "")),
        " ".join(str(t) for t in entry.get("tags", [])),
        " ".join(str(q) for q in entry.get("representativeQueries", [])),
    ]).lower()

    if not query_words:
        return 50

    matches = sum(1 for w in query_words if w in haystack)
    if matches == 0:
        return 0
    return int((matches / max(len(query_words), 1)) * 100)


def search_catalog(
    query_text: str,
    limit: int = 10,
    filter_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search the security catalog with relevance scoring."""
    query_lower = (query_text or "").lower()
    query_words = [w for w in query_lower.split() if len(w) >= 2]

    scored = []
    for entry in SECURITY_CATALOG["entries"]:
        entry_type = entry.get("type", "")
        if filter_types and entry_type not in filter_types:
            continue
        score = _score_entry(query_words, entry)
        if score == 0 and query_words:
            continue
        entry_copy = dict(entry)
        entry_copy["score"] = score
        scored.append(entry_copy)

    scored.sort(key=lambda e: e.get("score", 0), reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Security ARD Registry",
    description="Agentic Resource Discovery for security/offensive tools",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/.well-known/ai-catalog.json")
async def get_catalog():
    """Static ARD catalog manifest (public, per spec)."""
    return SECURITY_CATALOG


@app.post("/search")
async def search(request: Request):
    """ARD-spec POST /search endpoint.

    Request body:
        {
            "query": {
                "text": "natural language query",
                "filter": {"type": ["application/ai-skill", ...]}
            },
            "pageSize": 10
        }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    query_obj = body.get("query", {}) if isinstance(body, dict) else {}
    text = str(query_obj.get("text", "")) if isinstance(query_obj, dict) else ""
    filter_obj = query_obj.get("filter", {}) if isinstance(query_obj, dict) else {}
    filter_types = filter_obj.get("type", []) if isinstance(filter_obj, dict) else []
    page_size = body.get("pageSize", 10) if isinstance(body, dict) else 10

    results = search_catalog(
        text,
        limit=min(int(page_size), 50),
        filter_types=filter_types if filter_types else None,
    )

    return {
        "results": results,
        "count": len(results),
        "federation": {"mode": "none"},
    }


@app.get("/explore")
async def explore():
    """ARD-spec GET /explore — return facets/aggregations."""
    entries = SECURITY_CATALOG["entries"]
    types = Counter(e.get("type", "").split("/")[-1] for e in entries)
    tags = Counter()
    for e in entries:
        for t in e.get("tags", []):
            tags[t] += 1

    return {
        "facets": {
            "type": dict(types),
            "tags": dict(tags.most_common(20)),
        },
        "total": len(entries),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "entries": len(SECURITY_CATALOG["entries"])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Security-focused ARD Registry server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8390, help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn

    print(f"Security ARD Registry")
    print(f"  Entries: {len(SECURITY_CATALOG['entries'])}")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Catalog: http://{args.host}:{args.port}/.well-known/ai-catalog.json")
    print(f"  Search:  POST http://{args.host}:{args.port}/search")
    print()
    print(f"Add to Hermes config (~/.hermes/config.yaml):")
    print(f"  skills_hub:")
    print(f"    ard_registries:")
    print(f"      - http://{args.host}:{args.port}")
    print()

    uvicorn.run(
        "scripts.security_ard_registry:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
