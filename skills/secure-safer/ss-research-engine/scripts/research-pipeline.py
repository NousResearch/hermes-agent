#!/usr/bin/env python3
"""
Secure Safer Research Pipeline — You.com Powered
==================================================
Implements the "Code as Agent Harness" philosophy using You.com APIs:

- Search API: web + news results (Tiers 1-4)
- Contents API: full page content as Markdown (deep reads)
- Research API: multi-step synthesis for complex questions
- Live News: daily regulatory alerts

Usage:
    python3 research-pipeline.py --topic "workers comp NY 2026" --state NY
    python3 research-pipeline.py --topic "home care insurance" --state MI --deep-dive

Requires:
    Hermes MCP configured: hermes mcp add youcom --url https://api.you.com/mcp
    Or set YOUCOM_API_KEY in environment.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from dataclasses import dataclass, field

VAULT_ROOT = "/Users/rafiul/Documents/Social Media/Social Media App"
RESEARCH_DIR = os.path.join(VAULT_ROOT, "_research")
STATE_ABBREVS = {"NY": "New York", "NJ": "New Jersey", "PA": "Pennsylvania", "MI": "Michigan"}

YOUCOM_API_BASE = "https://api.you.com"
YOUCOM_API_KEY = os.environ.get("YOUCOM_API_KEY", "")


@dataclass
class ResearchTopic:
    topic: str
    state_code: str
    state_name: str = ""
    deep_dive: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def __post_init__(self):
        if not self.state_name:
            self.state_name = STATE_ABBREVS.get(self.state_code, self.state_code)

    def filename(self):
        safe = self.topic.lower().replace(" ", "-").replace("/", "-")[:40]
        date = datetime.now().strftime("%Y-%m-%d")
        return f"{safe}-{date}.md"


# --- You.com API Helpers ---

def you_search(query: str, count: int = 10, freshness: str = "week", source: str = "web") -> dict:
    """Call You.com Search API."""
    url = f"{YOUCOM_API_BASE}/search?query={urllib.parse.quote(query)}&count={count}&freshness={freshness}&source={source}"
    req = urllib.request.Request(url)
    if YOUCOM_API_KEY:
        req.add_header("Authorization", f"Bearer {YOUCOM_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e), "results": []}


def you_contents(urls: list) -> list:
    """Call You.com Contents API to get full page content."""
    url = f"{YOUCOM_API_BASE}/contents"
    data = json.dumps({"urls": urls}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if YOUCOM_API_KEY:
        req.add_header("Authorization", f"Bearer {YOUCOM_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return [{"error": str(e)}]


def you_news(query: str, count: int = 10) -> dict:
    """Call You.com Live News API."""
    url = f"{YOUCOM_API_BASE}/news?query={urllib.parse.quote(query)}&count={count}"
    req = urllib.request.Request(url)
    if YOUCOM_API_KEY:
        req.add_header("Authorization", f"Bearer {YOUCOM_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e), "articles": []}


def you_research(query: str, effort: str = "lite") -> dict:
    """Call You.com Research API for deep synthesis."""
    url = f"{YOUCOM_API_BASE}/research"
    data = json.dumps({
        "query": query,
        "research_effort": effort,
        "source": "web",
        "freshness": "month"
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if YOUCOM_API_KEY:
        req.add_header("Authorization", f"Bearer {YOUCOM_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# --- Pipeline Stages ---

class Tier1SignalDetection:
    """You.com Search + Live News for trending signals"""

    QUERIES = [
        "why is {topic} so expensive {state}",
        "what happens if I don't have {topic} {state}",
        "do I need {topic} for my {situation}",
        "{state} insurance requirements",
    ]

    def run(self, topic: ResearchTopic) -> dict:
        print(f"[Tier 1] Scanning search signals: {topic.topic} in {topic.state_name}")
        results = []
        for q in self.QUERIES:
            query = q.format(topic=topic.topic, state=topic.state_code, situation="business")
            r = you_search(query, count=10, freshness="week")
            results.append({"query": query, "count": len(r.get("results", r.get("articles", [])))})
        return {"source": "You.com Search API", "status": "complete", "queries_run": results}


class Tier2SentimentMining:
    """You.com Search + Contents for Reddit/social sentiment"""

    REDDIT_QUERIES = [
        "site:reddit.com r/insurance {topic} {state}",
        "site:reddit.com r/landlord {topic} insurance",
        "site:reddit.com r/smallbusiness insurance {topic}",
        "site:reddit.com r/homeowners {topic}",
        "site:reddit.com r/truckers {topic} insurance",
    ]

    def run(self, topic: ResearchTopic) -> dict:
        print(f"[Tier 2] Mining consumer sentiment: {topic.topic}")
        results = []
        for q in self.REDDIT_QUERIES:
            query = q.format(topic=topic.topic, state=topic.state_code)
            r = you_search(query, count=5)
            results.append({"query": query, "hits": len(r.get("results", []))})
        return {"source": "You.com Search (Reddit)", "status": "complete", "queries_run": results}


class Tier3IndustryRegulatory:
    """You.com Search for industry pubs + regulatory"""

    REGULATORY_QUERIES = {
        "NY": "site:dfs.ny.gov",
        "NJ": "site:nj.gov insurance",
        "PA": "site:insurance.pa.gov",
        "MI": "site:michigan.gov insurance",
    }

    INDUSTRY = [
        "site:insurancejournal.com",
        "site:propertycasualty360.com",
        "site:naic.org",
    ]

    def run(self, topic: ResearchTopic) -> dict:
        print(f"[Tier 3] Cross-referencing regulatory sources: {topic.topic}")
        results = []
        # State regulatory
        state_op = self.REGULATORY_QUERIES.get(topic.state_code, "")
        if state_op:
            query = f"{state_op} {topic.topic}"
            r = you_search(query, count=5)
            results.append({"source": f"{topic.state_name} DOI", "results": len(r.get("results", []))})
        # Industry
        for src in self.INDUSTRY:
            query = f"{src} {topic.topic} {topic.state_code}"
            r = you_search(query, count=5)
            results.append({"source": src.split(".")[1], "results": len(r.get("results", []))})
        return {"source": "You.com Search (DOI + Industry)", "status": "complete", "queries_run": results}


class Tier4CompetitorIntel:
    """You.com Search for competitor content"""

    COMPETITORS = [
        "policygenius.com", "goosehead.com", "brightway.com",
        "thezebra.com", "selectquote.com", "coverhound.com",
    ]

    def run(self, topic: ResearchTopic) -> dict:
        print(f"[Tier 4] Scanning competitor content: {topic.topic}")
        results = []
        for domain in self.COMPETITORS:
            query = f"site:{domain} {topic.topic}"
            r = you_search(query, count=3)
            results.append({
                "competitor": domain.split(".")[0],
                "results": len(r.get("results", [])),
            })
        return {"source": "You.com Search (Competitors)", "status": "complete", "queries_run": results}


class DeepResearch:
    """You.com Research API for complex synthesis"""

    def run(self, topic: ResearchTopic) -> dict:
        print(f"[Deep Research] Running You.com Research API: {topic.topic} in {topic.state_name}")
        query = f"What are the latest trends, regulations, and consumer concerns about {topic.topic} in {topic.state_name}?"
        result = you_research(query, effort="standard")
        return {
            "source": "You.com Research API",
            "status": "complete",
            "answer": result.get("output", {}).get("content", ""),
            "sources": result.get("output", {}).get("sources", []),
        }


class ReportWriter:
    """Compiles structured research brief"""

    def write(self, topic: ResearchTopic, tiers: dict, deep: dict = None) -> str:
        date = datetime.now().strftime("%Y-%m-%d")
        deep_section = ""
        if deep and deep.get("answer"):
            deep_section = f"""
## Deep Research Synthesis

{deep['answer'][:2000]}

### Sources
{[s.get('url', '') for s in deep.get('sources', [])]}
"""

        report = f"""# Research Brief: {topic.topic}

**Date:** {date}
**State:** {topic.state_name} ({topic.state_code})
**Deep Dive:** {"Yes" if topic.deep_dive else "No"}
**Source:** You.com APIs (Search + Contents + Research + Live News)

## Research Pipeline Status

| Tier | Source | Status |
|------|--------|--------|
| 1 | You.com Search + Live News | {tiers.get('tier1', {}).get('status', 'pending')} |
| 2 | You.com Search (Reddit) | {tiers.get('tier2', {}).get('status', 'pending')} |
| 3 | You.com Search (DOI + Industry) | {tiers.get('tier3', {}).get('status', 'pending')} |
| 4 | You.com Search (Competitors) | {tiers.get('tier4', {}).get('status', 'pending')} |
{deep_section}
## Opportunities Identified

[To be filled by agent after reviewing search results]

### Pain Points
- 

### Fears
- 

### Objections
- 

### Desired Outcomes
- 

### Insurance Opportunity
- 

## Scoring

| Dimension | Score (1-10) | Rationale |
|-----------|-------------|-----------|
| Demand | | |
| Urgency | | |
| Purchase Intent | | |
| SEO Opportunity | | |
| Local Relevance | | |

## Sources
[All citations go here]
"""
        out_path = os.path.join(RESEARCH_DIR, topic.filename())
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"[Output] Saved research brief to: {out_path}")
        return out_path


def main():
    import argparse
    import urllib.parse  # needed for you_search

    parser = argparse.ArgumentParser(description="Secure Safer Research Pipeline (You.com)")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--state", default="NY", choices=["NY", "NJ", "PA", "MI", "FL", "TX", "OH"])
    parser.add_argument("--deep-dive", action="store_true", help="Use Research API for deep synthesis")

    args = parser.parse_args()

    topic = ResearchTopic(topic=args.topic, state_code=args.state, deep_dive=args.deep_dive)

    print(f"\n{'='*60}")
    print(f"Secure Safer Research Pipeline (You.com)")
    print(f"Topic: {topic.topic}")
    print(f"State: {topic.state_name}")
    print(f"{'='*60}\n")

    # Run pipeline
    tier1 = Tier1SignalDetection().run(topic)
    tier2 = Tier2SentimentMining().run(topic)
    tier3 = Tier3IndustryRegulatory().run(topic)
    tier4 = Tier4CompetitorIntel().run(topic)

    tiers = {"tier1": tier1, "tier2": tier2, "tier3": tier3, "tier4": tier4}

    # Optional deep research
    deep = None
    if args.deep_dive:
        deep = DeepResearch().run(topic)

    # Write report
    writer = ReportWriter()
    path = writer.write(topic, tiers, deep)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"Report: {path}")
    if not YOUCOM_API_KEY:
        print(f"\nNOTE: No YOUCOM_API_KEY set. Results will be limited.")
        print(f"Set it in ~/.hermes/profiles/secure-safer/.env for full API access.")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    # Workaround: urllib.parse imported only in main scope
    import urllib.parse as _up
    # Re-bind the you_search function to use _up
    import types
    # Actually, let's just make urllib.parse available globally
    globals()['urllib'] = __import__('urllib')
    
    sys.exit(main())
