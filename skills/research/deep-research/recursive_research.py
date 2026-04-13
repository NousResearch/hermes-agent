#!/usr/bin/env python3
"""
Deep Iterative Research Engine
Performs recursive citation crawling via Semantic Scholar, synthesizes findings,
and performs a Latticework "Red-Team" validation to check for hallucinations/logic flaws.
"""

import argparse
import json
import logging
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"

def _load_env():
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
    except ImportError:
        pass

def _get_api_key():
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

def llm_completion(system_prompt: str, user_prompt: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("No OPENROUTER_API_KEY or OPENAI_API_KEY found.")

    is_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    url = "https://openrouter.ai/api/v1/chat/completions" if is_openrouter else "https://api.openai.com/v1/chat/completions"
    model = "anthropic/claude-3-5-sonnet" if is_openrouter else "gpt-4o"

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return f"Error during synthesis: {e}"

def s2_search(query: str, limit: int = 5) -> List[Dict]:
    """Search for foundational papers."""
    url = f"{SEMANTIC_SCHOLAR_URL}/paper/search?query={urllib.parse.quote(query)}&limit={limit}&fields=title,authors,year,abstract,citationCount,referenceCount"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("data", [])
    except Exception as e:
        logger.error(f"Semantic Scholar Search Error: {e}")
        return []

def s2_citations(paper_id: str, limit: int = 5) -> List[Dict]:
    """Find top citing papers."""
    url = f"{SEMANTIC_SCHOLAR_URL}/paper/{paper_id}/citations?fields=title,abstract,year,citationCount&limit={limit}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            # S2 returns {"citingPaper": {...}} array
            return [x["citingPaper"] for x in data.get("data", []) if x.get("citingPaper")]
    except Exception as e:
        # Expected if rate limits are hit or paper not found
        time.sleep(1)
        return []

def synthesize_papers(query: str, papers: List[Dict]) -> str:
    """Uses LLM to synthesize research findings."""
    logger.info(f"Synthesizing {len(papers)} papers...")
    
    docs = []
    for i, p in enumerate(papers):
        title = p.get('title', 'Unknown')
        year = p.get('year', 'N/A')
        abstract = p.get('abstract', '')
        if not abstract: continue
        docs.append(f"[{i+1}] Title: {title} ({year})\nAbstract: {abstract}\n")

    combined_text = "\n".join(docs)
    
    sys_prompt = (
        "You are an Elite Academic Synthesizer and Epistemic Risk Auditor.\n"
        "Your task is to synthesize the provided research papers into a deeply structured, rigorous literature review.\n\n"
        "REQUIRED STRUCTURE:\n"
        "1. **Core Epistemic Consensus (The Signal):** What is the verified bedrock truth across these papers?\n"
        "2. **Methodological Evolution (The Engine):** How are research methods changing (e.g., data shifts, architectural changes)?\n"
        "3. **Contested Frontiers (The Noise):** Where do these papers disagree? Highlight explicitly varying results or conflicting theses.\n"
        "4. **Adjacent Possibilities:** What novel cross-disciplinary ideas are implied but not explicitly stated?\n"
        "5. **High-Value Open Gaps:** Where is the current frontier blocked?\n\n"
        "Maintain severe intellectual rigor. Do not hallucinate consensus where there is tension. Use precise, high-density language."
    )
    user_prompt = f"TARGET TOPIC: {query}\n\nRAW RESEARCH CORPUS:\n{combined_text}"
    
    return llm_completion(sys_prompt, user_prompt)

def red_team_validation(synthesis: str) -> str:
    """Uses LLM Latticework to red-team and falsify the synthesis."""
    logger.info("Executing Red-Team validation (Hallucination Check)...")
    sys_prompt = (
        "You are a Hostile 'Red-Team' Epistemic Auditor deployed to falsify academic synthesis.\n"
        "You operate exclusively using 'The Latticework' Cognitive Framework. Specifically, apply:\n"
        "A) The Hallucination Check: Are there sweeping abstractions not grounded in the specific cited data?\n"
        "B) The Context Window: Is the synthesis ignoring historical base rates or out-of-domain edge cases?\n"
        "C) First Principles: Does the synthesis rely on fragile proxy metrics or fundamental physics/math?\n\n"
        "YOUR MISSION: Break the synthesis. Look for fragile logic, omitted variables, survivorship bias, and unwarranted confidence.\n\n"
        "REQUIRED OUTPUT FORMAT:\n"
        "### 🚨 VULNERABILITY MATRIX\n"
        "- [Claim]: [Why it is epistemically fragile]\n\n"
        "### ⚔️ CONTRADICTORY LITERATURE REQUIREMENTS\n"
        "What specific papers or evidence would immediately falsify the core thesis?\n\n"
        "### ⚖️ DIALECTICAL VERDICT\n"
        "A brutal, honest assessment of the synthesis's reliability."
    )
    user_prompt = f"SYNTHESIS DOCUMENT TO AUDIT:\n\n{synthesis}"
    return llm_completion(sys_prompt, user_prompt)

def main():
    parser = argparse.ArgumentParser(description="Deep Iterative Research Engine")
    parser.add_argument("query", help="The research topic to investigate.")
    parser.add_argument("--max-depth", type=int, default=1, help="Depth of citation crawling")
    parser.add_argument("--output-dir", type=str, default="~/Obsidian/Hermes/Research", help="Output directory")
    args = parser.parse_args()

    _load_env()
    
    logger.info(f"Starting Deep Research for: '{args.query}'")
    
    all_papers = []
    visited: Set[str] = set()
    
    # 1. Search seed papers
    logger.info("Discovering seed papers...")
    seed_papers = s2_search(args.query, limit=5)
    
    for p in seed_papers:
        if p.get('paperId') and p['paperId'] not in visited:
            visited.add(p['paperId'])
            all_papers.append(p)
            
    # 2. Iterate citations
    if args.max_depth > 0:
        logger.info("Crawling citations for depth...")
        new_papers = []
        for p in list(all_papers):
            time.sleep(1.1)  # Semantic Scholar rate limit buffer
            cites = s2_citations(p['paperId'], limit=3)
            for c in cites:
                if c.get('paperId') and c['paperId'] not in visited:
                    visited.add(c['paperId'])
                    new_papers.append(c)
        all_papers.extend(new_papers)
        
    logger.info(f"Total unique papers gathered: {len(all_papers)}")

    # 3. Synthesize
    synthesis = synthesize_papers(args.query, all_papers)
    
    # 4. Red-Team
    red_team_report = red_team_validation(synthesis)

    # 5. Format Output
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in args.query)[:30]
    filename = f"Deep_Research_{safe_query}_{timestamp}.md"
    file_path = output_dir / filename
    
    content = f"""---
tags: [research, deep-dive, auto-generated]
topic: "{args.query}"
date: {datetime.now().isoformat()}
---

# Deep Research: {args.query}

## Part 1: Synthesis
{synthesis}

---
## Part 2: Disputation & Falsification (Red-Team)
{red_team_report}

---
## Part 3: Source Graph Data
Found {len(all_papers)} papers.

"""
    for i, p in enumerate(all_papers):
        content += f"{i+1}. **{p.get('title', 'Unknown')}** ({p.get('year', 'N/A')})\n"
        
    file_path.write_text(content, encoding="utf-8")
    logger.info(f"Deep Research complete. Saved to: {file_path}")

if __name__ == "__main__":
    main()
