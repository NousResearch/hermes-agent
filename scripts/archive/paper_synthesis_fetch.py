#!/usr/bin/env python3
"""Research Paper Synthesis v2.1.0 — Multi-source fetch + score pipeline.
Fetches arXiv, Semantic Scholar, HuggingFace Daily Papers, Papers With Code.
Scores and deduplicates. Outputs structured JSON for downstream processing.
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import quote

# ── Config ──────────────────────────────────────────────────────────
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper"
HF_DAILY = "https://huggingface.co/api/daily_papers"
PWC_API = "https://paperswithcode.com/api/v1/papers/"
MAX_PER_QUERY = 30
D14_CUTOFF = datetime.now(timezone.utc) - timedelta(days=14)
ARXIV_NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
TOP_VENUES = {'NeurIPS', 'ICLR', 'ICML', 'ACL', 'EMNLP', 'NAACL', 'AAAI', 'IJCAI', 'CVPR', 'ECCV', 'ICCV'}

# Sahil's domain keywords for relevance scoring
HIGH_RELEVANCE_PATTERNS = [
    'hermes agent', 'mcp server', 'tool calling', 'context compression',
    'coding agent', 'llm agent', 'agent memory', 'prompt optimization',
    'context engineering', 'agent orchestration', 'multi-agent coordination',
    'agent workflow', 'tool output', 'blob storage', 'context rescue',
    'instruction drift', 'prompt re-assertion', 'code review agent',
    'adversarial testing agent', 'subagent spawning', 'parallel execution',
    'agent governance', 'lead worker pattern', 'ai tutoring',
]
MED_RELEVANCE_PATTERNS = [
    'fine-tuning', 'model serving', 'local llm', 'knowledge graph',
    'vector search', 'memory system', 'benchmark', 'evaluation',
    'react native ai', 'voice ai', 'kitchen ai', 'football ai',
    'coaching ai', 'sports analytics', 'ai education',
    'saas ai', 'proptech ai',
]
LOW_RELEVANCE_PATTERNS = [
    'transformer', 'large language model', 'reinforcement learning',
    'deep learning', 'neural network', 'generative ai',
]

# ── Helpers ─────────────────────────────────────────────────────────
def fetch_url(url, headers=None, method='GET', data=None, retries=2):
    """Fetch URL with retries."""
    if headers is None:
        headers = {'User-Agent': 'KenseiAgent-ResearchPipeline/2.1'}
    for attempt in range(retries + 1):
        try:
            req = Request(url, data=data, headers=headers, method=method)
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except (URLError, HTTPError) as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

def parse_arxiv_xml(xml_text):
    """Parse arXiv Atom XML into paper dicts."""
    papers = []
    if not xml_text:
        return papers
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return papers
    
    for entry in root.findall('a:entry', ARXIV_NS):
        try:
            title_el = entry.find('a:title', ARXIV_NS)
            title = title_el.text.strip().replace('\n', ' ') if title_el is not None else ''
            
            id_el = entry.find('a:id', ARXIV_NS)
            raw_id = id_el.text.strip().split('/abs/')[-1] if id_el is not None else ''
            # Strip version suffix for canonical ID
            arxiv_id = raw_id.split('v')[0] if 'v' in raw_id else raw_id
            
            published_el = entry.find('a:published', ARXIV_NS)
            published = published_el.text[:10] if published_el is not None else ''
            
            summary_el = entry.find('a:summary', ARXIV_NS)
            summary = summary_el.text.strip().replace('\n', ' ') if summary_el is not None else ''
            
            # Check for withdrawn
            if 'withdrawn' in summary.lower() or 'retracted' in summary.lower():
                continue
            if not summary or not title:
                continue
            
            authors = [a.find('a:name', ARXIV_NS).text for a in entry.findall('a:author', ARXIV_NS)]
            cats = [c.get('term') for c in entry.findall('a:category', ARXIV_NS)]
            
            # Get primary category
            primary_el = entry.find('arxiv:primary_category', ARXIV_NS)
            primary_cat = primary_el.get('term') if primary_el is not None else (cats[0] if cats else '')
            
            papers.append({
                'arxiv_id': arxiv_id,
                'versioned_id': raw_id,
                'title': title,
                'authors': authors,
                'published': published,
                'categories': cats,
                'primary_category': primary_cat,
                'summary': summary,
                'source': 'arxiv',
            })
        except Exception:
            continue
    return papers

def get_semantic_batch(arxiv_ids):
    """Batch fetch Semantic Scholar metadata."""
    results = {}
    if not arxiv_ids:
        return results
    # Process in batches of 10 via the batch endpoint
    for i in range(0, len(arxiv_ids), 10):
        batch = arxiv_ids[i:i+10]
        ids_param = ','.join(f'arXiv:{aid}' for aid in batch)
        url = f"{SEMANTIC_API}/batch?fields=title,citationCount,influentialCitationCount,publicationVenue,year,externalIds,abstract&ids={ids_param}"
        data = fetch_url(url)
        if data:
            try:
                parsed = json.loads(data)
                for paper in parsed if isinstance(parsed, list) else []:
                    if paper and 'externalIds' in paper and 'ArXiv' in paper['externalIds']:
                        aid = paper['externalIds']['ArXiv']
                        results[aid] = {
                            'citationCount': paper.get('citationCount', 0) or 0,
                            'influentialCitationCount': paper.get('influentialCitationCount', 0) or 0,
                            'venue': paper.get('publicationVenue', '') or '',
                            'year': paper.get('year'),
                            's2_title': paper.get('title', ''),
                            'externalIds': paper.get('externalIds', {}),
                        }
            except json.JSONDecodeError:
                pass
        time.sleep(1.1)  # Rate limit: 1 req/s
    return results

def get_hf_daily():
    """Fetch HuggingFace Daily Papers."""
    data = fetch_url(f"{HF_DAILY}?limit=30")
    if not data:
        return []
    try:
        papers = json.loads(data)
    except json.JSONDecodeError:
        return []
    
    results = []
    for p in papers:
        paper_link = p.get('paper', {}).get('id', '')
        arxiv_id = ''
        if 'arxiv.org/abs/' in paper_link:
            arxiv_id = paper_link.split('/abs/')[-1].split('v')[0]
        elif 'arxiv.org/pdf/' in paper_link:
            arxiv_id = paper_link.split('/pdf/')[-1].split('.pdf')[0].split('v')[0]
        
        if not arxiv_id:
            continue
        
        results.append({
            'arxiv_id': arxiv_id,
            'title': p.get('paper', {}).get('title', ''),
            'upvotes': p.get('paper', {}).get('upvotes', 0),
            'source': 'hf-daily',
            'discussion_url': p.get('discussionUrl', ''),
        })
    return results

def get_paperswithcode(arxiv_id):
    """Check Papers With Code for implementation."""
    url = f"{PWC_API}?arxiv_id={arxiv_id}"
    data = fetch_url(url)
    if not data:
        return None
    try:
        parsed = json.loads(data)
        results = parsed.get('results', [])
        if not results:
            return None
        paper = results[0]
        return {
            'repository_url': paper.get('repository_url', ''),
            'stars': paper.get('stars', 0),
            'framework': paper.get('framework', ''),
        }
    except json.JSONDecodeError:
        return None

# ── Phase 1A: arXiv Discovery ───────────────────────────────────────
def fetch_arxiv_papers():
    """Fetch papers from arXiv across multiple queries."""
    all_papers = []
    seen_ids = set()
    
    queries = [
        # Category searches
        ('cat:cs.AI', 'category'),
        ('cat:cs.CL', 'category'),
        ('cat:cs.LG', 'category'),
        ('cat:cs.SE', 'category'),
        ('cat:cs.HC', 'category'),
        # Keyword searches
        ('all:"coding+agent"+OR+all:"LLM+agent"+OR+all:"MCP+server"+OR+all:"tool+calling"', 'keyword'),
        ('all:"context+window"+OR+all:"prompt+engineering"+OR+all:"agent+memory"+OR+all:"agent+orchestration"', 'keyword'),
        ('all:"multi-agent"+OR+all:"AI+workflow"+OR+all:"local+LLM"+OR+all:"fine-tuning"', 'keyword'),
        ('all:"AI+product"+OR+all:"AI+SaaS"+OR+all:"voice+AI"+OR+all:"coaching+AI"', 'keyword'),
    ]
    
    for query, qtype in queries:
        url = f"{ARXIV_API}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_PER_QUERY}"
        print(f"  Fetching: {qtype} — {query[:60]}...", file=sys.stderr)
        xml_data = fetch_url(url)
        papers = parse_arxiv_xml(xml_data)
        
        # Filter by date
        for p in papers:
            try:
                pub_date = datetime.strptime(p['published'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if pub_date < D14_CUTOFF:
                    continue
            except ValueError:
                continue
            
            if p['arxiv_id'] not in seen_ids:
                seen_ids.add(p['arxiv_id'])
                all_papers.append(p)
        
        time.sleep(3)  # arXiv rate limit: ~1 req/3s
    
    print(f"  arXiv total unique papers in D-14 window: {len(all_papers)}", file=sys.stderr)
    return all_papers

# ── Phase 1C: HuggingFace Daily Papers ──────────────────────────────
def fetch_hf_papers():
    """Fetch HuggingFace Daily Papers."""
    print("  Fetching: HuggingFace Daily Papers...", file=sys.stderr)
    return get_hf_daily()

# ── Phase 2: Scoring ────────────────────────────────────────────────
def compute_base_relevance(paper):
    """Compute base relevance score (1-5)."""
    text = f"{paper['title'].lower()} {paper['summary'].lower()}"
    
    score = 1  # Default, will be dropped
    
    high_hits = sum(1 for pat in HIGH_RELEVANCE_PATTERNS if pat.lower() in text)
    med_hits = sum(1 for pat in MED_RELEVANCE_PATTERNS if pat.lower() in text)
    low_hits = sum(1 for pat in LOW_RELEVANCE_PATTERNS if pat.lower() in text)
    
    # Exact phrase matching for high-relevance terms
    direct_terms = ['hermes', 'toolaria', 'mcp server', 'context compression', 
                    'coding agent', 'tool calling', 'agent memory', 'prompt optimization']
    direct_hits = sum(1 for t in direct_terms if t in text)
    
    if direct_hits >= 2:
        score = 5
    elif direct_hits == 1 and high_hits >= 3:
        score = 5
    elif high_hits >= 4:
        score = 4
    elif high_hits >= 2 or (high_hits >= 1 and med_hits >= 2):
        score = 4
    elif med_hits >= 3 or high_hits >= 1:
        score = 3
    elif med_hits >= 1 or low_hits >= 3:
        score = 2
    else:
        score = 1
    
    # Downgrade football/coaching-only papers
    coaching_terms = ['football', 'coaching', 'soccer']
    has_coaching = any(t in text for t in coaching_terms)
    if has_coaching and high_hits == 0 and med_hits <= 1:
        score = max(1, score - 1)
    
    return score

def compute_quality_weight(citation_count, hf_featured, venue):
    """Compute quality weight (0.3-1.2)."""
    is_top_venue = any(v.lower() in venue.lower() for v in TOP_VENUES) if venue else False
    
    if hf_featured and is_top_venue:
        return 1.2
    elif citation_count >= 100 and hf_featured:
        return 1.1
    elif citation_count >= 50 and is_top_venue:
        return 1.1
    elif citation_count >= 50 and hf_featured:
        return 1.0
    elif citation_count >= 50:
        return 0.9
    elif 11 <= citation_count < 50 and hf_featured:
        return 0.9
    elif 0 <= citation_count <= 10 and hf_featured:
        return 0.8
    elif 11 <= citation_count < 50:
        return 0.7
    elif 3 <= citation_count <= 10:
        return 0.5
    else:  # 0-2
        return 0.3

def compute_implementation_multiplier(pwc_data):
    """Compute implementation multiplier (1.0-1.3)."""
    if not pwc_data:
        return 1.0
    
    stars = pwc_data.get('stars', 0)
    repo_url = pwc_data.get('repository_url', '')
    
    if not repo_url:
        return 1.0
    
    if stars >= 500:
        return 1.25
    elif 50 <= stars < 500:
        return 1.2
    else:
        return 1.1

def score_papers(arxiv_papers, sem_data, hf_papers, pwc_data_map):
    """Score all papers."""
    hf_ids = {p['arxiv_id']: p for p in hf_papers}
    
    scored = []
    for paper in arxiv_papers:
        aid = paper['arxiv_id']
        s2 = sem_data.get(aid, {})
        hf = hf_ids.get(aid)
        pwc = pwc_data_map.get(aid)
        
        base_rel = compute_base_relevance(paper)
        if base_rel < 2:
            continue
        
        citations = s2.get('citationCount', 0)
        venue = s2.get('venue', '')
        hf_featured = hf is not None
        
        qw = compute_quality_weight(citations, hf_featured, venue)
        imp = compute_implementation_multiplier(pwc)
        final_score = round(base_rel * qw * imp, 2)
        
        # Determine action
        if final_score >= 5.5:
            action = 'write_now'
        elif final_score >= 3.0:
            action = 'ask_first'
        elif final_score >= 1.5:
            action = 'file'
        else:
            action = 'skip'
        
        scored.append({
            **paper,
            'base_relevance': base_rel,
            'quality_weight': qw,
            'implementation_multiplier': imp,
            'final_score': final_score,
            'action': action,
            'citations': citations,
            'influential_citations': s2.get('influentialCitationCount', 0),
            'venue': venue,
            'year': s2.get('year'),
            'hf_featured': hf_featured,
            'hf_upvotes': hf.get('upvotes', 0) if hf else 0,
            'hf_discussion_url': hf.get('discussion_url', '') if hf else '',
            'pwc_data': pwc,
        })
    
    # Sort by final score descending
    scored.sort(key=lambda x: x['final_score'], reverse=True)
    return scored

# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=== Research Paper Synthesis v2.1.0 ===", file=sys.stderr)
    print(f"Date window: {D14_CUTOFF.strftime('%Y-%m-%d')} to {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", file=sys.stderr)
    
    # Phase 1A: arXiv
    print("\n── Phase 1A: arXiv Discovery ──", file=sys.stderr)
    arxiv_papers = fetch_arxiv_papers()
    
    if not arxiv_papers:
        print("ERROR: No arXiv papers found", file=sys.stderr)
        print(json.dumps({'error': 'no_papers', 'papers': []}))
        return
    
    # Phase 1B: Semantic Scholar (top 30 by recency)
    print("\n── Phase 1B: Semantic Scholar ──", file=sys.stderr)
    # Sort by published date descending, take top 30
    arxiv_papers.sort(key=lambda x: x['published'], reverse=True)
    top_30_ids = [p['arxiv_id'] for p in arxiv_papers[:30]]
    print(f"  Looking up {len(top_30_ids)} papers...", file=sys.stderr)
    sem_data = get_semantic_batch(top_30_ids)
    print(f"  Got {len(sem_data)} Semantic Scholar records", file=sys.stderr)
    
    # Phase 1C: HuggingFace
    print("\n── Phase 1C: HuggingFace Daily Papers ──", file=sys.stderr)
    hf_papers = fetch_hf_papers()
    print(f"  Got {len(hf_papers)} HF Daily papers", file=sys.stderr)
    
    # Phase 2: Score all papers
    print("\n── Phase 2: Scoring ──", file=sys.stderr)
    # Quick Papers With Code check for top relevance papers
    pwc_data_map = {}
    high_rel = [p for p in arxiv_papers if compute_base_relevance(p) >= 4]
    print(f"  Checking Papers With Code for {len(high_rel)} high-relevance papers...", file=sys.stderr)
    for i, p in enumerate(high_rel[:15]):  # Cap at 15 lookups
        time.sleep(0.5)
        pwc = get_paperswithcode(p['arxiv_id'])
        if pwc:
            pwc_data_map[p['arxiv_id']] = pwc
    
    scored = score_papers(arxiv_papers, sem_data, hf_papers, pwc_data_map)
    
    # Summary stats
    write_now = [p for p in scored if p['action'] == 'write_now']
    ask_first = [p for p in scored if p['action'] == 'ask_first']
    file_only = [p for p in scored if p['action'] == 'file']
    skipped = [p for p in scored if p['action'] == 'skip']
    
    print(f"\n── Results ──", file=sys.stderr)
    print(f"  Write Now (≥5.5): {len(write_now)}", file=sys.stderr)
    print(f"  Ask First (3.0-5.4): {len(ask_first)}", file=sys.stderr)
    print(f"  File (1.5-2.9): {len(file_only)}", file=sys.stderr)
    print(f"  Skip (<1.5): {len(skipped)}", file=sys.stderr)
    
    for p in write_now[:5]:
        print(f"  ★ {p['final_score']:.1f} | {p['citations']} cit | {p['title'][:80]}", file=sys.stderr)
    
    # Add HF papers not in arXiv set
    arxiv_ids = {p['arxiv_id'] for p in arxiv_papers}
    extra_hf = [p for p in hf_papers if p['arxiv_id'] not in arxiv_ids]
    if extra_hf:
        print(f"  + {len(extra_hf)} HF Daily papers not in arXiv results", file=sys.stderr)
    
    # Output JSON
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'd14_cutoff': D14_CUTOFF.strftime('%Y-%m-%d'),
        'total_fetched': len(arxiv_papers),
        'total_scored': len(scored),
        'write_now_count': len(write_now),
        'ask_first_count': len(ask_first),
        'file_count': len(file_only),
        'skip_count': len(skipped),
        'sem_scholar_records': len(sem_data),
        'hf_daily_papers': len(hf_papers),
        'extra_hf_papers': len(extra_hf),
        'pwc_checks': len(pwc_data_map),
        'papers': scored,
        'extra_hf': extra_hf,
    }
    
    print(json.dumps(output, ensure_ascii=False))
    
    # Also save to file
    out_path = f"/home/kensei/.hermes/runbooks/paper-synthesis/papers-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
