#!/usr/bin/env python3
"""
SearXNG Search Script
Makes HTTP requests to SearXNG instances and returns structured results.
"""

import urllib.request
import urllib.parse
import json
import sys
import random
from typing import Optional

# Default public instances - curated from https://searx.space/
# Prioritized by: uptime, TLS grade A+, Vanilla HTML grade, performance
DEFAULT_INSTANCES = [
    # Tier 1: Best overall (fast, reliable, A+ TLS, Vanilla grade)
    "https://priv.au",                    # Fastest (0.199s), 99% uptime, A+ TLS, AU
    "https://search.rhscz.eu",            # 100% uptime, Onion service, A+ TLS, CZ
    "https://search.hbubli.cc",           # 100% uptime, hosted in DE
    "https://paulgo.io",                  # High perf (0.445s), 98% uptime, Hetzner DE

    # Tier 2: Reliable alternatives
    "https://search.serpensin.com",       # Onion service, good uptime
    "https://search.indst.eu",            # Onion service, IPv6
    "https://searx.namejeff.xyz",         # IPv6 support
    "https://search.2b9t.xyz",            # IPv6 support
    "https://searx.tiekoetter.com",       # Onion service, long-running
    "https://kantan.cat",                 # Onion service
    "https://search.anoni.net",           # Onion service

    # Tier 3: Regional options
    "https://searxng.canine.tools",       # Customized UI
    "https://search.freestater.org",      # Customized UI
    "https://ooglester.com",              # Alternative option
    "https://search.pi.vps.pw",           # Customized UI
    "https://searx.be",                   # Popular BE instance
    "https://search.demoniak.ch",         # CH option
    "https://freesearch.club",            # US-based
    "https://search.fr3ak.net",           # DE option
]


def searxng_search(
    query: str,
    instance_url: Optional[str] = None,
    categories: Optional[str] = None,
    engines: Optional[str] = None,
    language: Optional[str] = None,
    time_range: Optional[str] = None,
    safesearch: int = 0,
    pageno: int = 1,
    format: str = "json",
    timeout: int = 30,
    fallback: bool = True,
):
    """
    Search using SearXNG metasearch engine.
    
    Args:
        query: Search query string
        instance_url: SearXNG instance URL (uses random public if not provided)
        categories: Filter by category (general, images, videos, news, etc.)
        engines: Comma-separated list of specific engines
        language: Language code (en, ru, de, etc.)
        time_range: Filter by time (day, month, year)
        safesearch: 0=None, 1=Moderate, 2=Strict
        pageno: Page number for pagination
        format: Output format (json, csv, rss)
        timeout: Request timeout in seconds
        fallback: If True, try multiple instances on failure
    
    Returns:
        dict: Parsed JSON response or error dict
    """
    
    instances_to_try = []
    
    if instance_url:
        instances_to_try.append(instance_url)
    
    if fallback or not instance_url:
        # Add remaining instances in random order
        remaining = [i for i in DEFAULT_INSTANCES if i not in instances_to_try]
        random.shuffle(remaining)
        instances_to_try.extend(remaining)
    
    last_error = None
    
    for inst in instances_to_try:
        result = _try_search_instance(
            query=query,
            instance_url=inst,
            categories=categories,
            engines=engines,
            language=language,
            time_range=time_range,
            safesearch=safesearch,
            pageno=pageno,
            format=format,
            timeout=timeout,
        )
        
        if result['success']:
            return result
        
        # Don't retry on 403 (JSON disabled) - it won't help
        if result.get('error', '').startswith('403'):
            continue
            
        last_error = result
    
    # All instances failed
    if last_error:
        return last_error
    
    return {
        'success': False,
        'error': 'All instances failed',
        'instance': None
    }


def _try_search_instance(
    query: str,
    instance_url: str,
    categories: Optional[str],
    engines: Optional[str],
    language: Optional[str],
    time_range: Optional[str],
    safesearch: int,
    pageno: int,
    format: str,
    timeout: int,
):
    """Try a single instance search."""
    
    instance_url = instance_url.rstrip('/')
    
    params = {
        'q': query,
        'format': format,
    }
    
    if categories:
        params['categories'] = categories
    if engines:
        params['engines'] = engines
    if language:
        params['language'] = language
    if time_range:
        params['time_range'] = time_range
    if safesearch:
        params['safesearch'] = safesearch
    if pageno > 1:
        params['pageno'] = pageno
    
    query_string = urllib.parse.urlencode(params)
    url = f"{instance_url}/search?{query_string}"
    
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/html;q=0.9',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            }
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            
            # Handle gzip if present
            if response.headers.get('Content-Encoding') == 'gzip':
                import gzip
                content = gzip.decompress(content)
            
            if format == 'json':
                data = json.loads(content.decode('utf-8'))
                return {
                    'success': True,
                    'instance': instance_url,
                    'data': data
                }
            else:
                return {
                    'success': True,
                    'instance': instance_url,
                    'data': content.decode('utf-8')
                }
                
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return {
                'success': False,
                'error': '403 Forbidden - JSON API disabled. Run your own instance: docker run -p 8080:8080 searxng/searxng',
                'instance': instance_url
            }
        elif e.code == 429:
            return {
                'success': False,
                'error': '429 Too Many Requests - Public instance rate-limited. Options: 1) Wait and retry 2) Use your own instance 3) Try with --instance for specific URL',
                'instance': instance_url
            }
        else:
            return {
                'success': False,
                'error': f'HTTP {e.code}: {e.reason}',
                'instance': instance_url
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'instance': instance_url
        }


def format_results(result: dict, max_results: int = 10) -> str:
    """Format search results for display."""
    
    if not result['success']:
        return f"ERROR: {result['error']}\nInstance: {result['instance']}"
    
    data = result['data']
    output = []
    
    # Header
    output.append(f"Search: {data.get('query', 'N/A')}")
    output.append(f"Instance: {result['instance']}")
    output.append("")
    
    # Results
    results = data.get('results', [])
    if results:
        output.append(f"Found {len(results)} results:\n")
        for i, r in enumerate(results[:max_results], 1):
            title = r.get('title', 'No title')
            url = r.get('url', 'No URL')
            content = r.get('content', '')
            engines = r.get('engines', [])
            
            output.append(f"{i}. {title}")
            output.append(f"   URL: {url}")
            if content:
                # Truncate long content
                if len(content) > 200:
                    content = content[:200] + "..."
                output.append(f"   {content}")
            if engines:
                output.append(f"   [via: {', '.join(engines)}]")
            output.append("")
    else:
        output.append("No results found.")
    
    # Suggestions
    suggestions = data.get('suggestions', [])
    if suggestions:
        output.append(f"\nSuggestions: {', '.join(suggestions)}")
    
    return '\n'.join(output)


def mock_search_result(query: str) -> dict:
    """Return mock result for testing/demo purposes."""
    return {
        'success': True,
        'instance': 'http://localhost:8080 (mock demo)',
        'data': {
            'query': query,
            'number_of_results': 3,
            'results': [
                {
                    'title': 'Python Tutorial - W3Schools',
                    'url': 'https://www.w3schools.com/python/',
                    'content': 'Learn Python. Python is a popular programming language. Python can be used on a server to create web applications. Start learning Python now ».',
                    'engines': ['google', 'duckduckgo']
                },
                {
                    'title': 'The Python Tutorial — Python 3.12.0 documentation',
                    'url': 'https://docs.python.org/3/tutorial/',
                    'content': 'Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming.',
                    'engines': ['google', 'bing']
                },
                {
                    'title': 'Learn Python - Free Interactive Python Tutorial',
                    'url': 'https://www.learnpython.org/',
                    'content': 'Whether you are an experienced programmer or not, this website is intended for everyone who wishes to learn the Python programming language.',
                    'engines': ['duckduckgo']
                }
            ],
            'suggestions': ['python tutorial for beginners', 'python tutorial pdf', 'python tutorial w3schools'],
            'infoboxes': []
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Search using SearXNG')
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--instance', '-i', help='SearXNG instance URL')
    parser.add_argument('--categories', '-c', help='Filter by category')
    parser.add_argument('--language', '-l', help='Language code')
    parser.add_argument('--time-range', '-t', choices=['day', 'month', 'year'], 
                        help='Time range filter')
    parser.add_argument('--max-results', '-n', type=int, default=10,
                        help='Maximum results to display (default: 10)')
    parser.add_argument('--raw', '-r', action='store_true',
                        help='Output raw JSON')
    parser.add_argument('--demo', '-d', action='store_true',
                        help='Show demo/mock output (no network request)')
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo or not args.query:
        query = args.query or 'python tutorial'
        result = mock_search_result(query)
        print("=== DEMO MODE (no network request) ===")
        print(format_results(result, max_results=args.max_results))
        print("\n=== To search real instances, run your own SearXNG:")
        print("docker run --rm -d -p 8080:8080 searxng/searxng")
        print("python3 searxng_search.py 'your query' -i http://localhost:8080")
        sys.exit(0)
    
    # Run search
    result = searxng_search(
        query=args.query,
        instance_url=args.instance,
        categories=args.categories,
        language=args.language,
        time_range=args.time_range,
    )
    
    # Output
    if args.raw:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_results(result, max_results=args.max_results))
