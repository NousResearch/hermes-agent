"""
Mock Web Tools for Testing WebSocket Reconnection

This module provides mock implementations of web_search and web_extract
that simulate long-running operations without making real API calls.

Perfect for testing WebSocket timeout/reconnection behavior without:
- Wasting API credits
- Waiting for real web crawling
- Network dependencies
"""

import time
import json
from typing import List


def mock_web_search(query: str, delay: int = 2) -> str:
    """
    Mock web search that returns fake results after a delay.
    
    Args:
        query: Search query (ignored, just for API compatibility)
        delay: Seconds to sleep (default: 2s)
    
    Returns:
        JSON string with fake search results
    """
    print(f"🔍 [MOCK] Searching for: '{query}' (will take {delay}s)...")
    time.sleep(delay)
    
    result = {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://example.com/article1",
                    "title": "Mock Article 1 - Water Utilities",
                    "description": "This is a mock search result for testing purposes. Real data would appear here.",
                    "category": None
                },
                {
                    "url": "https://example.com/article2",
                    "title": "Mock Article 2 - AI Data Centers",
                    "description": "Another mock result. This simulates web_search without making real API calls.",
                    "category": None
                },
                {
                    "url": "https://example.com/article3",
                    "title": "Mock Article 3 - Investment Opportunities",
                    "description": "Third mock result for testing. Query was: " + query,
                    "category": None
                }
            ]
        }
    }
    
    print(f"✅ [MOCK] Search completed with {len(result['data']['web'])} results")
    return json.dumps(result, indent=2)


def mock_web_extract(urls: List[str], delay: int = 60) -> str:
    """
    Mock web extraction that simulates long-running crawl.
    
    This is perfect for testing WebSocket timeout/reconnection because:
    - Default 60s delay triggers the ~30s WebSocket timeout
    - No actual web requests made
    - No API credits consumed
    - Predictable, reproducible behavior
    
    Args:
        urls: List of URLs to "extract" (ignored)
        delay: Seconds to sleep (default: 60s to trigger timeout)
    
    Returns:
        JSON string with fake extraction results
    """
    print(f"🌐 [MOCK] Extracting {len(urls)} URLs (will take {delay}s)...")
    print(f"📊 [MOCK] This will test WebSocket reconnection (timeout at ~30s)")
    
    # Simulate long-running operation
    # Show progress so user knows it's working
    for i in range(delay):
        if i % 10 == 0 and i > 0:
            print(f"  ⏱️  [MOCK] {i}/{delay}s elapsed...")
        time.sleep(1)
    
    # Generate fake but realistic-looking content
    result = {
        "success": True,
        "data": []
    }
    
    for idx, url in enumerate(urls, 1):
        result["data"].append({
            "url": url,
            "title": f"Mock Extracted Content {idx}",
            "content": f"# Mock Content from {url}\n\n"
                      f"This is simulated extracted content for testing purposes. "
                      f"In a real scenario, this would contain the full text from the webpage. "
                      f"\n\n## Key Points\n"
                      f"- Mock point 1 about water utilities\n"
                      f"- Mock point 2 about AI data centers\n"
                      f"- Mock point 3 about investment opportunities\n"
                      f"\n\nThis content took {delay} seconds to 'extract', which is long enough "
                      f"to trigger WebSocket timeout and test reconnection logic."
                      * 10,  # Make it longer to simulate real extraction
            "extracted_at": "2025-10-10T14:00:00Z"
        })
    
    json_result = json.dumps(result, indent=2)
    size_kb = len(json_result) / 1024
    
    print(f"✅ [MOCK] Extraction completed: {len(urls)} URLs, {size_kb:.1f} KB")
    return json_result


def mock_web_crawl(start_url: str, max_pages: int = 10, delay: int = 30) -> str:
    """
    Mock web crawling that simulates multi-page crawl.
    
    Args:
        start_url: Starting URL (ignored)
        max_pages: Max pages to crawl (just affects result count)
        delay: Seconds to sleep (default: 30s)
    
    Returns:
        JSON string with fake crawl results
    """
    print(f"🕷️  [MOCK] Crawling from: {start_url} (max {max_pages} pages, {delay}s)...")
    time.sleep(delay)
    
    result = {
        "success": True,
        "data": {
            "start_url": start_url,
            "pages_crawled": min(max_pages, 5),
            "pages": []
        }
    }
    
    for i in range(min(max_pages, 5)):
        result["data"]["pages"].append({
            "url": f"{start_url}/page{i+1}",
            "title": f"Mock Page {i+1}",
            "content": f"Mock content from page {i+1}. " * 50
        })
    
    print(f"✅ [MOCK] Crawl completed: {len(result['data']['pages'])} pages")
    return json.dumps(result, indent=2)


# Tool definitions for the agent (same format as real tools)
MOCK_WEB_TOOLS = [
    {
        "name": "web_search",
        "description": "[MOCK] Search the web for information. Returns fake results after 2s delay. Perfect for quick tests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "delay": {
                    "type": "integer",
                    "description": "Seconds to delay (default: 2)",
                    "default": 2
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "web_extract",
        "description": "[MOCK] Extract content from URLs. Simulates 60s delay to test WebSocket timeout/reconnection. Returns fake content without making real requests. PERFECT FOR TESTING!",
        "input_schema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to extract"
                },
                "delay": {
                    "type": "integer",
                    "description": "Seconds to delay (default: 60 to trigger timeout)",
                    "default": 60
                }
            },
            "required": ["urls"]
        }
    },
    {
        "name": "web_crawl",
        "description": "[MOCK] Crawl website starting from URL. Returns fake results after 30s delay.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_url": {
                    "type": "string",
                    "description": "Starting URL for crawl"
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Max pages to crawl (default: 10)",
                    "default": 10
                },
                "delay": {
                    "type": "integer",
                    "description": "Seconds to delay (default: 30)",
                    "default": 30
                }
            },
            "required": ["start_url"]
        }
    }
]


# Map function names to implementations
MOCK_TOOL_FUNCTIONS = {
    "web_search": mock_web_search,
    "web_extract": mock_web_extract,
    "web_crawl": mock_web_crawl
}


if __name__ == "__main__":
    # Demo/test the mock tools
    print("Testing Mock Web Tools")
    print("=" * 60)
    
    print("\n1. Mock web_search (2s delay):")
    result = mock_web_search("test query", delay=2)
    print(f"Result length: {len(result)} chars\n")
    
    print("\n2. Mock web_extract (5s delay for demo - normally 60s):")
    result = mock_web_extract(["https://example.com"], delay=5)
    print(f"Result length: {len(result)} chars\n")
    
    print("\n✅ All mock tools working!")

