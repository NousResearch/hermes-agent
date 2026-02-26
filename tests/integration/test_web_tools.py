#!/usr/bin/env python3
"""
Comprehensive Test Suite for Web Tools Module

This script tests all web tools functionality to ensure they work correctly.
Run this after any updates to the web_tools.py module or Firecrawl library.

Usage:
    python test_web_tools.py              # Run all tests
    python test_web_tools.py --no-llm     # Skip LLM processing tests
    python test_web_tools.py --verbose    # Show detailed output

Requirements:
    - FIRECRAWL_API_KEY environment variable must be set
    - NOUS_API_KEY environment variable (optional, for LLM tests)
"""

import pytest
pytestmark = pytest.mark.integration

import json
import asyncio
import sys
import os
import argparse
from datetime import datetime
from typing import List

# Import the web tools to test (updated path after moving tools/)
from tools.web_tools import (
    web_search_tool, 
    web_extract_tool, 
    web_crawl_tool,
    check_firecrawl_api_key,
    check_auxiliary_model,
    get_debug_session_info
)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    """Print a formatted section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}📌 {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text: str, indent: int = 0):
    """Print info message"""
    indent_str = "  " * indent
    print(f"{indent_str}{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


class WebToolsTester:
    """Test suite for web tools"""
    
    def __init__(self, verbose: bool = False, test_llm: bool = True):
        self.verbose = verbose
        self.test_llm = test_llm
        self.test_results = {
            "passed": [],
            "failed": [],
            "skipped": []
        }
        self.start_time = None
        self.end_time = None
    
    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        if status == "passed":
            self.test_results["passed"].append(result)
            print_success(f"{test_name}: {details}" if details else test_name)
        elif status == "failed":
            self.test_results["failed"].append(result)
            print_error(f"{test_name}: {details}" if details else test_name)
        elif status == "skipped":
            self.test_results["skipped"].append(result)
            print_warning(f"{test_name} skipped: {details}" if details else f"{test_name} skipped")
    
    def test_environment(self) -> bool:
        """Test environment setup and API keys"""
        print_section("Environment Check")
        
        # Check Firecrawl API key
        if not check_firecrawl_api_key():
            self.log_result("Firecrawl API Key", "failed", "FIRECRAWL_API_KEY not set")
            return False
        else:
            self.log_result("Firecrawl API Key", "passed", "Found")
        
        # Check Nous API key (optional)
        if not check_auxiliary_model():
            self.log_result("Nous API Key", "skipped", "NOUS_API_KEY not set (LLM tests will be skipped)")
            self.test_llm = False
        else:
            self.log_result("Nous API Key", "passed", "Found")
        
        # Check debug mode
        debug_info = get_debug_session_info()
        if debug_info["enabled"]:
            print_info(f"Debug mode enabled - Session: {debug_info['session_id']}")
            print_info(f"Debug log: {debug_info['log_path']}")
        
        return True
    
    def test_web_search(self) -> List[str]:
        """Test web search functionality"""
        print_section("Test 1: Web Search")
        
        test_queries = [
            ("Python web scraping tutorial", 5),
            ("Firecrawl API documentation", 3),
            ("inflammatory arthritis symptoms treatment", 8)  # Test medical query from your example
        ]
        
        extracted_urls = []
        
        for query, limit in test_queries:
            try:
                print(f"\n  Testing search: '{query}' (limit={limit})")
                
                if self.verbose:
                    print(f"  Calling web_search_tool(query='{query}', limit={limit})")
                
                # Perform search
                result = web_search_tool(query, limit)
                
                # Parse result
                try:
                    data = json.loads(result)
                except json.JSONDecodeError as e:
                    self.log_result(f"Search: {query[:30]}...", "failed", f"Invalid JSON: {e}")
                    if self.verbose:
                        print(f"    Raw response (first 500 chars): {result[:500]}...")
                    continue
                
                if "error" in data:
                    self.log_result(f"Search: {query[:30]}...", "failed", f"API error: {data['error']}")
                    continue
                
                # Check structure
                if "success" not in data or "data" not in data:
                    self.log_result(f"Search: {query[:30]}...", "failed", "Missing success or data fields")
                    if self.verbose:
                        print(f"    Response keys: {list(data.keys())}")
                    continue
                
                web_results = data.get("data", {}).get("web", [])
                
                if not web_results:
                    self.log_result(f"Search: {query[:30]}...", "failed", "Empty web results array")
                    if self.verbose:
                        print(f"    data.web content: {data.get('data', {}).get('web')}")
                    continue
                
                # Validate each result
                valid_results = 0
                missing_fields = []
                
                for i, result in enumerate(web_results):
                    required_fields = ["url", "title", "description"]
                    has_all_fields = all(key in result for key in required_fields)
                    
                    if has_all_fields:
                        valid_results += 1
                        # Collect URLs for extraction test
                        if len(extracted_urls) < 3:
                            extracted_urls.append(result["url"])
                        
                        if self.verbose:
                            print(f"    Result {i+1}: ✓ {result['title'][:50]}...")
                            print(f"      URL: {result['url'][:60]}...")
                    else:
                        missing = [f for f in required_fields if f not in result]
                        missing_fields.append(f"Result {i+1} missing: {missing}")
                        if self.verbose:
                            print(f"    Result {i+1}: ✗ Missing fields: {missing}")
                
                # Log results
                if valid_results == len(web_results):
                    self.log_result(
                        f"Search: {query[:30]}...", 
                        "passed", 
                        f"All {valid_results} results valid"
                    )
                else:
                    self.log_result(
                        f"Search: {query[:30]}...", 
                        "failed", 
                        f"Only {valid_results}/{len(web_results)} valid. Issues: {'; '.join(missing_fields[:3])}"
                    )
                    
            except Exception as e:
                self.log_result(f"Search: {query[:30]}...", "failed", f"Exception: {type(e).__name__}: {str(e)}")
                if self.verbose:
                    import traceback
                    print(f"    Traceback: {traceback.format_exc()}")
        
        if self.verbose and extracted_urls:
            print(f"\n  URLs collected for extraction test: {len(extracted_urls)}")
            for url in extracted_urls:
                print(f"    - {url}")
        
        return extracted_urls
    
    async def test_web_extract(self, urls: List[str] = None):
        """Test web content extraction"""
        print_section("Test 2: Web Extract (without LLM)")
        
        # Use provided URLs or defaults
        if not urls:
            urls = [
                "https://docs.firecrawl.dev/introduction",
                "https://www.python.org/about/"
            ]
            print(f"  Using default URLs for testing")
        else:
            print(f"  Using {len(urls)} URLs from search results")
        
        # Test extraction
        if urls:
            try:
                test_urls = urls[:2]  # Test with max 2 URLs
                print(f"\n  Extracting content from {len(test_urls)} URL(s)...")
                for url in test_urls:
                    print(f"    - {url}")
                
                if self.verbose:
                    print(f"  Calling web_extract_tool(urls={test_urls}, format='markdown', use_llm_processing=False)")
                
                result = await web_extract_tool(
                    test_urls,
                    format="markdown",
                    use_llm_processing=False
                )
                
                # Parse result
                try:
                    data = json.loads(result)
                except json.JSONDecodeError as e:
                    self.log_result("Extract (no LLM)", "failed", f"Invalid JSON: {e}")
                    if self.verbose:
                        print(f"    Raw response (first 500 chars): {result[:500]}...")
                    return
                
                if "error" in data:
                    self.log_result("Extract (no LLM)", "failed", f"API error: {data['error']}")
                    return
                
                results = data.get("results", [])
                
                if not results:
                    self.log_result("Extract (no LLM)", "failed", "No results in response")
                    if self.verbose:
                        print(f"    Response keys: {list(data.keys())}")
                    return
                
                # Validate each result
                valid_results = 0
                failed_results = 0
                total_content_length = 0
                extraction_details = []
                
                for i, result in enumerate(results):
                    title = result.get("title", "No title")
                    content = result.get("content", "")
                    error = result.get("error")
                    
                    if error:
                        failed_results += 1
                        extraction_details.append(f"Page {i+1}: ERROR - {error}")
                        if self.verbose:
                            print(f"    Page {i+1}: ✗ Error - {error}")
                    elif content:
                        content_len = len(content)
                        total_content_length += content_len
                        valid_results += 1
                        extraction_details.append(f"Page {i+1}: {title[:40]}... ({content_len} chars)")
                        if self.verbose:
                            print(f"    Page {i+1}: ✓ {title[:50]}... - {content_len} characters")
                            print(f"      First 100 chars: {content[:100]}...")
                    else:
                        extraction_details.append(f"Page {i+1}: {title[:40]}... (EMPTY)")
                        if self.verbose:
                            print(f"    Page {i+1}: ⚠ {title[:50]}... - Empty content")
                
                # Log results
                if valid_results > 0:
                    self.log_result(
                        "Extract (no LLM)", 
                        "passed", 
                        f"{valid_results}/{len(results)} pages extracted, {total_content_length} total chars"
                    )
                else:
                    self.log_result(
                        "Extract (no LLM)", 
                        "failed", 
                        f"No valid content. {failed_results} errors, {len(results) - failed_results} empty"
                    )
                    if self.verbose:
                        print(f"\n  Extraction details:")
                        for detail in extraction_details:
                            print(f"    {detail}")
                    
            except Exception as e:
                self.log_result("Extract (no LLM)", "failed", f"Exception: {type(e).__name__}: {str(e)}")
                if self.verbose:
                    import traceback
                    print(f"    Traceback: {traceback.format_exc()}")
    
    async def test_web_extract_with_llm(self, urls: List[str] = None):
        """Test web extraction with LLM processing"""
        print_section("Test 3: Web Extract (with Gemini LLM)")
        
        if not self.test_llm:
            self.log_result("Extract (with LLM)", "skipped", "LLM testing disabled")
            return
        
        # Use a URL likely to have substantial content
        test_url = urls[0] if urls else "https://docs.firecrawl.dev/features/scrape"
        
        try:
            print(f"\n  Extracting and processing: {test_url}")
            
            result = await web_extract_tool(
                [test_url],
                format="markdown",
                use_llm_processing=True,
                min_length=1000  # Lower threshold for testing
            )
            
            data = json.loads(result)
            
            if "error" in data:
                self.log_result("Extract (with LLM)", "failed", data["error"])
                return
            
            results = data.get("results", [])
            
            if not results:
                self.log_result("Extract (with LLM)", "failed", "No results returned")
                return
            
            result = results[0]
            content = result.get("content", "")
            
            if content:
                content_len = len(content)
                
                # Check if content was actually processed (should be shorter than typical raw content)
                if content_len > 0:
                    self.log_result(
                        "Extract (with LLM)", 
                        "passed", 
                        f"Content processed: {content_len} chars"
                    )
                    
                    if self.verbose:
                        print(f"\n    First 300 chars of processed content:")
                        print(f"    {content[:300]}...")
                else:
                    self.log_result("Extract (with LLM)", "failed", "No content after processing")
            else:
                self.log_result("Extract (with LLM)", "failed", "No content field in result")
                
        except json.JSONDecodeError as e:
            self.log_result("Extract (with LLM)", "failed", f"Invalid JSON: {e}")
        except Exception as e:
            self.log_result("Extract (with LLM)", "failed", str(e))
    
    async def test_web_crawl(self):
        """Test web crawling functionality"""
        print_section("Test 4: Web Crawl")
        
        test_sites = [
            ("https://docs.firecrawl.dev", None, 2),  # Test docs site
            ("https://firecrawl.dev", None, 3),  # Test main site
        ]
        
        for url, instructions, expected_min_pages in test_sites:
            try:
                print(f"\n  Testing crawl of: {url}")
                if instructions:
                    print(f"  Instructions: {instructions}")
                else:
                    print(f"  No instructions (general crawl)")
                print(f"  Expected minimum pages: {expected_min_pages}")
                
                # Show what's being called
                if self.verbose:
                    print(f"  Calling web_crawl_tool(url='{url}', instructions={instructions}, use_llm_processing=False)")
                
                result = await web_crawl_tool(
                    url,
                    instructions=instructions,
                    use_llm_processing=False  # Disable LLM for faster testing
                )
                
                # Check if result is valid JSON
                try:
                    data = json.loads(result)
                except json.JSONDecodeError as e:
                    self.log_result(f"Crawl: {url}", "failed", f"Invalid JSON response: {e}")
                    if self.verbose:
                        print(f"    Raw response (first 500 chars): {result[:500]}...")
                    continue
                
                # Check for errors
                if "error" in data:
                    self.log_result(f"Crawl: {url}", "failed", f"API error: {data['error']}")
                    continue
                
                # Get results
                results = data.get("results", [])
                
                if not results:
                    self.log_result(f"Crawl: {url}", "failed", "No pages in results array")
                    if self.verbose:
                        print(f"    Full response: {json.dumps(data, indent=2)[:1000]}...")
                    continue
                
                # Analyze pages
                valid_pages = 0
                empty_pages = 0
                total_content = 0
                page_details = []
                
                for i, page in enumerate(results):
                    content = page.get("content", "")
                    title = page.get("title", "Untitled")
                    error = page.get("error")
                    
                    if error:
                        page_details.append(f"Page {i+1}: ERROR - {error}")
                    elif content:
                        valid_pages += 1
                        content_len = len(content)
                        total_content += content_len
                        page_details.append(f"Page {i+1}: {title[:40]}... ({content_len} chars)")
                    else:
                        empty_pages += 1
                        page_details.append(f"Page {i+1}: {title[:40]}... (EMPTY)")
                
                # Show detailed results if verbose
                if self.verbose:
                    print(f"\n  Crawl Results:")
                    print(f"    Total pages returned: {len(results)}")
                    print(f"    Valid pages (with content): {valid_pages}")
                    print(f"    Empty pages: {empty_pages}")
                    print(f"    Total content size: {total_content} characters")
                    print(f"\n  Page Details:")
                    for detail in page_details[:10]:  # Show first 10 pages
                        print(f"    - {detail}")
                    if len(page_details) > 10:
                        print(f"    ... and {len(page_details) - 10} more pages")
                
                # Determine pass/fail
                if valid_pages >= expected_min_pages:
                    self.log_result(
                        f"Crawl: {url}", 
                        "passed", 
                        f"{valid_pages}/{len(results)} valid pages, {total_content} chars total"
                    )
                else:
                    self.log_result(
                        f"Crawl: {url}", 
                        "failed", 
                        f"Only {valid_pages} valid pages (expected >= {expected_min_pages}), {empty_pages} empty, {len(results)} total"
                    )
                    
            except Exception as e:
                self.log_result(f"Crawl: {url}", "failed", f"Exception: {type(e).__name__}: {str(e)}")
                if self.verbose:
                    import traceback
                    print(f"    Traceback:")
                    print("    " + "\n    ".join(traceback.format_exc().split("\n")))
    
    async def run_all_tests(self):
        """Run all tests"""
        self.start_time = datetime.now()
        
        print_header("WEB TOOLS TEST SUITE")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test environment
        if not self.test_environment():
            print_error("\nCannot proceed without required API keys!")
            return False
        
        # Test search and collect URLs
        urls = self.test_web_search()
        
        # Test extraction
        await self.test_web_extract(urls if urls else None)
        
        # Test extraction with LLM
        if self.test_llm:
            await self.test_web_extract_with_llm(urls if urls else None)
        
        # Test crawling
        await self.test_web_crawl()
        
        # Print summary
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print_header("TEST SUMMARY")
        print(f"Duration: {duration:.2f} seconds")
        print(f"\n{Colors.GREEN}Passed: {len(self.test_results['passed'])}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {len(self.test_results['failed'])}{Colors.ENDC}")
        print(f"{Colors.WARNING}Skipped: {len(self.test_results['skipped'])}{Colors.ENDC}")
        
        # List failed tests
        if self.test_results["failed"]:
            print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
            for test in self.test_results["failed"]:
                print(f"  - {test['test']}: {test['details']}")
        
        # Save results to file
        self.save_results()
        
        return len(self.test_results["failed"]) == 0
    
    def save_results(self):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_web_tools_{timestamp}.json"
        
        results = {
            "test_suite": "Web Tools",
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "summary": {
                "passed": len(self.test_results["passed"]),
                "failed": len(self.test_results["failed"]),
                "skipped": len(self.test_results["skipped"])
            },
            "results": self.test_results,
            "environment": {
                "firecrawl_api_key": check_firecrawl_api_key(),
                "nous_api_key": check_auxiliary_model(),
                "debug_mode": get_debug_session_info()["enabled"]
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print_info(f"Test results saved to: {filename}")
        except Exception as e:
            print_warning(f"Failed to save results: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Web Tools Module")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM processing tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for web tools")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        os.environ["WEB_TOOLS_DEBUG"] = "true"
        print_info("Debug mode enabled for web tools")
    
    # Create tester
    tester = WebToolsTester(
        verbose=args.verbose,
        test_llm=not args.no_llm
    )
    
    # Run tests
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
