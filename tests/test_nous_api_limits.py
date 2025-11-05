#!/usr/bin/env python3
"""
Test script to diagnose Nous API 400 errors with gemini-2.5-flash model.
This tests various content lengths and parameters to identify what causes failures.
"""

import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Nous API client
nous_client = AsyncOpenAI(
    api_key=os.getenv("NOUS_API_KEY"),
    base_url="https://inference-api.nousresearch.com/v1"
)

MODEL = "gemini-2.5-flash"

async def test_api_call(test_name: str, content_length: int, **kwargs):
    """Test an API call with specific parameters."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"Content length: {content_length:,} characters")
    print(f"Additional params: {kwargs}")
    print(f"{'='*60}")
    
    # Generate test content
    content = "A" * content_length
    
    system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

    user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        
        result = response.choices[0].message.content
        print(f"✅ SUCCESS")
        print(f"   Response length: {len(result)} characters")
        print(f"   Model used: {response.model}")
        print(f"   Usage: {response.usage}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("Testing Nous API with gemini-2.5-flash model")
    print(f"API Key present: {'Yes' if os.getenv('NOUS_API_KEY') else 'No'}")
    
    results = {}
    
    # Test 1: Small content (should always work)
    results['small'] = await test_api_call(
        "Small content (5,000 chars)",
        5000,
        temperature=0.1,
        max_tokens=4000
    )
    await asyncio.sleep(1)
    
    # Test 2: Medium content (around what was failing)
    results['medium'] = await test_api_call(
        "Medium content (20,000 chars)",
        20000,
        temperature=0.1,
        max_tokens=4000
    )
    await asyncio.sleep(1)
    
    # Test 3: Large content (79,625 chars like the error)
    results['large'] = await test_api_call(
        "Large content (79,625 chars)",
        79625,
        temperature=0.1,
        max_tokens=4000
    )
    await asyncio.sleep(1)
    
    # Test 4: Very large content (100k chars)
    results['very_large'] = await test_api_call(
        "Very large content (100,000 chars)",
        100000,
        temperature=0.1,
        max_tokens=4000
    )
    await asyncio.sleep(1)
    
    # Test 5: Same as working case but different max_tokens
    results['diff_max_tokens'] = await test_api_call(
        "Medium content with higher max_tokens",
        20000,
        temperature=0.1,
        max_tokens=8000
    )
    await asyncio.sleep(1)
    
    # Test 6: No max_tokens specified
    results['no_max_tokens'] = await test_api_call(
        "Medium content without max_tokens",
        20000,
        temperature=0.1
    )
    await asyncio.sleep(1)
    
    # Test 7: With actual web content (mixed characters)
    mixed_content = """
    This is a test of web content with various characters:
    - Unicode: 你好世界 🌍 
    - Special chars: <>&"'
    - Numbers: 123456789
    - Markdown: **bold** _italic_ `code`
    - URLs: https://example.com
    """ * 1000  # Repeat to make it ~79k chars
    
    print(f"\n{'='*60}")
    print(f"Test: Mixed content (real-world scenario)")
    print(f"Content length: {len(mixed_content):,} characters")
    print(f"{'='*60}")
    
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize this content."},
                {"role": "user", "content": mixed_content}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
        results['mixed_content'] = True
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        results['mixed_content'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS:")
    print(f"{'='*60}")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    asyncio.run(main())

