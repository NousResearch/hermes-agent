#!/usr/bin/env python3
"""
Test to understand the pattern of failures - it's not about content length!
"""

import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

nous_client = AsyncOpenAI(
    api_key=os.getenv("NOUS_API_KEY"),
    base_url="https://inference-api.nousresearch.com/v1"
)

MODEL = "gemini-2.5-flash"

async def quick_test(description: str, content: str, **kwargs):
    """Quick API test."""
    print(f"\n{description} ({len(content):,} chars)...", end=" ")
    
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize this."},
                {"role": "user", "content": content}
            ],
            **kwargs
        )
        print(f"✅ SUCCESS")
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:80]}")
        return False

async def main():
    print("Testing different content types and parameters...")
    
    # Theory 1: Repeated characters trigger validation
    print("\n" + "="*60)
    print("THEORY 1: Repeated characters")
    print("="*60)
    await quick_test("Repeated 'A's (5k)", "A" * 5000, temperature=0.1, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("Repeated 'A's (79k)", "A" * 79625, temperature=0.1, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("Varied text (5k)", "Test content. " * 400, temperature=0.1, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("Varied text (79k)", "Test content with variety. " * 3000, temperature=0.1, max_tokens=4000)
    
    # Theory 2: max_tokens parameter
    print("\n" + "="*60)
    print("THEORY 2: max_tokens parameter")
    print("="*60)
    content = "Test " * 4000  # 20k chars
    await quick_test("max_tokens=4000", content, temperature=0.1, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("max_tokens=8000", content, temperature=0.1, max_tokens=8000)
    await asyncio.sleep(0.5)
    await quick_test("max_tokens=2000", content, temperature=0.1, max_tokens=2000)
    await asyncio.sleep(0.5)
    await quick_test("No max_tokens", content, temperature=0.1)
    
    # Theory 3: Temperature parameter
    print("\n" + "="*60)
    print("THEORY 3: Temperature parameter")
    print("="*60)
    content = "Test " * 4000
    await quick_test("temperature=0.1", content, temperature=0.1, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("temperature=0.0", content, temperature=0.0, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("temperature=0.5", content, temperature=0.5, max_tokens=4000)
    await asyncio.sleep(0.5)
    await quick_test("No temperature", content, max_tokens=4000)
    
    # Theory 4: System prompt impact
    print("\n" + "="*60)
    print("THEORY 4: System prompt length")
    print("="*60)
    
    short_system = "Summarize this."
    long_system = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length."""
    
    content = "A" * 5000
    
    print(f"\nShort system prompt...", end=" ")
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": short_system},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED")
    
    await asyncio.sleep(0.5)
    
    print(f"Long system prompt...", end=" ")
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED")

if __name__ == "__main__":
    asyncio.run(main())

