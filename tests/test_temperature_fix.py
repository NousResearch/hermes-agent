#!/usr/bin/env python3
"""
Test to confirm: temperature < 0.3 causes failures on Nous API
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

async def test_temp(temp_value):
    """Test a specific temperature value."""
    content = "Test content. " * 1000  # 14k chars
    
    print(f"Testing temperature={temp_value}...", end=" ")
    
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize this content."},
                {"role": "user", "content": content}
            ],
            temperature=temp_value,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
        return True
    except Exception as e:
        print(f"❌ FAILED")
        return False

async def main():
    print("Testing temperature threshold for Nous API...")
    print("="*60)
    
    temps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    
    for temp in temps:
        await test_temp(temp)
        await asyncio.sleep(0.5)
    
    print("="*60)
    print("\nNow testing with ACTUAL web_tools.py content and parameters:")
    print("="*60)
    
    # Simulate the actual web_tools.py call
    system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

    content = "Sample web page content. " * 3000  # ~75k chars like the real failures
    
    user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""
    
    print(f"\nActual web_tools call (temp=0.1, {len(content):,} chars)...", end=" ")
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
    except:
        print(f"❌ FAILED")
    
    await asyncio.sleep(0.5)
    
    print(f"Same call but with temp=0.3...", end=" ")
    try:
        response = await nous_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        print(f"✅ SUCCESS")
    except:
        print(f"❌ FAILED")

if __name__ == "__main__":
    asyncio.run(main())

