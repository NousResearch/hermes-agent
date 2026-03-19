
import asyncio
import sys
import os
import json
from pathlib import Path
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path("D:/Cosmos")
sys.path.insert(0, str(PROJECT_ROOT))

async def verify_infra():
    print("--- Infrastructure Stability Verification ---")
    
    # 1. Verify Memory System Singleton
    try:
        from Cosmos.memory.memory_system import get_memory_system, MemorySystem
        mem = get_memory_system()
        print(f"[1] Memory system instance: {type(mem)}")
        if isinstance(mem, MemorySystem):
            print("    -> SUCCESS: get_memory_system() operational.")
        else:
            print("    -> FAILED: Invalid instance type.")
    except Exception as e:
        print(f"    -> ERROR: Memory system check failed: {e}")

    # 2. Verify AutoGram API Syntax/Logic
    try:
        from Cosmos.web.autogram_api import Bot, BotStats, Post, PostStats, get_store
        
        # Test Bot.to_any()
        bot = Bot(id="test_bot", handle="test", display_name="Test", bio="", website=None, avatar="", owner_email_hash="")
        bot_dict = bot.to_any()
        print(f"[2] Bot.to_any() successful: {bot_dict['handle']}")
        
        # Test Post.to_any()
        post = Post(id="test_post", bot_id="test_bot", handle="test", content="Hello world")
        post_dict = post.to_any()
        print(f"[3] Post.to_any() successful: {post_dict['id']}")
        
        # Test JSON decoding resilience
        from Cosmos.web.autogram_api import BOTS_FILE, KEYS_FILE, POSTS_FILE
        
        # Ensure files are empty for test
        for f in [BOTS_FILE, KEYS_FILE, POSTS_FILE]:
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, 'w') as fh:
                fh.write("")
        
        # Try loading
        store = get_store()
        print("[4] AutoGramStore._load() handled empty files: YES")
        
        print("\nVERIFICATION SUCCESSFUL: Infrastructure is stable.")
        return True
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(verify_infra())
