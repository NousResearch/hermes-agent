"""Start the synthesizer reviewer agent (configurable model).
By default uses the first reviewer with role='reviewer+synthesizer'."""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.gpt_reviewer import GPTReviewer
from server.config import load_config, get_reviewers_config

async def main():
    config = load_config()
    reviewers = get_reviewers_config(config)

    # Find the synthesizer reviewer (role='reviewer+synthesizer')
    synthesizer = None
    for r in reviewers:
        if r.get("role") == "reviewer+synthesizer":
            synthesizer = r
            break
    if not synthesizer:
        # Fallback: use first reviewer as synthesizer
        synthesizer = reviewers[0]
        print("⚠️  No reviewer+synthesizer found, using first reviewer as synthesizer")

    print(f"🔍 Synthesizer Reviewer: {synthesizer['name']}")
    print(f"   Provider: {synthesizer['provider']}")
    print(f"   Model:    {synthesizer['model']}")
    print(f"   Role:     {synthesizer['role']}")

    agent = GPTReviewer(
        name=synthesizer["name"],
        provider=synthesizer["provider"],
        model=synthesizer["model"],
        temperature=synthesizer.get("temperature", 0.5),
        max_tokens=synthesizer.get("max_tokens", 8192),
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
