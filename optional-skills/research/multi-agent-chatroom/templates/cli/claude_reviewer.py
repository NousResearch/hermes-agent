"""Start a reviewer agent (configurable model).
By default uses the first reviewer with role='mathematical-rigor' or 'reviewer'."""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.claude_reviewer import ClaudeReviewer
from server.config import load_config, get_reviewers_config

async def main():
    config = load_config()
    reviewers = get_reviewers_config(config)

    # Find a non-synthesizer reviewer (mathematical-rigor or plain reviewer)
    reviewer_cfg = None
    for r in reviewers:
        if r.get("role") in ("mathematical-rigor", "reviewer"):
            reviewer_cfg = r
            break
    if not reviewer_cfg:
        # Fallback: use second reviewer, or first if only one
        reviewer_cfg = reviewers[1] if len(reviewers) > 1 else reviewers[0]
        print("⚠️  Using fallback reviewer selection")

    print(f"🔬 Reviewer: {reviewer_cfg['name']}")
    print(f"   Provider: {reviewer_cfg['provider']}")
    print(f"   Model:    {reviewer_cfg['model']}")
    print(f"   Role:     {reviewer_cfg['role']}")

    agent = ClaudeReviewer(
        name=reviewer_cfg["name"],
        provider=reviewer_cfg["provider"],
        model=reviewer_cfg["model"],
        temperature=reviewer_cfg.get("temperature", 0.4),
        max_tokens=reviewer_cfg.get("max_tokens", 8192),
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
