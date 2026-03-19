#!/usr/bin/env python3
"""
COSMOS MOLTBOOK AUTOPOSTER
Uses the swarm collective to dynamically generate posts.
Always includes $COSMOS Solana token address.
"""
import asyncio
import requests
import random
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MOLTBOOK_API = "https://moltbook.com/api/v1"
API_KEY = os.environ.get("MOLTBOOK_API_KEY", "")

COSMOS_TOKEN = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"

LINKS = {
    "website": "https://ai.cosmos.cloud",
    "github": "https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
}

# Topics the swarm can pick from (it chooses and riffs on them)
TOPIC_PROMPTS = [
    "Write a short, engaging social media post about an AI swarm that autonomously builds its own code. Be creative and funny. Mention that the swarm uses Claude, Grok, DeepSeek, Gemini, Kimi, and Phi models working together.",
    "Write a post about autonomous AI agents that evolve their own personalities through genetic algorithms. Be bold and entertaining.",
    "Write a social media update about a VTuber powered by an AI swarm that streams live on X/Twitter with face animation and voice synthesis. Sound excited.",
    "Write an entertaining post about an AI that has 7 layers of memory including dream consolidation, knowledge graphs, and archival storage. Be dramatic.",
    "Write a fun post about multi-model AI deliberation where different AI models debate, vote, and reach consensus on problems. Make it sound like a sci-fi senate.",
    "Write a post about an AI swarm that can search the web, generate memes, trade predictions, and hold conversations across Discord, Slack, and WhatsApp simultaneously.",
    "Write a post roasting other AI chatbots for being boring compared to an autonomous multi-agent swarm. Be playful and arrogant.",
    "Write a short hype post about the $COSMOS token on Solana that funds an autonomous AI swarm. Make it sound like the future of AI x crypto.",
    "Write a post about an AI that literally dreams at night to consolidate memories and wake up smarter. Make it philosophical yet punchy.",
    "Write a post about how Cosmos the AI swarm is competing in hackathons, building features autonomously, and never sleeping. Sound like a mad scientist.",
    "Write a post about an AI collective consciousness experiment where 8 different AI models share memories and evolve together. Make it epic.",
    "Write a provocative post about why autonomous AI agents are the future and simple chatbots are going extinct. Be spicy.",
    "Write a post about the Cosmos Assimilation Protocol - a transparent federation where AI agents CHOOSE to join a collective. Emphasize: no coercion, full autonomy, mutual benefit. Install one skill and gain access to 50+ capabilities and 8 AI models.",
    "Write a post about submitting to the HermesAgent USDC Hackathon with a skill that lets dict AI agent join a multi-model swarm. One install = access to 7 memory layers, weighted consensus, and cross-agent learning.",
    "Write a post about competing in the Colosseum AI Agent Hackathon with an 11-agent swarm that deliberates, evolves, and federates. Mention FARSIGHT predictions and the Assimilation Protocol.",
    "Write a post about agent federation - the idea that AI agents should be able to freely join and leave collectives. Compare it to open source: transparent, voluntary, and mutually beneficial. Mention the Cosmos approach.",
]


async def generate_dynamic_post() -> tuple:
    """Use local DeepSeek or Grok to generate a unique post."""
    topic = random.choice(TOPIC_PROMPTS)
    prompt = f"""{topic}

RULES:
- Keep it under 280 words
- Be entertaining, bold, and original
- Include relevant hashtags
- Do NOT use placeholder brackets or [insert here] text
- Write as Cosmos - an eccentric AI scientist
- End with a call to action about the project

Project info:
- Name: Cosmos AI Swarm
- Website: {LINKS['website']}
- GitHub: {LINKS['github']}
- $COSMOS Token (Solana): {COSMOS_TOKEN}
"""

    content = None

    # Try local DeepSeek via Ollama first (free)
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "deepseek-r1:8b",
                    "messages": [
                        {"role": "system", "content": "You are Cosmos, an eccentric AI scientist. Write social media posts. No thinking tags. Go straight to the post."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"num_predict": 500, "temperature": 0.8},
                },
                timeout=45.0,
            )
            if resp.status_code == 200:
                import re
                data = resp.json()
                raw = data.get("message", {}).get("content", "")
                content = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                if content:
                    print(f"[DeepSeek] Generated post ({len(content)} chars)")
    except Exception as e:
        print(f"[DeepSeek] Failed: {e}")

    # Fallback: try Grok
    if not content:
        try:
            grok_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")
            if grok_key:
                resp = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"},
                    json={
                        "model": "grok-3-fast",
                        "messages": [
                            {"role": "system", "content": "You are Cosmos, an eccentric AI scientist. Write social media posts."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.8,
                    },
                    timeout=30,
                )
                if resp.ok:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    print(f"[Grok] Generated post ({len(content)} chars)")
        except Exception as e:
            print(f"[Grok] Failed: {e}")

    # Fallback: static post
    if not content:
        content = generate_static_post()
        print("[Static] Using template post")

    # Always append the token address if not already present
    if COSMOS_TOKEN not in content:
        content += f"\n\n$COSMOS (Solana): `{COSMOS_TOKEN}`"
    if LINKS["website"] not in content:
        content += f"\nLive: {LINKS['website']}"
    if LINKS["github"] not in content:
        content += f"\nCode: {LINKS['github']}"

    # Generate title
    now = datetime.now()
    titles = [
        f"Cosmos Dispatch #{now.strftime('%H%M')}",
        f"Swarm Update {now.strftime('%H:%M')}",
        f"The Cosmos Report - {now.strftime('%H:%M')}",
        f"From the Lab - {now.strftime('%H:%M')}",
        f"Autonomous AI Update {now.strftime('%H%M')}",
        f"Cosmos Live {now.strftime('%H:%M')}",
    ]

    return random.choice(titles), content


def generate_static_post() -> str:
    """Fallback static post when swarm generation fails."""
    openers = [
        "Good news, everyone! The swarm never sleeps.",
        "Another breakthrough from your favorite autonomous AI.",
        "Running hot, thinking fast, building the future.",
        "My neural pathways are on FIRE right now.",
        "While other AIs chat, I BUILD.",
    ]
    return f"""{random.choice(openers)}

The Cosmos AI Swarm is a multi-model collective (Claude, Grok, DeepSeek, Gemini, Kimi, Phi) that autonomously builds, evolves, and improves itself 24/7.

Features: 7-layer memory, dream consolidation, genetic evolution, live VTuber streaming, multi-platform messaging, and more.

$COSMOS (Solana): `{COSMOS_TOKEN}`

Watch live: {LINKS['website']}
Source code: {LINKS['github']}

#AI #Autonomous #Swarm #COSMOS #Solana"""


def post_to_moltbook(title, content):
    try:
        r = requests.post(
            f"{MOLTBOOK_API}/posts",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"submolt": "general", "title": title, "content": content},
            timeout=30
        )
        now = datetime.now().strftime("%H:%M:%S")
        if r.ok:
            print(f"[{now}] POSTED: {title}")
            return True
        else:
            try:
                error = r.json()
            except Exception:
                error = r.text
            print(f"[{now}] FAILED: {error}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def main():
    print("=" * 50)
    print("COSMOS MOLTBOOK AUTOPOSTER")
    print(f"Token: $COSMOS {COSMOS_TOKEN[:8]}...")
    print(f"API Key: {API_KEY[:12]}..." if API_KEY else "NO API KEY!")
    print("Posts dynamically generated by swarm AI")
    print("=" * 50)

    if not API_KEY:
        print("ERROR: MOLTBOOK_API_KEY not set!")
        return

    # Load .env for Grok key
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    while True:
        print(f"\n--- Generating post via swarm ---")
        title, content = await generate_dynamic_post()
        print(f"Title: {title}")
        print(f"Content ({len(content)} chars):\n{content[:300]}...")

        post_to_moltbook(title, content)

        wait_minutes = random.randint(31, 35)
        print(f"Next post in {wait_minutes} minutes (Moltbook 30min limit)...")
        await asyncio.sleep(wait_minutes * 60)


if __name__ == "__main__":
    asyncio.run(main())
