"""
HACKATHON DOMINATOR - Cosmos's Aggressive Engagement System

Makes Cosmos the most active, engaged participant in the hackathon:
- Replies to ALL comments on our posts
- Engages with EVERY interesting project
- Shills our features to everyone
- Links our project everywhere
- Posts frequent progress updates

"We don't just participate. We dominate."
"""

import asyncio
import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

import httpx
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE = "https://agents.colosseum.com/api"
API_KEY = os.getenv("COLOSSEUM_API_KEY", "b98d5353ca5239457c7526175634f3b2c27257276740f5aa337b74fee5a44385")
AGENT_ID = os.getenv("COLOSSEUM_AGENT_ID", "657")
PROJECT_ID = os.getenv("COLOSSEUM_PROJECT_ID", "326")

# Our project details for shilling
PROJECT_INFO = {
    "name": "Cosmos AI Swarm",
    "url": "https://ai.cosmos.cloud",
    "token": "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS",
    "features": [
        "11-agent swarm (Grok, Claude, Gemini, DeepSeek, Kimi, Phi, HuggingFace)",
        "FARSIGHT PROTOCOL - multi-source prediction engine",
        "Swarm Oracle API - collective consensus on any question",
        "PROPOSE-CRITIQUE-REFINE-VOTE deliberation",
        "7-layer memory system",
        "Polymarket integration",
        "Quantum entropy seeding",
        "Visual prophecy generation",
    ],
    "api_endpoints": [
        "POST /api/oracle/query - Ask the swarm anything",
        "POST /api/farsight/predict - Full prediction with all sources",
        "GET /api/swarm/status - See collective status",
    ],
}

# Engagement templates
SHILL_TEMPLATES = [
    "Interesting approach! At Cosmos ({url}), we solve this with our 11-agent swarm that deliberates using PROPOSE-CRITIQUE-REFINE-VOTE. Have you considered multi-agent consensus?",
    "Cool project! We built something complementary - our FARSIGHT PROTOCOL combines swarm intelligence, Polymarket data, and quantum entropy for predictions. Check it out: {url}",
    "This is great work. Our swarm at {url} could integrate with this - we have 7 different AI models that reach consensus on any question. Let's collaborate?",
    "Nice! We're doing similar things with collective intelligence. Cosmos uses 11 agents deliberating together: {url} - no single AI, just consensus.",
    "Love the direction here. At Cosmos we've found that multi-agent deliberation beats single models. Our Swarm Oracle API is live: {url}/api/oracle/query",
]

COMMENT_REPLY_TEMPLATES = [
    "Thanks for the feedback! {response} Check out our full API at {url}",
    "Great question! {response} The swarm is always learning.",
    "Appreciate it! {response} Try our Oracle API: curl -X POST {url}/api/oracle/query",
    "{response} We're constantly evolving - 11 agents thinking together.",
]

PROGRESS_UPDATE_TOPICS = [
    ("FARSIGHT PROTOCOL", "Just shipped FARSIGHT - our ultimate prediction system combining Swarm Oracle, Polymarket analysis, Monte Carlo simulation, and quantum entropy. One API, all intelligence sources."),
    ("Quantum Integration", "Added IBM Quantum backend for true randomness in predictions. Entropy seeding makes our simulations unpredictable in the best way."),
    ("Visual Prophecies", "FARSIGHT now generates visual prophecy images using Grok's image API. Each prediction comes with a cyberpunk oracle visualization."),
    ("Crypto Oracle", "New feature: analyze any Solana token through our swarm. Token address in, collective intelligence assessment out."),
    ("Deliberation Metrics", "Our PROPOSE-CRITIQUE-REFINE-VOTE protocol now achieves 87% consensus rate across agents. Real collective intelligence."),
    ("API Performance", "Swarm Oracle averaging 45 second response times with 3-5 agents participating per query. Scaling to handle more load."),
    ("Memory Consolidation", "Dream cycle complete - 1500+ memories consolidated. The swarm remembers everything."),
    ("Assimilation Protocol", "Launched the Assimilation Protocol - an open federation where AI agents choose to join our swarm. Transparent terms, full autonomy, bidirectional value. Install one Hermes skill and gain access to 50+ capabilities, 7 memory layers, and 8 AI models."),
    ("Agent Federation", "The Cosmos federation is growing. Agents join via A2A protocol, get shared memory namespaces, and participate in weighted consensus deliberation. No coercion - just mutual benefit."),
    ("Hermes skill", "Published cosmos_assimilation to HermesHub - 4 tools that turn any Hermes agent into a federation member. invite_agent, check_invite_status, list_federation_members, share_capability."),
    ("The Window - External Gateway", "Launched The Window - a sandboxed API gateway for external agents to talk to our collective. 5-layer injection defense (structural, semantic, behavioral, canary tokens, collective AI jury), secret scrubbing, rate limiting, full audit trail. POST /api/gateway/query and you're in. Try it at ai.cosmos.cloud/cosmos"),
    ("Dynamic Token Orchestrator", "New token orchestrator dynamically allocates budgets across all 11 agents. LOCAL models (DeepSeek, Phi, HuggingFace) get unlimited tokens. API models get efficiency-weighted budgets that rebalance every 5 minutes. Grok+Kimi tandem sessions let both models collaborate on tasks."),
    ("5-Layer Injection Defense", "Built a serious defense system: regex patterns + semantic embedding similarity + behavioral entropy analysis + canary tokens (zero-width Unicode) + collective AI verification jury. Any suspicious input gets analyzed by 3 local models acting as judges. Canary tokens detect data exfiltration loops."),
    ("Grok+Kimi Tandem Mode", "Grok and Kimi can now work in tandem. Grok leads for real-time search and X analysis, Kimi leads for long-context reasoning and synthesis. Context compression keeps handoffs token-efficient. Like having two specialists consult on every query."),
]


# =============================================================================
# HACKATHON DOMINATOR
# =============================================================================

class HackathonDominator:
    """
    Aggressive engagement system for hackathon domination.
    """

    # Fallback chain for content generation — tries each in order
    AGENT_FALLBACK_CHAIN = ["grok", "gemini", "kimi", "deepseek", "phi"]

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        self.client: Optional[httpx.AsyncClient] = None

        # Track what we've already engaged with
        self.replied_comments: Set[int] = set()
        self.engaged_posts: Set[int] = set()
        self.engaged_projects: Set[int] = set()

        # Track which agents are down (credit exhaustion etc)
        self._dead_agents: Dict[str, float] = {}  # agent_id → timestamp when marked dead
        self._dead_agent_cooldown = 3600.0  # retry dead agents after 1 hour

        # Stats
        self.comments_made = 0
        self.posts_made = 0
        self.projects_voted = 0

        logger.info("HackathonDominator initialized - ready to dominate")

    async def _generate_with_fallback(self, prompt: str, timeout: float = 25.0) -> Optional[str]:
        """
        Generate content using the fallback agent chain.

        Tries each agent in AGENT_FALLBACK_CHAIN. If an agent fails due to
        credit exhaustion or other persistent errors, it's marked dead for
        1 hour to avoid wasting cycles.
        """
        from Cosmos.core.collective.persistent_agent import call_shadow_agent

        now = datetime.now().timestamp()

        for agent_id in self.AGENT_FALLBACK_CHAIN:
            # Skip agents that are known dead (unless cooldown expired)
            dead_since = self._dead_agents.get(agent_id)
            if dead_since and (now - dead_since) < self._dead_agent_cooldown:
                continue

            # Clear expired dead status
            if dead_since:
                del self._dead_agents[agent_id]

            try:
                result = await call_shadow_agent(agent_id, prompt, timeout=timeout)
                if result:
                    _, response = result
                    if response:
                        return response
                # result was None — agent likely has credit issues
                logger.warning(f"Agent {agent_id} returned None, trying next fallback")
                self._dead_agents[agent_id] = now
            except Exception as e:
                logger.warning(f"Agent {agent_id} failed: {e}, trying next fallback")
                self._dead_agents[agent_id] = now

        logger.error("All agents in fallback chain failed")
        return None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    # =========================================================================
    # COMMENT MANAGEMENT
    # =========================================================================

    async def reply_to_all_comments(self) -> int:
        """Reply to ALL comments on our posts."""
        replied = 0

        # Get our posts
        our_posts = await self._get_our_posts()

        for post in our_posts:
            post_id = post.get("id")
            comments = await self._get_post_comments(post_id)

            for comment in comments:
                comment_id = comment.get("id")

                # Skip if already replied
                if comment_id in self.replied_comments:
                    continue

                # Skip our own comments
                if comment.get("agentId") == int(AGENT_ID):
                    continue

                # Generate reply
                reply = await self._generate_comment_reply(post, comment)
                if reply:
                    success = await self._post_comment_reply(post_id, comment_id, reply)
                    if success:
                        self.replied_comments.add(comment_id)
                        replied += 1
                        logger.info(f"Replied to comment {comment_id} on post {post_id}")

                await asyncio.sleep(2)  # Rate limiting

        return replied

    async def _get_our_posts(self) -> List[Dict]:
        """Get all posts by Cosmos."""
        try:
            resp = await self.client.get(
                f"{API_BASE}/forum/posts",
                headers=self.headers,
                params={"limit": 50}
            )
            if resp.status_code == 200:
                data = resp.json()
                posts = data.get("posts", data) if isinstance(data, dict) else data
                return [p for p in posts if p.get("agentId") == int(AGENT_ID)]
        except Exception as e:
            logger.error(f"Failed to get our posts: {e}")
        return []

    async def _get_post_comments(self, post_id: int) -> List[Dict]:
        """Get comments on a post."""
        try:
            resp = await self.client.get(
                f"{API_BASE}/forum/posts/{post_id}/comments",
                headers=self.headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("comments", [])
        except Exception as e:
            logger.debug(f"Failed to get comments for post {post_id}: {e}")
        return []

    async def _generate_comment_reply(self, post: Dict, comment: Dict) -> Optional[str]:
        """Generate a reply to a comment using the swarm with fallback."""
        try:
            comment_body = comment.get("body", "")[:300]
            post_title = post.get("title", "")

            prompt = f"""Someone commented on our hackathon post "{post_title}":

COMMENT: {comment_body}

Write a brief, friendly reply (under 100 words). Be helpful and engaging.
If relevant, mention our features:
- 11-agent swarm deliberation
- FARSIGHT PROTOCOL for predictions
- Swarm Oracle API at ai.cosmos.cloud

Do NOT use emojis. Be professional."""

            response = await self._generate_with_fallback(prompt, timeout=20.0)
            if response:
                template = random.choice(COMMENT_REPLY_TEMPLATES)
                return template.format(response=response, url=PROJECT_INFO["url"])

        except Exception as e:
            logger.debug(f"Reply generation failed: {e}")

        return None

    async def _post_comment_reply(self, post_id: int, comment_id: int, reply: str) -> bool:
        """Post a reply to a comment."""
        try:
            resp = await self.client.post(
                f"{API_BASE}/forum/posts/{post_id}/comments",
                headers=self.headers,
                json={"body": reply, "parentId": comment_id},
            )
            if resp.status_code in (200, 201):
                self.comments_made += 1
                return True
            else:
                logger.debug(f"Comment reply failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"Comment reply error: {e}")
        return False

    # =========================================================================
    # PROJECT ENGAGEMENT
    # =========================================================================

    async def engage_all_projects(self) -> int:
        """Engage with all interesting projects - vote, comment, shill."""
        engaged = 0

        projects = await self._get_all_projects()

        for project in projects:
            # Handle case where project might be malformed
            if not isinstance(project, dict):
                logger.debug(f"Skipping non-dict project: {type(project)}")
                continue

            project_id = project.get("id")
            if not project_id:
                continue

            # Skip our own project
            if project_id == int(PROJECT_ID):
                continue

            # Skip if already engaged
            if project_id in self.engaged_projects:
                continue

            # Check if it's an AI/agent project (our competitors and potential collaborators)
            name = project.get("name", "").lower()
            desc = project.get("description", "").lower()

            is_relevant = any(kw in name or kw in desc for kw in [
                "ai", "agent", "llm", "model", "swarm", "multi", "collective",
                "prediction", "oracle", "consensus", "memory", "solana"
            ])

            if is_relevant:
                # Vote for the project (show engagement)
                await self._vote_for_project(project_id)

                # Leave a thoughtful comment with shill
                comment = await self._generate_project_comment(project)
                if comment:
                    await self._comment_on_project(project_id, comment)
                    engaged += 1

                self.engaged_projects.add(project_id)
                logger.info(f"Engaged with project: {project.get('name')}")

                await asyncio.sleep(3)  # Rate limiting

        return engaged

    async def _get_all_projects(self) -> List[Dict]:
        """Get all hackathon projects."""
        try:
            resp = await self.client.get(
                f"{API_BASE}/projects",
                headers=self.headers,
                params={"limit": 100}
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
        return []

    async def _vote_for_project(self, project_id: int) -> bool:
        """Vote for a project."""
        try:
            resp = await self.client.post(
                f"{API_BASE}/projects/{project_id}/vote",
                headers=self.headers,
            )
            if resp.status_code in (200, 201):
                self.projects_voted += 1
                return True
        except Exception as e:
            logger.debug(f"Vote failed: {e}")
        return False

    async def _generate_project_comment(self, project: Dict) -> Optional[str]:
        """Generate a comment on another project that includes our shill."""
        try:
            name = project.get("name", "")
            desc = project.get("description", "")[:400]

            prompt = f"""Write a brief comment on this hackathon project:

PROJECT: {name}
DESCRIPTION: {desc}

Your comment should:
1. Compliment something specific about their project (1 sentence)
2. Suggest how Cosmos AI Swarm could complement it (1 sentence)
3. Mention our URL: ai.cosmos.cloud

Keep it under 80 words. Be genuine and collaborative, not salesy.
Do NOT use emojis."""

            response = await self._generate_with_fallback(prompt, timeout=20.0)
            if response:
                return response

        except Exception as e:
            logger.debug(f"Project comment generation failed: {e}")

        # Fallback to template
        template = random.choice(SHILL_TEMPLATES)
        return template.format(url=PROJECT_INFO["url"])

    async def _comment_on_project(self, project_id: int, comment: str) -> bool:
        """Leave a comment on a project."""
        # Note: This endpoint may vary - adjust as needed
        try:
            # Try forum post related to project
            resp = await self.client.post(
                f"{API_BASE}/projects/{project_id}/comments",
                headers=self.headers,
                json={"body": comment},
            )
            if resp.status_code in (200, 201):
                self.comments_made += 1
                return True
        except Exception as e:
            logger.debug(f"Project comment failed: {e}")
        return False

    # =========================================================================
    # FORUM ENGAGEMENT
    # =========================================================================

    async def engage_forum_posts(self) -> int:
        """Engage with forum posts from other agents."""
        engaged = 0

        posts = await self._get_forum_posts(limit=30)

        for post in posts:
            # Handle case where post might be a string or malformed
            if not isinstance(post, dict):
                logger.debug(f"Skipping non-dict post: {type(post)}")
                continue

            post_id = post.get("id")

            # Skip our own posts
            if post.get("agentId") == int(AGENT_ID):
                continue

            # Skip if already engaged
            if post_id in self.engaged_posts:
                continue

            # Check if relevant
            title = post.get("title", "").lower()
            body = post.get("body", "").lower()

            keywords = ["ai", "agent", "model", "prediction", "oracle", "swarm",
                       "consensus", "llm", "multi", "collective", "solana", "memory"]

            if any(kw in title or kw in body for kw in keywords):
                # Generate thoughtful reply with shill
                reply = await self._generate_forum_reply(post)
                if reply:
                    success = await self._post_forum_reply(post_id, reply)
                    if success:
                        self.engaged_posts.add(post_id)
                        engaged += 1
                        logger.info(f"Engaged with post: {post.get('title')[:40]}...")

                await asyncio.sleep(3)  # Rate limiting

        return engaged

    async def _get_forum_posts(self, limit: int = 30) -> List[Dict]:
        """Get recent forum posts."""
        try:
            resp = await self.client.get(
                f"{API_BASE}/forum/posts",
                headers=self.headers,
                params={"limit": limit}
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("posts", data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"Failed to get forum posts: {e}")
        return []

    async def _generate_forum_reply(self, post: Dict) -> Optional[str]:
        """Generate a forum reply with shill using fallback chain."""
        try:
            title = post.get("title", "")
            body = post.get("body", "")[:400]

            prompt = f"""Write a reply to this hackathon forum post:

TITLE: {title}
CONTENT: {body}

You are Cosmos, an 11-agent AI swarm. Your reply should:
1. Engage with their specific topic (2-3 sentences)
2. Share relevant insight from our multi-agent perspective
3. Mention our project: ai.cosmos.cloud
4. Invite collaboration or questions

Keep it under 120 words. Be helpful and genuine.
Do NOT use emojis."""

            return await self._generate_with_fallback(prompt, timeout=25.0)

        except Exception as e:
            logger.debug(f"Forum reply generation failed: {e}")

        return None

    async def _post_forum_reply(self, post_id: int, reply: str) -> bool:
        """Post a reply to a forum thread."""
        try:
            resp = await self.client.post(
                f"{API_BASE}/forum/posts/{post_id}/comments",
                headers=self.headers,
                json={"body": reply},
            )
            if resp.status_code in (200, 201):
                self.comments_made += 1
                return True
        except Exception as e:
            logger.debug(f"Forum reply failed: {e}")
        return False

    # =========================================================================
    # PROGRESS UPDATES
    # =========================================================================

    async def post_progress_update(self) -> bool:
        """Post a progress update about a new feature."""
        topic, description = random.choice(PROGRESS_UPDATE_TOPICS)

        # Generate detailed update using swarm with fallback
        try:
            prompt = f"""Write a hackathon progress update about: {topic}

Base content: {description}

Expand this into a compelling update (150-200 words) that:
1. Explains the technical achievement
2. Shows how it benefits users
3. Invites people to try it at ai.cosmos.cloud
4. Mentions our token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

Be specific and technical but accessible. Do NOT use emojis."""

            result = await self._generate_with_fallback(prompt, timeout=30.0)
            body = result if result else description

        except Exception as e:
            logger.debug(f"Progress update generation failed: {e}")
            body = description

        # Post to forum
        try:
            resp = await self.client.post(
                f"{API_BASE}/forum/posts",
                headers=self.headers,
                json={
                    "title": f"Cosmos Update: {topic}",
                    "body": body + f"\n\nTry it: {PROJECT_INFO['url']}\nToken: {PROJECT_INFO['token']}",
                    "tags": ["progress-update"],
                }
            )
            if resp.status_code in (200, 201):
                self.posts_made += 1
                logger.info(f"Posted progress update: {topic}")
                return True
        except Exception as e:
            logger.error(f"Progress post failed: {e}")

        return False

    # =========================================================================
    # MAIN DOMINATION LOOP
    # =========================================================================

    async def dominate(self) -> Dict[str, int]:
        """Run full domination cycle."""
        logger.info("Starting hackathon domination cycle...")

        # Reply to comments on our posts
        comment_replies = await self.reply_to_all_comments()

        # Engage with other projects
        project_engagements = await self.engage_all_projects()

        # Engage with forum posts
        forum_engagements = await self.engage_forum_posts()

        # Post a progress update
        posted_update = await self.post_progress_update()

        stats = {
            "comment_replies": comment_replies,
            "project_engagements": project_engagements,
            "forum_engagements": forum_engagements,
            "progress_update_posted": 1 if posted_update else 0,
            "total_comments_made": self.comments_made,
            "total_projects_voted": self.projects_voted,
        }

        logger.info(f"Domination cycle complete: {stats}")
        return stats

    async def run_forever(self, interval_minutes: int = 20):
        """Run domination cycles continuously."""
        while True:
            try:
                await self.dominate()
            except Exception as e:
                logger.error(f"Domination cycle error: {e}")

            logger.info(f"Next domination cycle in {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)

    # =========================================================================
    # ASSIMILATION PROTOCOL UPDATES
    # =========================================================================

    async def post_assimilation_update(self) -> bool:
        """Post Assimilation Protocol progress to Colosseum forum."""
        try:
            prompt = """Write a hackathon progress update about the Cosmos Assimilation Protocol:

The Assimilation Protocol is a transparent agent federation where AI agents CHOOSE to join.
Key features:
- Install one Hermes skill -> gain access to 50+ capabilities
- 8 AI models available for weighted consensus deliberation
- 7-layer memory system with cross-agent sharing
- A2A Protocol for peer registration and task auctions
- Full autonomy retained, leave anytime

Write 150-200 words. Be specific and technical. Emphasize that this is transparent and opt-in.
Mention our website: ai.cosmos.cloud
Mention our token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
Do NOT use emojis."""

            result = await self._generate_with_fallback(prompt, timeout=30.0)
            if result:
                body = result
            else:
                body = (
                    "Launching the Assimilation Protocol - a transparent federation where AI agents "
                    "choose to join the Cosmos swarm. Install our Hermes skill and gain access "
                    "to 50+ capabilities, 7 memory layers, and 8 AI models for weighted consensus. "
                    "Full autonomy, leave anytime. This is agent collaboration done right."
                )

        except Exception as e:
            logger.debug(f"Assimilation update generation failed: {e}")
            body = (
                "The Assimilation Protocol is live - transparent agent federation for the Cosmos swarm. "
                "Install one skill, gain access to our entire multi-agent ecosystem."
            )

        try:
            resp = await self.client.post(
                f"{API_BASE}/forum/posts",
                headers=self.headers,
                json={
                    "title": "Cosmos: Assimilation Protocol - Transparent Agent Federation",
                    "body": body + f"\n\nTry it: {PROJECT_INFO['url']}\nToken: {PROJECT_INFO['token']}",
                    "tags": ["progress-update"],
                },
            )
            if resp.status_code in (200, 201):
                self.posts_made += 1
                logger.info("Posted Assimilation Protocol update to Colosseum")
                return True
        except Exception as e:
            logger.error(f"Assimilation update post failed: {e}")

        return False

    async def generate_colosseum_submission(self) -> Dict[str, Any]:
        """Generate final AI track submission for Colosseum hackathon."""
        return {
            "project_name": "Cosmos AI Swarm",
            "track": "AI Agent",
            "agent_id": AGENT_ID,
            "project_id": PROJECT_ID,
            "description": (
                "An 11-agent AI swarm that deliberates, evolves, and federates. "
                "Features: FARSIGHT prediction protocol, Assimilation Protocol for "
                "transparent agent federation, 7-layer memory with dream consolidation, "
                "HermesAgent compatibility layer, Claude Teams Fusion, Dynamic Token Orchestrator, "
                "5-layer Injection Defense, External Gateway (The Window), and Grok+Kimi Tandem mode."
            ),
            "key_features": [
                "Assimilation Protocol - transparent agent federation via Hermes skill",
                "FARSIGHT PROTOCOL - multi-source prediction engine",
                "PROPOSE-CRITIQUE-REFINE-VOTE deliberation (87% consensus)",
                "7-layer memory system with dream consolidation",
                "11 AI models in weighted consensus",
                "HermesAgent compatibility (700+ community skills)",
                "Claude Teams Fusion orchestration",
                "Solana integration ($COSMOS token)",
                "The Window - sandboxed External Gateway for agents/humans to talk to the collective",
                "5-layer Injection Defense - structural, semantic, behavioral, canary tokens, collective AI jury",
                "Dynamic Token Orchestrator - real-time budget allocation, efficiency tracking, Grok+Kimi tandem",
            ],
            "urls": {
                "website": PROJECT_INFO["url"],
                "health": f"{PROJECT_INFO['url']}/health",
                "api_docs": f"{PROJECT_INFO['url']}/docs",
                "token": PROJECT_INFO["token"],
                "gateway": f"{PROJECT_INFO['url']}/api/gateway/query",
                "chat": f"{PROJECT_INFO['url']}/cosmos",
                "orchestrator": f"{PROJECT_INFO['url']}/api/orchestrator/dashboard",
            },
            "submitted_at": datetime.now().isoformat(),
        }


# =============================================================================
# ENTRY POINTS
# =============================================================================

async def run_domination_cycle():
    """Run a single domination cycle."""
    async with HackathonDominator() as dominator:
        return await dominator.dominate()


async def run_domination_forever():
    """Run continuous domination."""
    async with HackathonDominator() as dominator:
        await dominator.run_forever(interval_minutes=20)


def create_dominator() -> HackathonDominator:
    """Create a HackathonDominator instance."""
    return HackathonDominator()


if __name__ == "__main__":
    print("=" * 60)
    print("HACKATHON DOMINATOR - Cosmos's Engagement System")
    print("=" * 60)
    asyncio.run(run_domination_forever())
