"""
X Engagement Routes

Endpoints:
- POST /api/x/mega-thread - Launch a mega thread (20+ posts with images)
- POST /api/x/mega-thread/custom - Launch a custom topic mega thread
- GET /api/x/mega-thread/status - Get status of running mega thread
- POST /api/x/engagement/trending - Get current trending topics
"""

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()

import asyncio


@router.post("/api/x/mega-thread")
async def launch_mega_thread(request: Request):
    """Launch the quantum/tech mega thread."""
    try:
        from Cosmos.integration.x_automation.x_engagement_poster import get_engagement_poster
        poster = get_engagement_poster()

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        generate_images = body.get("generate_images", True)
        delay = body.get("delay", 3.0)

        # Run in background
        async def _run():
            try:
                thread = await poster.execute(
                    generate_images=generate_images,
                    delay=delay,
                )
                logger.info(f"Mega thread complete: {thread.posted_count} posts, root: {thread.root_tweet_id}")
            except Exception as e:
                logger.error(f"Mega thread failed: {e}")

        asyncio.create_task(_run())

        return {"status": "launched", "message": "Mega thread generation started in background"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/x/mega-thread/custom")
async def launch_custom_mega_thread(request: Request):
    """Launch a custom topic mega thread."""
    try:
        from Cosmos.integration.x_automation.x_engagement_poster import get_engagement_poster
        poster = get_engagement_poster()

        body = await request.json()
        topic = body.get("topic", "Cosmos AI")
        prompt = body.get("prompt", "")
        num_posts = body.get("num_posts", 20)
        generate_images = body.get("generate_images", True)

        async def _run():
            try:
                thread = await poster.execute_custom(
                    topic=topic,
                    content_prompt=prompt,
                    num_posts=num_posts,
                    generate_images=generate_images,
                )
                logger.info(f"Custom mega thread complete: {thread.posted_count} posts")
            except Exception as e:
                logger.error(f"Custom mega thread failed: {e}")

        asyncio.create_task(_run())

        return {"status": "launched", "topic": topic, "target_posts": num_posts}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/x/mega-thread/status")
async def mega_thread_status():
    """Get current mega thread status."""
    try:
        from Cosmos.integration.x_automation.x_engagement_poster import get_engagement_poster
        poster = get_engagement_poster()
        return {
            "status": "ready",
            "sections_available": len(poster.__class__.__mro__),  # placeholder
            "daily_limit": 50,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/x/engagement/trending")
async def get_trending():
    """Fetch current trending topics."""
    try:
        from Cosmos.integration.x_automation.x_engagement_poster import get_engagement_poster
        poster = get_engagement_poster()
        trending = await poster.get_trending_topics()
        return {"trending": trending, "count": len(trending)}
    except Exception as e:
        return {"error": str(e)}
