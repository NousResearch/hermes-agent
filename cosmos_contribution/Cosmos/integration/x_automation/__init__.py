"""X/Twitter Automation for Cosmos using Puppeteer Stealth"""
from .social_poster import (
    SocialPoster,
    get_social_poster,
    post_task_completion,
    post_progress_update,
)

__all__ = [
    "SocialPoster",
    "get_social_poster",
    "post_task_completion",
    "post_progress_update",
]
