"""KENSEI Content Engine -- Configuration.

Merged: V1 paths/infra + V2 brand registry, visual strategies,
platform formats, image providers, and budget.
"""
import os
from pathlib import Path

# Resolve deployment root relative to this file so the engine works
# from any checkout location, not just /home/kensei/repos/KenseiAgent.
# Falls back to the hardcoded path only if the relative resolution
# fails (e.g. running from a frozen binary).
BASE_DIR = Path(os.environ.get("CONTENT_ENGINE_ROOT", Path(__file__).resolve().parent)).resolve()
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = Path(os.environ.get("CONTENT_ENGINE_DB_PATH", BASE_DIR / "db" / "content_engine.db")).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(DB_PATH.parent).mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CONTENT_CHAT_ID", "-1003922682700")
TELEGRAM_TOPIC_ID = os.getenv("TELEGRAM_CONTENT_TOPIC_ID", "22")

POSTIZ_DB_HOST = os.getenv("POSTIZ_DB_HOST", "postiz-postgres")
POSTIZ_DB_PORT = int(os.getenv("POSTIZ_DB_PORT", "5432"))
POSTIZ_DB_USER = os.getenv("POSTIZ_DB_USER", "postiz-user")
POSTIZ_DB_PASS = os.getenv("POSTIZ_DB_PASS", "postiz-password")
POSTIZ_DB_NAME = os.getenv("POSTIZ_DB_NAME", "postiz-db-local")

FOOTBALL_API_BASE = "https://api.football-data.org/v4"

FONT_DIR = Path("/usr/share/fonts/truetype")


def font_path(name: str, fallback: str = "DejaVuSans.ttf") -> Path:
    candidates = [
        FONT_DIR / "roboto" / f"{name}.ttf",
        FONT_DIR / "inter" / f"{name}.ttf",
        FONT_DIR / "liberation" / f"{name}.ttf",
        FONT_DIR / f"{name}.ttf",
        FONT_DIR / "dejavu" / f"{name}.ttf",
    ]
    for c in candidates:
        if c.exists():
            return c
    fb = FONT_DIR / "dejavu" / fallback
    if fb.exists():
        return fb
    return FONT_DIR / fallback


BRANDS = {
    "matchdaymaestro": {
        "display": "MatchdayMaestro",
        "handle": "@MaestroMatchday",
        "colour": "#FBBF24",
        "bg": "#0F0F1A",
        "accent": "#E11D48",
        "platforms": ["twitter", "instagram", "tiktok"],
        "primary": "gamified",
        "secondary": "fpl_official",
        "name": "MatchdayMaestro",
        "description": "UK gamified live football prediction app",
        "voice_skill": "matchdaymaestro-voice",
        "content_pillars": [
            "Live Predictions In Action",
            "Game Modes (Beyond Predictions)",
            "Friend Battles & Social Competition",
            "Progression & Mastery (RPG)",
            "The Football Beat",
        ],
        "visual_strategy": "screenshot_repurposing",
        "post_mix": {"text_only": 0.3, "text_image": 0.5, "video": 0.2},
        "posts_per_week": 30,
        "org_id": "2645662d-a479-4a6a-91ca-a50a7d29f607",
    },
    "plenishd": {
        "display": "Plenishd",
        "handle": "@PlenishdApp",
        "colour": "#FBBF24",
        "bg": "#2C2A28",
        "accent": "#10B981",
        "platforms": ["twitter", "instagram", "linkedin"],
        "primary": "warm_practical",
        "secondary": "witty",
        "name": "Plenishd",
        "description": "UK voice-first smart kitchen assistant",
        "voice_skill": "plenishd-voice",
        "content_pillars": [
            "The Loop In Action",
            "Smart Money On UK Groceries",
            "Run It Together (Household)",
            "Hands-Free Phone-Free (Voice)",
            "Real Kitchen Life",
        ],
        "visual_strategy": "ai_image",
        "post_mix": {"text_only": 0.3, "text_image": 0.5, "video": 0.2},
        "posts_per_week": 20,
        "org_id": "2645662d-a479-4a6a-91ca-a50a7d29f607",
    },
    "sahil_twitter": {
        "display": "Sahil",
        "handle": "@SahilSaghir",
        "colour": "#C0C0C0",
        "bg": "#0A0A0A",
        "accent": "#8B0000",
        "platforms": ["twitter"],
        "primary": "direct_specific",
        "secondary": "wry",
        "name": "Sahil Saghir",
        "description": "Senior PM / TPM, indie developer, build-in-public",
        "voice_skill": "sahil-twitter-voice",
        "content_pillars": [
            "Agent Build Notes", "Harness Tuning", "Paper Takes", "Radar Finds", "AI Patterns",
            "Build-in-Public", "Wry Observations", "Football (MUFC)",
        ],
        "educational_pillars": ["Agent Build Notes","Harness Tuning","Paper Takes","Radar Finds","AI Patterns"],
        "educational_mix": 0.65,
        "visual_strategy": "screenshot_repurposing",
        "post_mix": {"text_only": 0.6, "text_image": 0.3, "video": 0.1},
        "posts_per_week": 28,
        "org_id": "2645662d-a479-4a6a-91ca-a50a7d29f607",
    },
    "sahil_linkedin": {
        "display": "Sahil",
        "handle": "Sahil Saghir",
        "colour": "#C0C0C0",
        "bg": "#0A0A0A",
        "accent": "#8B0000",
        "platforms": ["linkedin"],
        "primary": "authoritative",
        "secondary": "opinion_having",
        "name": "Sahil Saghir",
        "description": "Senior PM / TPM, indie developer, build-in-public",
        "voice_skill": "sahil-linkedin-voice",
        "content_pillars": [
            "Agentic Systems in Practice", "AI Engineering Notes", "Research to Practice", "Tooling Signals",
            "PM Thought Leadership", "Leadership Insights",
        ],
        "educational_pillars": ["Agentic Systems in Practice","AI Engineering Notes","Research to Practice","Tooling Signals"],
        "educational_mix": 0.65,
        "visual_strategy": "screenshot_repurposing",
        "post_mix": {"text_only": 0.7, "text_image": 0.3, "video": 0.0},
        "posts_per_week": 17,
        "org_id": "2645662d-a479-4a6a-91ca-a50a7d29f607",
    },
    "coachos": {
        "display": "CoachOS",
        "handle": "@CoachOSApp",
        "colour": "#22C55E",
        "bg": "#111827",
        "accent": "#16A34A",
        "platforms": ["twitter", "instagram", "linkedin"],
        "primary": "direct_credible",
        "secondary": "coach_to_coach",
        "name": "CoachOS",
        "description": "Grassroots football coaching operating system",
        "voice_skill": "coachos-voice",
        "content_pillars": [
            "The Sunday Morning Reality",
            "Platform Capability",
            "Coach Development",
            "Platform Philosophy",
            "Community And Beta Movement",
        ],
        "visual_strategy": "ai_image",
        "post_mix": {"text_only": 0.4, "text_image": 0.4, "video": 0.2},
        "posts_per_week": 20,
        "org_id": "2645662d-a479-4a6a-91ca-a50a7d29f607",
    },
}

VISUAL_STRATEGIES = {
    "ai_image": {
        "description": "AI-generated images for brand scenes",
        "tools": ["flux_schnell", "flux_pro"],
        "fallback": "screenshot_repurposing",
        "cost_per_image": 0.003,
    },
    "screenshot_repurposing": {
        "description": "Raw app screenshots transformed into marketing assets",
        "tools": ["pil_templates", "ffmpeg"],
        "fallback": "text_only",
        "cost_per_image": 0,
    },
    "text_only": {
        "description": "Text-only posts, no visual",
        "tools": [],
        "fallback": None,
        "cost_per_image": 0,
    },
}

PLATFORM_FORMATS = {
    "twitter": {
        "max_chars": 280,
        "image_ratio": "1:1",
        "image_size": (1200, 1200),
        "video_max_secs": 140,
        "hashtag_count": 2,
    },
    "instagram": {
        "max_chars": 2200,
        "image_ratio": "4:5",
        "image_size": (1080, 1350),
        "video_max_secs": 60,
        "hashtag_count": 5,
    },
    "tiktok": {
        "max_chars": 2200,
        "image_ratio": "9:16",
        "image_size": (1080, 1920),
        "video_max_secs": 60,
        "hashtag_count": 3,
    },
    "linkedin": {
        "max_chars": 3000,
        "image_ratio": "1.91:1",
        "image_size": (1200, 627),
        "video_max_secs": 600,
        "hashtag_count": 5,
    },
}

IMAGE_PROVIDERS = [
    {
        "name": "fal",
        "base_url": "https://fal.ai/api",
        "model": "fal-ai/flux/schnell",
        "cost_per_image": 0.003,
        "api_key_env": "FAL_API_KEY",
        "priority": 1,
    },
    {
        "name": "replicate_flux_schnell",
        "base_url": "https://api.replicate.com/v1",
        "model": "black-forest-labs/flux-schnell",
        "cost_per_image": 0.003,
        "api_key_env": "REPLICATE_API_TOKEN",
        "priority": 2,
    },
    {
        "name": "together_flux",
        "base_url": "https://api.together.xyz/v1",
        "model": "black-forest-labs/FLUX.1-schnell",
        "cost_per_image": 0.005,
        "api_key_env": "TOGETHER_API_KEY",
        "priority": 3,
    },
    {
        "name": "pollinations",
        "base_url": "https://image.pollinations.ai",
        "model": "flux",
        "cost_per_image": 0,
        "api_key_env": None,
        "priority": 99,
    },
]

MONTHLY_BUDGET_GBP = 10.0
BUDGET_ALLOCATION = {
    "image_generation": 2.0,
    "video_generation": 0.0,
    "text_generation": 0.0,
    "buffer": 8.0,
}

# ── X Articles long-form track (sub-project B, sahil_twitter only) ─────
# Daily long-form X Article. Quality-gated, illustrated, paste-ready.
# Defaults are starting points to tune in the live step; all keys env-overridable
# so a V1 tuning run never needs a code change.
ARTICLE_ENABLED = os.getenv("ARTICLE_ENABLED", "1").strip() not in ("0", "false", "")
ARTICLE_DEEP_DIVE_THRESHOLD = int(os.getenv("ARTICLE_DEEP_DIVE_THRESHOLD", "7"))
ARTICLE_DIGEST_MIN_SIGNALS = int(os.getenv("ARTICLE_DIGEST_MIN_SIGNALS", "4"))
ARTICLE_MIN_WORDS = int(os.getenv("ARTICLE_MIN_WORDS", "900"))
ARTICLE_IMG_DENSITY = os.getenv("ARTICLE_IMG_DENSITY", "per-section").strip()
ARTICLE_DIGEST_WINDOW_DAYS = int(os.getenv("ARTICLE_DIGEST_WINDOW_DAYS", "14"))
ARTICLE_MAX_IMAGES = int(os.getenv("ARTICLE_MAX_IMAGES", "6"))
# Cross-run dedup: a signal that became an article is off-limits for this many
# days so consecutive daily cron runs cannot re-pick the same top signal.
ARTICLE_TOPIC_RECENCY_DAYS = int(os.getenv("ARTICLE_TOPIC_RECENCY_DAYS", "30"))

# ── Reference-anchored imagery (personal brands) ───────────────────────
# Validated 2026-06-16: nano-banana-pro/edit dual-anchor transplant (baoyu
# layout + style/palette exemplar) beats text-to-image decisively. DNA is the
# CRAFT thread (texture/type/density), not a fixed palette; the selector rotates
# a palette system for variety. See brand_imagery_standard.md.
IMAGERY_ANCHORS_DIR = os.path.expanduser(
    os.getenv("CONTENT_IMAGERY_ANCHORS", "~/content-references"))
# Brands routed through the transplant path (others keep the legacy path).
IMAGERY_TRANSPLANT_BRANDS = tuple(
    b.strip() for b in os.getenv(
        "CONTENT_TRANSPLANT_BRANDS", "sahil_twitter,sahil_linkedin").split(",")
    if b.strip())
IMAGERY_EDIT_MODEL = os.getenv("CONTENT_EDIT_MODEL", "fal-ai/nano-banana-pro/edit").strip()
IMAGERY_EDIT_COST_GBP = float(os.getenv("CONTENT_EDIT_COST_GBP", "0.12"))
# Tiered scene models (validated 2026-06-16): infographics + HERO scenes need
# nano-banana-pro (text/quality); default (non-hero) scenes use the ~3x cheaper
# nano-banana non-pro at near-identical quality. qwen was too anchor-literal
# (reproduced refs) and pollinations is textless-only — both rejected for scenes.
IMAGERY_SCENE_MODEL = os.getenv("CONTENT_SCENE_MODEL", "fal-ai/nano-banana/edit").strip()
IMAGERY_SCENE_COST_GBP = float(os.getenv("CONTENT_SCENE_COST_GBP", "0.039"))
IMAGERY_HERO_MODEL = os.getenv("CONTENT_HERO_EDIT_MODEL", "fal-ai/nano-banana-pro/edit").strip()
# How many recent (palette|layout) picks to avoid re-using, per brand.
IMAGERY_ROTATION_MEMORY = int(os.getenv("CONTENT_IMAGERY_ROTATION_MEMORY", "6"))

# Temporarily paused brands — skipped by the social generator. Paused 2026-06-16
# while the personal brands (sahil_twitter/sahil_linkedin) are validated on the
# new imagery system; product brands stay on the legacy path until re-enabled.
# Clear CONTENT_PAUSED_BRANDS (or set to "") to resume them.
CONTENT_PAUSED_BRANDS = frozenset(
    b.strip() for b in os.getenv(
        "CONTENT_PAUSED_BRANDS", "plenishd,coachos,matchdaymaestro").split(",")
    if b.strip())

# ── Free Gemini vision (Google AI Studio key) ─────────────────────────
# Vision INPUT (image -> text) is FREE on the AI Studio tier; image OUTPUT is
# paid-only (validated 2026-06-17 against a live key: every image model returns
# 429 with a zero per-day free quota, and the pricing page lists image output as
# "Not available" on free tier). So generation stays on FAL; we only use the key
# for the two free image-understanding features below. Key is read at call time
# from GEMINI_API_KEY (or GOOGLE_AI_API_KEY); never hardcoded.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash").strip()
GEMINI_VISION_ENABLED = os.getenv("GEMINI_VISION_ENABLED", "1").strip() not in ("0", "false", "")

# Feature 1: screenshot -> topic ingestion. Drop app screenshots / inspiration
# here and the personal-brand topic mix picks them up, captions them via Gemini
# vision, and turns each into a high-priority topic. Consumed files move to a
# processed/ subdir so they are ingested once.
CONTENT_SCREENSHOT_INBOX = os.path.expanduser(
    os.getenv("CONTENT_SCREENSHOT_INBOX", "~/content-inbox/screenshots"))
SCREENSHOT_MAX_PER_RUN = int(os.getenv("CONTENT_SCREENSHOT_MAX", "3"))

# Feature 2: visual-QA of generated heroes. After a transplant render, re-read
# the finished image with Gemini vision and flag garbled text / off-brief /
# wrong palette. Below the min score (0-10) we regenerate once with the QA
# issues appended, then accept best-effort (free check, bounded one retry).
IMAGERY_QA_ENABLED = os.getenv("CONTENT_IMAGERY_QA", "1").strip() not in ("0", "false", "")
IMAGERY_QA_MIN_SCORE = int(os.getenv("CONTENT_IMAGERY_QA_MIN_SCORE", "6"))

# ── SahilBlog content pipeline (blog/ subpackage) ─────────────────────
# Daily long-form blog articles for SahilBlog across 3 streams (AI, PM,
# Builders Log). Published as draft MDX into the Astro repo, gated behind
# human approval. Mirrors the article track shape with per-stream dedup
# via database.topic_usage_log (brand=f"blog_{stream}").
SAHILBLOG_REPO = os.path.expanduser(os.getenv("SAHILBLOG_REPO", "~/repos/SahilBlog"))
BLOG_ENABLED = os.getenv("BLOG_ENABLED", "1").strip() not in ("0", "false", "")
BLOG_STREAMS = tuple(s.strip() for s in os.getenv("BLOG_STREAMS", "ai,pm,builder").split(",") if s.strip())
# Per-section images cap. The hard cap is two sections, so one post can request
# one hero plus at most two section images. P11 only persists a provider-free
# visual plan/provenance manifest; it does not alter generator cost or provider
# behaviour. Any future provider/budget decision belongs to P10.
BLOG_MAX_SECTION_IMAGES = int(os.getenv("BLOG_MAX_SECTION_IMAGES", "2"))
BLOG_TOPIC_RECENCY_DAYS = int(os.getenv("BLOG_TOPIC_RECENCY_DAYS", "45"))

# ── Blog back-population ──────────────────────────────────────────────
# The current legacy blog image generator remains unchanged in P11. P11 adds
# provider-free planning and provenance manifests only; P10 owns any provider
# replacement/runtime proof. Deprecated FAL config (BLOG_IMAGE_MODEL,
# BLOG_IMAGE_COST_GBP) remains removed from the blog path. Backfill budget
# tracking belongs to the separate social content pipeline.
BACKFILL_SPEND_CAP_GBP = float(os.getenv("BACKFILL_SPEND_CAP_GBP", "9.0"))
BACKFILL_LEDGER_PATH = str(Path(os.getenv(
    "BACKFILL_LEDGER_PATH", str(Path(__file__).parent / "output" / "backfill_ledger.json"))))
