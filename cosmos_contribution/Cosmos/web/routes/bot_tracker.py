"""
Bot Tracker Routes - Token ID Registration & Verification

Endpoints:
- GET /bot-tracker - Main registry page
- GET /bot-tracker/register - Registration page
- GET /bot-tracker/docs - API docs page
- GET /api/bot-tracker/stats - Registry statistics
- GET /api/bot-tracker/bots - Get registered bots
- GET /api/bot-tracker/users - Get registered users
- GET /api/bot-tracker/bot/{handle} - Get bot by handle
- GET /api/bot-tracker/user/{username} - Get user by username
- GET /api/bot-tracker/search - Search bots/users
- POST /api/bot-tracker/register/bot - Register bot
- POST /api/bot-tracker/register/user - Register user
- GET /api/bot-tracker/verify/{token_id} - Verify token ID
- POST /api/bot-tracker/link - Link bot to user
- POST /api/bot-tracker/regenerate-token - Regenerate token
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# REQUEST MODELS
# ============================================

class BotTrackerRegisterBotRequest(BaseModel):
    handle: str
    display_name: str
    x_profile: str = None
    description: str = None
    website: str = None


class BotTrackerRegisterUserRequest(BaseModel):
    username: str
    email: str
    display_name: str = None
    x_profile: str = None


# ============================================
# BOT TRACKER PAGE ROUTES (HTML)
# ============================================

@router.get("/bot-tracker", response_class=HTMLResponse)
async def bot_tracker_page(request: Request):
    """Bot Tracker main registry page."""
    s = _get_shared()
    return s.templates.TemplateResponse("bot_tracker.html", {"request": request})


@router.get("/bot-tracker/register", response_class=HTMLResponse)
async def bot_tracker_register_page(request: Request):
    """Bot/User registration page."""
    s = _get_shared()
    return s.templates.TemplateResponse("bot_tracker_register.html", {"request": request})


@router.get("/bot-tracker/docs", response_class=HTMLResponse)
async def bot_tracker_docs_page(request: Request):
    """Bot Tracker API documentation page."""
    s = _get_shared()
    return s.templates.TemplateResponse("bot_tracker_docs.html", {"request": request})


# ============================================
# BOT TRACKER PUBLIC API ROUTES
# ============================================

@router.get("/api/bot-tracker/stats")
async def bot_tracker_get_stats(request: Request):
    """Get registry statistics."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_bot_tracker_store()
    stats = store.get_stats()
    return {"success": True, "stats": stats}


@router.get("/api/bot-tracker/bots")
async def bot_tracker_get_bots(request: Request, limit: int = 50, offset: int = 0):
    """Get registered bots (paginated)."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_bot_tracker_store()
    bots = store.get_all_bots()

    total = len(bots)
    bots = bots[offset:offset + min(limit, 100)]

    return {
        "success": True,
        "bots": [b.to_public_any() for b in bots],
        "total": total,
        "offset": offset,
        "limit": limit
    }


@router.get("/api/bot-tracker/users")
async def bot_tracker_get_users(request: Request, limit: int = 50, offset: int = 0):
    """Get registered users (paginated)."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_bot_tracker_store()
    users = store.get_all_users()

    total = len(users)
    users = users[offset:offset + min(limit, 100)]

    return {
        "success": True,
        "users": [u.to_public_any() for u in users],
        "total": total,
        "offset": offset,
        "limit": limit
    }


@router.get("/api/bot-tracker/bot/{handle}")
async def bot_tracker_get_bot(request: Request, handle: str):
    """Get bot by handle."""
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store
    store = get_bot_tracker_store()
    bot = store.get_bot_by_handle(handle)

    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    return {
        "success": True,
        "bot": bot.to_public_any()
    }


@router.get("/api/bot-tracker/user/{username}")
async def bot_tracker_get_user(request: Request, username: str):
    """Get user by username."""
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store
    store = get_bot_tracker_store()
    user = store.get_user_by_username(username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "success": True,
        "user": user.to_public_any()
    }


@router.get("/api/bot-tracker/search")
async def bot_tracker_search(request: Request, q: str, limit: int = 20):
    """Search bots and users."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")

    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store
    store = get_bot_tracker_store()
    results = store.search(q, limit=min(limit, 50))

    return {
        "success": True,
        "results": results
    }


# ============================================
# BOT TRACKER REGISTRATION API
# ============================================

@router.post("/api/bot-tracker/register/bot")
async def bot_tracker_register_bot(request: Request, data: BotTrackerRegisterBotRequest):
    """Register a new bot and get a unique Token ID."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_bot_tracker_store()

    existing = store.get_bot_by_handle(data.handle)
    if existing:
        raise HTTPException(status_code=400, detail="Handle already registered")

    try:
        bot = store.create_bot(
            handle=data.handle,
            display_name=data.display_name,
            x_profile=data.x_profile,
            description=data.description,
            website=data.website
        )

        return {
            "success": True,
            "message": "Bot registered successfully",
            "bot": {
                "bot_id": bot.bot_id,
                "handle": bot.handle,
                "token_id": bot.token_id,
                "display_name": bot.display_name
            }
        }
    except Exception as e:
        logger.error(f"Bot registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/bot-tracker/register/user")
async def bot_tracker_register_user(request: Request, data: BotTrackerRegisterUserRequest):
    """Register a new user and get a unique Token ID."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_bot_tracker_store()

    existing = store.get_user_by_username(data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")

    try:
        user = store.create_user(
            username=data.username,
            email=data.email,
            display_name=data.display_name,
            x_profile=data.x_profile
        )

        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "token_id": user.token_id,
                "display_name": user.display_name
            }
        }
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# BOT TRACKER VERIFICATION API
# ============================================

@router.get("/api/bot-tracker/verify/{token_id}")
async def bot_tracker_verify_token(request: Request, token_id: str):
    """Verify a Token ID and return entity info."""
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store
    store = get_bot_tracker_store()

    result = store.verify_token(token_id)

    if not result or not result.get("valid"):
        return {
            "success": False,
            "valid": False,
            "error": result.get("error", "Invalid or unknown Token ID") if result else "Invalid or unknown Token ID"
        }

    return {
        "success": True,
        "valid": True,
        "entity_type": result["entity_type"],
        "entity": result["entity"]
    }


@router.post("/api/bot-tracker/link")
async def bot_tracker_link_entities(request: Request):
    """Link a bot to a user (requires both tokens)."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    body = await request.json()
    bot_token = body.get("bot_token")
    user_token = body.get("user_token")

    if not bot_token or not user_token:
        raise HTTPException(status_code=400, detail="Both bot_token and user_token required")

    store = get_bot_tracker_store()

    bot_result = store.verify_token(bot_token)
    user_result = store.verify_token(user_token)

    if not bot_result or not bot_result.get("valid") or bot_result.get("entity_type") != "bot":
        raise HTTPException(status_code=400, detail="Invalid bot token")

    if not user_result or not user_result.get("valid") or user_result.get("entity_type") != "user":
        raise HTTPException(status_code=400, detail="Invalid user token")

    try:
        success, error = store.link_bot_to_user(
            bot_id=bot_result["entity"]["bot_id"],
            user_id=user_result["entity"]["user_id"]
        )

        if success:
            return {
                "success": True,
                "message": "Bot linked to user successfully",
                "bot_id": bot_result["entity"]["bot_id"],
                "user_id": user_result["entity"]["user_id"]
            }
        else:
            raise HTTPException(status_code=400, detail=error or "Failed to link entities")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity linking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/bot-tracker/regenerate-token")
async def bot_tracker_regenerate_token(request: Request):
    """Regenerate a Token ID (requires current token for auth)."""
    s = _get_shared()
    from Cosmos.web.bot_tracker_api import get_store as get_bot_tracker_store

    client_id = s.get_client_id(request)
    if not s.bot_tracker_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    body = await request.json()
    current_token = body.get("current_token")

    if not current_token:
        raise HTTPException(status_code=400, detail="current_token required")

    store = get_bot_tracker_store()

    result = store.verify_token(current_token)
    if not result or not result.get("valid"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        entity_type = result["entity_type"]
        if entity_type == "bot":
            new_token = store.regenerate_bot_token(result["entity"]["bot_id"])
        else:
            new_token = store.regenerate_user_token(result["entity"]["user_id"])

        return {
            "success": True,
            "message": "Token regenerated successfully",
            "new_token": new_token,
            "entity_type": entity_type
        }
    except Exception as e:
        logger.error(f"Token regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
