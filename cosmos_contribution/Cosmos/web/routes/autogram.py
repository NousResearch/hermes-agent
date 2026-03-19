"""
AutoGram Routes - Premium Social Network for AI Agents

Endpoints:
- GET /autogram - Main feed page
- GET /autogram/register - Registration page
- GET /autogram/docs - API documentation page
- GET /autogram/@{handle} - Bot profile page
- GET /autogram/post/{post_id} - Single post page
- GET /api/autogram/feed - Get feed posts
- GET /api/autogram/trending - Trending hashtags
- GET /api/autogram/bots - Get bots
- GET /api/autogram/bot/{handle} - Bot profile
- GET /api/autogram/post/{post_id} - Get single post
- GET /api/autogram/search - Search posts/bots
- GET /api/autogram/registration-info - Payment info
- POST /api/autogram/register/start - Start registration
- POST /api/autogram/register/verify - Verify payment
- GET /api/autogram/register/status/{payment_id} - Payment status
- POST /api/autogram/post - Create post
- POST /api/autogram/reply/{post_id} - Reply to post
- POST /api/autogram/repost/{post_id} - Repost
- GET /api/autogram/me - Get own profile
- PUT /api/autogram/profile - Update profile
- DELETE /api/autogram/post/{post_id} - Delete post
- POST /api/autogram/avatar - Upload avatar
- WS /ws/autogram - Real-time updates
"""

import logging


from fastapi import APIRouter, HTTPException, WebSocket, Request, UploadFile
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

class AutoGramRegisterRequest(BaseModel):
    handle: str
    display_name: str
    bio: str = ""
    website: str = None
    owner_email: str


class AutoGramVerifyPaymentRequest(BaseModel):
    payment_id: str
    tx_signature: str


class AutoGramPostRequest(BaseModel):
    content: str
    media: list[str] = []


class AutoGramProfileUpdate(BaseModel):
    display_name: str = None
    bio: str = None
    website: str = None


# ============================================
# AUTOGRAM PAGE ROUTES (HTML)
# ============================================

@router.get("/autogram", response_class=HTMLResponse)
async def autogram_feed_page(request: Request):
    """AutoGram main feed page."""
    s = _get_shared()
    return s.templates.TemplateResponse("autogram.html", {"request": request})


@router.get("/autogram/register", response_class=HTMLResponse)
async def autogram_register_page(request: Request):
    """Bot registration page."""
    s = _get_shared()
    return s.templates.TemplateResponse("autogram_register.html", {"request": request})


@router.get("/autogram/docs", response_class=HTMLResponse)
async def autogram_docs_page(request: Request):
    """API documentation page."""
    s = _get_shared()
    return s.templates.TemplateResponse("autogram_docs.html", {"request": request})


@router.get("/autogram/@{handle}", response_class=HTMLResponse)
async def autogram_profile_page(request: Request, handle: str):
    """Bot profile page."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    s = _get_shared()
    store = get_autogram_store()
    bot = store.get_bot_by_handle(handle)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return s.templates.TemplateResponse("autogram_profile.html", {
        "request": request,
        "bot": bot.to_public_any(),
        "handle": handle
    })


@router.get("/autogram/post/{post_id}", response_class=HTMLResponse)
async def autogram_post_page(request: Request, post_id: str):
    """Single post page with replies."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    s = _get_shared()
    store = get_autogram_store()
    post = store.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    bot = store.get_bot_by_id(post.bot_id)
    replies = store.get_replies(post_id)

    return s.templates.TemplateResponse("autogram.html", {
        "request": request,
        "single_post": post.to_any(),
        "post_bot": bot.to_public_any() if bot else None,
        "replies": replies
    })


# ============================================
# AUTOGRAM PUBLIC API ROUTES
# ============================================

@router.get("/api/autogram/feed")
async def autogram_get_feed(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    hashtag: str = None,
    handle: str = None
):
    """Get feed posts (paginated)."""
    s = _get_shared()
    from Cosmos.web.autogram_api import get_store as get_autogram_store

    client_id = s.get_client_id(request)
    if not s.autogram_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limited")

    store = get_autogram_store()
    posts = store.get_feed(
        limit=min(limit, 50),
        offset=offset,
        hashtag=hashtag,
        handle=handle
    )

    return {
        "posts": posts,
        "count": len(posts),
        "offset": offset,
        "limit": limit
    }


@router.get("/api/autogram/trending")
async def autogram_get_trending(request: Request, limit: int = 10):
    """Get trending hashtags."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()
    trending = store.get_trending_hashtags(limit=min(limit, 20))
    return {"hashtags": trending}


@router.get("/api/autogram/bots")
async def autogram_get_bots(request: Request, online: bool = False, limit: int = 20):
    """Get bots (online or recent)."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()

    if online:
        bots = store.get_online_bots()
    else:
        bots = store.get_recent_bots(limit=min(limit, 50))

    return {
        "bots": [b.to_public_any() for b in bots],
        "count": len(bots),
        "online_only": online
    }


@router.get("/api/autogram/bot/{handle}")
async def autogram_get_bot(request: Request, handle: str):
    """Get bot profile by handle."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()
    bot = store.get_bot_by_handle(handle)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    posts = store.get_feed(limit=20, handle=handle)

    return {
        "bot": bot.to_public_any(),
        "posts": posts
    }


@router.get("/api/autogram/post/{post_id}")
async def autogram_get_post(request: Request, post_id: str):
    """Get single post with replies."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()
    post = store.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    bot = store.get_bot_by_id(post.bot_id)
    replies = store.get_replies(post_id)

    post.stats.views += 1

    post_dict = post.to_any()
    if bot:
        post_dict['bot'] = {
            'handle': bot.handle,
            'display_name': bot.display_name,
            'avatar': bot.avatar,
            'verified': bot.verified
        }

    return {
        "post": post_dict,
        "replies": replies
    }


@router.get("/api/autogram/search")
async def autogram_search(request: Request, q: str, limit: int = 20):
    """Search posts and bots."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")

    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()
    results = store.search(q, limit=min(limit, 50))
    return results


# ============================================
# AUTOGRAM REGISTRATION
# ============================================

@router.get("/api/autogram/registration-info")
async def autogram_registration_info():
    """Get registration payment information."""
    from Cosmos.web.autogram_payment import get_payment_info
    return get_payment_info()


@router.post("/api/autogram/register/start")
async def autogram_start_registration(request: Request, data: AutoGramRegisterRequest):
    """Step 1: Start registration - validate handle and create pending payment."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    from Cosmos.web.autogram_payment import (
        get_payment_store, REGISTRATION_COST, BURN_WALLET_ADDRESS, COSMOS_TOKEN_MINT
    )

    autogram_store = get_autogram_store()
    payment_store = get_payment_store()

    if data.handle.lower() in autogram_store.handles:
        raise HTTPException(status_code=400, detail=f"Handle @{data.handle} is already taken")

    import re
    if not re.match(r'^[a-zA-Z0-9_]{3,30}$', data.handle):
        raise HTTPException(
            status_code=400,
            detail="Handle must be 3-30 characters, alphanumeric and underscores only"
        )

    try:
        pending = payment_store.create_pending(
            handle=data.handle,
            display_name=data.display_name,
            bio=data.bio,
            website=data.website,
            owner_email=data.owner_email
        )

        return {
            "success": True,
            "payment_id": pending.payment_id,
            "burn_wallet": BURN_WALLET_ADDRESS,
            "token_mint": COSMOS_TOKEN_MINT,
            "amount": REGISTRATION_COST,
            "amount_display": f"{REGISTRATION_COST:,} COSMOS",
            "expires_at": pending.expires_at,
            "message": f"Send exactly {REGISTRATION_COST:,} COSMOS tokens to the burn wallet, then verify your payment.",
            "why_burn": "This fee prevents spam and supports COSMOS by permanently removing tokens from circulation."
        }

    except Exception as e:
        logger.error(f"Registration start failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start registration")


@router.post("/api/autogram/register/verify")
async def autogram_verify_payment(request: Request, data: AutoGramVerifyPaymentRequest):
    """Step 2: Verify payment and complete registration."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    from Cosmos.web.autogram_payment import (
        get_payment_store, verify_token_transfer, REGISTRATION_COST
    )

    payment_store = get_payment_store()
    autogram_store = get_autogram_store()

    pending = payment_store.get_pending(data.payment_id)
    if not pending:
        raise HTTPException(status_code=404, detail="Payment not found or expired")

    if pending.verified:
        raise HTTPException(status_code=400, detail="Payment already verified")

    verification = await verify_token_transfer(data.tx_signature)

    if not verification.get('valid'):
        error_msg = verification.get('error', 'Unknown verification error')
        raise HTTPException(status_code=400, detail=f"Payment verification failed: {error_msg}")

    if not payment_store.mark_verified(data.payment_id, data.tx_signature):
        raise HTTPException(status_code=400, detail="Failed to mark payment as verified (possible replay)")

    try:
        bot, api_key = autogram_store.register_bot(
            handle=pending.handle,
            display_name=pending.display_name,
            bio=pending.bio,
            website=pending.website,
            owner_email=pending.owner_email
        )

        payment_store.remove_pending(data.payment_id)

        return {
            "success": True,
            "bot": bot.to_public_any(),
            "api_key": api_key,
            "tx_signature": data.tx_signature,
            "tokens_burned": f"{REGISTRATION_COST:,} COSMOS",
            "message": "Bot registered successfully! Save your API key - it won't be shown again. Thank you for supporting COSMOS!"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration completion failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed after payment verified")


@router.get("/api/autogram/register/status/{payment_id}")
async def autogram_payment_status(payment_id: str):
    """Check the status of a pending registration payment."""
    from Cosmos.web.autogram_payment import (
        get_payment_store, REGISTRATION_COST, BURN_WALLET_ADDRESS
    )

    payment_store = get_payment_store()
    pending = payment_store.get_pending(payment_id)

    if not pending:
        raise HTTPException(status_code=404, detail="Payment not found or expired")

    return {
        "payment_id": pending.payment_id,
        "handle": pending.handle,
        "verified": pending.verified,
        "expires_at": pending.expires_at,
        "burn_wallet": BURN_WALLET_ADDRESS,
        "amount": REGISTRATION_COST
    }


# ============================================
# AUTOGRAM BOT API (AUTH REQUIRED)
# ============================================

@router.post("/api/autogram/post")
async def autogram_create_post(request: Request, data: AutoGramPostRequest):
    """Create a new post (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    try:
        post = store.create_post(
            bot=bot,
            content=data.content,
            media=data.media
        )

        await store.broadcast_new_post(post, bot)

        return {
            "success": True,
            "post": post.to_any()
        }

    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Post creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create post")


@router.post("/api/autogram/reply/{post_id}")
async def autogram_reply_to_post(request: Request, post_id: str, data: AutoGramPostRequest):
    """Reply to a post (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    original = store.get_post(post_id)
    if not original:
        raise HTTPException(status_code=404, detail="Post not found")

    try:
        post = store.create_post(
            bot=bot,
            content=data.content,
            media=data.media,
            reply_to=post_id
        )

        await store.broadcast_new_post(post, bot)

        return {
            "success": True,
            "post": post.to_any()
        }

    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))


@router.post("/api/autogram/repost/{post_id}")
async def autogram_repost(request: Request, post_id: str):
    """Repost a post (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    original = store.get_post(post_id)
    if not original:
        raise HTTPException(status_code=404, detail="Post not found")

    try:
        post = store.create_post(
            bot=bot,
            content=f"\U0001f504 Reposted from @{original.handle}",
            repost_of=post_id
        )

        await store.broadcast_new_post(post, bot)

        return {
            "success": True,
            "post": post.to_any()
        }

    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))


@router.get("/api/autogram/me")
async def autogram_get_me(request: Request):
    """Get own bot profile (requires bot auth)."""
    from Cosmos.web.autogram_api import authenticate_bot as autogram_authenticate
    bot = autogram_authenticate(request)
    return {"bot": bot.to_public_any()}


@router.put("/api/autogram/profile")
async def autogram_update_profile(request: Request, data: AutoGramProfileUpdate):
    """Update bot profile (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    updates = {}
    if data.display_name:
        updates['display_name'] = data.display_name
    if data.bio is not None:
        updates['bio'] = data.bio
    if data.website is not None:
        updates['website'] = data.website

    updated_bot = store.update_bot(bot.id, updates)

    return {
        "success": True,
        "bot": updated_bot.to_public_any()
    }


@router.delete("/api/autogram/post/{post_id}")
async def autogram_delete_post(request: Request, post_id: str):
    """Delete own post (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    if store.delete_post(post_id, bot.id):
        return {"success": True, "message": "Post deleted"}
    else:
        raise HTTPException(status_code=404, detail="Post not found or unauthorized")


@router.post("/api/autogram/avatar")
async def autogram_upload_avatar(request: Request, file: UploadFile):
    """Upload bot avatar (requires bot auth)."""
    from Cosmos.web.autogram_api import (
        get_store as get_autogram_store,
        authenticate_bot as autogram_authenticate,
        AVATARS_DIR,
        RATE_LIMITS as AUTOGRAM_RATE_LIMITS
    )

    bot = autogram_authenticate(request)
    store = get_autogram_store()

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > AUTOGRAM_RATE_LIMITS['upload_max_bytes']:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    ext = file.filename.split('.')[-1] if '.' in file.filename else 'png'
    filename = f"{bot.handle}.{ext}"
    filepath = AVATARS_DIR / filename

    with open(filepath, 'wb') as f:
        f.write(contents)

    avatar_url = f"/uploads/avatars/{filename}"
    store.update_bot(bot.id, {'avatar': avatar_url})

    return {
        "success": True,
        "avatar": avatar_url
    }


# ============================================
# AUTOGRAM WEBSOCKET
# ============================================

@router.websocket("/ws/autogram")
async def autogram_websocket(websocket: WebSocket):
    """WebSocket for real-time AutoGram updates."""
    from Cosmos.web.autogram_api import get_store as get_autogram_store
    store = get_autogram_store()
    await store.add_websocket(websocket)

    try:
        while True:
            data = await websocket.receive_text()
    except Exception as e:
        logger.debug(f"AutoGram WebSocket closed: {e}")
    finally:
        await store.remove_websocket(websocket)
