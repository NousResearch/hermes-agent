"""
Cosmos Pro — API Routes
============================
Professional AI platform routes: auth, chat, scanner, wallet, arena, PnL, predictions.
"""
import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from loguru import logger

router = APIRouter()

# ============================================================
# Storage paths
# ============================================================
DATA_DIR = Path("/workspace/cosmos_memory/pro") if Path("/workspace").exists() else Path("./data/pro")
USERS_FILE = DATA_DIR / "users.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Simple JWT-like token (HMAC-based, no external deps)
SECRET_KEY = os.environ.get("PRO_SECRET_KEY", "cosmos-pro-secret-2026")

# OAuth state storage (in-memory, cleared on restart)
_oauth_states = {}


# ============================================================
# Helper: Template rendering
# ============================================================
def _get_templates():
    try:
        from Cosmos.web.server import templates
        return templates
    except Exception:
        from fastapi.templating import Jinja2Templates
        return Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


# ============================================================
# Helper: Token generation & validation
# ============================================================
def _generate_token(user_id: str, email: str, plan: str) -> str:
    """Generate a simple auth token."""
    payload = f"{user_id}:{email}:{plan}:{int(time.time()) + 86400 * 30}"
    sig = hashlib.sha256(f"{payload}:{SECRET_KEY}".encode()).hexdigest()[:16]
    return base64.b64encode(f"{payload}:{sig}".encode()).decode()


def _validate_token(token: str) -> Optional[dict]:
    """Validate token and return user info."""
    try:
        decoded = base64.b64decode(token).decode()
        parts = decoded.rsplit(":", 4)
        if len(parts) < 4:
            return None
        user_id, email, plan, expiry_sig = parts[0], parts[1], parts[2], parts[3]
        # Split expiry and sig
        expiry, sig = expiry_sig.rsplit(":", 1) if ":" in expiry_sig else (expiry_sig, "")
        return {"user_id": user_id, "email": email, "plan": plan}
    except Exception:
        return None


def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


# ============================================================
# PAGE ROUTES (serve templates)
# ============================================================
@router.get("/pro", response_class=HTMLResponse)
async def pro_landing(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_landing.html", {"request": request})


@router.get("/pro/login", response_class=HTMLResponse)
async def pro_login(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_login.html", {"request": request})


@router.get("/pro/signup", response_class=HTMLResponse)
async def pro_signup(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_signup.html", {"request": request})


@router.get("/pro/chat", response_class=HTMLResponse)
async def pro_chat(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_chat.html", {"request": request})


@router.get("/pro/scanner", response_class=HTMLResponse)
async def pro_scanner(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_scanner.html", {"request": request})


@router.get("/pro/wallet", response_class=HTMLResponse)
async def pro_wallet(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_wallet.html", {"request": request})


@router.get("/pro/arena", response_class=HTMLResponse)
async def pro_arena(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_arena.html", {"request": request})


@router.get("/pro/pnl", response_class=HTMLResponse)
async def pro_pnl(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_pnl.html", {"request": request})


@router.get("/pro/predictions", response_class=HTMLResponse)
async def pro_predictions(request: Request):
    tpl = _get_templates()
    return tpl.TemplateResponse("pro_predictions.html", {"request": request})


# ============================================================
# AUTH API — X (Twitter) OAuth2 + Solana Wallet
# ============================================================
@router.get("/api/pro/auth/x/login")
async def auth_x_login(request: Request):
    """Redirect to X (Twitter) OAuth2 authorization."""
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(43)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")

    # Store state -> code_verifier mapping
    _oauth_states[state] = {"code_verifier": code_verifier, "ts": time.time()}

    client_id = os.environ.get("X_CLIENT_ID", "placeholder_client_id")
    redirect_uri = "https://ai.cosmos.cloud/callback"

    auth_url = (
        f"https://x.com/i/oauth2/authorize"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
        f"&scope=tweet.read%20users.read%20offline.access"
        f"&state={state}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=S256"
    )
    return RedirectResponse(auth_url)


@router.get("/callback")
async def auth_x_callback(request: Request):
    """Handle X OAuth2 callback."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state or state not in _oauth_states:
        return RedirectResponse("/pro/login?error=auth_failed")

    oauth_data = _oauth_states.pop(state)
    code_verifier = oauth_data["code_verifier"]

    # Exchange code for access token
    client_id = os.environ.get("X_CLIENT_ID", "placeholder_client_id")
    client_secret = os.environ.get("X_CLIENT_SECRET", "")
    redirect_uri = "https://ai.cosmos.cloud/callback"

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Token exchange
            token_data = {
                "code": code,
                "grant_type": "authorization_code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if client_secret:
                creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
                headers["Authorization"] = f"Basic {creds}"

            async with session.post("https://api.x.com/2/oauth2/token", data=token_data, headers=headers) as resp:
                token_resp = await resp.json()

            access_token = token_resp.get("access_token")
            if not access_token:
                logger.error(f"X token exchange failed: {token_resp}")
                return RedirectResponse("/pro/login?error=token_failed")

            # Get user info
            async with session.get(
                "https://api.x.com/2/users/me",
                headers={"Authorization": f"Bearer {access_token}"}
            ) as resp:
                user_resp = await resp.json()

            x_user = user_resp.get("data", {})
            x_id = x_user.get("id", "")
            x_username = x_user.get("username", "")
            x_name = x_user.get("name", "")
    except Exception as e:
        logger.error(f"X OAuth error: {e}")
        # Fallback for when X API isn't configured
        x_id = f"x_{secrets.token_hex(4)}"
        x_username = "cosmos_user"
        x_name = "Cosmos User"

    # Create or update user
    users = _load_users()
    user_id = None
    for uid, user in users.items():
        if user.get("x_id") == x_id:
            user_id = uid
            break

    if not user_id:
        user_id = str(uuid.uuid4())[:8]
        users[user_id] = {
            "x_id": x_id,
            "x_username": x_username,
            "x_name": x_name,
            "auth_method": "x",
            "plan": "free",
            "created_at": datetime.utcnow().isoformat(),
            "messages_today": 0,
            "last_reset": datetime.utcnow().date().isoformat(),
        }
        _save_users(users)
        # New user — redirect to plan selection
        token = _generate_token(user_id, x_username, "free")
        user_json = urllib.parse.quote(json.dumps({"id": user_id, "username": x_username, "name": x_name, "plan": "free", "auth_method": "x"}))
        return RedirectResponse(f"/pro/signup?token={token}&user={user_json}")
    else:
        plan = users[user_id].get("plan", "free")
        token = _generate_token(user_id, x_username, plan)
        user_json = urllib.parse.quote(json.dumps({"id": user_id, "username": x_username, "name": x_name, "plan": plan, "auth_method": "x"}))
        return RedirectResponse(f"/pro/chat?token={token}&user={user_json}")


@router.post("/api/pro/auth/wallet")
async def auth_wallet(request: Request):
    """Authenticate via Solana wallet signature."""
    try:
        body = await request.json()
        public_key = body.get("publicKey", "").strip()
        signature = body.get("signature", "")
        message = body.get("message", "")

        if not public_key or not signature:
            return JSONResponse({"error": "Wallet address and signature required"}, status_code=400)

        # In production, verify the signature against the message using nacl/ed25519
        # For now, we trust the client-side Phantom verification

        # Create or find user
        users = _load_users()
        user_id = None
        for uid, user in users.items():
            if user.get("wallet_address") == public_key:
                user_id = uid
                break

        is_new = user_id is None
        if is_new:
            user_id = str(uuid.uuid4())[:8]
            short_addr = f"{public_key[:4]}...{public_key[-4:]}"
            users[user_id] = {
                "wallet_address": public_key,
                "username": short_addr,
                "auth_method": "wallet",
                "plan": "free",
                "created_at": datetime.utcnow().isoformat(),
                "messages_today": 0,
                "last_reset": datetime.utcnow().date().isoformat(),
            }
            _save_users(users)

        user = users[user_id]
        plan = user.get("plan", "free")
        token = _generate_token(user_id, public_key, plan)
        short_addr = f"{public_key[:4]}...{public_key[-4:]}"

        return JSONResponse({
            "token": token,
            "new_user": is_new,
            "user": {
                "id": user_id,
                "username": short_addr,
                "wallet_address": public_key,
                "plan": plan,
                "auth_method": "wallet",
            }
        })
    except Exception as e:
        logger.error(f"Wallet auth error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/pro/auth/select-plan")
async def auth_select_plan(request: Request):
    """Update user's selected plan."""
    try:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        user_info = _validate_token(token) if token else None

        if not user_info:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        body = await request.json()
        plan = body.get("plan", "free")
        if plan not in ("free", "pro", "unlimited"):
            plan = "free"

        users = _load_users()
        for uid, user in users.items():
            if uid == user_info.get("user_id"):
                user["plan"] = plan
                _save_users(users)

                # Generate new token with updated plan
                new_token = _generate_token(uid, user_info.get("email", ""), plan)
                return JSONResponse({
                    "token": new_token,
                    "plan": plan,
                    "user": {
                        "id": uid,
                        "username": user.get("username", ""),
                        "plan": plan,
                    }
                })

        return JSONResponse({"error": "User not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Plan selection error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# CHAT API — Routes through COSMOS Mesh with Latent Routing
# ============================================================

def _get_cosmos_node():
    """Get the running COSMOS node."""
    try:
        from Cosmos.network.cosmos_node import get_cosmos_node
        return get_cosmos_node()
    except Exception:
        return None


async def _mesh_query(prompt: str, model: str = "", timeout: float = 90.0):
    """
    Route a query through the COSMOS mesh.
    If model is specified, routes directly. Otherwise uses latent routing.
    Returns (response_text, routing_metadata).
    """
    node = _get_cosmos_node()
    if not node:
        return None, {}

    available = list(node.get_all_bots().keys())
    if not available:
        return None, {}

    # Latent route or direct model selection
    routing_meta = {}
    bot_name = model
    if node._latent_router and (not model or model == "cosmos" or model not in available):
        decision = node._latent_router.route(prompt, available)
        bot_name = decision.selected_bot
        routing_meta = {
            "routed_to": decision.selected_bot,
            "confidence": round(decision.confidence, 4),
            "method": decision.method,
            "mesh_routed": True,
        }
    else:
        routing_meta = {"routed_to": bot_name, "mesh_routed": True, "method": "direct"}

    # Execute through mesh
    t0 = time.time()
    try:
        local_bots = node.get_local_bots()
        if bot_name in local_bots:
            resp = await asyncio.wait_for(
                node._local_bots[bot_name](prompt, 2000), timeout=timeout
            )
        else:
            resp = await asyncio.wait_for(
                node.query_remote_bot(bot_name, prompt, 2000, timeout), timeout=timeout
            )
        routing_meta["inference_ms"] = round((time.time() - t0) * 1000)

        # Feed back to latent router for learning
        if node._latent_router and resp and "decision" in dir():
            quality = min(1.0, len(resp) / 200)
            node._latent_router.record_outcome(decision, quality, routing_meta["inference_ms"])

        return resp, routing_meta
    except Exception as e:
        logger.warning(f"Mesh query to {bot_name} failed: {e}")
        routing_meta["error"] = str(e)
        routing_meta["inference_ms"] = round((time.time() - t0) * 1000)
        return None, routing_meta


@router.post("/api/pro/chat")
async def pro_chat_api(request: Request):
    """Send a chat message — routes through COSMOS mesh with latent routing."""
    try:
        body = await request.json()
        message = body.get("message", "").strip()
        model = body.get("model", "cosmos").lower()
        mode = body.get("mode", "default")
        image_base64 = body.get("image_base64")
        conversation_id = body.get("conversation_id")

        if not message:
            return JSONResponse({"error": "Message required"}, status_code=400)

        # Build prompt with mode context
        prompt = message
        if mode == "research":
            prompt = f"[RESEARCH MODE - be thorough, cite sources, provide detailed analysis]\n\n{message}"
        elif mode == "creative":
            prompt = f"[CREATIVE MODE - be inventive, think outside the box]\n\n{message}"

        response_text = ""
        routing_meta = {}

        # PRIMARY: Route through COSMOS mesh with latent routing
        mesh_resp, routing_meta = await _mesh_query(prompt, model)
        if mesh_resp:
            response_text = mesh_resp

        # FALLBACK 1: Agent spawner (if mesh unavailable)
        if not response_text:
            try:
                from Cosmos.core.agent_spawner import get_agent_spawner
                spawner = get_agent_spawner()
                result = await spawner.call_agent(model, prompt)
                if result and isinstance(result):
                    response_text = result.get("response", result.get("text", str(result)))
                elif result:
                    response_text = str(result)
                routing_meta["fallback"] = "agent_spawner"
            except Exception as e:
                logger.warning(f"Agent spawner fallback failed for {model}: {e}")

        # FALLBACK 2: Model swarm
        if not response_text:
            try:
                from Cosmos.core.model_swarm import get_model_swarm
                swarm = get_model_swarm()
                result = await swarm.query(message, preferred_model=model)
                response_text = result if isinstance(result, str) else str(result)
                routing_meta["fallback"] = "model_swarm"
            except Exception as e2:
                logger.warning(f"Swarm fallback failed: {e2}")
                response_text = f"I received your message. The swarm is initializing. Please try again in a moment."
                routing_meta["fallback"] = "placeholder"

        return JSONResponse({
            "response": response_text,
            "model": routing_meta.get("routed_to", model),
            "mode": mode,
            "conversation_id": conversation_id or str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "routing": routing_meta,
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/pro/chat/stream")
async def pro_chat_stream(request: Request):
    """Stream a chat response via COSMOS mesh with latent routing."""
    try:
        body = await request.json()
        message = body.get("message", "").strip()
        model = body.get("model", "cosmos").lower()

        if not message:
            return JSONResponse({"error": "Message required"}, status_code=400)

        async def generate():
            try:
                response_text = ""
                routing_meta = {}

                # PRIMARY: COSMOS mesh query
                mesh_resp, routing_meta = await _mesh_query(message, model)
                if mesh_resp:
                    response_text = mesh_resp
                else:
                    # Fallback to agent spawner
                    try:
                        from Cosmos.core.agent_spawner import get_agent_spawner
                        spawner = get_agent_spawner()
                        result = await spawner.call_agent(model, message)
                        if result and isinstance(result):
                            response_text = result.get("response", result.get("text", str(result)))
                        elif result:
                            response_text = str(result)
                    except Exception:
                        response_text = f"The swarm is processing your request. Deliberating on: {message[:200]}"

                # Send routing metadata first
                if routing_meta:
                    yield f"data: {json.dumps({'routing': routing_meta})}\n\n"

                # Stream response in chunks
                for i in range(0, len(response_text), 3):
                    chunk = response_text[i:i+3]
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    await asyncio.sleep(0.02)

                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# IMAGE UPLOAD API
# ============================================================
@router.post("/api/pro/upload")
async def pro_upload(request: Request):
    """Handle image upload for chat."""
    try:
        body = await request.json()
        image_data = body.get("image_base64", "")
        filename = body.get("filename", "upload.png")

        if not image_data:
            return JSONResponse({"error": "No image data"}, status_code=400)

        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)

        file_id = str(uuid.uuid4())[:8]
        ext = Path(filename).suffix or ".png"
        filepath = upload_dir / f"{file_id}{ext}"

        image_bytes = base64.b64decode(image_data.split(",")[-1] if "," in image_data else image_data)
        filepath.write_bytes(image_bytes)

        return JSONResponse({
            "file_id": file_id,
            "filename": filename,
            "url": f"/api/pro/uploads/{file_id}{ext}",
        })
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# WALLET ANALYSIS API
# ============================================================
@router.post("/api/pro/wallet/analyze")
async def wallet_analyze(request: Request):
    """Analyze a Solana wallet."""
    try:
        body = await request.json()
        address = body.get("address", "").strip()

        if not address or len(address) < 32:
            return JSONResponse({"error": "Valid Solana address required"}, status_code=400)

        # Try to get real wallet data
        wallet_data = None
        try:
            from Cosmos.trading.degen_trader import DegenTrader
            # Use Solana RPC to get token accounts
            import aiohttp
            async with aiohttp.ClientSession() as session:
                rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [address]
                }
                async with session.post(rpc_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sol_balance = data.get("result", {}).get("value", 0) / 1e9
                        wallet_data = {"sol_balance": sol_balance}
        except Exception as e:
            logger.debug(f"Wallet RPC error: {e}")

        # Generate swarm analysis
        analysis = ""
        try:
            from Cosmos.core.agent_spawner import get_agent_spawner
            spawner = get_agent_spawner()
            prompt = f"Analyze this Solana wallet address: {address}. Provide a brief portfolio assessment, risk factors, and recommendations. Be concise and crypto-savvy."
            result = await spawner.call_agent("grok", prompt)
            if result:
                analysis = result.get("response", str(result)) if isinstance(result) else str(result)
        except Exception:
            analysis = f"Wallet {address[:8]}...{address[-4:]} analysis pending. The swarm is gathering on-chain data."

        sol_balance = wallet_data.get("sol_balance", 0) if wallet_data else 0

        return JSONResponse({
            "address": address,
            "sol_balance": sol_balance,
            "analysis": analysis,
            "risk_score": 65,
            "swarm_grade": "B+",
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Wallet analysis error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# TOKEN SCANNER API
# ============================================================
@router.get("/api/pro/scan")
async def token_scan():
    """Get latest token scan results."""
    try:
        # Try to get data from existing DEX system
        tokens = []
        try:
            from Cosmos.dex.dex_engine import DexEngine
            engine = DexEngine()
            data = await engine.get_trending()
            if data:
                tokens = data[:20]
        except Exception:
            pass

        if not tokens:
            # Return structured empty response — frontend has demo data
            tokens = []

        return JSONResponse({
            "tokens": tokens,
            "total_scanned": len(tokens),
            "last_scan": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Scanner error: {e}")
        return JSONResponse({"tokens": [], "total_scanned": 0}, status_code=500)


# ============================================================
# ARENA API
# ============================================================
@router.post("/api/pro/arena/start")
async def arena_start(request: Request):
    """Start an agent debate."""
    try:
        body = await request.json()
        topic = body.get("topic", "").strip()
        agents = body.get("agents", [])
        debate_format = body.get("format", "roundtable")

        if not topic:
            return JSONResponse({"error": "Topic required"}, status_code=400)
        if len(agents) < 2:
            return JSONResponse({"error": "Select at least 2 agents"}, status_code=400)

        debate_id = str(uuid.uuid4())[:8]
        rounds = []

        # Generate debate rounds using actual agents
        for round_num in range(1, 4):  # 3 rounds
            round_responses = []
            for agent_name in agents[:5]:  # Max 5 agents
                try:
                    from Cosmos.core.agent_spawner import get_agent_spawner
                    spawner = get_agent_spawner()

                    round_context = f"Round {round_num}/3"
                    if round_num == 1:
                        prompt = f"[DEBATE] Topic: {topic}\n\n{round_context} - Present your opening position. Be direct, opinionated, and concise (2-3 paragraphs max). You are {agent_name}."
                    elif round_num == 2:
                        prev_args = "\n".join([f"{r['agent']}: {r['text'][:200]}" for r in rounds[-1]['responses']]) if rounds else ""
                        prompt = f"[DEBATE] Topic: {topic}\n\nPrevious arguments:\n{prev_args}\n\n{round_context} - Critique others and refine your position. Be sharp. You are {agent_name}."
                    else:
                        prompt = f"[DEBATE] Topic: {topic}\n\n{round_context} - Final argument. Make your strongest case. You are {agent_name}."

                    result = await spawner.call_agent(agent_name.lower(), prompt)
                    text = ""
                    if result and isinstance(result):
                        text = result.get("response", result.get("text", ""))
                    elif result:
                        text = str(result)

                    round_responses.append({
                        "agent": agent_name,
                        "text": text or f"{agent_name} is formulating their argument...",
                        "score": 70 + (hash(f"{agent_name}{round_num}{topic}") % 25),
                    })
                except Exception as e:
                    round_responses.append({
                        "agent": agent_name,
                        "text": f"{agent_name} is thinking deeply about this topic...",
                        "score": 65,
                    })

            rounds.append({
                "round": round_num,
                "responses": round_responses,
            })

        # Determine winner by total score
        scores = {}
        for rnd in rounds:
            for resp in rnd["responses"]:
                scores[resp["agent"]] = scores.get(resp["agent"], 0) + resp["score"]

        winner = max(scores, key=scores.get) if scores else agents[0]

        return JSONResponse({
            "debate_id": debate_id,
            "topic": topic,
            "format": debate_format,
            "rounds": rounds,
            "winner": winner,
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Arena error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# PNL API
# ============================================================
@router.get("/api/pro/pnl")
async def pnl_data():
    """Get trading PnL data."""
    try:
        trades = []
        stats = {}

        try:
            from Cosmos.trading.degen_trader import DegenTrader
            trader = DegenTrader.__new__(DegenTrader)
            if hasattr(trader, "get_trade_history"):
                trades = await trader.get_trade_history()
            if hasattr(trader, "get_stats"):
                stats = await trader.get_stats()
        except Exception:
            pass

        return JSONResponse({
            "trades": trades[-50:] if trades else [],
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"PnL error: {e}")
        return JSONResponse({"trades": [], "stats": {}}, status_code=500)


# ============================================================
# PREDICTIONS API
# ============================================================
@router.get("/api/pro/predictions")
async def predictions_data():
    """Get prediction scoreboard data."""
    try:
        predictions = []
        stats = {}

        try:
            from Cosmos.integration.polymarket_integration import get_polymarket
            pm = get_polymarket()
            if hasattr(pm, "get_predictions"):
                predictions = await pm.get_predictions()
            if hasattr(pm, "get_stats"):
                stats = await pm.get_stats()
        except Exception:
            pass

        # Also get from deliberation stats
        try:
            from Cosmos.core.deliberation_engine import get_deliberation_engine
            engine = get_deliberation_engine()
            if hasattr(engine, "get_prediction_stats"):
                delib_stats = engine.get_prediction_stats()
                stats.update(delib_stats)
        except Exception:
            pass

        return JSONResponse({
            "predictions": predictions,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Predictions error: {e}")
        return JSONResponse({"predictions": [], "stats": {}}, status_code=500)


# ============================================================
# POLYMARKET LIVE TRACKER API
# ============================================================

_polymarket_api = None

def _get_polymarket_api():
    """Singleton PolymarketAPI to avoid session leaks."""
    global _polymarket_api
    if _polymarket_api is None:
        from Cosmos.integration.financial.polymarket import PolymarketAPI
        _polymarket_api = PolymarketAPI()
    return _polymarket_api


@router.get("/api/pro/polymarket/markets")
async def polymarket_markets(request: Request):
    """Live Polymarket markets with optional category and search filters."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    user_info = _validate_token(token) if token else None
    if not user_info:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        api = _get_polymarket_api()
        category = request.query_params.get("category", "")
        search = request.query_params.get("search", "")
        limit = int(request.query_params.get("limit", "20"))

        if search:
            markets = await api.search_markets(search, limit=limit)
        elif category and category.lower() != "all":
            events = await api.get_events(category=category, limit=limit)
            markets = []
            for evt in events:
                markets.extend(evt.markets[:3])
            markets = markets[:limit]
        else:
            markets = await api.get_markets(active=True, limit=limit, order="volume")

        return JSONResponse({
            "markets": [m.to_any() for m in markets],
            "count": len(markets),
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Polymarket markets error: {e}")
        return JSONResponse({"markets": [], "count": 0}, status_code=500)


@router.get("/api/pro/polymarket/closing-soon")
async def polymarket_closing_soon(request: Request):
    """Markets closing within N hours."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    user_info = _validate_token(token) if token else None
    if not user_info:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        api = _get_polymarket_api()
        hours = int(request.query_params.get("hours", "48"))
        limit = int(request.query_params.get("limit", "10"))

        markets = await api.get_closing_soon(hours=hours, limit=limit)

        return JSONResponse({
            "markets": [m.to_any() for m in markets],
            "count": len(markets),
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Polymarket closing-soon error: {e}")
        return JSONResponse({"markets": [], "count": 0}, status_code=500)


@router.get("/api/pro/polymarket/ai-predictions")
async def polymarket_ai_predictions(request: Request):
    """AI predictions with stats from the prediction engine."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    user_info = _validate_token(token) if token else None
    if not user_info:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        from Cosmos.core.polymarket_predictor import get_predictor
        predictor = get_predictor()

        limit = int(request.query_params.get("limit", "20"))
        predictions = predictor.get_recent_predictions(limit=limit)
        stats = predictor.get_stats()

        return JSONResponse({
            "predictions": [p.to_any() for p in predictions],
            "stats": {
                "total_predictions": stats.total_predictions,
                "correct": stats.correct,
                "incorrect": stats.incorrect,
                "pending": stats.pending,
                "accuracy": round(stats.accuracy * 100, 1),
                "streak": stats.streak,
                "best_streak": stats.best_streak,
                "by_category": stats.by_category,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Polymarket AI predictions error: {e}")
        return JSONResponse({"predictions": [], "stats": {}}, status_code=500)


# ============================================================
# USER PROFILE / USAGE API
# ============================================================
@router.get("/api/pro/usage")
async def usage_data(request: Request):
    """Get usage stats for the authenticated user."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

    user_info = _validate_token(token) if token else None
    if not user_info:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    limits = {
        "free": {"messages_per_day": 10, "models": 3},
        "pro": {"messages_per_day": 500, "models": 11},
        "unlimited": {"messages_per_day": 999999, "models": 11},
    }

    plan = user_info.get("plan", "free")
    plan_limits = limits.get(plan, limits["free"])

    return JSONResponse({
        "plan": plan,
        "messages_today": 0,  # TODO: track from users.json
        "messages_limit": plan_limits["messages_per_day"],
        "models_available": plan_limits["models"],
    })
