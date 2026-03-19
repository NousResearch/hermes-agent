"""
Chat & Deliberation Routes

Endpoints:
- POST /api/chat - Main chat with swarm
- GET /api/status - Server status
- POST /api/memory/remember - Store memory
- POST /api/memory/recall - Recall memories
- GET /api/memory/stats - Memory statistics
- GET /api/notes - list notes
- POST /api/notes - Create note
- DELETE /api/notes/{note_id} - Delete note
- GET /api/snippets - list snippets
- POST /api/snippets - Create snippet
- GET /api/focus/status - Focus timer status
- POST /api/focus/start - Start focus timer
- POST /api/focus/stop - Stop focus timer
- GET /api/profiles - list profiles
- POST /api/profiles/switch - Switch profile
- GET /api/health/summary - Health summary
- GET /api/health/metrics/{metric_type} - Health metrics
- POST /api/think - Sequential thinking
- GET /api/tools - list tools
- POST /api/tools/execute - Execute tool
- POST /api/tools/whale-track - Whale tracking
- POST /api/tools/rug-check - Rug check
- POST /api/tools/token-scan - Token scan
- GET /api/tools/market-sentiment - Market sentiment
- POST /api/oracle/query - Oracle query
- GET /api/oracle/queries - list oracle queries
- GET /api/oracle/query/{query_id} - Get oracle query
- GET /api/oracle/stats - Oracle stats
- POST /api/farsight/predict - FarSight prediction
- POST /api/farsight/crypto - FarSight crypto
- GET /api/farsight/stats - FarSight stats
- GET /api/farsight/predictions - FarSight predictions
- GET /api/solana/scan/{token_address} - Solana token scan
- POST /api/solana/defi/recommend - Solana DeFi recommendations
- GET /api/solana/wallet/{wallet_address} - Solana wallet info
- GET /api/solana/swap/quote - Solana swap quote
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# MAIN CHAT ENDPOINT
# ============================================

@router.post("/api/chat")
async def chat(request: Request):
    """Handle chat messages with security validation and crypto query detection."""
    s = _get_shared()
    from Cosmos.web.server import ChatRequest
    import json

    # Rate limiting check
    client_id = s.get_client_id(request)
    if not s.chat_rate_limiter.is_allowed(client_id):
        retry_after = s.chat_rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(retry_after) + 1)}
        )

    try:
        body = await request.json()
        chat_request = ChatRequest(**body)

        if not chat_request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Security: Validate input is safe
        is_safe, error_msg = s.is_safe_input(chat_request.message)
        if not is_safe:
            s.logger.warning(f"Blocked unsafe input attempt: {chat_request.message[:100]}")
            return JSONResponse({
                "response": f"*adjusts spectacles nervously* Wha? I'm a chat assistant, not a code execution engine! {error_msg}",
                "blocked": True,
                "demo_mode": s.DEMO_MODE
            })

        original_message = chat_request.message

        # Pause free discussion while user is active
        try:
            from Cosmos.core.collective.free_discussion import get_free_discussion_engine
            fde = get_free_discussion_engine()
            if fde:
                fde.notify_user_activity()
        except Exception:
            pass

        # Self-awareness: Check intent BEFORE upgrading
        intent = s.detect_intent(original_message)

        # Handle self-awareness intents directly (bypass upgrader)
        if intent["primary_intent"] in ["self_examine", "swarm_query", "evolution_query"]:
            s.logger.info(f"Self-awareness intent detected: {intent['primary_intent']}")
            response = s.spawn_task_from_intent(original_message, intent)
            if response:
                return JSONResponse({
                    "response": response,
                    "demo_mode": s.DEMO_MODE,
                    "features_available": True,
                    "self_awareness_query": True,
                    "intent_detected": intent["primary_intent"]
                })

        # Automatically upgrade user prompt to professional quality
        upgraded_message = chat_request.message
        prompt_was_upgraded = False

        if s.PROMPT_UPGRADER_AVAILABLE and s.upgrade_prompt:
            try:
                upgraded_message = await s.upgrade_prompt(chat_request.message)
                if upgraded_message != original_message:
                    prompt_was_upgraded = True
                    s.logger.info(f"Prompt upgraded: '{original_message[:50]}...' -> '{upgraded_message[:50]}...'")
            except Exception as e:
                s.logger.warning(f"Prompt upgrade failed, using original: {e}")
                upgraded_message = original_message

        # Check for crypto/token queries (use original message for pattern matching)
        parsed = s.crypto_parser.parse(original_message)

        if parsed['has_crypto_query']:
            # Execute the appropriate crypto tool
            tool_result = await s.crypto_parser.execute_tool(parsed)

            if tool_result and tool_result.get('success'):
                # Combine tool result with AI commentary
                ai_intro = s.generate_ai_response(
                    f"User asked about {parsed['intent']} for {parsed['query']}. Provide brief commentary.",
                    []
                )
                response = f"{ai_intro}\n\n{tool_result['formatted']}"

                return JSONResponse({
                    "response": response,
                    "demo_mode": s.DEMO_MODE,
                    "features_available": True,
                    "tool_used": tool_result['tool_used'],
                    "crypto_query": True
                })

        # 🧠 INJECT PERSISTENT MEMORY CONTEXT
        try:
            memory_sys = s.get_memory_system()
            if memory_sys:
                memories = await memory_sys.recall(query=upgraded_message, limit=3)
                if memories:
                    mem_text = "\n".join([f"- {m.get('content', '')}" for m in memories])
                    upgraded_message += f"\n\n[PERSISTENT MEMORY RECALL - Use these facts from past conversations]:\n{mem_text}"
        except Exception as e:
            s.logger.debug(f"Memory recall error for chat: {e}")

        # Regular chat response - COSMOS IS THE COLLECTIVE
        collective_result = await s.generate_ai_response_collective(
            upgraded_message,
            chat_request.history or []
        )

        response_data = {
            "response": collective_result["response"],
            "demo_mode": s.DEMO_MODE,
            "features_available": True,
            "collective_active": collective_result.get("collective_active", False),
        }

        # Include collective metadata
        if collective_result.get("collective_active"):
            response_data["agents_count"] = collective_result.get("agents_count", 0)
            response_data["winning_agent"] = collective_result.get("winning_agent")
            response_data["consensus"] = collective_result.get("consensus", False)

        # Include upgrade info if prompt was enhanced
        if prompt_was_upgraded:
            response_data["prompt_upgraded"] = True
            response_data["original_prompt"] = original_message[:100]
            response_data["upgraded_prompt"] = upgraded_message[:200]

        return JSONResponse(response_data)

    except Exception as e:
        s.logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================
# STATUS ENDPOINT
# ============================================

@router.get("/api/status")
async def status():
    """Get server status with feature availability."""
    s = _get_shared()
    return JSONResponse({
        "status": "online",
        "version": "2.9.3",
        "demo_mode": s.DEMO_MODE,
        "ollama_available": s.OLLAMA_AVAILABLE,
        "solana_available": s.SOLANA_AVAILABLE,
        "features": {
            "memory": s.get_memory_system() is not None,
            "notes": s.get_notes_manager() is not None,
            "snippets": s.get_snippet_manager() is not None,
            "focus_timer": s.get_focus_timer() is not None,
            "profiles": s.get_context_profiles() is not None,
            "evolution": s.EVOLUTION_AVAILABLE and s.evolution_engine is not None,
            "tools": s.get_tool_router() is not None,
            "thinking": s.get_sequential_thinking() is not None,
        },
        "multi_model": {
            "enabled": True,
            "providers": {
                "ollama": {
                    "available": s.OLLAMA_AVAILABLE,
                    "bots": ["Cosmos", "DeepSeek", "Phi", "Swarm-Mind"]
                },
                "claude_code": {
                    "available": s.CLAUDE_CODE_AVAILABLE,
                    "description": "Claude via CLI (uses Claude Max subscription)",
                    "bots": ["Claude"]
                },
                "kimi": {
                    "available": s.KIMI_AVAILABLE,
                    "description": "Moonshot AI (256k context, Eastern philosophy)",
                    "bots": ["Kimi"]
                }
            },
            "active_bots": s.ACTIVE_SWARM_BOTS
        },
        "cosmos_persona": True,
        "voice_enabled": True
    })


# ============================================
# MEMORY SYSTEM API
# ============================================

@router.post("/api/memory/remember")
async def remember(request: Request):
    """Store information in memory."""
    s = _get_shared()
    from Cosmos.web.server import MemoryRequest

    client_id = s.get_client_id(request)
    if not s.api_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        body = await request.json()
        memory_request = MemoryRequest(**body)

        memory = s.get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "message": "Memory system not available. Install dependencies locally.",
                "demo_mode": True
            })

        result = await memory.remember(
            content=memory_request.content,
            tags=memory_request.tags or [],
            importance=memory_request.importance
        )

        await s.ws_manager.emit_event(s.EventType.MEMORY_STORED, {
            "content": memory_request.content[:100] + "..." if len(memory_request.content) > 100 else memory_request.content,
            "tags": memory_request.tags
        })

        return JSONResponse({
            "success": True,
            "message": "Good news, everyone! Stored in the Memory-Matic 3000!",
            "memory_id": str(result)
        })

    except Exception as e:
        s.logger.error(f"Memory store error: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Memory storage failed: {str(e)}"
        })


@router.post("/api/memory/recall")
async def recall(request: Request):
    """Search and recall memories."""
    s = _get_shared()
    from Cosmos.web.server import RecallRequest

    try:
        body = await request.json()
        req = RecallRequest(**body)

        memory = s.get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "memories": [],
                "message": "Memory system not available. Install dependencies locally."
            })

        results = await memory.recall(
            query=req.query,
            top_k=req.limit
        )

        await s.ws_manager.emit_event(s.EventType.MEMORY_RECALLED, {
            "query": req.query,
            "count": len(results) if results else 0
        })

        return JSONResponse({
            "success": True,
            "memories": results if results else [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        s.logger.error(f"Memory recall error: {e}")
        return JSONResponse({
            "success": False,
            "memories": [],
            "message": f"Memory recall failed: {str(e)}"
        })


@router.get("/api/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    s = _get_shared()
    try:
        memory = s.get_memory_system()
        if memory is None:
            return JSONResponse({"available": False})

        stats = memory.get_stats() if hasattr(memory, 'get_stats') else {}
        return JSONResponse({
            "available": True,
            "stats": stats
        })

    except Exception as e:
        s.logger.error(f"Memory stats error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


# ============================================
# NOTES API
# ============================================

@router.get("/api/notes")
async def list_notes():
    """list all notes."""
    s = _get_shared()
    try:
        notes = s.get_notes_manager()
        if notes is None:
            return JSONResponse({"notes": [], "message": "Notes not available"})
        return JSONResponse({"notes": notes.get_all()})
    except Exception as e:
        s.logger.error(f"Notes list error: {e}")
        return JSONResponse({"notes": [], "error": str(e)})


@router.post("/api/notes")
async def create_note(request: Request):
    """Create a new note."""
    s = _get_shared()
    from Cosmos.web.server import NoteRequest

    try:
        body = await request.json()
        note_req = NoteRequest(**body)

        notes = s.get_notes_manager()
        if notes is None:
            return JSONResponse({"success": False, "message": "Notes not available"})

        result = notes.add(content=note_req.content, tags=note_req.tags)

        await s.ws_manager.emit_event(s.EventType.NOTE_ADDED, {
            "content": note_req.content[:100]
        })

        return JSONResponse({
            "success": True,
            "note": result
        })
    except Exception as e:
        s.logger.error(f"Note creation error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    s = _get_shared()
    try:
        notes = s.get_notes_manager()
        if notes is None:
            return JSONResponse({"success": False, "message": "Notes not available"})

        success = notes.delete(note_id)
        return JSONResponse({"success": success})
    except Exception as e:
        s.logger.error(f"Note deletion error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# SNIPPETS API
# ============================================

@router.get("/api/snippets")
async def list_snippets():
    """list all snippets."""
    s = _get_shared()
    try:
        snippets = s.get_snippet_manager()
        if snippets is None:
            return JSONResponse({"snippets": [], "message": "Snippets not available"})
        return JSONResponse({"snippets": snippets.get_all()})
    except Exception as e:
        s.logger.error(f"Snippets list error: {e}")
        return JSONResponse({"snippets": [], "error": str(e)})


@router.post("/api/snippets")
async def create_snippet(request: Request):
    """Create a new snippet."""
    s = _get_shared()
    from Cosmos.web.server import SnippetRequest

    try:
        body = await request.json()
        snippet_req = SnippetRequest(**body)

        snippets = s.get_snippet_manager()
        if snippets is None:
            return JSONResponse({"success": False, "message": "Snippets not available"})

        result = snippets.add(
            code=snippet_req.code,
            language=snippet_req.language,
            description=snippet_req.description,
            tags=snippet_req.tags
        )
        return JSONResponse({"success": True, "snippet": result})
    except Exception as e:
        s.logger.error(f"Snippet creation error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# FOCUS TIMER API
# ============================================

@router.get("/api/focus/status")
async def focus_status():
    """Get focus timer status."""
    s = _get_shared()
    try:
        timer = s.get_focus_timer()
        if timer is None:
            return JSONResponse({"active": False, "message": "Focus timer not available"})
        return JSONResponse(timer.get_status())
    except Exception as e:
        s.logger.error(f"Focus status error: {e}")
        return JSONResponse({"active": False, "error": str(e)})


@router.post("/api/focus/start")
async def focus_start(request: Request):
    """Start focus timer."""
    s = _get_shared()
    from Cosmos.web.server import FocusRequest

    try:
        body = await request.json()
        focus_req = FocusRequest(**body)

        timer = s.get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        result = timer.start(task=focus_req.task, duration=focus_req.duration_minutes)

        await s.ws_manager.emit_event(s.EventType.FOCUS_START, {
            "task": focus_req.task,
            "duration": focus_req.duration_minutes
        })

        return JSONResponse({"success": True, **result})
    except Exception as e:
        s.logger.error(f"Focus start error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/api/focus/stop")
async def focus_stop():
    """Stop focus timer."""
    s = _get_shared()
    try:
        timer = s.get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        result = timer.stop()

        await s.ws_manager.emit_event(s.EventType.FOCUS_END, result)

        return JSONResponse({"success": True, **result})
    except Exception as e:
        s.logger.error(f"Focus stop error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# PROFILES API
# ============================================

@router.get("/api/profiles")
async def list_profiles():
    """list available profiles."""
    s = _get_shared()
    try:
        profiles = s.get_context_profiles()
        if profiles is None:
            return JSONResponse({"profiles": [], "message": "Profiles not available"})
        return JSONResponse({"profiles": profiles.list_profiles()})
    except Exception as e:
        s.logger.error(f"Profiles list error: {e}")
        return JSONResponse({"profiles": [], "error": str(e)})


@router.post("/api/profiles/switch")
async def switch_profile(request: Request):
    """Switch active profile."""
    s = _get_shared()
    from Cosmos.web.server import ProfileRequest

    try:
        body = await request.json()
        profile_req = ProfileRequest(**body)

        profiles = s.get_context_profiles()
        if profiles is None:
            return JSONResponse({"success": False, "message": "Profiles not available"})

        result = profiles.switch(profile_req.profile_id)
        return JSONResponse({"success": True, **result})
    except Exception as e:
        s.logger.error(f"Profile switch error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# HEALTH TRACKING API
# ============================================

@router.get("/api/health/summary")
async def health_summary():
    """Get health summary."""
    s = _get_shared()
    try:
        analyzer = s.get_health_analyzer()
        if analyzer is None:
            return JSONResponse({"available": False, "message": "Health analyzer not available"})
        summary = analyzer.get_summary()
        return JSONResponse({"available": True, "summary": summary})
    except Exception as e:
        s.logger.error(f"Health summary error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@router.get("/api/health/metrics/{metric_type}")
async def health_metrics(metric_type: str):
    """Get specific health metrics."""
    s = _get_shared()
    try:
        analyzer = s.get_health_analyzer()
        if analyzer is None:
            return JSONResponse({"available": False})
        metrics = analyzer.get_metrics(metric_type)
        return JSONResponse({"available": True, "metric_type": metric_type, "data": metrics})
    except Exception as e:
        s.logger.error(f"Health metrics error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


# ============================================
# SEQUENTIAL THINKING API
# ============================================

@router.post("/api/think")
async def think(request: Request):
    """Sequential thinking endpoint."""
    s = _get_shared()
    from Cosmos.web.server import ThinkingRequest

    try:
        body = await request.json()
        think_req = ThinkingRequest(**body)

        engine = s.get_sequential_thinking()

        await s.ws_manager.emit_event(s.EventType.THINKING_START, {
            "problem": think_req.problem[:100]
        })

        if engine:
            steps = engine.think(think_req.problem, max_steps=think_req.max_steps)
            for step in steps:
                await s.ws_manager.emit_event(s.EventType.THINKING_STEP, step)

            await s.ws_manager.emit_event(s.EventType.THINKING_END, {"steps": len(steps)})

            return JSONResponse({
                "success": True,
                "steps": steps,
                "summary": steps[-1] if steps else "No conclusion"
            })
        else:
            # Fallback
            await s.ws_manager.emit_event(s.EventType.THINKING_END, {
                "steps": 1,
                "fallback": True
            })
            return JSONResponse({
                "success": True,
                "steps": [{"step": 1, "thought": "Sequential thinking engine not available. Using basic analysis.", "conclusion": think_req.problem}],
                "summary": "Basic analysis completed"
            })

    except Exception as e:
        s.logger.error(f"Thinking error: {e}")
        await s.ws_manager.emit_event(s.EventType.ERROR, {"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================
# TOOLS API
# ============================================

@router.get("/api/tools")
async def list_tools():
    """list available tools."""
    s = _get_shared()
    try:
        router_obj = s.get_tool_router()
        if router_obj is None:
            return JSONResponse({"tools": [], "message": "Tool router not available"})
        return JSONResponse({"tools": router_obj.list_tools()})
    except Exception as e:
        s.logger.error(f"Tools list error: {e}")
        return JSONResponse({"tools": [], "error": str(e)})


@router.post("/api/tools/execute")
async def execute_tool(request: Request):
    """Execute a specific tool."""
    s = _get_shared()
    from Cosmos.web.server import ToolRequest

    try:
        body = await request.json()
        tool_req = ToolRequest(**body)

        router_obj = s.get_tool_router()
        if router_obj is None:
            return JSONResponse({"success": False, "message": "Tool router not available"})

        await s.ws_manager.emit_event(s.EventType.TOOL_CALL, {
            "tool": tool_req.tool_name,
            "args": tool_req.args
        })

        result = router_obj.execute(tool_req.tool_name, tool_req.args or {})

        await s.ws_manager.emit_event(s.EventType.TOOL_RESULT, {
            "tool": tool_req.tool_name,
            "success": True
        })

        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        s.logger.error(f"Tool execution error: {e}")
        await s.ws_manager.emit_event(s.EventType.TOOL_RESULT, {
            "tool": "unknown",
            "success": False,
            "error": str(e)
        })
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# CRYPTO TOOLS API
# ============================================

@router.post("/api/tools/whale-track")
async def whale_track(request: Request):
    """Track whale wallet."""
    s = _get_shared()
    from Cosmos.web.server import WhaleTrackRequest

    try:
        body = await request.json()
        req = WhaleTrackRequest(**body)
        result = await s.CryptoQueryParser._whale_track(req.wallet_address)
        formatted = s.CryptoQueryParser._format_whale_track(result, req.wallet_address)
        return JSONResponse({"success": True, "data": result, "formatted": formatted})
    except Exception as e:
        s.logger.error(f"Whale track error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/api/tools/rug-check")
async def rug_check(request: Request):
    """Check token for rug risks."""
    s = _get_shared()
    from Cosmos.web.server import RugCheckRequest

    try:
        body = await request.json()
        req = RugCheckRequest(**body)
        result = await s.CryptoQueryParser._rug_check(req.mint_address)
        formatted = s.CryptoQueryParser._format_rug_check(result, req.mint_address)
        return JSONResponse({"success": True, "data": result, "formatted": formatted})
    except Exception as e:
        s.logger.error(f"Rug check error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/api/tools/token-scan")
async def token_scan(request: Request):
    """Scan token info."""
    s = _get_shared()
    from Cosmos.web.server import TokenScanRequest

    try:
        body = await request.json()
        req = TokenScanRequest(**body)
        result = await s.CryptoQueryParser._token_lookup(req.query)
        formatted = s.CryptoQueryParser._format_token_info(result, req.query)
        return JSONResponse({"success": True, "data": result, "formatted": formatted})
    except Exception as e:
        s.logger.error(f"Token scan error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/api/tools/market-sentiment")
async def market_sentiment():
    """Get market sentiment."""
    s = _get_shared()
    try:
        result = await s.CryptoQueryParser._market_sentiment()
        formatted = s.CryptoQueryParser._format_sentiment(result)
        return JSONResponse({"success": True, "data": result, "formatted": formatted})
    except Exception as e:
        s.logger.error(f"Market sentiment error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# ORACLE API
# ============================================

@router.post("/api/oracle/query")
async def oracle_query(request: Request):
    """Submit oracle query."""
    s = _get_shared()
    try:
        body = await request.json()
        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        from Cosmos.core.oracle import get_oracle
        oracle = get_oracle()
        if not oracle:
            return JSONResponse({"success": False, "message": "Oracle not available"})

        result = await oracle.query(query)
        return JSONResponse({"success": True, **result})
    except HTTPException:
        raise
    except Exception as e:
        s.logger.error(f"Oracle query error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/api/oracle/queries")
async def oracle_queries():
    """list recent oracle queries."""
    s = _get_shared()
    try:
        from Cosmos.core.oracle import get_oracle
        oracle = get_oracle()
        if not oracle:
            return JSONResponse({"queries": []})
        return JSONResponse({"queries": oracle.get_recent_queries()})
    except Exception as e:
        s.logger.error(f"Oracle queries error: {e}")
        return JSONResponse({"queries": [], "error": str(e)})


@router.get("/api/oracle/query/{query_id}")
async def oracle_get_query(query_id: str):
    """Get specific oracle query."""
    s = _get_shared()
    try:
        from Cosmos.core.oracle import get_oracle
        oracle = get_oracle()
        if not oracle:
            raise HTTPException(status_code=503, detail="Oracle not available")
        result = oracle.get_query(query_id)
        if not result:
            raise HTTPException(status_code=404, detail="Query not found")
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        s.logger.error(f"Oracle get query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/oracle/stats")
async def oracle_stats():
    """Get oracle statistics."""
    s = _get_shared()
    try:
        from Cosmos.core.oracle import get_oracle
        oracle = get_oracle()
        if not oracle:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, "stats": oracle.get_stats()})
    except Exception as e:
        s.logger.error(f"Oracle stats error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


# ============================================
# FARSIGHT API
# ============================================

@router.post("/api/farsight/predict")
async def farsight_predict(request: Request):
    """FarSight prediction."""
    s = _get_shared()
    try:
        body = await request.json()
        from Cosmos.core.farsight import get_farsight
        farsight = get_farsight()
        if not farsight:
            return JSONResponse({"success": False, "message": "FarSight not available"})
        result = await farsight.predict(body.get("question", ""))
        return JSONResponse({"success": True, **result})
    except Exception as e:
        s.logger.error(f"FarSight predict error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/api/farsight/crypto")
async def farsight_crypto(request: Request):
    """FarSight crypto prediction."""
    s = _get_shared()
    try:
        body = await request.json()
        from Cosmos.core.farsight import get_farsight
        farsight = get_farsight()
        if not farsight:
            return JSONResponse({"success": False, "message": "FarSight not available"})
        result = await farsight.crypto_prediction(body.get("token", ""))
        return JSONResponse({"success": True, **result})
    except Exception as e:
        s.logger.error(f"FarSight crypto error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/api/farsight/stats")
async def farsight_stats():
    """FarSight stats."""
    s = _get_shared()
    try:
        from Cosmos.core.farsight import get_farsight
        farsight = get_farsight()
        if not farsight:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, "stats": farsight.get_stats()})
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)})


@router.get("/api/farsight/predictions")
async def farsight_predictions():
    """FarSight recent predictions."""
    s = _get_shared()
    try:
        from Cosmos.core.farsight import get_farsight
        farsight = get_farsight()
        if not farsight:
            return JSONResponse({"predictions": []})
        return JSONResponse({"predictions": farsight.get_recent_predictions()})
    except Exception as e:
        return JSONResponse({"predictions": [], "error": str(e)})


# ============================================
# SOLANA API
# ============================================

@router.get("/api/solana/scan/{token_address}")
async def solana_scan(token_address: str):
    """Scan Solana token."""
    s = _get_shared()
    try:
        result = await s.CryptoQueryParser._token_lookup(token_address)
        formatted = s.CryptoQueryParser._format_token_info(result, token_address)
        return JSONResponse({"success": True, "data": result, "formatted": formatted})
    except Exception as e:
        s.logger.error(f"Solana scan error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/api/solana/defi/recommend")
async def solana_defi_recommend(request: Request):
    """Solana DeFi recommendations."""
    s = _get_shared()
    try:
        body = await request.json()
        from Cosmos.integration.solana.degen_mob import DeGenMob
        degen = DeGenMob()
        result = await degen.get_defi_recommendations(body)
        return JSONResponse({"success": True, **result})
    except ImportError:
        return JSONResponse({"success": False, "message": "Solana DeFi module not available"})
    except Exception as e:
        s.logger.error(f"DeFi recommend error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/api/solana/wallet/{wallet_address}")
async def solana_wallet(wallet_address: str):
    """Get Solana wallet info."""
    s = _get_shared()
    try:
        from Cosmos.integration.solana.degen_mob import DeGenMob
        degen = DeGenMob()
        result = await degen.get_wallet_info(wallet_address)
        return JSONResponse({"success": True, **result})
    except ImportError:
        return JSONResponse({"success": False, "message": "Solana module not available"})
    except Exception as e:
        s.logger.error(f"Wallet info error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/api/solana/swap/quote")
async def solana_swap_quote(request: Request):
    """Get Solana swap quote."""
    s = _get_shared()
    try:
        from Cosmos.integration.solana.degen_mob import DeGenMob
        degen = DeGenMob()
        params = dict(request.query_params)
        result = await degen.get_swap_quote(params)
        return JSONResponse({"success": True, **result})
    except ImportError:
        return JSONResponse({"success": False, "message": "Solana module not available"})
    except Exception as e:
        s.logger.error(f"Swap quote error: {e}")
        return JSONResponse({"success": False, "error": str(e)})
