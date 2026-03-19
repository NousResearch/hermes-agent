"""
Swarm Chat Routes - WebSocket & API

Endpoints:
- WS /ws/swarm - Swarm Chat WebSocket
- GET /api/swarm/status - Swarm chat status
- GET /api/swarm/history - Swarm chat history
- GET /api/swarm/learning - Learning statistics
- GET /api/swarm/concepts - Extracted concepts
- GET /api/swarm/users - User patterns
- POST /api/swarm/inject - Inject message
- POST /api/swarm/learn - Trigger learning
- POST /api/swarm-memory/enable - Enable swarm memory
- POST /api/swarm-memory/disable - Disable swarm memory
- GET /api/swarm-memory/stats - Swarm memory stats
- POST /api/swarm-memory/recall - Recall swarm context
- GET /api/turn-taking/stats - Turn-taking stats
- POST /api/memory/dedup/enable - Enable deduplication
- POST /api/memory/dedup/disable - Disable deduplication
- GET /api/memory/dedup/stats - Dedup statistics
- POST /api/memory/dedup/check - Check for duplicates
- GET /api/deliberations/stats - Deliberation statistics
- GET /api/limits - Get dynamic limits
- POST /api/limits/model/{model_id} - Update model limits
- POST /api/limits/session/{session_type} - Update session limits
- POST /api/limits/deliberation - Update deliberation limits
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# SWARM CHAT WEBSOCKET
# ============================================

@router.websocket("/ws/swarm")
async def websocket_swarm(websocket: WebSocket):
    """WebSocket endpoint for Swarm Chat - community shared chat."""
    import uuid
    import html
    import random
    s = _get_shared()

    user_id = str(uuid.uuid4())
    user_name = None

    def sanitize_username(name: str, max_length: int = 32) -> str:
        """Sanitize user name to prevent XSS and enforce limits."""
        if not name or not isinstance(name, str):
            return f"Anon_{user_id[:6]}"
        name = html.escape(name.strip())
        name = "".join(c for c in name if c.isalnum() or c in " _-.")
        name = name[:max_length].strip()
        return name if name else f"Anon_{user_id[:6]}"

    try:
        await websocket.accept()
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        user_name = sanitize_username(init_data.get("user_name", ""))

        s.swarm_manager.connections[user_id] = websocket
        s.swarm_manager.user_names[user_id] = user_name

        await s.swarm_manager.broadcast_system(f"\U0001f7e2 {user_name} joined the swarm!")
        await websocket.send_json({
            "type": "swarm_connected",
            "user_id": user_id,
            "user_name": user_name,
            "messages": s.swarm_manager.chat_history[-50:],
            "online_users": s.swarm_manager.get_online_users(),
            "active_models": s.swarm_manager.active_models,
            "online_count": s.swarm_manager.get_online_count()
        })

        logger.info(f"Swarm Chat: {user_name} connected. Total: {s.swarm_manager.get_online_count()}")

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "swarm_message":
                    content = data.get("content", "").strip()
                    logger.info(f"Swarm message received from {user_name}: '{content[:100] if content else 'EMPTY'}'")
                    if content:
                        is_safe, error_msg = s.is_safe_input(content)
                        if not is_safe:
                            logger.warning(f"Swarm: Blocked unsafe input from {user_name}: {content[:100]}")
                            await websocket.send_json({
                                "type": "swarm_error",
                                "message": "This is a chat interface - code execution is not allowed.",
                                "blocked": True
                            })
                            continue

                        await s.swarm_manager.broadcast_user_message(user_id, content)

                        responses = await s.generate_swarm_responses(
                            content,
                            s.swarm_manager.chat_history
                        )

                        logger.info(f"Swarm responses generated: {len(responses)} responses")
                        last_bot_message = None
                        last_bot_name = None
                        for resp in responses:
                            bot_content = resp.get("content", "").strip()
                            logger.info(f"Bot {resp.get('bot_name')}: content length={len(bot_content)}, preview={bot_content[:50] if bot_content else 'EMPTY'}")
                            if not bot_content:
                                logger.warning(f"Skipping empty response from {resp.get('bot_name')}")
                                continue
                            await s.swarm_manager.broadcast_typing(resp["bot_name"], True)
                            await asyncio.sleep(0.3)
                            await s.swarm_manager.broadcast_bot_message(
                                resp["bot_name"],
                                bot_content
                            )
                            await s.swarm_manager.broadcast_typing(resp["bot_name"], False)
                            last_bot_message = bot_content
                            last_bot_name = resp["bot_name"]

                        # Autonomous bot-to-bot conversation continuation
                        continuation_rounds = 0
                        max_rounds = random.randint(1, 3)
                        while last_bot_message and last_bot_name and continuation_rounds < max_rounds:
                            await asyncio.sleep(random.uniform(1.5, 3.0))

                            followup = await s.generate_bot_followup(
                                last_bot_name,
                                last_bot_message,
                                s.swarm_manager.chat_history
                            )

                            if not followup:
                                break

                            followup_content = followup.get("content", "").strip()
                            if not followup_content:
                                break

                            logger.info(f"Bot followup: {followup['bot_name']} responding to {last_bot_name}")
                            await s.swarm_manager.broadcast_typing(followup["bot_name"], True)
                            await asyncio.sleep(0.3)
                            await s.swarm_manager.broadcast_bot_message(
                                followup["bot_name"],
                                followup_content
                            )
                            await s.swarm_manager.broadcast_typing(followup["bot_name"], False)

                            last_bot_message = followup_content
                            last_bot_name = followup["bot_name"]
                            continuation_rounds += 1

                        # Share conversation with P2P planetary network
                        if continuation_rounds > 0 and s.P2P_FABRIC_AVAILABLE and s.swarm_fabric:
                            try:
                                recent_bot_msgs = [
                                    {"bot": m.get("bot_name"), "content": m.get("content", "")[:200]}
                                    for m in s.swarm_manager.chat_history[-10:]
                                    if m.get("type") == "swarm_bot"
                                ]
                                if recent_bot_msgs:
                                    await s.swarm_fabric.broadcast_conversation(recent_bot_msgs)
                                    logger.info(f"P2P: Shared {len(recent_bot_msgs)} bot messages to planetary network")
                            except Exception as e:
                                logger.debug(f"P2P conversation share failed: {e}")

                        if len(s.swarm_manager.learning_queue) >= 10:
                            await s.swarm_manager.store_learnings()

                elif data.get("type") == "get_online":
                    await websocket.send_json({
                        "type": "online_update",
                        "online_users": s.swarm_manager.get_online_users(),
                        "online_count": s.swarm_manager.get_online_count()
                    })

                elif data.get("type") == "audio_complete":
                    bot_name = data.get("bot_name", "")
                    s.audio_complete_signal(bot_name)

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        name = s.swarm_manager.disconnect(user_id)
        await s.swarm_manager.broadcast_system(f"\U0001f534 {name} left the swarm")
    except Exception as e:
        logger.error(f"Swarm WebSocket error: {e}")
        s.swarm_manager.disconnect(user_id)


# ============================================
# SWARM STATUS & MANAGEMENT API
# ============================================

@router.get("/api/swarm/status")
async def swarm_status():
    """Get Swarm Chat status."""
    s = _get_shared()
    return JSONResponse({
        "online_count": s.swarm_manager.get_online_count(),
        "online_users": s.swarm_manager.get_online_users(),
        "active_models": s.swarm_manager.active_models,
        "message_count": len(s.swarm_manager.chat_history),
        "learning_queue_size": len(s.swarm_manager.learning_queue)
    })


@router.get("/api/swarm/history")
async def swarm_history(limit: int = 50):
    """Get recent Swarm Chat history."""
    s = _get_shared()
    return JSONResponse({
        "messages": s.swarm_manager.chat_history[-limit:],
        "total": len(s.swarm_manager.chat_history)
    })


@router.get("/api/swarm/learning")
async def swarm_learning_stats():
    """Get real-time learning statistics from Swarm Chat."""
    s = _get_shared()
    return JSONResponse({
        "learning_stats": s.swarm_manager.get_learning_stats(),
        "status": "active",
        "description": "Real-time learning from community interactions"
    })


@router.get("/api/swarm/concepts")
async def swarm_concepts():
    """Get extracted concepts from Swarm Chat conversations."""
    s = _get_shared()
    stats = s.swarm_manager.get_learning_stats()
    return JSONResponse({
        "concepts": stats.get("top_concepts", []),
        "total": stats.get("concept_count", 0)
    })


@router.get("/api/swarm/users")
async def swarm_user_patterns():
    """Get user behavior patterns learned from Swarm Chat."""
    s = _get_shared()
    return JSONResponse({
        "online_users": s.swarm_manager.get_online_users(),
        "online_count": s.swarm_manager.get_online_count(),
        "patterns_tracked": len(s.swarm_learning.user_patterns)
    })


@router.post("/api/swarm/inject")
async def inject_swarm_message(request: dict):
    """Inject a message into swarm chat programmatically."""
    s = _get_shared()
    try:
        bot_name = request.get("bot_name", "System")
        content = request.get("content", "")
        is_thinking = request.get("is_thinking", False)

        if not content:
            return JSONResponse({
                "success": False,
                "message": "Content is required"
            })

        msg_id = await s.swarm_manager.broadcast_bot_message(
            bot_name=bot_name,
            content=content,
            is_thinking=is_thinking
        )

        try:
            from Cosmos.core.swarm_memory_integration import process_swarm_interaction_for_memory
            await process_swarm_interaction_for_memory({
                "role": "assistant",
                "name": bot_name,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "source": "api_inject"
            })
        except Exception as e:
            logger.debug(f"Memory bridge not available: {e}")

        return JSONResponse({
            "success": True,
            "message": "Message injected to swarm",
            "msg_id": msg_id,
            "bot_name": bot_name
        })

    except Exception as e:
        logger.error(f"Failed to inject swarm message: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


@router.post("/api/swarm/learn")
async def trigger_learning():
    """Force a learning cycle to process buffered interactions."""
    s = _get_shared()
    await s.swarm_manager.force_learning_cycle()
    return JSONResponse({
        "success": True,
        "message": "Learning cycle triggered",
        "stats": s.swarm_manager.get_learning_stats()
    })


# ============================================
# SWARM MEMORY BRIDGE
# ============================================

@router.post("/api/swarm-memory/enable")
async def enable_swarm_memory_endpoint():
    """Enable swarm memory bridge for persistent conversation storage."""
    s = _get_shared()
    try:
        from Cosmos.core.swarm_memory_integration import enable_swarm_memory
        memory = s.get_memory_system()
        await enable_swarm_memory(memory)
        return JSONResponse({
            "success": True,
            "message": "Swarm memory bridge enabled - conversations will be stored!"
        })
    except Exception as e:
        logger.error(f"Failed to enable swarm memory: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


@router.post("/api/swarm-memory/disable")
async def disable_swarm_memory_endpoint():
    """Disable swarm memory bridge."""
    try:
        from Cosmos.core.swarm_memory_integration import disable_swarm_memory
        await disable_swarm_memory()
        return JSONResponse({
            "success": True,
            "message": "Swarm memory bridge disabled"
        })
    except Exception as e:
        logger.error(f"Failed to disable swarm memory: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


@router.get("/api/swarm-memory/stats")
async def swarm_memory_stats():
    """Get swarm memory bridge statistics."""
    try:
        from Cosmos.core.swarm_memory_integration import get_swarm_memory_stats
        stats = await get_swarm_memory_stats()
        return JSONResponse({
            "available": True,
            **stats
        })
    except Exception as e:
        logger.error(f"Failed to get swarm memory stats: {e}")
        return JSONResponse({
            "available": False,
            "message": f"Not available: {str(e)}"
        })


@router.post("/api/swarm-memory/recall")
async def recall_swarm_memory(request: dict):
    """Recall relevant past swarm conversations."""
    try:
        from Cosmos.core.swarm_memory_integration import recall_swarm_context
        topic = request.get("topic", "")
        limit = request.get("limit", 5)

        context = await recall_swarm_context(topic, limit)
        return JSONResponse({
            "success": True,
            "context": context,
            "count": len(context)
        })
    except Exception as e:
        logger.error(f"Failed to recall swarm memory: {e}")
        return JSONResponse({
            "success": False,
            "context": [],
            "message": f"Failed: {str(e)}"
        })


@router.get("/api/turn-taking/stats")
async def turn_taking_stats():
    """Get smart turn-taking statistics."""
    try:
        from Cosmos.core.smart_turn_taking import get_turn_stats
        stats = get_turn_stats()
        return JSONResponse({
            "available": True,
            **stats
        })
    except Exception as e:
        logger.error(f"Failed to get turn stats: {e}")
        return JSONResponse({
            "available": False,
            "message": f"Not available: {str(e)}"
        })


# ============================================
# SEMANTIC DEDUPLICATION
# ============================================

@router.post("/api/memory/dedup/enable")
async def enable_memory_dedup(request: dict):
    """Enable semantic deduplication for memory storage."""
    s = _get_shared()
    try:
        from Cosmos.memory.dedup_integration import enable_deduplication
        auto_merge = request.get("auto_merge", False)
        memory = s.get_memory_system()
        await enable_deduplication(memory, auto_merge)
        return JSONResponse({
            "success": True,
            "message": f"Deduplication enabled (auto_merge: {auto_merge})"
        })
    except Exception as e:
        logger.error(f"Failed to enable dedup: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


@router.post("/api/memory/dedup/disable")
async def disable_memory_dedup():
    """Disable semantic deduplication."""
    try:
        from Cosmos.memory.dedup_integration import disable_deduplication
        disable_deduplication()
        return JSONResponse({
            "success": True,
            "message": "Deduplication disabled"
        })
    except Exception as e:
        logger.error(f"Failed to disable dedup: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


@router.get("/api/memory/dedup/stats")
async def memory_dedup_stats():
    """Get semantic deduplication statistics."""
    try:
        from Cosmos.memory.dedup_integration import get_deduplication_stats
        stats = get_deduplication_stats()
        return JSONResponse({
            "available": True,
            **stats
        })
    except Exception as e:
        logger.error(f"Failed to get dedup stats: {e}")
        return JSONResponse({
            "available": False,
            "message": f"Not available: {str(e)}"
        })


@router.post("/api/memory/dedup/check")
async def check_memory_duplicate(request: dict):
    """Check if content would be a duplicate before storing."""
    try:
        from Cosmos.memory.semantic_deduplication import check_for_duplicate
        content = request.get("content", "")

        if not content:
            return JSONResponse({
                "is_duplicate": False,
                "message": "No content provided"
            })

        match = check_for_duplicate(content)

        if match:
            return JSONResponse({
                "is_duplicate": match.is_duplicate,
                "similarity": match.similarity,
                "existing_id": match.memory_id,
                "message": "Duplicate" if match.is_duplicate else "Similar content found"
            })
        else:
            return JSONResponse({
                "is_duplicate": False,
                "similarity": 0.0,
                "message": "No duplicates found"
            })

    except Exception as e:
        logger.error(f"Failed to check duplicate: {e}")
        return JSONResponse({
            "is_duplicate": False,
            "message": f"Check failed: {str(e)}"
        })


# ============================================
# DELIBERATION & LIMITS
# ============================================

@router.get("/api/deliberations/stats")
async def deliberation_stats():
    """AGI v1.8: Get deliberation memory statistics."""
    try:
        from Cosmos.core.collective.dialogue_memory import get_dialogue_memory
        memory = get_dialogue_memory()

        stats = memory.get_stats()
        consensus_patterns = await memory.get_consensus_patterns()

        recent = await memory.get_recent_exchanges(limit=20)
        recent_summary = [
            {
                "id": e.exchange_id,
                "timestamp": e.timestamp,
                "winner": e.winning_agent,
                "consensus": e.consensus_reached,
                "participants": e.participating_agents,
                "session_type": e.session_type,
                "prompt_preview": e.prompt[:100] + "..." if len(e.prompt) > 100 else e.prompt,
            }
            for e in recent
        ]

        return JSONResponse({
            "success": True,
            "stats": stats,
            "consensus_patterns": consensus_patterns,
            "recent_exchanges": recent_summary,
            "storage_path": str(memory.storage_path),
        })

    except Exception as e:
        logger.error(f"Failed to get deliberation stats: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "DialogueMemory may not be initialized yet"
        }, status_code=500)


@router.get("/api/limits")
async def get_dynamic_limits():
    """AGI v1.8: Get all dynamic limits configuration."""
    try:
        from Cosmos.core.dynamic_limits import get_all_limits
        return JSONResponse({
            "success": True,
            **get_all_limits()
        })
    except Exception as e:
        logger.error(f"Failed to get dynamic limits: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/limits/model/{model_id}")
async def update_model_limits(model_id: str, request: Request):
    """AGI v1.8: Update limits for a specific model."""
    try:
        from Cosmos.core.dynamic_limits import update_model_limits as _update
        body = await request.json()

        success = _update(model_id, **body)
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Updated limits for {model_id}",
                "updates": body
            })
        else:
            return JSONResponse({
                "success": False,
                "error": f"Unknown model: {model_id}"
            }, status_code=404)
    except Exception as e:
        logger.error(f"Failed to update model limits: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/limits/session/{session_type}")
async def update_session_limits(session_type: str, request: Request):
    """AGI v1.8: Update limits for a specific session type."""
    try:
        from Cosmos.core.dynamic_limits import update_session_limits as _update
        body = await request.json()

        success = _update(session_type, **body)
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Updated limits for {session_type}",
                "updates": body
            })
        else:
            return JSONResponse({
                "success": False,
                "error": f"Unknown session type: {session_type}"
            }, status_code=404)
    except Exception as e:
        logger.error(f"Failed to update session limits: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/limits/deliberation")
async def update_deliberation_limits(request: Request):
    """AGI v1.8: Update deliberation character limits."""
    try:
        from Cosmos.core.dynamic_limits import update_deliberation_limits as _update
        body = await request.json()

        _update(
            critique=body.get("critique"),
            refine=body.get("refine"),
            propose=body.get("propose")
        )
        return JSONResponse({
            "success": True,
            "message": "Updated deliberation limits",
            "updates": body
        })
    except Exception as e:
        logger.error(f"Failed to update deliberation limits: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
