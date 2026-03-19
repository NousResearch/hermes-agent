"""
Media Routes - TTS, Voice Cloning, Code Analysis, AirLLM

Endpoints:
- POST /api/speak - Generate speech with voice cloning
- GET /api/speak - Retrieve cached audio
- GET /api/speak/stats - TTS cache statistics
- POST /api/speak/bot - Generate speech as specific bot
- GET /api/voices - list available voices
- GET /api/voices/queue - Speech queue status
- POST /api/voices/queue/add - Add to speech queue
- POST /api/voices/queue/complete - Mark speech complete
- POST /api/code/analyze - Analyze Python code
- POST /api/code/analyze-project - Analyze project directory
- GET /api/airllm/stats - AirLLM statistics
- POST /api/airllm/start - Start AirLLM swarm
- POST /api/airllm/stop - Stop AirLLM swarm
- POST /api/airllm/queue - Queue AirLLM task
- GET /api/airllm/result/{task_id} - Get AirLLM result
"""

import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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

class AirLLMTaskRequest(BaseModel):
    task_type: str = "analyze"
    prompt: str
    priority: int = 5


# ============================================
# TEXT-TO-SPEECH WITH VOICE CLONING
# ============================================

@router.post("/api/speak")
async def speak_text_api():
    """Generate speech using XTTS v2 voice cloning with Cosmos's voice."""
    s = _get_shared()
    request = s.SpeakRequest  # Forward to the original handler
    # This endpoint is complex with many dependencies - delegate to server
    raise HTTPException(status_code=501, detail="Use server.py speak endpoint directly")


@router.post("/api/speak_impl")
async def speak_text_api_impl(request):
    """
    Generate speech using XTTS v2 voice cloning with Cosmos's voice.

    Uses Planetary Audio Shard for distributed caching.
    """
    s = _get_shared()
    try:
        text = request.text[:500]

        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        audio_dir = s.STATIC_DIR / "audio"
        reference_audio = audio_dir / "cosmos_reference.wav"

        text_hash = hashlib.md5(text.encode()).hexdigest()

        audio_shard = s.get_planetary_audio_shard()

        if audio_shard:
            local_path = audio_shard.get_audio(text_hash)
            if local_path and local_path.exists():
                logger.info(f"TTS: Local shard hit for {text_hash[:8]}...")
                return FileResponse(str(local_path), media_type="audio/wav")

            if audio_shard.has_remote_audio(text_hash):
                logger.info(f"TTS: Requesting {text_hash[:8]}... from P2P peer")
                peer_audio = await audio_shard.request_audio_from_peer(text_hash, timeout=5.0)
                if peer_audio:
                    local_path = audio_shard.get_audio(text_hash)
                    if local_path and local_path.exists():
                        logger.info(f"TTS: P2P cache hit for {text_hash[:8]}...")
                        return FileResponse(str(local_path), media_type="audio/wav")

        cache_dir = audio_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        simple_cache_path = cache_dir / f"{text_hash}.wav"

        if simple_cache_path.exists():
            logger.info(f"TTS: Simple cache hit for {text_hash[:8]}...")
            return FileResponse(str(simple_cache_path), media_type="audio/wav")

        model = s.get_tts_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="TTS model not available. Install TTS package."
            )

        if not reference_audio.exists():
            raise HTTPException(
                status_code=503,
                detail="Reference audio not found. Add cosmos_reference.wav to static/audio/"
            )

        logger.info(f"TTS: Generating speech for: {text[:50]}...")

        temp_path = cache_dir / f"{text_hash}_temp.wav"

        model.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language="en",
            file_path=str(temp_path)
        )

        try:
            import numpy as np
            import soundfile as sf
            data, sr = sf.read(str(temp_path))
            speed_factor = 1.15
            new_length = int(len(data) / speed_factor)
            indices = np.linspace(0, len(data) - 1, new_length).astype(int)
            sped_up = data[indices]
            sf.write(str(temp_path), sped_up, sr)
        except Exception as e:
            logger.debug(f"Could not speed up audio: {e}")

        with open(temp_path, "rb") as f:
            audio_data = f.read()

        if audio_shard and s.AUDIO_SHARD_AVAILABLE:
            final_path = await audio_shard.store_audio(
                text_hash=text_hash,
                audio_data=audio_data,
                voice_id="cosmos",
                scope=s.AudioScope.PLANETARY
            )
            temp_path.unlink(missing_ok=True)
            logger.info(f"TTS: Generated and stored in shard: {text_hash[:8]}...")
            return FileResponse(str(final_path), media_type="audio/wav")
        else:
            temp_path.rename(simple_cache_path)
            logger.info(f"TTS: Generated and cached: {text_hash[:8]}...")
            return FileResponse(str(simple_cache_path), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/speak")
async def get_cached_audio(text_hash: str = None):
    """Retrieve cached audio by text hash."""
    s = _get_shared()
    if not text_hash:
        raise HTTPException(status_code=400, detail="text_hash parameter required")

    try:
        audio_shard = s.get_planetary_audio_shard()
        if audio_shard:
            local_path = audio_shard.get_audio(text_hash)
            if local_path and Path(local_path).exists():
                logger.info(f"TTS GET: Shard hit for {text_hash[:8]}...")
                return FileResponse(str(local_path), media_type="audio/wav")

        cache_dir = s.STATIC_DIR / "audio" / "cache"
        cache_path = cache_dir / f"{text_hash}.wav"
        if cache_path.exists():
            logger.info(f"TTS GET: Cache hit for {text_hash[:8]}...")
            return FileResponse(str(cache_path), media_type="audio/wav")

        # Read from the file
        cache_dir = Path.home() / ".cosmos" / "tts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_cache = cache_dir / f"{text_hash}.wav"
        if temp_cache.exists():
            logger.info(f"TTS GET: Temp cache hit for {text_hash[:8]}...")
            return FileResponse(str(temp_cache), media_type="audio/wav")

        raise HTTPException(
            status_code=202,
            detail="Audio still generating, retry in a moment"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS GET error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/speak/stats")
async def get_tts_stats():
    """Get TTS cache statistics including P2P network info."""
    s = _get_shared()
    audio_shard = s.get_planetary_audio_shard()

    if audio_shard:
        stats = audio_shard.get_stats()
        stats["tts_available"] = s.TTS_AVAILABLE
        stats["p2p_enabled"] = s.AUDIO_SHARD_AVAILABLE
        return JSONResponse(stats)

    return JSONResponse({
        "local_entries": 0,
        "global_entries": 0,
        "total_size_mb": 0,
        "tts_available": s.TTS_AVAILABLE,
        "p2p_enabled": False
    })


# ============================================
# MULTI-VOICE TTS
# ============================================

@router.post("/api/speak/bot")
async def speak_as_bot(request):
    """Generate speech using a specific bot's voice."""
    s = _get_shared()
    if not s.MULTI_VOICE_AVAILABLE:
        return await speak_text_api_impl(request)

    try:
        bot_name = request.bot_name or "Cosmos"
        text = request.text[:1000]

        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        voice_system = s.get_multi_voice_system()
        audio_path = await voice_system.generate_speech(text, bot_name)

        if audio_path and audio_path.exists():
            return FileResponse(
                str(audio_path),
                media_type="audio/wav",
                headers={
                    "X-Bot-Name": bot_name,
                    "X-Voice-System": "multi-voice",
                }
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Voice generation failed for {bot_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-voice TTS error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/voices")
async def list_voices():
    """list all available swarm voices."""
    s = _get_shared()
    if not s.MULTI_VOICE_AVAILABLE:
        return JSONResponse({
            "available": False,
            "voices": []
        })

    voice_system = s.get_multi_voice_system()
    voices = voice_system.get_available_voices()

    for name, config in voices.items():
        ref_path = voice_system._find_reference_audio(
            voice_system.get_voice_config(name)
        )
        voices[name]["has_reference_audio"] = ref_path is not None

    return JSONResponse({
        "available": True,
        "qwen3_tts": s.QWEN3_TTS_AVAILABLE if s.MULTI_VOICE_AVAILABLE else False,
        "fish_speech": s.FISH_SPEECH_AVAILABLE if s.MULTI_VOICE_AVAILABLE else False,
        "xtts": s.MULTI_XTTS_AVAILABLE if s.MULTI_VOICE_AVAILABLE else False,
        "voices": voices
    })


@router.get("/api/voices/queue")
async def get_speech_queue_status():
    """Get current speech queue status for sequential playback."""
    s = _get_shared()
    if not s.MULTI_VOICE_AVAILABLE:
        return JSONResponse({"queue": [], "is_speaking": False})

    queue = s.get_speech_queue()
    return JSONResponse(queue.get_status())


@router.post("/api/voices/queue/add")
async def add_to_speech_queue(request):
    """Add speech to the sequential playback queue."""
    s = _get_shared()
    if not s.MULTI_VOICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Multi-voice not available")

    queue = s.get_speech_queue()
    voice_system = s.get_multi_voice_system()

    bot_name = request.bot_name or "Cosmos"
    text = request.text[:1000]

    audio_path = await voice_system.generate_speech(text, bot_name)

    if audio_path:
        position = await queue.add_to_queue(
            bot_name=bot_name,
            text=text,
            audio_url=f"/api/speak/cached/{audio_path.stem}",
        )

        return JSONResponse({
            "success": True,
            "position": position,
            "bot_name": bot_name,
            "audio_ready": True
        })
    else:
        raise HTTPException(status_code=503, detail="Failed to generate audio")


@router.post("/api/voices/queue/complete")
async def mark_speech_complete(bot_name: str):
    """Mark that a bot has finished speaking."""
    s = _get_shared()
    if not s.MULTI_VOICE_AVAILABLE:
        return JSONResponse({"success": True})

    queue = s.get_speech_queue()
    await queue.mark_complete(bot_name)

    next_speaker = await queue.get_next_speaker()

    return JSONResponse({
        "success": True,
        "next_speaker": next_speaker["bot_name"] if next_speaker else None,
        "next_audio_url": next_speaker.get("audio_url") if next_speaker else None
    })


# ============================================
# CODE ANALYSIS API
# ============================================

ALLOWED_ANALYSIS_DIRS = [
    "/workspace/Cosmos",
    "/workspace",
]


def validate_path_security(path: str, allowed_bases: list = None) -> bool:
    """Validate path is within allowed directories to prevent traversal attacks."""
    if allowed_bases is None:
        allowed_bases = ALLOWED_ANALYSIS_DIRS

    try:
        resolved = Path(path).resolve()
        resolved_str = str(resolved)

        for base in allowed_bases:
            base_resolved = str(Path(base).resolve())
            if resolved_str.startswith(base_resolved):
                return True
        return False
    except Exception:
        return False


@router.post("/api/code/analyze")
async def analyze_code_api(request):
    """Analyze Python code for complexity, security issues, and patterns."""
    from fastapi import Request as FastAPIRequest
    try:
        from Cosmos.tools.code_analyzer import analyze_python_code, analyze_python_file, scan_code_security

        body = await request.json()
        code = body.get("code")
        filepath = body.get("file")

        if code:
            metrics = analyze_python_code(code)
            security = scan_code_security(code)
        elif filepath:
            if not validate_path_security(filepath):
                raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

            metrics = analyze_python_file(filepath)
            with open(filepath, 'r') as f:
                security = scan_code_security(f.read())
        else:
            raise HTTPException(status_code=400, detail="Provide 'code' or 'file' in request body")

        if not metrics:
            raise HTTPException(status_code=400, detail="Failed to parse code")

        return JSONResponse({
            "success": True,
            "metrics": {
                "path": metrics.path,
                "lines": metrics.num_lines,
                "functions": metrics.num_functions,
                "classes": metrics.num_classes,
                "imports": list(metrics.imports),
                "todos": metrics.todos,
                "fixmes": metrics.fixmes,
                "function_details": [
                    {
                        "name": f.name,
                        "line": f.lineno,
                        "complexity": f.complexity,
                        "cognitive_complexity": f.cognitive_complexity,
                        "params": f.num_params,
                        "lines": f.num_lines
                    }
                    for f in metrics.functions
                ]
            },
            "security_issues": security
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/api/code/analyze-project")
async def analyze_project_api(request):
    """Analyze an entire project directory."""
    try:
        from Cosmos.tools.code_analyzer import analyze_project

        body = await request.json()
        directory = body.get("directory", "/workspace/Cosmos")

        if not validate_path_security(directory):
            raise HTTPException(status_code=403, detail="Access denied: directory outside allowed paths")

        report = analyze_project(directory)

        return JSONResponse({
            "success": True,
            "report": report
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze project: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Failed: {str(e)}"
        })


# ============================================
# AIRLLM BACKGROUND PROCESSING
# ============================================

@router.get("/api/airllm/stats")
async def airllm_stats():
    """Get AirLLM side swarm statistics."""
    try:
        from Cosmos.core.airllm_swarm import get_airllm_swarm
        swarm = get_airllm_swarm()
        if swarm:
            return swarm.get_stats()
        return {"available": False, "message": "AirLLM swarm not initialized"}
    except ImportError:
        return {"available": False, "message": "AirLLM module not installed"}


@router.post("/api/airllm/start")
async def airllm_start():
    """Initialize and start the AirLLM side swarm."""
    try:
        from Cosmos.core.airllm_swarm import initialize_airllm_swarm
        swarm = await initialize_airllm_swarm()
        return {"success": True, "message": "AirLLM side swarm started", "stats": swarm.get_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/airllm/stop")
async def airllm_stop():
    """Stop the AirLLM side swarm."""
    try:
        from Cosmos.core.airllm_swarm import get_airllm_swarm
        swarm = get_airllm_swarm()
        if swarm:
            await swarm.stop()
            return {"success": True, "message": "AirLLM side swarm stopped"}
        return {"success": False, "message": "AirLLM swarm not running"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/airllm/queue")
async def airllm_queue_task(request: AirLLMTaskRequest):
    """Queue a task for background processing by AirLLM."""
    try:
        from Cosmos.core.airllm_swarm import get_airllm_swarm, initialize_airllm_swarm

        swarm = get_airllm_swarm()
        if not swarm:
            swarm = await initialize_airllm_swarm()

        task_id = swarm.queue_task(
            task_type=request.task_type,
            prompt=request.prompt,
            priority=request.priority
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Task queued for background processing",
            "queue_size": len(swarm.task_queue)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/api/airllm/result/{task_id}")
async def airllm_get_result(task_id: str):
    """Get result of a background task."""
    try:
        from Cosmos.core.airllm_swarm import get_airllm_swarm
        swarm = get_airllm_swarm()
        if not swarm:
            return {"success": False, "message": "AirLLM swarm not running"}

        result = swarm.get_result(task_id)
        if result:
            return {"success": True, "result": result}
        return {"success": False, "message": "Task not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# IMAGE GENERATION API
# ============================================

@router.get("/api/media/generate_image")
async def generate_image_api(prompt: str):
    """Generate an image via integration tools and return the FileResponse directly."""
    try:
        from Cosmos.integration.image_gen.generator import get_image_generator
        generator = get_image_generator()
        
        # Pull live IBM Quantum Entropy for Phera's creativity
        fez_entropy = 0.0
        try:
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            bridge = get_quantum_bridge()
            if bridge:
                physics = {}
                try:
                    field = s.FIELD if hasattr(s, 'FIELD') else None
                    if field:
                        physics = field.user_physics
                except: pass
                fez_entropy = bridge.get_entropy(physics)
        except Exception as e:
            logger.debug(f"[MEDIA] Quantum bridge unavailable for entropy: {e}")
            
        if fez_entropy > 0.3:
            logger.info(f"[QUANTUM] Injecting FEZ variance into image prompt (entropy={fez_entropy:.4f})")
            prompt += f". [Quantum Variance: {fez_entropy:.4f}]. Inject unpredictable, subtle creative anomalies and geometric variance into the composition based on this level of quantum chaos."

        # Save to a temporary file
        from pathlib import Path
        import time
        
        # We need a stable hash to prevent duplicate generation for the same prompt during UI reconnects
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cache_dir = Path.home() / ".cosmos" / "image_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{prompt_hash}.jpg"
        
        if out_path.exists():
            return FileResponse(str(out_path), media_type="image/jpeg")
            
        logger.info(f"[IMAGE GEN] Generating brand new image for: '{prompt}'")
        
        # Use underlying API
        image_bytes = None
        
        if generator.grok.is_available():
            image_bytes = await generator.grok.generate(prompt)
            
        # Fallback to gemini if grok failed or unavailable
        if not image_bytes and generator.gemini.is_available():
            # gemini returns bytes directly from _generate_imagen
            res = await generator.gemini.generate(prompt)
            if res:
                image_bytes = res
                
        if not image_bytes:
            raise HTTPException(status_code=503, detail="No valid image generator backend configured.")
            
        with open(out_path, "wb") as f:
            f.write(image_bytes)
            
        return FileResponse(str(out_path), media_type="image/jpeg", headers={"Cache-Control": "max-age=31536000"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[IMAGE GEN] Failed to generate visual: {e}")
        raise HTTPException(status_code=500, detail="Internal server failure while generating image")
