import os
import json
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Hermes imports
from run_agent import AIAgent
from hermes_cli.config import load_config, save_config
from hermes_state import SessionDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hermes_dashboard")

app = FastAPI(title="Hermes Agent Dashboard")

# Setup templates and static files
templates = Jinja2Templates(directory="dashboard/templates")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Global state
active_sessions: Dict[str, AIAgent] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None
    theme: Optional[str] = None
    language: Optional[str] = None

def get_agent(session_id: str) -> AIAgent:
    if session_id in active_sessions:
        return active_sessions[session_id]

    config = load_config()
    model_cfg = config.get("model", {})

    # Extract model and provider from config
    if isinstance(model_cfg, dict):
        model = model_cfg.get("default", "")
        provider = model_cfg.get("provider", "auto")
        base_url = model_cfg.get("base_url", "")
    else:
        model = model_cfg
        provider = "auto"
        base_url = ""

    session_db = SessionDB()

    agent = AIAgent(
        model=model,
        provider=provider,
        base_url=base_url,
        session_id=session_id,
        platform="dashboard",
        session_db=session_db,
        quiet_mode=True
    )
    active_sessions[session_id] = agent
    return agent

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/config")
async def get_dashboard_config():
    config = load_config()
    return {
        "model": config.get("model", {}).get("default") if isinstance(config.get("model"), dict) else config.get("model"),
        "provider": config.get("model", {}).get("provider") if isinstance(config.get("model"), dict) else "auto",
        "theme": config.get("display", {}).get("theme", "dark"),
        "language": config.get("display", {}).get("language", "en")
    }

@app.post("/api/config")
async def update_dashboard_config(update: ConfigUpdate):
    config = load_config()

    if update.model or update.provider:
        if not isinstance(config.get("model"), dict):
            config["model"] = {"default": config.get("model", "")}
        if update.model:
            config["model"]["default"] = update.model
        if update.provider:
            config["model"]["provider"] = update.provider

    if update.theme or update.language:
        if "display" not in config:
            config["display"] = {}
        if update.theme:
            config["display"]["theme"] = update.theme
        if update.language:
            config["display"]["language"] = update.language

    save_config(config)
    return {"status": "success"}

@app.get("/api/sessions")
async def list_sessions():
    try:
        db = SessionDB()
        sessions = db.list_sessions_rich(limit=20)
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return []

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get the history of a specific session."""
    try:
        db = SessionDB()
        history = db.get_messages_as_conversation(session_id)
        if not history:
            # Check if it exists
            session = db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"history": []}
        return {"history": history}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or f"web_{uuid.uuid4().hex[:8]}"
    agent = get_agent(session_id)

    # Non-streaming chat for simplicity in first iteration
    result = agent.run_conversation(request.message)
    return {
        "session_id": session_id,
        "response": result.get("final_response"),
        "history": result.get("messages")
    }

@app.get("/api/chat/stream")
async def chat_stream(message: str, session_id: Optional[str] = None):
    sid = session_id or f"web_{uuid.uuid4().hex[:8]}"
    agent = get_agent(sid)

    def event_generator():
        # This is a bit tricky with the current sync run_conversation
        # We might need a separate thread or use the stream_delta_callback

        yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\n\n"

        def stream_cb(delta):
            if delta:
                # We need to escape the delta for SSE if it's not handled by json.dumps
                data = json.dumps({'type': 'delta', 'content': delta})
                # Using a queue would be better for thread safety
                # But for now let's try a simple yield (Wait, this won't work inside a callback called from another thread easily)
                pass

        # For the sake of this task, I'll implement a basic version that runs in a thread
        # and pushes to a queue that the generator reads from.

        import queue
        q = queue.Queue()

        def run_agent():
            try:
                def inner_cb(delta):
                    if delta:
                        q.put({'type': 'delta', 'content': delta})

                result = agent.run_conversation(message, stream_delta_callback=inner_cb)
                q.put({'type': 'done', 'response': result.get('final_response')})
            except Exception as e:
                q.put({'type': 'error', 'message': str(e)})
            finally:
                q.put(None)

        threading.Thread(target=run_agent).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
