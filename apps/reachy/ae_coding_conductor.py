from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
WORKSPACE = r"C:\æ\hermes-fork"

class PlanRequest(BaseModel):
    intent: str
    target_dir: str | None = WORKSPACE

class RunRequest(BaseModel):
    agent: str = "hermes-local"
    intent: str
    files: list[str] | None = None

@router.get("/health")
def health():
    return {"ok": True, "surface": "ae-coding-conductor", "workdir": WORKSPACE}

@router.get("/plan")
def plan_get(intent: str = "", target_dir: str = WORKSPACE):
    return {
        "mode": "plan",
        "intent": intent,
        "target": target_dir,
        "steps": ["scope", "trace", "execute", "verify", "commit"],
        "agents": ["hermes-agent", "claude-code", "codex", "opencode"],
    }

@router.post("/plan")
def plan_post(req: PlanRequest):
    return {
        "mode": "plan",
        "intent": req.intent,
        "target": req.target_dir,
        "steps": ["scope", "trace", "execute", "verify", "commit"],
        "agents": ["hermes-agent", "claude-code", "codex", "opencode"],
    }

@router.post("/run")
def run(req: RunRequest):
    return {
        "ok": True,
        "mode": "run",
        "agent": req.agent,
        "intent": req.intent,
        "scope": {"workdir": WORKSPACE, "recent_files": []},
    }

@router.post("/trace")
def trace(intent: str = "", payload: dict | None = None):
    return {
        "ok": True,
        "mode": "trace",
        "intent": intent,
        "path": ["intent", "plan", "execute", "verify", "surface"],
        "latest_payload": payload or {},
    }

@router.get("/skills")
def skills():
    return {
        "skills": [
            "hermes-operator-grammar",
            "subagent-driven-development",
            "test-driven-development",
            "systematic-debugging",
            "code-review",
            "plan",
            "requesting-code-review",
        ]
    }

@router.get("/agents")
def agents():
    return {
        "agents": [
            "hermes-agent",
            "claude-code",
            "codex",
            "opencode",
        ]
    }

@router.get("/workspace")
def workspace():
    return {"workdir": WORKSPACE}
