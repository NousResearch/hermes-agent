from typing import Any

from fastapi import APIRouter, HTTPException

from backend.services.codex_bridge import CodexBridge
from backend.services.hermes_bridge import HermesBridge
from backend.services.settings_store import get_settings

router = APIRouter(prefix="/api/adapters", tags=["adapters"])


@router.get("")
async def get_adapters() -> list[dict[str, str]]:
    hermes_status = "offline"
    hermes_version = "unknown"
    hermes_model = "unknown"
    codex_status = "offline"
    codex_version = "unknown"
    codex_model = "unknown"

    try:
        if HermesBridge().ping() == "pong":
            hermes_status = "online"
            hermes_version = "0.1.0"
            hermes_model = "Hermes Adapter"
    except Exception:
        pass

    try:
        codex = CodexBridge()
        if codex.ping() == "pong":
            codex_status = "online"
            status = codex.status()
            codex_version = str(status.get("version", "unknown"))
            codex_model = str(status.get("model", "unknown"))
    except Exception:
        pass

    return [
        {
            "name": "hermes",
            "status": hermes_status,
            "version": hermes_version,
            "model": hermes_model,
        },
        {
            "name": "codex",
            "status": codex_status,
            "version": codex_version,
            "model": codex_model,
        },
    ]


@router.post("/codex/exec")
async def exec_codex(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        settings = get_settings()
        return CodexBridge().exec(
            prompt=prompt,
            workdir=str(payload.get("workdir") or settings.get("codex_workdir") or r"D:\Codex"),
            timeout=int(payload.get("timeout") or 120),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to execute Codex adapter: {exc}") from exc


@router.get("/codex/session-last")
async def get_codex_session_last() -> dict[str, Any] | None:
    try:
        return CodexBridge().session_last()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to read Codex session: {exc}") from exc
