from typing import Any

from fastapi import APIRouter

from backend.services.settings_store import get_settings, save_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def read_settings() -> dict[str, Any]:
    return get_settings()


@router.put("")
async def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    return save_settings(payload)
