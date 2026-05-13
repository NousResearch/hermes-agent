from fastapi import APIRouter

from ..config import VERSION
from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)
