"""Example dashboard plugin — backend API routes.

Mounted at /api/plugins/example/ by the dashboard plugin system.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/hello")
async def hello():
    """Simple greeting endpoint to demonstrate plugin API routes."""
    return {"message": "Hello from the example plugin!", "plugin": "example", "version": "1.0.0"}
