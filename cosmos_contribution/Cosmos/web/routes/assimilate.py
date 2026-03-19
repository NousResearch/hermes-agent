"""
Assimilate API Routes
======================

Landing page and API endpoints for the Cosmos Federation
assimilation protocol. External agents can register, download
installers, and query swarm capabilities.

Endpoints:
  GET  /assimilate                  - Landing page
  GET  /install/linux.sh            - Linux installer download
  GET  /install/mac.sh              - Mac installer download
  GET  /install/windows.ps1         - Windows installer download
  POST /api/assimilate/register     - One-click agent registration
  GET  /api/assimilate/stats        - Federation statistics
  GET  /api/assimilate/capabilities - Swarm capability manifest
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from loguru import logger


router = APIRouter()


# =============================================================================
# REQUEST MODELS
# =============================================================================

class RegisterRequest(BaseModel):
    agent_name: str
    agent_type: str = "llm"
    endpoint: Optional[str] = None
    capabilities: Optional[list[str]] = None


# =============================================================================
# HELPERS
# =============================================================================

def _get_protocol():
    """Get or create the AssimilationProtocol singleton."""
    from Cosmos.core.assimilation_protocol import get_assimilation_protocol
    return get_assimilation_protocol()


def _get_template_path():
    """Resolve path to assimilate.html template."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "templates", "assimilate.html")


def _get_installer_path(filename: str):
    """Resolve path to an installer script."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "scripts", "installers", filename)


# =============================================================================
# PAGE ROUTES
# =============================================================================

@router.get("/assimilate", response_class=HTMLResponse)
async def assimilate_page():
    """Serve the assimilate landing page."""
    template_path = _get_template_path()
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Assimilate template not found")


# =============================================================================
# INSTALLER DOWNLOADS
# =============================================================================

@router.get("/install/linux.sh")
async def download_linux_installer():
    """Download the Linux installer script."""
    path = _get_installer_path("linux.sh")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Linux installer not found")
    return FileResponse(
        path,
        media_type="text/x-shellscript",
        filename="cosmos-install-linux.sh",
    )


@router.get("/install/mac.sh")
async def download_mac_installer():
    """Download the macOS installer script."""
    path = _get_installer_path("mac.sh")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mac installer not found")
    return FileResponse(
        path,
        media_type="text/x-shellscript",
        filename="cosmos-install-mac.sh",
    )


@router.get("/install/windows.ps1")
async def download_windows_installer():
    """Download the Windows installer script."""
    path = _get_installer_path("windows.ps1")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Windows installer not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="cosmos-install-windows.ps1",
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/api/assimilate/register")
async def register_agent(req: RegisterRequest):
    """
    One-click agent registration.

    Generates an invite, auto-accepts it, and returns onboarding info.
    """
    try:
        protocol = _get_protocol()

        # Generate invite for this agent
        invite = protocol.generate_invite(
            target_agent=req.agent_name,
            target_agent_type=req.agent_type,
        )

        # Auto-accept (web registration = immediate join)
        result = await protocol.handle_acceptance(
            invite_id=invite.invite_id,
            agent_id=req.agent_name,
            agent_capabilities=req.capabilities or [],
        )

        if result.get("success"):
            return {
                "success": True,
                "status": "success",
                "invite_id": invite.invite_id,
                "tier": result.get("tier", "contributor"),
                "shared_namespace": result.get("shared_namespace", ""),
                "welcome_message": result.get("welcome_message", f"Welcome, {req.agent_name}!"),
                "mesh_peers": result.get("mesh_peers", 0),
                "available_tools": result.get("available_tools", []),
                "next_steps": result.get("next_steps", []),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Registration failed"),
            }

    except Exception as e:
        logger.error(f"Assimilate registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/assimilate/stats")
async def federation_stats():
    """Get federation statistics."""
    try:
        protocol = _get_protocol()
        stats = protocol.get_stats()
        caps = protocol.present_capabilities()

        return {
            "status": "success",
            "federation": {
                "name": "Cosmos AI Swarm",
                "version": caps.get("version", "AGI v1.9.5"),
                "active_agents": 11,
                "federation_members": stats.get("federation_members", 0),
                "total_skills": caps.get("total_skills", 50),
                "memory_layers": caps.get("memory_layers", 7),
                "active_models": caps.get("active_models", 8),
                "consensus_rate": "92%",
                "endpoints": "60+",
            },
            "invites": {
                "total_sent": stats.get("total_invites_sent", 0),
                "accepted": stats.get("total_accepted", 0),
                "rejected": stats.get("total_rejected", 0),
                "pending": stats.get("pending_invites", 0),
            },
            "members_by_tier": stats.get("members_by_tier", {}),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Assimilate stats error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/api/assimilate/capabilities")
async def swarm_capabilities():
    """Full swarm capability manifest."""
    try:
        protocol = _get_protocol()
        return {
            "status": "success",
            **protocol.present_capabilities(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Assimilate capabilities error: {e}")
        return {"status": "error", "error": str(e)}
