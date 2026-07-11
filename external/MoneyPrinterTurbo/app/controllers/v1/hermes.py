"""Hermes sidecar identity endpoints."""

from __future__ import annotations

import os

from app.config import config
from app.controllers.v1.base import new_router
from app.utils import utils


router = new_router()


@router.get("/hermes/health")
def hermes_health():
    return utils.get_response(
        200,
        {
            "managed": bool(os.getenv("MONEYPRINTER_HERMES_TOKEN", "").strip()),
            "protocol_version": 1,
            "service": "moneyprinterturbo",
            "version": config.project_version,
        },
    )
