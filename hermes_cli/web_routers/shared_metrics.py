"""``/api/shared-metrics/consent``: the dashboard's first-run shared-metrics offer.

The twin of Desktop's composer strip and the terminal's offer: one answer per profile, the same
two config.yaml keys (``telemetry.shared_metrics.enabled`` / ``.send``), read and written through
``hermes_cli.observability.shared_metrics_consent`` for the ``?profile=`` being managed.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from hermes_cli.web_routers._common import config_write_scope, scoped_to_thread

router = APIRouter()


class ConsentAnswer(BaseModel):
    enabled: bool
    send: bool = False


def _read_consent() -> dict:
    """``{enabled, send, decided, managed}``; a managed install cannot save, so it is never offered."""
    from hermes_cli.config import is_managed, read_raw_config
    from hermes_cli.observability.shared_metrics_consent import consent_state

    return {**consent_state(read_raw_config()), "managed": is_managed()}


@router.get("/api/shared-metrics/consent")
async def get_shared_metrics_consent(profile: Optional[str] = None):
    return await scoped_to_thread(profile, _read_consent)


@router.put("/api/shared-metrics/consent")
async def put_shared_metrics_consent(body: ConsentAnswer, profile: Optional[str] = None):
    from hermes_cli.observability.shared_metrics_consent import save_consent

    def _write() -> dict:
        with config_write_scope(profile):
            save_consent(body.enabled, body.send)
            return _read_consent()

    return await asyncio.to_thread(_write)
