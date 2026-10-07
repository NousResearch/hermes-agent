"""``/api/shared-metrics/consent``: the dashboard's first-run shared-metrics offer.

The twin of Desktop's composer strip and the terminal's offer: one answer per profile, the same
config.yaml key (``telemetry.shared_metrics.enabled``), read and written through
``rabbit_cli.observability.shared_metrics_consent`` for the ``?profile=`` being managed.
Collection is local-only; nothing is uploaded.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from rabbit_cli.web_routers._common import config_write_scope, scoped_to_thread

router = APIRouter()


class ConsentAnswer(BaseModel):
    enabled: bool


def _read_consent() -> dict:
    """``{enabled, decided, managed}``; a managed install cannot save, so it is never offered."""
    from rabbit_cli.config import is_managed, read_raw_config
    from rabbit_cli.observability.shared_metrics_consent import consent_state

    return {**consent_state(read_raw_config()), "managed": is_managed()}


@router.get("/api/shared-metrics/consent")
async def get_shared_metrics_consent(profile: Optional[str] = None):
    return await scoped_to_thread(profile, _read_consent)


@router.put("/api/shared-metrics/consent")
async def put_shared_metrics_consent(body: ConsentAnswer, profile: Optional[str] = None):
    from rabbit_cli.observability.shared_metrics_consent import save_consent

    def _write() -> dict:
        with config_write_scope(profile):
            save_consent(body.enabled)
            return _read_consent()

    return await asyncio.to_thread(_write)
