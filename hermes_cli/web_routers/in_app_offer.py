"""``/api/in-app-offer``: the Nous Portal's in-app offer for the dashboard (``hermes dashboard``).

The dashboard is a separate SPA over REST, so it reads the same per-profile memo that
``free_tier.status`` serves Hermes Desktop and the Webapp (``hermes_cli.in_app_offer``), for the
``?profile=`` being managed. One route, two shapes:

- ``GET`` answers from the memo: a pure local read, safe to poll.
- ``GET ?refresh=1`` first fetches the account once with the stored token (never refreshing it), the
  same network leg as the ``in_app_offer.refresh`` RPC, so the offer reaches a user whose sessions
  never fetch the account. The dashboard calls it after boot settles and on a slow cadence.

Both answer ``{"in_app_offer": {...}}`` or ``{"in_app_offer": null}``, never an error: an offer is
optional and a failed fetch keeps the last answer. ``POST /api/in-app-offer/event`` reports a show or a
click for the Portal's counts, through the same ``report_event`` the Desktop RPC uses.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

logger = logging.getLogger(__name__)
router = APIRouter()


def _read(refresh: bool) -> dict:
    from hermes_cli import in_app_offer

    try:
        offer = in_app_offer.refresh_from_portal() if refresh else in_app_offer.pending_offer()
    except Exception:  # an offer is optional: log and answer "none"
        logger.warning("in-app offer read failed", exc_info=True)
        offer = None
    return {"in_app_offer": offer}


@router.get("/api/in-app-offer")
async def get_in_app_offer(profile: Optional[str] = None, refresh: bool = False):
    return await scoped_to_thread(profile, lambda: _read(refresh))


class InAppOfferEvent(BaseModel):
    offer_id: str
    event: str


@router.post("/api/in-app-offer/event")
async def report_in_app_offer_event(body: InAppOfferEvent, profile: Optional[str] = None):
    """Tell the Portal the offer was shown or clicked (its campaign counts), for the managed
    profile. Best-effort like the Desktop RPC: ``{"reported": false}`` on any failure."""
    from hermes_cli import in_app_offer

    def _report() -> dict:
        try:
            return {"reported": in_app_offer.report_event(body.offer_id, body.event)}
        except Exception:  # a count is never worth an error
            logger.warning("in-app offer report failed", exc_info=True)
            return {"reported": False}

    return await scoped_to_thread(profile, _report)
