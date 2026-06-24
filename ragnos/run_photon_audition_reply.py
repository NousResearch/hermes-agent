#!/usr/bin/env python3
"""RAGnos sibling-service entrypoint: Photon inbound adapter -> Keez CoS over text.

The inbound runner. It installs the UNIFIED conversational-CoS handler from
``home_agent.cos_text_handler.build_live_cos_handler``, gated by
``KEEZ_TEXT_COS_ENABLED``:
  off (default) -> delegates verbatim to the audition-reply handler (today's
                   behavior: reply to the most-recent audition alert, confirm,
                   send via gws). Nothing else is answered.
  on            -> full conversational CoS: text Keez anything; it answers drawing
                   from the entire system (brain + Librarian + live sources) and
                   takes actions (auto low-risk / confirm high-risk). Audition
                   reply is absorbed as the email-reply path.

Sidecar policy (CRITICAL): this runner does NOT spawn a sidecar. It sets
``PHOTON_SIDECAR_AUTOSTART=false`` so it connects to the EXISTING durable
``io.ragnos.photon-sidecar`` (127.0.0.1:8789) and consumes its ``/inbound``
stream. Two sidecars would fight over the port; there must be exactly one
(the launchd-managed photon-sidecar). See scripts/launchd/photon-sidecar.sh.

Inbound:  iMessage -> Photon -> (existing) sidecar -> this adapter -> audition
          reply handler -> gws gmail reply. The adapter auto-sends the handler's
          returned confirm/sent/cancel string back to the operator's chat.

Env (rendered from Infisical in production):
  PHOTON_PROJECT_ID, PHOTON_PROJECT_SECRET   from ``hermes photon setup``
  PHOTON_SIDECAR_TOKEN                        shared secret (sidecar control)
  PHOTON_ALLOWED_USERS                        E.164 allowlist (operator number)
  HOME_AGENT_SRC                              path to RAGnos tools/home-agent/src
  RAGNOS_WORKSPACE                            repo root (gws.js + threads file)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys


def _allowlist() -> list[str]:
    raw = os.environ.get("PHOTON_ALLOWED_USERS", "")
    return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]


async def _run() -> int:
    logging.basicConfig(
        level=os.environ.get("PHOTON_KEEZ_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("photon-audition-reply")

    # Never spawn a competing sidecar: consume the durable photon-sidecar's stream.
    os.environ.setdefault("PHOTON_SIDECAR_AUTOSTART", "false")

    home_src = os.environ.get("HOME_AGENT_SRC")
    if home_src and home_src not in sys.path:
        sys.path.insert(0, home_src)

    from gateway.config import PlatformConfig
    from plugins.platforms.photon.adapter import PhotonAdapter
    from home_agent.cos_text_handler import build_live_cos_handler

    allow = _allowlist()
    if not allow:
        log.error("PHOTON_ALLOWED_USERS empty -- refusing to start (would block all). Set it in Infisical.")
        return 78  # EX_CONFIG: do not crash-loop fast on a config gap

    # Unified conversational-CoS handler. Gated by KEEZ_TEXT_COS_ENABLED:
    #   off (default) -> delegates verbatim to the audition-reply handler (today's behavior).
    #   on            -> full CoS (brain + live sources + audition absorbed).
    adapter = PhotonAdapter(PlatformConfig(extra={}))
    adapter.set_message_handler(build_live_cos_handler(os.environ))

    if not await adapter.connect():
        log.error("adapter.connect() failed (missing creds or sidecar). See logs above.")
        return 1
    cos_enabled = os.environ.get("KEEZ_TEXT_COS_ENABLED", "").strip().lower()
    mode = "full CoS" if cos_enabled in {"1", "true", "yes", "on"} else "audition-reply"
    log.info("connected; %s inbound active (allowlist=%d). Ctrl-C to stop.", mode, len(allow))
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shutting down")
    finally:
        await adapter.disconnect()
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
