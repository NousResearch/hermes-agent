"""Point discord.py at a local Discord stand-in (tests only).

discord.py has no base-URL knob: REST URLs are built from the class attribute
``discord.http.Route.BASE`` and the gateway socket from
``discord.gateway.DiscordWebSocket.DEFAULT_GATEWAY`` (non-sharded ``Client.connect`` never calls
``GET /gateway/bot``; the resume URL comes from READY's ``resume_gateway_url``, which the stand-in
also points at itself). This module is put on a child process's ``PYTHONPATH`` by the Discord driver
and does nothing unless ``HERMES_STANDIN_DISCORD_API`` is set. It patches the SDK only; no Hermes
code is touched. Side effect: when set, discord.py (and yarl/aiohttp) are imported eagerly at
interpreter startup, before Hermes runs, rather than lazily by the adapter.
"""

import os

_api = os.environ.get("HERMES_STANDIN_DISCORD_API")
_gw = os.environ.get("HERMES_STANDIN_DISCORD_GATEWAY")
if _api:
    try:
        import yarl
        import discord.gateway
        import discord.http

        discord.http.Route.BASE = _api.rstrip("/")
        if _gw:
            discord.gateway.DiscordWebSocket.DEFAULT_GATEWAY = yarl.URL(_gw)
        os.environ["HERMES_STANDIN_DISCORD_SHIM_LOADED"] = "1"
    except ImportError:  # discord.py absent in this interpreter: nothing to redirect
        pass
