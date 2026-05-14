"""Local OpenAI-compatible proxy that forwards to OAuth-authenticated upstreams.

Lets external apps (OpenViking, Karakeep, Open WebUI, ...) ride the user's
already-logged-in provider subscription instead of needing a static API key
copy-pasted into each app's config.

The proxy listens on ``127.0.0.1:<port>``, accepts any bearer (the client's
``Authorization`` header is discarded), and attaches the user's real
upstream credential to the forwarded request. The credential is refreshed
automatically when it approaches expiry.

First-class adapter:
  - ``nous`` — Nous Portal (https://inference-api.nousresearch.com/v1)

Future adapters can plug in by implementing ``UpstreamAdapter``.
"""

from hermes_cli.proxy.adapters.base import UpstreamAdapter

__all__ = ["UpstreamAdapter"]
