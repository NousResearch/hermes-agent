"""Request identity of the OpenCode client — shared by every keyless Zen free-tier path.

The Zen relay's free tier (``https://opencode.ai/zen/v1``, served anonymously) answers HTTP 429
``FreeUsageLimitError`` to requests carrying the canonical Hermes attribution set
(``User-Agent: HermesAgent/<v>`` + ``HTTP-Referer: https://hermes-agent.nousresearch.com`` +
``X-Title: Hermes Agent``), while the same model returns 200 when the request looks like the
OpenCode client (#106495). Keyed Zen/Go traffic keeps the Hermes attribution — only the anonymous
free tier needs this fingerprint.

Two callers need these values and they must never drift:
``hermes_cli.models.opencode_zen_free_headers`` (the runtime/client-build owner for keyless routes)
and the ``opencode-free`` provider profile's ``default_headers``. This module is deliberately
stdlib-only: model-provider plugin discovery runs *during* ``hermes_cli.models``' own import, so a
plugin importing ``hermes_cli.models`` at module scope would import a half-built module.
"""

from __future__ import annotations

import uuid

OPENCODE_CLIENT_USER_AGENT = "opencode/0.20.5"
OPENCODE_CLIENT_REFERER = "https://opencode.ai/"
OPENCODE_CLIENT_TITLE = "opencode"


def opencode_client_headers() -> dict[str, str]:
    """The OpenCode client fingerprint, with a fresh opaque session id.

    ``X-Session-ID`` only has to be opaque and stable for the client's lifetime; conversation
    affinity across main and auxiliary calls stays on ``x-opencode-session``
    (``agent.opencode_affinity``), which rides on every OpenCode request either way.
    """
    return {
        "HTTP-Referer": OPENCODE_CLIENT_REFERER,
        "X-Title": OPENCODE_CLIENT_TITLE,
        "User-Agent": OPENCODE_CLIENT_USER_AGENT,
        "X-Session-ID": str(uuid.uuid4()),
    }
