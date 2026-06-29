"""apex-overlay plugin — boot hook that applies ApexNodes seams onto upstream.

This bundled plugin is the *load-time entry point* for the ``apex_overlay``
package. Its only job is to call each overlay seam's ``apply()`` during plugin
discovery, which the CLI runs early in startup (``cli.py`` deferred-startup,
before the ``/model`` picker cache prewarm) and the gateway runs at boot
(``gateway/run.py``). That ordering is what lets the hc-392 provider denylist
take effect *before* the picker's background prewarm would otherwise fetch a
disabled provider's catalog.

Why a plugin (and not an in-place import in model_switch.py)?
------------------------------------------------------------
The whole point of ``apex_overlay/`` is to keep our behavioral debt OUT of hot
upstream files so they merge cleanly on the next bump. A bundled plugin +
``apply()`` keeps the upstream files byte-for-byte upstream; the only non-code
touch is one line in ``cli-config.yaml.example`` adding ``apex-overlay`` to
``plugins.enabled`` — the cleanest seam tier (config > plugin > in-place).

See ``apex_overlay/README.md`` for the full pattern.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:  # noqa: ARG001 — ctx unused; this is a boot hook
    """Apply all apex_overlay seams. Called once during plugin discovery.

    Each ``apply()`` is idempotent and fail-safe: a seam that can't bind to its
    upstream target logs an error and returns False (the seam-test turns that
    into a hard CI failure) but never raises, so a single broken overlay can't
    take down plugin discovery or the host.
    """
    try:
        from apex_overlay import provider_filter

        if not provider_filter.apply():
            logger.warning(
                "apex-overlay: hc-392 provider denylist seam did not fully "
                "apply (see prior error). Disabled providers may still be "
                "probed/fetched at startup."
            )
    except Exception:
        logger.warning("apex-overlay: provider_filter seam failed to load", exc_info=True)

    try:
        from apex_overlay import gateway_bootstrap

        if not gateway_bootstrap.apply():
            logger.warning(
                "apex-overlay: hc-384/385 gateway background-startup seam did "
                "not fully apply (see prior error). Slow adapters (Feishu) may "
                "block the gateway's conversation-ready signal at startup."
            )
    except Exception:
        logger.warning("apex-overlay: gateway_bootstrap seam failed to load", exc_info=True)

    try:
        from apex_overlay import feishu_supervisor

        if not feishu_supervisor.apply():
            logger.warning(
                "apex-overlay: hc-384 Feishu self-reconnect supervisor seam did "
                "not fully apply (see prior error). Feishu falls back to the "
                "lark SDK's broken single-shot auto-reconnect, which left bots "
                "dead for hours in prod."
            )
    except Exception:
        logger.warning("apex-overlay: feishu_supervisor seam failed to load", exc_info=True)
