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
        from apex_overlay import models_dev_fast

        if not models_dev_fast.apply():
            logger.warning(
                "apex-overlay: non-blocking models.dev fetch seam did not fully "
                "apply (see prior error). The /model picker may block on a live "
                "~2.4MB models.dev catalog fetch (7-15s from mainland China)."
            )
    except Exception:
        logger.warning("apex-overlay: models_dev_fast seam failed to load", exc_info=True)

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

    # hc-401: seams porting the four cloud build-time in-place patches into the
    # overlay so the cloud container image can be built from this fork (unify the
    # two runtime assembly lines). Each is idempotent + fail-safe.

    try:
        from apex_overlay import web_tools_extract

        if not web_tools_extract.apply():
            logger.warning(
                "apex-overlay: hc-401 web_extract capability-gate seam did not "
                "fully apply (see prior error). web_extract stays gated by the "
                "broad check_web_api_key and may hang on search-only backends "
                "like ddgs (model calls it, request spins to Feishu timeout)."
            )
    except Exception:
        logger.warning("apex-overlay: web_tools_extract seam failed to load", exc_info=True)

    try:
        from apex_overlay import stt_no_lazy_install

        if not stt_no_lazy_install.apply():
            logger.warning(
                "apex-overlay: hc-401 managed-runtime STT lazy-install guard seam "
                "did not fully apply (see prior error). A managed (768MB) container "
                "may OOM-kill the gateway lazy-installing faster-whisper on first "
                "transcription."
            )
    except Exception:
        logger.warning("apex-overlay: stt_no_lazy_install seam failed to load", exc_info=True)

    try:
        from apex_overlay import first_turn_ack

        if not first_turn_ack.apply():
            logger.warning(
                "apex-overlay: hc-401 first_turn_ack seam did not fully apply "
                "(see prior error). Native no-edit CN-IM entries "
                "(wecom/weixin/dingtalk/qqbot) will sit in ~14.6s of first-turn "
                "silence with no 'received, working' ack."
            )
    except Exception:
        logger.warning("apex-overlay: first_turn_ack seam failed to load", exc_info=True)

    try:
        from apex_overlay import cn_im_messages

        if not cn_im_messages.apply():
            logger.warning(
                "apex-overlay: hc-401 CN-IM message localization seam did not "
                "fully apply (see prior error). Gateway control/status messages "
                "(busy-ack, gateway-busy, no-home-channel, mid-run rejections) "
                "may leak English to wecom/weixin/dingtalk/qqbot users."
            )
    except Exception:
        logger.warning("apex-overlay: cn_im_messages seam failed to load", exc_info=True)

    try:
        from apex_overlay import cn_mirror_env

        if not cn_mirror_env.apply():
            logger.warning(
                "apex-overlay: hc-476 runtime CN mirror env seam did not fully "
                "apply (see prior error). Runtime lazy downloads (Playwright "
                "Chromium autoinstall, managed-Node reinstall) may pull from "
                "foreign hosts on mainland-China machines."
            )
    except Exception:
        logger.warning("apex-overlay: cn_mirror_env seam failed to load", exc_info=True)
