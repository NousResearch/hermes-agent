"""Shared application scheduling for canonical provider-catalogue cache warming."""
from __future__ import annotations

import logging
import threading
from contextvars import copy_context

logger = logging.getLogger(__name__)
_picker_prewarm_done = threading.Event()
_picker_prewarm_lock = threading.Lock()


def prewarm_picker_cache_async() -> threading.Thread | None:
    """Once per process, warm the inventory's existing provider-model cache off-thread."""
    with _picker_prewarm_lock:
        if _picker_prewarm_done.is_set():
            return None
        _picker_prewarm_done.set()
    context = copy_context()

    def _warm() -> None:
        try:
            from hermes_cli.inventory import build_models_payload, load_picker_context
            build_models_payload(
                load_picker_context(),
                probe_custom_providers=True,
                probe_current_custom_provider=False,
                non_blocking_catalogs=False,
            )
        except Exception:
            logger.debug("picker cache prewarm failed", exc_info=True)

    thread = threading.Thread(
        target=lambda: context.run(_warm), daemon=True, name="picker-cache-prewarm",
    )
    thread.start()
    return thread
