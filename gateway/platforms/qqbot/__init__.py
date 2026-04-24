# -*- coding: utf-8 -*-
"""QQBot platform package — public API.

External callers only need::

    from gateway.platforms.qqbot import QQAdapter, check_qq_requirements
    from gateway.platforms.qqbot import (
        BindStatus, create_bind_task, poll_bind_result,
        build_connect_url, decrypt_secret, ONBOARD_POLL_INTERVAL,
    )

``core/`` and ``adapter.py`` are internal implementation details.
"""

# ── hermes adapter entry point ────────────────────────────────────────
from .adapter import QQAdapter, check_qq_requirements  # noqa: F401

# ── QR-code onboard flow (used by hermes_cli/gateway.py) ─────────────
from .core.onboard import (  # noqa: F401
    BindStatus,
    build_connect_url,
    create_bind_task,
    poll_bind_result,
    qr_register,
)
from .core.crypto import decrypt_secret  # noqa: F401
from .core.constants import ONBOARD_POLL_INTERVAL  # noqa: F401

# ── Backward-compat re-exports (used by tests and external callers) ───
from .core.websocket import QQCloseError  # noqa: F401
from .core.utils import coerce_list as _coerce_list  # noqa: F401
from gateway.platforms.base import _ssrf_redirect_guard  # noqa: F401
