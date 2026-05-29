"""QQBot package-level constants shared across adapter, onboard, and other modules."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# QQBot adapter version — bump on functional changes to the adapter package.
# ---------------------------------------------------------------------------

QQBOT_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

# The portal domain is configurable via QQ_API_HOST for corporate proxies
# or test environments.  Default: q.qq.com (production).
PORTAL_HOST = os.getenv("QQ_PORTAL_HOST", "q.qq.com")

API_BASE = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
GATEWAY_URL_PATH = "/gateway"

# QR-code onboard endpoints (on the portal host)
ONBOARD_CREATE_PATH = "/lite/create_bind_task"
ONBOARD_POLL_PATH = "/lite/poll_bind_result"
QR_URL_TEMPLATE = (
    "https://q.qq.com/qqbot/openclaw/connect.html"
    "?task_id={task_id}&_wv=2&source=hermes"
)

# ---------------------------------------------------------------------------
# Timeouts & retry
# ---------------------------------------------------------------------------

DEFAULT_API_TIMEOUT = 30.0
FILE_UPLOAD_TIMEOUT = 120.0
CONNECT_TIMEOUT_SECONDS = 20.0
# If no WebSocket frame arrives within this window the connection is
# considered dead and the adapter will reconnect.  Covers CLOSE_WAIT
# half-open sockets where the transport has already failed but aiohttp
# hasn't detected it yet (no TCP RST, no close frame).
# Set well above 3x heartbeat_interval so transient lag doesn't trigger
# false reconnects.  QQ's default server-side heartbeat is 30 s;
# Hermes sends at 80 % = 24 s.  90 s = 3.7 heartbeat periods.
RECEIVE_TIMEOUT = 90.0

RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
MAX_RECONNECT_ATTEMPTS = 100
RATE_LIMIT_DELAY = 60  # seconds
QUICK_DISCONNECT_THRESHOLD = 5.0  # seconds
MAX_QUICK_DISCONNECT_COUNT = 3

ONBOARD_POLL_INTERVAL = 2.0  # seconds between poll_bind_result calls
ONBOARD_API_TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# Message limits
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 4000
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000

# ---------------------------------------------------------------------------
# QQ Bot message types
# ---------------------------------------------------------------------------

MSG_TYPE_TEXT = 0
MSG_TYPE_MARKDOWN = 2
MSG_TYPE_MEDIA = 7
MSG_TYPE_INPUT_NOTIFY = 6

# ---------------------------------------------------------------------------
# QQ Bot file media types
# ---------------------------------------------------------------------------

MEDIA_TYPE_IMAGE = 1
MEDIA_TYPE_VIDEO = 2
MEDIA_TYPE_VOICE = 3
MEDIA_TYPE_FILE = 4
