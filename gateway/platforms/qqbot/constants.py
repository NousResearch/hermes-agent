"""QQBot package-level constants shared across adapter, onboard, and other modules."""

from __future__ import annotations

import os

QQBOT_VERSION = "1.1.0"  # bump on functional changes to the adapter package
# Portal domain is overridable (QQ_PORTAL_HOST) for corporate proxies / test environments.
PORTAL_HOST = os.getenv("QQ_PORTAL_HOST", "q.qq.com")

API_BASE = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
GATEWAY_URL_PATH = "/gateway"
ONBOARD_CREATE_PATH = "/lite/create_bind_task"
ONBOARD_POLL_PATH = "/lite/poll_bind_result"
QR_URL_TEMPLATE = "https://q.qq.com/qqbot/openclaw/connect.html?task_id={task_id}&_wv=2&source=hermes"

DEFAULT_API_TIMEOUT = 30.0
FILE_UPLOAD_TIMEOUT = 120.0
CONNECT_TIMEOUT_SECONDS = 20.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
MAX_RECONNECT_ATTEMPTS = 100
RATE_LIMIT_DELAY = 60  # seconds
# Bounded budget for one open attempt (token fetch DEFAULT_API_TIMEOUT + gateway URL
# DEFAULT_API_TIMEOUT + ws_connect CONNECT_TIMEOUT_SECONDS). An UNBOUNDED open here left the
# adapter permanently deaf: the listener task never returned, so no further reconnect attempt
# was ever logged (op7/4009 close -> "Reconnecting in 2s (attempt 1)..." -> 30h of silence).
OPEN_WS_TIMEOUT_SECONDS = 60.0
# A live QQ gateway answers every op1 heartbeat with an op11 ACK (~30s cadence), so silence for
# this long means a half-open socket the kernel never reported: treat it as dead and reconnect.
READ_TIMEOUT_SECONDS = 120.0
QUICK_DISCONNECT_THRESHOLD = 5.0  # seconds
MAX_QUICK_DISCONNECT_COUNT = 3
ONBOARD_POLL_INTERVAL = 2.0  # seconds between poll_bind_result calls
ONBOARD_API_TIMEOUT = 10.0

MAX_MESSAGE_LENGTH = 4000
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000

# QQ Bot message types / file media types
MSG_TYPE_TEXT = 0
MSG_TYPE_MARKDOWN = 2
MSG_TYPE_MEDIA = 7
MSG_TYPE_INPUT_NOTIFY = 6
MEDIA_TYPE_IMAGE = 1
MEDIA_TYPE_VIDEO = 2
MEDIA_TYPE_VOICE = 3
MEDIA_TYPE_FILE = 4
