"""Hermes-side QQ Bot limits.

Protocol URLs, token refresh, reconnect, and upload live in qqbot-agent-sdk.
The media type numbers are the values ``MediaUploader.upload`` expects; they
stay here so the adapter can import without loading the SDK.
"""

from __future__ import annotations

MAX_MESSAGE_LENGTH = 4000
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000

MSG_TYPE_MEDIA = 7
MEDIA_TYPE_IMAGE = 1
MEDIA_TYPE_VIDEO = 2
MEDIA_TYPE_VOICE = 3
MEDIA_TYPE_FILE = 4
