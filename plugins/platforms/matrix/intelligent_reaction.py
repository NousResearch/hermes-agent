"""Compatibility import for the former Matrix-only policy module."""

from plugins.groupchat.relevance import *  # noqa: F401,F403
from plugins.groupchat.relevance import _load_env_key, _parse_score_delays
