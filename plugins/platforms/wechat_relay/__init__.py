"""Independent WeChat relay platform plugin for Hermes.

This is the 微信中转方案: a dedicated Android relay transport compatible with
OpenClaw's 9797 shape. It is intentionally separate from Hermes' built-in
``weixin`` adapter and must not import, patch, or reuse that adapter.
"""

from .adapter import register

__all__ = ["register"]
