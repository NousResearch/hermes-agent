"""Neutralise inherited HTTP(S) proxy env for the content pipeline.

run_daily.sh (and the gateway) source ~/.hermes/.env with `set -a`, which exports
a Privoxy proxy at 127.0.0.1:8118. That proxy is frequently down (connection
refused), and every outbound call in this pipeline goes to a public API (opencode
LLMs, FAL, Pollinations, Postiz, Telegram) that must NOT be routed through it. A
refused proxy was silently collapsing both text generation (to canned templates)
and image generation (to nothing).

Importing this module removes the proxy variables from this process's environment
so `requests`/`httpx` default sessions stop honouring the dead proxy. Import it at
the top of any module that makes outbound HTTP so the fix holds for every entry
point. (llm_generate also sets trust_env=False explicitly as belt-and-braces.)
"""
from __future__ import annotations

import os

_PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)

for _v in _PROXY_VARS:
    os.environ.pop(_v, None)
