"""Local OpenAI-compatible proxy that forwards to OAuth-authenticated upstreams."""

from rabbit_cli.proxy.adapters.base import UpstreamAdapter

__all__ = ["UpstreamAdapter"]
