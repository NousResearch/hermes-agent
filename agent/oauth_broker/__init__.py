"""Hermes OAuth broker: opt-in, loopback-only Codex OAuth multiplexer.

See docs/design/oauth-broker.md. This package is imported only by the
`hermes oauth-broker` command path; keep this module free of submodule
imports so ordinary Hermes startup never pays for (or requires) broker
dependencies such as aiohttp.
"""
