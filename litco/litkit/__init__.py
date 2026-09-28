"""LitKit toolset for the matter host.

``client`` talks to the firm's LitKit instance with the matter-pinned agent token and,
during a turn, the acting lawyer's user assertion. ``tools`` exposes the ``litkit``
Hermes toolset over that client; ``plugins/litkit`` registers it. ``context`` carries the
current turn's identity and working directory from the turn runner to the tools.
"""
