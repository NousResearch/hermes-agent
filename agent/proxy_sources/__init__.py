"""Egress proxy integrations.

Currently ships an iron-proxy (ironsh/iron-proxy) wrapper that intercepts
outbound traffic from remote terminal sandboxes and swaps proxy tokens
for real upstream credentials at the network edge.

Design notes live in :mod:`agent.proxy_sources.iron_proxy`.
"""
