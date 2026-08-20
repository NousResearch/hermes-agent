"""Deterministic fault injection for reliability-critical paths.

Companion to #54964 ("Enforce E2E Testing over Isolated Unit Mocks in Core
Agent Tests"): provides REAL primitives (a real local HTTP socket for
provider faults, a real separate process for SQLite lock contention)
instead of mocking the client/transport, so tests exercise the actual
error-surfacing code instead of a MagicMock standing in for it.

Scope of this first cut: provider HTTP faults (429, truncated stream) and
SQLite write-lock contention. MCP-disconnect and gateway-reconnect fault
scenarios are natural follow-ups once these primitives are in use.
"""
