"""Gateway-owned Windows launch metadata shared by parent and child processes."""

# Private launcher-to-child diagnostic state; not user configuration.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"

__all__ = ["_WINDOWS_GATEWAY_BREAKAWAY_ENV"]
