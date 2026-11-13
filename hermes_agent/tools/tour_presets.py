"""The built-in tour presets ``gui_tour`` start-without-steps accepts.

A leaf module with no ``registry.register``: the wire contract
(``hermes_agent/tui_gateway/contracts/server_requests.py``) imports this enum, and importing
``hermes_agent/tools/tour_tool.py`` there would register ``gui_tour`` as a side effect.
"""

from enum import StrEnum


class TourPreset(StrEnum):
    """Which built-in tour ``start`` without steps runs."""

    quick = "quick"
    full = "full"
