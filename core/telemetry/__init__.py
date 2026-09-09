"""
Hermes Core Telemetry Package.
Privacy-first, disabled by default, zero secrets logged.
"""


class TelemetryManager:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def track_event(self, event_name: str, properties: dict) -> None:
        if not self.enabled:
            return
        # Tracking logic when enabled


__all__ = ["TelemetryManager"]
