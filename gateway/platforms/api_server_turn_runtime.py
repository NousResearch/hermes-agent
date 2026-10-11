"""Turn-runtime metadata for the session-chat routes, extracted from ``api_server.py``.

Moved unchanged for the FILE_LINES ratchet (moved code keeps its cap); the parent class mixes
this in, so call sites still read ``self._effective_turn_runtime(...)``.
"""

from __future__ import annotations

from typing import Any


class TurnRuntimeMixin:
    """``_effective_turn_runtime`` and the two ladders it reads."""

    def _effective_turn_runtime(self, runtime_request: dict[str, Any], result: Any, usage: Any) -> dict[str, Any]:
        """Sanitized runtime metadata for a finished session-chat turn."""
        runtime = self._result_runtime(result, usage)
        return self._sanitize_runtime_metadata(
            # Same shared ladder /api/status uses. Before this was unified, the two endpoints disagreed on
            # the same page load — the sidebar strip read "running" (it probed GATEWAY_HEALTH_URL and scoped
            # to the requested profile) while the Channels page rendered "The gateway is not running" (it
            # did neither). Cross-container, profile-scoped, and launch-service-managed deployments each hit
            # that split. profile_home is passed when the request was scoped to a named profile:
            # gateway/status readers resolve process-level paths and do NOT follow the HERMES_HOME
            # contextvar override (#56986 / #69143), so the profile's directory has to be handed over
            # explicitly or messaging silently reports another profile's gateway (#71211).
            runtime=runtime,
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=self._model_lock_state(runtime_request, runtime))

    @staticmethod
    def _result_runtime(result: Any, usage: Any) -> dict[str, Any]:
        """Runtime metadata from the result dict, falling back to the usage dict."""
        runtime = (result.get("runtime") or {}) if isinstance(result, dict) else {}
        return runtime or ((usage.get("runtime") or {}) if isinstance(usage, dict) else {})

    @staticmethod
    def _model_lock_state(runtime_request: dict[str, Any], runtime: Any) -> str:
        """``confirmed`` once a runtime was observed under a lock, ``accepted`` before, else ``""``."""
        if not runtime_request.get("require_model_lock"):
            return ""
        return "confirmed" if runtime else "accepted"
