"""Profile admission policy for the generic webhook adapter."""

from typing import Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


# Sentinel returned by _resolve_request_profile when a /p/<profile>/ prefix
# names a profile this gateway does not serve (→ 404). Distinct from None
# (no prefix / multiplexing off → handle as the default profile).
_PROFILE_REJECTED = object()


class WebhookProfileAdmissionMixin:
    """Resolve and authorize profile-bound webhook requests."""

    def _resolve_request_profile(self, request: "web.Request"):
        """Resolve + validate the /p/<profile>/ URL prefix on a webhook request.

        Returns:
          - ``None`` when no profile prefix is present, or multiplexing is off
            (the prefix is ignored, request handled as the default profile).
          - the profile name (str) when present, multiplexing is on, and the
            profile is one this gateway serves.
          - ``_PROFILE_REJECTED`` when a prefix is present but the profile is
            unknown/unconfigured (handler returns 404).
        """
        profile = (request.match_info.get("profile") or "").strip()
        if not profile:
            return None
        runner = self.gateway_runner
        cfg = getattr(runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            # Prefix supplied but multiplexing is off — ignore it, behave as
            # the single-profile gateway (don't 404 a would-be valid route).
            return None
        try:
            from hermes_cli.profiles import profiles_to_serve
            served = {name for name, _ in profiles_to_serve(multiplex=True)}
        except Exception:
            return _PROFILE_REJECTED
        if profile not in served:
            return _PROFILE_REJECTED
        return profile

    @staticmethod
    def _route_allows_profile(
        route_config: dict,
        request_profile: Optional[str],
    ) -> bool:
        """Return whether a route is bound to the URL-selected profile.

        Omitting ``profile`` keeps a route on the default profile. An explicit
        null, blank, or non-string value is malformed and fails closed.
        """
        if "profile" not in route_config:
            configured_profile = "default"
        else:
            configured_profile = route_config.get("profile")
        if not isinstance(configured_profile, str):
            return False
        configured_profile = configured_profile.strip()
        if not configured_profile:
            return False
        effective_profile = request_profile or "default"
        return configured_profile == effective_profile
