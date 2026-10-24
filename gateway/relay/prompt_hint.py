"""Live relay hint lookup for new system-prompt builds."""

import logging

logger = logging.getLogger(__name__)


def negotiated_platform_hint(platform_key: str) -> str:
    """Read only the active profile's descriptor for this platform; fail open.

    The transport's map changes on re-handshake while the adapter's primary
    descriptor may still describe the old connection. Never cache this value
    or publish it into the process-global platform registry.
    """
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref
        from hermes_cli.profiles import get_active_profile_name

        runner = _gateway_runner_ref()
        if runner is None:
            return ""
        adapter = runner._adapters_for_profile(get_active_profile_name()).get(Platform.RELAY)
        if adapter is None:
            return ""
        resolve = getattr(adapter._transport, "descriptor_for_platform", None)
        descriptor = resolve(platform_key) if callable(resolve) else adapter.descriptor
        if descriptor is None or descriptor.platform != platform_key:
            return ""
        hint = descriptor.platform_hint
        return hint if isinstance(hint, str) else ""
    except Exception:
        logger.debug("Relay platform hint unavailable for %s", platform_key, exc_info=True)
        return ""
