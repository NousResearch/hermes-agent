"""Capability descriptors shared by the API server's feature-discovery response."""


def browser_control_features(adapter) -> dict:
    from gateway.platforms import api_server as api

    return {
        "enabled": adapter._browser_control_enabled(),
        "protocol_version": api._BROWSER_CONTROL_PROTOCOL_VERSION,
        "capabilities": sorted(api.BROWSER_CONTROL_CAPABILITIES),
        "artifact_capabilities": sorted(api.BROWSER_CONTROL_ARTIFACT_CAPABILITIES),
        "developer_capabilities": sorted(api.BROWSER_CONTROL_DEVELOPER_CAPABILITIES),
        "developer_mode": adapter._browser_control_developer_mode(),
        "artifact_transport": {
            "upload": {"method": "POST", "path": "/v1/artifacts/upload"},
            "download": {"method": "GET", "path": "/v1/artifacts/download/{artifact_id}"},
            "max_bytes": api.DEFAULT_MAX_ARTIFACT_BYTES,
            "ttl_seconds": api.DEFAULT_ARTIFACT_TTL_SECONDS,
            "allowed_mime_types": sorted(api.DEFAULT_ALLOWED_MIME_TYPES),
        },
        "real_browser_actions": True,
        "transports": {
            "local_vps": "websocket-subprotocol-ticket",
            "cloud": "authenticated-gateway-rpc",
        },
    }
