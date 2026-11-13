"""Authenticated API capabilities, including the exact mirror's bot identity."""
import asyncio


def bot_capability(adapter):
    from gateway.platforms.api_server import _api_request_profile
    from gateway.platforms.api_server_room_grants import _local_target, _local_room_catalog
    profile, installation_id = _local_target(None, _api_request_profile)
    from gateway.hosted_room_peer import HostedRoomPeerError
    from gateway.hosted_room_execution_policy import RoomExecutionPolicyError
    try:
        _, catalog = _local_room_catalog(adapter, profile, installation_id)
    except (HostedRoomPeerError, RoomExecutionPolicyError):
        # DM discovery remains available when the profile's policy disables RoomLink.
        catalog = None
    return {'profile': profile, 'installation_id': installation_id, 'room_link': catalog}


async def capabilities(self):
    from aiohttp import web
    from gateway.platforms.api_server import _api_runs, RunIdempotencyStore, _STATIC_FEATURE_FLAGS, _CAPABILITY_ENDPOINTS
    runner = getattr(self, "gateway_runner", None)
    features = {
        "chat_completions": True, "chat_completions_streaming": True,
        "responses_api": True, "responses_streaming": True, "run_submission": True,
        "runs_idempotency": _api_runs._idempotency_capabilities(self, store_type=RunIdempotencyStore),
        **_STATIC_FEATURE_FLAGS,
        "run_unknown_resolution": getattr(runner, "session_authority", None) is not None,
        "cors": bool(self._cors_origins)}
    bot = await asyncio.to_thread(bot_capability, self)
    return web.json_response({
        "bot": bot,
        "object": "hermes.api_server.capabilities", "platform": "hermes-agent",
        "model": self._model_name,
        "auth": {"type": "bearer", "required": bool(self._api_key)},
        "runtime": {
            "mode": "server_agent", "tool_execution": "server", "split_runtime": False,
            "description": (
                "The API server creates a server-side Hermes AIAgent; "
                "tools execute on the API-server host unless a future "
                "explicit split-runtime mode is enabled.")},
        "features": features,
        "endpoints": {
            name: {"method": entry[0], "path": entry[1]}
            for name, entry in _CAPABILITY_ENDPOINTS
            if len(entry) < 3 or features[entry[2]]},
    })
