"""Native/text ACP controls on the gateway's retained mutation boundary."""
from __future__ import annotations

import asyncio

from acp.schema import ConfigOptionUpdate, SetSessionConfigOptionResponse

from acp_adapter.config_options import parse_selection, project_options, text_selection
from hermes_cli.gateway_client import GatewayClientError


class GatewayControlsMixin:
    async def _config_options(self, session_id, info=None):
        from acp_adapter.gateway_session_meta import model_state
        if info is None:
            info = self._snapshots[session_id].get("info")
            if not info:
                info = await (await self._client()).rpc("session.info", session_id=session_id)
        models = await asyncio.to_thread(model_state, info)
        return await asyncio.to_thread(project_options, info, models)

    async def _send_config_options(self, session_id, info=None):
        options = await self._config_options(session_id, info)
        if self._conn:
            await self._conn.session_update(session_id=self._editor_id(session_id), update=ConfigOptionUpdate(
                session_update="config_option_update", config_options=options))
        return options

    async def set_config_option(self, config_id, session_id, value=None, **kwargs):
        from acp_adapter.gateway_session_meta import DEFAULT_EDIT_APPROVAL_POLICY, EDIT_APPROVAL_POLICY_CONFIG_ID
        self._held_session(session_id)
        session_id = self._aliases.get(session_id, session_id)
        if config_id == EDIT_APPROVAL_POLICY_CONFIG_ID:
            if value != DEFAULT_EDIT_APPROVAL_POLICY:
                raise GatewayClientError("acp_edit_policy_mutation_unavailable")
            return SetSessionConfigOptionResponse(config_options=await self._config_options(session_id))
        if config_id == "model":
            await self.set_session_model(value, session_id)
            return SetSessionConfigOptionResponse(config_options=await self._config_options(session_id))
        if config_id not in {"reasoning", "fast"}:
            from acp.exceptions import RequestError
            raise RequestError.invalid_params({"details": f"Unsupported config option: {config_id}"})
        client = await self._client()
        info = await client.rpc("session.info", session_id=session_id)
        if config_id == "reasoning" and value == "" and not info.get("reasoning_effort"):
            return SetSessionConfigOptionResponse(config_options=await self._send_config_options(session_id, info))
        try:
            value = await asyncio.to_thread(parse_selection, config_id, value, info)
        except ValueError as exc:
            from acp.exceptions import RequestError
            raise RequestError.invalid_params({"details": str(exc)}) from exc
        operation, payload = (("reasoning", {"effort": value}) if config_id == "reasoning" else
                              ("service_tier", {"service_tier": value}))
        self._unacked.pop(session_id, None)
        await self._mutations.apply(client, session_id, operation, payload)
        self._snapshots[session_id] = await client.rpc("session.resume", session_id=session_id)
        options = await self._send_config_options(session_id)
        self._mutations.acknowledge(session_id, operation, payload)
        return SetSessionConfigOptionResponse(config_options=options)

    async def _control_command(self, session_id, command, arg):
        try:
            value = text_selection(arg)
        except ValueError as exc:
            raise GatewayClientError(str(exc)) from exc
        if value and value != "status":
            response = await self.set_config_option(command, session_id, value)
            options = response.config_options
        else:
            info = await (await self._client()).rpc("session.info", session_id=session_id)
            options = await self._config_options(session_id, info)
        option = next(option for option in options if option.id == command)
        return f"{option.name}: {option.current_value or 'provider default'} (this session)"
