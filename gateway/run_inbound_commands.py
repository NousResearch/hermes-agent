"""Adapter-owned diagnostic commands, dispatched after normal gateway admission."""


class GatewayPlatformCommandsMixin:
    async def _hm_dispatch_idle_commands(self, event, source, _quick_key):
        """Resolve built-in, quick, plugin, platform and skill commands after admission."""
        _handled, _result, command, canonical = await self._hm_resolve_command(event, source, _quick_key)
        if not _handled:
            _handled, _result = await self._hm_dispatch_canonical_command(event, source, _quick_key, canonical)
        if not _handled:
            _handled, _result, command = await self._hm_dispatch_quick_and_plugin_commands(event, source, command)
        if not _handled:
            _handled, _result = await dispatch_platform_command(self, event, source, command)
        if not _handled:
            # Skill scans are disk-bound; carry the profile scope to the gateway's worker pool.
            _result = await self._run_in_executor_with_context(
                self._hm_skill_slash_rewrite, event, source, _quick_key, command)
            _handled = _result is not None
        return _handled, _result


async def dispatch_platform_command(runner, event, source, command):
    if not command:
        return False, None
    adapter = runner._intake_adapter_for(source)
    handler = getattr(type(adapter), "handle_local_command", None)
    if handler is None:
        return False, None
    denied = runner._check_slash_access(source, command)
    if denied is not None:
        return True, denied
    return await handler(adapter, event), None
