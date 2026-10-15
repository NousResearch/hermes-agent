"""Fresh membership checks for owner-held durable native inputs."""
import asyncio


class DiscordAuthorizationMixin:
    async def reauthorize_native_roles(self, source):
        from plugins.platforms.discord.adapter import _read_dm_role_auth_guild, _scoped_gate_env
        from plugins.platforms.discord.adapter_config_readers import _discord_yaml_gates
        from gateway.config import Platform
        from gateway.config_loader import effective_platform_extra
        from hermes_cli.config import _expand_env_vars, read_raw_config
        from hermes_cli import managed_scope
        from hermes_constants import get_hermes_home
        from discord import HTTPException

        # Unlike ingress caches, recovery reads current transport-scoped policy.
        def policy():
            extra = effective_platform_extra(
                managed_scope.apply_managed_overlay(_expand_env_vars(read_raw_config() or {})), Platform.DISCORD,
                managed=managed_scope.apply_managed_overlay({}), seed_extra=_discord_yaml_gates)
            raw = _scoped_gate_env('DISCORD_ALLOWED_ROLES') or extra.get('allowed_roles', [])
            roles = {int(str(item).strip()) for item in self._gate_csv_set(raw)
                     if str(item).strip().isdigit()}
            guild_id = _read_dm_role_auth_guild() if source.chat_type == 'dm' else source.scope_id
            return roles, guild_id

        roles, guild_id = policy()
        if not roles or guild_id is None or self._client is None:
            return False
        try:
            guild = self._client.get_guild(int(guild_id))
            uid = int(source.user_id)
        except (TypeError, ValueError):
            return False
        if guild is None:
            return False
        client = self._client
        try:
            member = await asyncio.wait_for(guild.fetch_member(uid), timeout=10)
        except (HTTPException, TimeoutError, OSError):
            return False
        # The entry-time ContextVar contains a secret snapshot. Reload the same
        # transport home after SDK I/O so an intervening .env edit is visible too.
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(get_hermes_home()):
            current_policy = policy()
        if self._client is not client or current_policy != (roles, guild_id):
            return False
        return (member.id == uid and member.guild.id == int(guild_id)
                and any(role.id in roles for role in member.roles))
