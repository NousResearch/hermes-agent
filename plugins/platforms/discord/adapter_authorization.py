"""Fresh membership checks for owner-held durable native inputs."""
import asyncio


class DiscordAuthorizationMixin:
    async def reauthorize_native_roles(self, source):
        from plugins.platforms.discord.adapter import _read_dm_role_auth_guild, _scoped_gate_env
        from hermes_cli.config import read_raw_config
        from discord import HTTPException

        # Unlike ingress caches, recovery reads current transport-scoped policy.
        def policy():
            cfg = read_raw_config() or {}
            discord_cfg = cfg.get('discord', {}) or {}
            raw = _scoped_gate_env('DISCORD_ALLOWED_ROLES') or discord_cfg.get('allowed_roles')
            if raw is None or raw == '':
                # Ingress resolves this gate from the per-profile platform block too:
                # PlatformConfig.from_dict promotes both ``platforms.discord.allowed_roles``
                # and ``platforms.discord.extra.allowed_roles`` into config.extra (#10206).
                # Durable re-authorization honored only the env and top-level spellings,
                # dropping every message from a role-authorized user (#139148). An
                # explicitly empty top-level list stays empty, so revocation still works.
                block = (cfg.get('platforms', {}) or {}).get('discord', {}) or {}
                extra = block.get('extra') if isinstance(block.get('extra'), dict) else {}
                raw = extra.get('allowed_roles') if 'allowed_roles' in extra else block.get('allowed_roles')
                if raw is None:
                    raw = []
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
        from hermes_constants import get_hermes_home
        with _profile_runtime_scope(get_hermes_home()):
            current_policy = policy()
        if self._client is not client or current_policy != (roles, guild_id):
            return False
        return (member.id == uid and member.guild.id == int(guild_id)
                and any(role.id in roles for role in member.roles))
