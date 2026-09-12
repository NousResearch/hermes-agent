"""Owned Discord SDK peer + actual public adapter/TurnRunner/HTTP execution."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import sys
import traceback
from types import SimpleNamespace

from shared_authority_peer import ModelPeer


async def probe(peer):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter
    from hermes_state_runtime import list_session_admissions

    class Guild:
        hook = None
        id = 100
        roles = [SimpleNamespace(id=700)]
        calls = 0

        async def fetch_member(self, uid):
            assert uid == 200
            # No SQLite transaction may span the network await.
            assert not authority.db._conn.in_transaction
            self.calls += 1
            await asyncio.sleep(0)
            roles = list(self.roles)
            if self.hook:
                self.hook()
            return SimpleNamespace(id=uid, guild=self, roles=roles)

        def get_member(self, uid):
            raise AssertionError('cached membership is not authorization')

    class Adapter(DiscordAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.deliveries.append(content)
            return SendResult(success=True, message_id='role-result')

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='role-peer')
    adapter = Adapter(PlatformConfig(enabled=True, token='owned-discord-peer'))
    adapter.deliveries = []
    guild = Guild()
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild if gid == guild.id else None)
    adapter.gateway_runner = runner
    runner.adapters[Platform.DISCORD] = adapter
    runner._wire_adapter_handlers(adapter)

    def event(identity, *, scope='100', role=True):
        return MessageEvent(text='ROLE_INPUT_' + identity, message_id=identity, source=adapter.build_source(
            chat_id='300', chat_type='group', user_id='200', scope_id=scope, role_authorized=role))

    async def send(ev):
        await adapter.handle_message(ev)
        async with asyncio.timeout(15):
            while adapter._active_sessions:
                await asyncio.sleep(0.01)

    if sys.argv[1] == 'multiplex':
        from agent.secret_scope import set_multiplex_active
        from gateway.profile_routing import ProfileRoute
        state = Path(os.environ['HERMES_HOME'])
        transport = state / 'profiles' / 'transport'
        transport.mkdir(parents=True)
        (state / '.env').write_text('DISCORD_ALLOWED_ROLES=999\n')
        (transport / '.env').write_text('DISCORD_ALLOWED_ROLES=700\n')
        (transport / 'config.yaml').write_text((state / 'config.yaml').read_text())
        runner.config.multiplex_profiles = True
        runner.config._runtime_profile_homes = [('default', state), ('transport', transport)]
        runner.config.profile_routes = [ProfileRoute(name='role-route', platform='discord',
                                                    profile='default', chat_id='300', bot_profile='transport')]
        set_multiplex_active(True)
        runner.adapters.clear()
        runner._profile_adapters = {'transport': {Platform.DISCORD: adapter}}
        runner._configure_profile_adapter(adapter, 'transport', Platform.DISCORD)

    if sys.argv[1] not in {'positive', 'multiplex'}:
        from native_role_fences import fences
        await fences(runner, authority, adapter, guild, event, peer, sys.argv[1])
        return

    ev = event('positive')
    await send(ev)
    entry = runner.session_store.get_or_create_session(ev.source)
    rows = list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)
    assert rows and rows[0]['outcome'] == 'completed', (rows, adapter.deliveries)
    assert len(peer.requests) == 1 and guild.calls >= 2, (peer.requests, guild.calls)
    guild.roles = []
    await send(event('revoked'))
    await send(event('wrong-guild', scope='999'))
    assert len(list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)) == 1
    assert len(peer.requests) == 1
    if sys.argv[1] == 'multiplex':
        guild.roles = [SimpleNamespace(id=700)]
        # Hot policy edits during SDK I/O must not reuse the entry-time secret scope.
        def revoke_policy():
            guild.hook = None
            (transport / '.env').write_text('DISCORD_ALLOWED_ROLES=999\n')
        guild.hook = revoke_policy
        from gateway.session_ingress_context import native_callback
        from hermes_state_runtime import RuntimeStoreError
        ev = event('policy-revoked-during-fetch')
        with native_callback(runner, ev, transport, 'transport'):
            try:
                await authority.admit_native(ev)
            except RuntimeStoreError as exc:
                assert exc.reason == 'permission_denied'
            else:
                raise AssertionError('revoked transport role policy was durably accepted')
        assert len(peer.requests) == 1
    if sys.argv[1] == 'positive':
        # Independent user/pairing grants survive membership revocation.
        calls = guild.calls
        os.environ['DISCORD_ALLOWED_USERS'] = '200'
        await send(event('direct-user'))
        assert len(peer.requests) == 2 and guild.calls == calls
        os.environ.pop('DISCORD_ALLOWED_USERS')
        code = runner.pairing_store.generate_code('discord', '200')
        assert runner.pairing_store.approve_code('discord', code)['user_id'] == '200'
        await send(event('paired-user'))
        assert len(peer.requests) == 3 and guild.calls == calls
        assert runner.pairing_store.revoke('discord', '200')
        guild.roles = [SimpleNamespace(id=700)]

        def dm(identity):
            return MessageEvent(text='ROLE_DM_' + identity, message_id=identity,
                                source=adapter.build_source(chat_id='400', chat_type='dm',
                                                            user_id='200', role_authorized=True))
        await send(dm('default-deny'))
        assert len(peer.requests) == 3 and guild.calls == calls
        config = Path(os.environ['HERMES_HOME'], 'config.yaml')
        original = config.read_text()
        config.write_text(original + '\ndiscord:\n  dm_role_auth_guild: 100\n')
        await send(dm('opted-in'))
        assert len(peer.requests) == 4 and guild.calls > calls
        config.write_text(original + '\ndiscord:\n  dm_role_auth_guild: 999\n')
        await send(dm('wrong-guild'))
        assert len(peer.requests) == 4
        config.write_text(original)
    print(json.dumps({'model_calls': len(peer.requests), 'fresh_fetches': guild.calls,
                      'outcome': rows[0]['outcome'], 'negative': ['revoked', 'wrong-guild']}), flush=True)


if __name__ == '__main__':
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base_url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='explicit-loopback-fixture', OPENAI_BASE_URL=base_url,
                      DISCORD_ALLOWED_ROLES='700')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {base_url}\n'
        f'auxiliary:\n  title_generation:\n    enabled: false\nterminal:\n  cwd: {os.environ["HERMES_HOME"]}\n')
    status = 0
    try:
        asyncio.run(probe(peer))
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        peer.shutdown()
        peer.server_close()
    os._exit(status)
