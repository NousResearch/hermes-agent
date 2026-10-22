"""A plugin wake cannot borrow another served profile's committed native envelope."""
import pytest


@pytest.mark.asyncio
async def test_plugin_wake_profile_a_b_a_keeps_native_envelopes_separate(tmp_path, monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import active_authority
    from gateway.session_authority import SessionAuthority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('beta',))
    beta = homes[1][1]
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    # Real per-profile authorization, not the ambient launch environment.
    (root / '.env').write_text('DISCORD_ALLOWED_USERS=100,200\n')
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    try:
        await initialize_gateway_runtime(runner)
        authority = runner.session_authorities.for_home(root)
        beta_authority = runner.session_authorities.for_home(beta)
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
        runner.adapters = {Platform.DISCORD: adapter}
        runner._wire_adapter_handlers(adapter)
        register_transport_home(runner, None, root)
        thread = dict(chat_id='42', chat_type='thread', thread_id='43')
        with _profile_runtime_scope(root):
            first = adapter.build_source(**thread, user_id='100', user_name='dispatcher')
            entry = runner.session_store.get_or_create_session(first)
            human = MessageEvent(text='human', message_id='human',
                source=adapter.build_source(**thread, user_id='200', user_name='human'))
            with native_callback(runner, human, root):
                await authority.admit_native(human)
        runner._running = True
        for home, accepted in ((root, True), (beta, False), (root, True)):
            with _profile_runtime_scope(home):
                assert active_authority(runner) is (authority if home == root else beta_authority)
                assert await runner._dispatch_plugin_message_injection(
                    session_key=entry.session_key, content='plugin notice', plugin_id='fixture-plugin',
                    plugin_home=home) is accepted
        rows = list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)
        wakes = [row for row in rows if row['payload']['native_text_v1'].get('automation')]
        assert len(wakes) == 2
        assert all(row['payload']['native_text_v1']['source']['user_id'] == '200' for row in wakes)
        with beta_authority.db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    finally:
        for authority in getattr(runner, 'session_authorities', []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
