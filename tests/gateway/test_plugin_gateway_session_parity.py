"""Plugin code running inside the gateway sees the session it runs for (LC-5).

* A key-less ``inject_message`` from a classic CLI session's turn lands in that session, as main
  queued it into the CLI REPL.
* Approval hooks report a CLI session's ``surface='cli'`` (as on main);
  ``ctx.session_cwd()`` / ``ctx.session_surface()`` read the session, never the daemon's cwd.
"""
import asyncio
import contextvars
import os
from types import SimpleNamespace

import pytest

from tests.gateway.test_plugin_origin_injection_authority import _context, _plugin_rows, _publish
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_keyless_injection_from_a_cli_session_turn_lands_in_that_session(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_context import set_session_vars
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_policy import _ACTIVE_POLICY
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = db
    runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    try:
        actor = Principal("test-owner", "default", frozenset({"session:create", "session:submit", "session:read"}),
                          "fixture")
        ref = create_local_session(authority, actor, dict(request_id="cli", source="cli", model="fixture",
                                                         toolsets=[]))
        live = authority.sessions[ref.session_id]
        ctx, manager = _context(tmp_path)
        _publish(runner, manager)

        def hook_in_the_cli_turn():
            # What a plugin hook or tool sees while the gateway runs this CLI session's turn.
            set_session_vars(platform="local", session_key=live.route, session_id=ref.session_id)
            _ACTIVE_POLICY.set(SimpleNamespace(platform="cli", cwd=str(tmp_path)))
            return ctx.inject_message("plugin note")

        accepted = await asyncio.to_thread(contextvars.copy_context().run, hook_in_the_cli_turn)
        assert accepted is True
        await async_wait_until(lambda: _plugin_rows(authority, ref.session_id, "plugin note"), timeout=10,
                               what="the plugin admission in the CLI session")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_keyed_local_injection_stays_inside_the_plugins_own_profile(tmp_path, monkeypatch):
    """Multiplexed gateway: a plugin of profile A may inject into A's live local sessions (key-less
    from the turn, or by key), never into profile B's by naming B's session key; that is refused
    up front (``False``, reason in gateway.log) and admits nothing."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from gateway.session_context import set_session_vars
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_policy import _ACTIVE_POLICY
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    launch = tmp_path / ".hermes"
    beta = launch / "profiles" / "beta"
    beta.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=launch / "sessions"))
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [("default", launch), ("beta", beta)]
    runner.session_authorities = SessionAuthorities(launch, multiplexed=True)
    authorities, sessions, contexts = {}, {}, {}
    for name, home in (("default", launch), ("beta", beta)):
        contexts[name] = _context(home)
        db = SessionDB(home / "state.db")
        authorities[name] = SessionAuthority(runner, profile_id=str(home), instance_id="owner", db=db,
                                             epoch=begin_runtime_epoch(db, instance_id="owner"))
        runner.session_authorities.add(home, authorities[name], name=name)
    runner.session_authority = authorities["default"]
    runner.session_store._db = runner._session_db = authorities["default"].db
    try:
        for name, home in (("default", launch), ("beta", beta)):
            authority = authorities[name]
            actor = Principal("test-owner", authority.profile_id,
                              frozenset({"session:create", "session:submit", "session:read"}), "fixture")
            with _profile_runtime_scope(home, hydrate_secrets=False):
                ref = create_local_session(authority, actor, dict(request_id="cli", source="cli", model="fixture",
                                                                 toolsets=[]))
            sessions[name] = (ref.session_id, authority.sessions[ref.session_id].route)
        for _ctx, manager in contexts.values():
            _publish(runner, manager)
        alpha_ctx, beta_ctx = contexts["default"][0], contexts["beta"][0]
        alpha_sid, alpha_key = sessions["default"]
        beta_sid, beta_key = sessions["beta"]

        def keyless_in_alphas_cli_turn():
            set_session_vars(platform="local", session_key=alpha_key, session_id=alpha_sid)
            _ACTIVE_POLICY.set(SimpleNamespace(platform="cli", cwd=str(tmp_path)))
            return alpha_ctx.inject_message("alpha keyless")

        assert await asyncio.to_thread(contextvars.copy_context().run, keyless_in_alphas_cli_turn) is True
        assert await asyncio.to_thread(alpha_ctx.inject_message, "alpha keyed", session_key=alpha_key) is True
        assert await asyncio.to_thread(beta_ctx.inject_message, "beta keyed", session_key=beta_key) is True
        for name, text in (("default", "alpha keyless"), ("default", "alpha keyed"), ("beta", "beta keyed")):
            await async_wait_until(lambda: _plugin_rows(authorities[name], sessions[name][0], text), timeout=10,
                                   what=f"the same-profile plugin admission {text!r}")

        with caplog_at("gateway.run") as records:
            assert await asyncio.to_thread(alpha_ctx.inject_message, "cross", session_key=beta_key) is False
            # Dispatch rechecks before entering the target profile's scope.
            assert await runner._dispatch_plugin_message_injection(
                session_key=beta_key, content="cross", plugin_id="desk", plugin_home=launch) is False

        def in_betas_scope(call):
            # Where beta's plugin code runs: the ambient home is beta's, which must not make a
            # default-profile session (its source carries no profile) look like beta's.
            with _profile_runtime_scope(beta, hydrate_secrets=False):
                return call()

        def keyless_in_beta_scope_for_alphas_turn():
            set_session_vars(platform="local", session_key=alpha_key, session_id=alpha_sid)
            _ACTIVE_POLICY.set(SimpleNamespace(platform="cli", cwd=str(tmp_path)))
            return in_betas_scope(lambda: beta_ctx.inject_message("cross"))

        with caplog_at("gateway.run") as beta_records:
            assert await asyncio.to_thread(
                in_betas_scope, lambda: beta_ctx.inject_message("cross", session_key=alpha_key)) is False
            assert await asyncio.to_thread(contextvars.copy_context().run, keyless_in_beta_scope_for_alphas_turn) is False
            assert await asyncio.to_thread(
                in_betas_scope, lambda: beta_ctx.inject_message("beta again", session_key=beta_key)) is True
        await async_wait_until(lambda: _plugin_rows(authorities["beta"], beta_sid, "beta again"), timeout=10,
                               what="the same-profile plugin admission from beta's scope")
        assert not _plugin_rows(authorities["beta"], beta_sid, "cross")
        assert not _plugin_rows(authorities["default"], alpha_sid, "cross")
        assert all("not a live local session of the plugin's profile" in record.getMessage()
                   for record in (*records, *beta_records) if "refused" in record.getMessage())
        assert sum("refused" in record.getMessage() for record in beta_records) == 2
    finally:
        for authority in authorities.values():
            authority.db.close()


@pytest.mark.asyncio
async def test_keyed_messaging_injection_stays_inside_the_plugins_own_profile(tmp_path, monkeypatch):
    """A messaging session's key names its profile: beta's plugin naming default's Telegram session
    gets ``False`` up front (reason in gateway.log) and nothing reaches the adapter, as for a local
    session; default's own plugin is still accepted and its turn reaches the session's adapter."""
    from datetime import datetime
    from unittest.mock import AsyncMock
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource

    launch = tmp_path / ".hermes"
    beta = launch / "profiles" / "beta"
    beta.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    runner = GatewayRunner(GatewayConfig(sessions_dir=launch / "sessions"))
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [("default", launch), ("beta", beta)]
    key, now = "agent:main:telegram:dm:42", datetime.now()  # default's namespace
    entry = SessionEntry(session_key=key, session_id="session-42", created_at=now, updated_at=now,
                         platform=Platform.TELEGRAM,
                         origin=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42"))
    monkeypatch.setattr(runner.session_store, "lookup_by_session_key", {key: entry}.get)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda _source: adapter)
    monkeypatch.setattr(runner, "_plugin_injection_authorized", lambda *_args, **_kwargs: True)
    (default_ctx, default_manager), (beta_ctx, beta_manager) = _context(launch), _context(beta)
    for manager in (default_manager, beta_manager):
        _publish(runner, manager)

    with caplog_at("gateway.run") as records:
        assert await asyncio.to_thread(beta_ctx.inject_message, "cross", session_key=key) is False
        # Dispatch rechecks.
        assert await runner._dispatch_plugin_message_injection(
            session_key=key, content="cross", plugin_id="desk", plugin_home=beta) is False
    refusals = [record.getMessage() for record in records if "refused" in record.getMessage()]
    assert len(refusals) == 2 and all("not the plugin's profile beta" in message for message in refusals)
    adapter.handle_message.assert_not_awaited()

    assert await asyncio.to_thread(default_ctx.inject_message, "own", session_key=key) is True
    await async_wait_until(lambda: adapter.handle_message.await_count, timeout=10,
                           what="the same-profile plugin turn at the session's adapter")
    assert [call.args[0].text for call in adapter.handle_message.await_args_list] == ["own"]


class caplog_at:
    """Collect one logger's records without depending on the caplog fixture's propagation."""

    def __init__(self, name):
        import logging
        self.logger, self.records = logging.getLogger(name), []
        self.handler = logging.Handler()
        self.handler.emit = self.records.append

    def __enter__(self):
        self.logger.addHandler(self.handler)
        return self.records

    def __exit__(self, *_exc):
        self.logger.removeHandler(self.handler)


def test_gateway_approval_hooks_report_a_cli_sessions_surface(tmp_path, monkeypatch):
    from gateway.session_policy import _ACTIVE_POLICY
    from tools.approval_gateway_wait import _await_gateway_decision

    seen = []
    approval_hooks = {"pre_approval_request", "post_approval_response"}
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook",
                        lambda name, **kw: seen.append((name, kw)) if name in approval_hooks else None)

    def boom(_data):
        raise RuntimeError("no client")

    def decide(policy):
        if policy is not None:
            _ACTIVE_POLICY.set(policy)
        return _await_gateway_decision("k", boom, {"command": "rm -rf x", "description": "d", "pattern_key": "rm"})

    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = contextvars.copy_context().run(decide, SimpleNamespace(platform="cli", cwd=str(workspace)))
    assert result["notify_failed"] is True
    assert [(name, kw["surface"]) for name, kw in seen] == [
        ("pre_approval_request", "cli"), ("post_approval_response", "cli")]

    # TUI/Desktop and messaging sessions keep the documented gateway surface.
    seen.clear()
    contextvars.copy_context().run(decide, SimpleNamespace(platform="tui", cwd=str(workspace)))
    contextvars.copy_context().run(decide, None)
    assert {kw["surface"] for _name, kw in seen} == {"gateway"}


def test_plugin_context_reads_the_calling_sessions_cwd_and_surface(tmp_path):
    from agent.runtime_cwd import set_session_cwd
    from gateway.session_context import set_session_vars
    from gateway.session_policy import _ACTIVE_POLICY

    ctx, manager = _context(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def in_local_turn():
        _ACTIVE_POLICY.set(SimpleNamespace(platform="tui", cwd=str(workspace)))
        set_session_cwd(str(workspace))
        return ctx.session_cwd(), ctx.session_surface()

    def in_messaging_turn():
        set_session_vars(platform="telegram", session_key="agent:main:telegram:dm:1")
        return ctx.session_surface()

    assert contextvars.copy_context().run(in_local_turn) == (str(workspace), "tui")
    assert contextvars.copy_context().run(in_messaging_turn) == "telegram"
    # The classic CLI process itself still runs in the user's launch directory.
    manager._cli_ref = SimpleNamespace()
    assert (ctx.session_cwd(), ctx.session_surface()) == (os.getcwd(), "cli")
