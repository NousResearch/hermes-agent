"""Contract: the CLI, gateway and TUI /status renderers report the SAME common facts.

Every surface renders through ``hermes_cli.status_report.build_status_fields``; each keeps
its own header, labels (the gateway is i18n) and extras. This test feeds the three real
renderers the same session facts and asserts every common value appears in each output —
a contract between one builder and three renderers, not a snapshot of any layout.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_cli.cli_session_mixin import CLISessionMixin

SESSION_ID = "status-contract-7f3a"
MODEL = "vendor/distinctive-model-9000"
PROVIDER = "distinctive-provider"
TITLE = "Distinctive status title"
CREATED = datetime(2031, 3, 14, 15, 9)
UPDATED = datetime(2031, 3, 15, 1, 5)
TOKENS = 1_234_567
HOME = "~/.hermes-status-contract"

COMMON_VALUES = (
    SESSION_ID, MODEL, PROVIDER, TITLE,
    CREATED.strftime("%Y-%m-%d %H:%M"), UPDATED.strftime("%Y-%m-%d %H:%M"), f"{TOKENS:,}",
)


def _meta() -> dict:
    return {"title": TITLE, "started_at": CREATED.timestamp(), "updated_at": UPDATED.timestamp()}


def _agent():
    return SimpleNamespace(model=MODEL, provider=PROVIDER, session_total_tokens=TOKENS, reasoning_config=None)


def _render_cli() -> str:
    db = MagicMock()
    db.get_session.return_value = _meta()
    rendered: list[str] = []
    cli = SimpleNamespace(
        _session_db=db, session_id=SESSION_ID, session_start=datetime(2000, 1, 1), agent=_agent(),
        provider=PROVIDER, model=MODEL, _agent_running=True, reasoning_config=None, show_reasoning=None,
        session_key="", _get_status_bar_snapshot=dict,
        _console_print=lambda text, **_kw: rendered.append(text),
    )
    CLISessionMixin._show_session_status(cli)
    return "\n".join(rendered)


def _render_gateway() -> str:
    from gateway.run import GatewayRunner

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    entry = SessionEntry(session_key="telegram:c1", session_id=SESSION_ID, created_at=CREATED, updated_at=UPDATED,
                         platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {"telegram:c1": _agent()}
    runner._queue_depth = lambda *_a, **_k: 0
    runner._run_in_executor_with_context = AsyncMock(return_value=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner._session_db = None
    # Token totals come from the SQLite row (never from SessionEntry); model route from the live agent.
    runner._status_session_db_facts = AsyncMock(return_value=(TITLE, {}, TOKENS, {}))
    event = MessageEvent(text="/status", source=source, message_id="m1")
    return asyncio.run(runner._handle_status_command(event))


def _render_tui() -> str:
    from tui_gateway import server

    class _DB:
        def get_session(self, key):
            return _meta() if key == SESSION_ID else {}

    session = {
        "agent": _agent(), "session_key": SESSION_ID, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
    }
    server._sessions["status-contract-sid"] = session
    try:
        with patch.object(server, "_get_db", lambda: _DB()):
            resp = server.handle_request(
                {"id": "1", "method": "session.status", "params": {"session_id": "status-contract-sid"}})
    finally:
        server._sessions.pop("status-contract-sid", None)
    return resp["result"]["output"]


def test_three_status_surfaces_report_the_same_common_fields():
    with patch("hermes_constants.display_hermes_home", return_value=HOME), \
         patch("tools.approval_context._get_approval_mode", side_effect=RuntimeError("n/a")):
        outputs = {"cli": _render_cli(), "gateway": _render_gateway(), "tui": _render_tui()}

    for surface, text in outputs.items():
        for value in COMMON_VALUES:
            assert value in text, f"{surface} /status lost {value!r}:\n{text}"
    # Path is English-only (the gateway catalog has no path line); running flag is per-surface wording.
    assert HOME in outputs["cli"] and HOME in outputs["tui"]
    assert "Agent Running: Yes" in outputs["cli"] and "Agent Running: Yes" in outputs["tui"]
    assert "Yes" in outputs["gateway"]


def test_tui_status_bar_and_session_status_agree_on_identity():
    """The Ink TUI status bar (``_session_info``) and ``/status`` (``session.status``)
    must report the SAME (model, provider) for one session.

    Regression: each picked its own source order — the bar preferred the compute-host
    metadata mirror, ``/status`` preferred the live agent — so after a fallback or a
    model switch the bar named a stale provider while ``/status`` named the working one.
    Both now resolve through ``resolve_display_identity``.
    """
    from hermes_cli.status_report import resolve_display_identity
    from tui_gateway import server

    LIVE_MODEL, LIVE_PROVIDER = "vendor/live-model", "live-provider"
    STALE_PROVIDER = "stale-provider"
    live_agent = SimpleNamespace(model=LIVE_MODEL, provider=LIVE_PROVIDER)

    class _DB:
        def get_session(self, key):
            return {}

    def _bar_and_status(extra):
        session = {
            "agent": live_agent, "session_key": SESSION_ID, "history": [],
            "history_lock": threading.Lock(), "history_version": 0, "running": True,
            "attached_images": [], "image_counter": 0, "cols": 80,
            "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
            **extra,
        }
        server._sessions["identity-contract-sid"] = session
        try:
            with patch.object(server, "_get_db", lambda: _DB()):
                status = server.handle_request(
                    {"id": "1", "method": "session.status",
                     "params": {"session_id": "identity-contract-sid"}})["result"]["output"]
                info = server._session_info(live_agent, session)
            return info, status
        finally:
            server._sessions.pop("identity-contract-sid", None)

    # The mirror is the host's own view of the route and outranks the in-process agent,
    # so both surfaces name it — never the agent's stale or sentinel value.
    info, status = _bar_and_status({"_metadata_mirror": {"model": LIVE_MODEL, "provider": STALE_PROVIDER}})
    assert (info["model"], info["provider"]) == (LIVE_MODEL, STALE_PROVIDER), info
    assert f"{LIVE_MODEL} ({STALE_PROVIDER})" in status, status
    assert LIVE_PROVIDER not in status

    # No mirror at all: the live agent is the only identity there is, on both surfaces.
    info, status = _bar_and_status({})
    assert (info["model"], info["provider"]) == (LIVE_MODEL, LIVE_PROVIDER), info

    # A queued switch outranks the live agent on both, and does not blip back to it.
    info, _ = _bar_and_status({
        "pending_model_switch": {"display_model": "vendor/queued", "display_provider": "queued-provider"}})
    assert (info["model"], info["provider"]) == ("vendor/queued", "queued-provider"), info

    # Compute-host isolation: the in-process agent is a stale proxy, so the mirror is
    # authoritative — the same rule `session.status` applies when it drops the agent.
    info, status = _bar_and_status({
        "_compute_host_active": True,
        "_metadata_mirror": {"model": LIVE_MODEL, "provider": STALE_PROVIDER}})
    assert (info["model"], info["provider"]) == (LIVE_MODEL, STALE_PROVIDER), info
    assert f"{LIVE_MODEL} ({STALE_PROVIDER})" in status, status

    # The resolver's own contract: model and provider move as a pair, mirror before agent.
    assert resolve_display_identity(
        agent=SimpleNamespace(model="m1", provider="p1"),
        mirror={"model": "m2", "provider": "p2"},
        configured_model="m3", configured_provider="p3") == ("m2", "p2")
    assert resolve_display_identity(
        agent=None, mirror={"model": "m2", "provider": "p2"},
        configured_model="m3", configured_provider="p3") == ("m2", "p2")
    assert resolve_display_identity(
        agent=None, mirror=None, configured_model="m3", configured_provider="p3") == ("m3", "p3")
    assert resolve_display_identity(agent=None, mirror=None) == ("", "")
    # A pending switch outranks both, as a pair.
    assert resolve_display_identity(
        pending={"display_model": "pm", "display_provider": "pp"},
        agent=SimpleNamespace(model="m1", provider="p1"),
        mirror={"model": "m2", "provider": "p2"}) == ("pm", "pp")
    # A tier that names a model but no provider still takes the first provider on offer
    # — a model-only switch keeps the current provider rather than showing none.
    assert resolve_display_identity(
        agent=SimpleNamespace(model="m1", provider=""),
        mirror={"provider": "p2"}) == ("m1", "p2")
    # ...but a tier that names both never has its provider donated to a lower tier.
    assert resolve_display_identity(
        agent=SimpleNamespace(model="m1", provider="p1"),
        mirror={"provider": "p2"}) == ("m1", "p1")

    # The "custom" sentinel resolves identically on both surfaces when a resolver is
    # supplied — the bar must never print `custom:route` where `/status` prints `custom`.
    custom_agent = SimpleNamespace(model=LIVE_MODEL, provider="custom")
    resolve = lambda a: resolve_display_identity(
        agent=a, custom_resolver=lambda _: "custom:resolved")
    assert resolve(custom_agent) == (LIVE_MODEL, "custom:resolved")
    assert resolve_display_identity(agent=custom_agent) == (LIVE_MODEL, "custom")
