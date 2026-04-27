import os

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from tools.session_context import (
    get_chat_id,
    get_chat_name,
    get_platform,
    get_thread_id,
    set_session,
    clear_session,
)


def test_set_session_env_populates_contextvars(monkeypatch):
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    runner._set_session_env(context)

    # ContextVars are the single source of truth now.
    assert get_platform() == "telegram"
    assert get_chat_id() == "-1001"
    assert get_chat_name() == "Group"
    assert get_thread_id() == "17585"

    # os.environ is intentionally NOT written — that was the H-1 race.
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_CHAT_ID") is None
    assert os.getenv("HERMES_SESSION_CHAT_NAME") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

    clear_session()


def test_clear_session_env_clears_contextvars():
    set_session(
        platform="telegram",
        chat_id="-1001",
        chat_name="Group",
        thread_id="17585",
    )
    runner = object.__new__(GatewayRunner)
    runner._clear_session_env()

    assert get_platform() is None
    assert get_chat_id() is None
    assert get_chat_name() is None
    assert get_thread_id() is None


def test_concurrent_handlers_do_not_clobber_session(monkeypatch):
    """Two concurrent message handlers must each see their own session scope.

    This is the H-1 race: pre-fix `_set_session_env` wrote os.environ which is
    process-global, so a context switch during an `await` let handler-B's
    chat_id leak into handler-A's tool calls. ContextVars are asyncio-task-local
    and survive task switches.
    """
    import asyncio

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    runner = object.__new__(GatewayRunner)

    def _ctx_for(chat_id):
        return SessionContext(
            source=SessionSource(
                platform=Platform.SLACK,
                chat_id=chat_id,
                chat_name=f"dm-{chat_id}",
                chat_type="dm",
            ),
            connected_platforms=[],
            home_channels={},
        )

    observed = {}

    async def handler(name, chat_id):
        runner._set_session_env(_ctx_for(chat_id))
        # Yield, letting the other handler run and mutate global state.
        await asyncio.sleep(0)
        # Read both surfaces. ContextVar is task-local (always correct).
        # os.environ is process-global — the H-1 attack vector. After fix
        # it must always be empty for these keys.
        observed[name] = {
            "ctx_chat_id": get_chat_id(),
            "env_chat_id": os.getenv("HERMES_SESSION_CHAT_ID"),
        }
        runner._clear_session_env()

    async def main():
        await asyncio.gather(
            handler("alice", "D_ALICE"),
            handler("bob", "D_BOB"),
        )

    asyncio.run(main())

    # ContextVar isolation: each handler sees its own chat_id.
    assert observed["alice"]["ctx_chat_id"] == "D_ALICE"
    assert observed["bob"]["ctx_chat_id"] == "D_BOB"

    # The H-1 assertion: os.environ must NOT carry session data — that is
    # the process-global channel that races. Pre-fix, after the await, both
    # handlers would see whichever chat_id was written last (typically Bob's).
    assert observed["alice"]["env_chat_id"] is None, (
        "os.environ leaked session chat_id across the await — H-1 regression"
    )
    assert observed["bob"]["env_chat_id"] is None, (
        "os.environ leaked session chat_id across the await — H-1 regression"
    )

    # And after both clear, no residue remains.
    assert os.getenv("HERMES_SESSION_CHAT_ID") is None
