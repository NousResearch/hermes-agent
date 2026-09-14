"""Real durable admission for route tests; only execution is held at scheduling."""
import asyncio
from contextlib import suppress


def mount_authority(app, adapter):
    authority = None
    async def start(app):
        nonlocal authority
        from gateway.config import Platform
        from gateway.run import GatewayRunner
        from gateway.session_authority import initialize_session_authority
        from gateway.session import SessionStore
        from hermes_constants import get_hermes_home
        runner = GatewayRunner()
        runner.session_store = SessionStore(sessions_dir=get_hermes_home() / "sessions", config=runner.config)
        from hermes_state import SessionDB
        runner.session_store._db = SessionDB(db_path=get_hermes_home() / "state.db")
        runner._session_db = runner.session_store._db
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='route-test')
        runner.adapters[Platform.WEBHOOK] = adapter
        runner._wire_adapter_handlers(adapter)
        # These tests inspect routing/rendering, not inference. Leave the actual
        # committed FIFO untouched; execution is covered by test_webhook_authority.
        authority._schedule = lambda ref: None
        original = authority.admit_native
        async def capture(event):
            receipt = await original(event)
            observer = adapter.__dict__.get('handle_message')
            if observer is not None:
                await observer(event)
            return receipt
        authority.admit_native = capture

    async def cleanup(app):
        for task in list(adapter._background_tasks):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        authority.db.close()

    app.on_startup.append(start)
    app.on_cleanup.append(cleanup)
