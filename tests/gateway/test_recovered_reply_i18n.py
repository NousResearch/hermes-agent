"""The recovered-reply marker on a redelivered final is rendered when it is SENT, in the active
display.language — a row claimed earlier carries only its cause, never English text frozen at claim time.
Real ledger rows in a temp state.db, a real user overlay under a temp HERMES_HOME, no i18n mocks."""

from unittest.mock import AsyncMock, MagicMock

import hermes_yaml as yaml
import pytest

from agent import i18n
from gateway import delivery_ledger as dl

_GERMAN = "♻️ Wiederhergestellte Antwort — möglicherweise ein Duplikat:"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    i18n.reset_language_cache()
    yield home
    i18n.reset_language_cache()


def _runner(adapter):
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    runner.session_store = None
    runner._async_session_store = store
    return runner


@pytest.mark.asyncio
async def test_recovered_marker_uses_the_language_active_at_send_time(home):
    dl.record_obligation(obligation_id="ob-1", session_key="agent:main:slack:channel:C1", platform="slack",
                         chat_id="C1", thread_id=None, content="the final answer", adapter_profile=None)
    dl.mark_attempting("ob-1")  # crashed mid-send: the redelivery must be marked
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1")
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
    runner = _runner(adapter)
    claimed = await runner._claim_pending_obligations()  # claimed while the profile is still English
    assert claimed

    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"language": "de"}}), encoding="utf-8")
    (home / "locales" / "de.yaml").write_text(
        yaml.safe_dump({"gateway": {"recovered_reply": {"restart": _GERMAN}}}, allow_unicode=True), encoding="utf-8")
    i18n.reset_language_cache()

    assert await runner._redeliver_claimed_obligations(claimed) == 1
    assert adapter.send.call_args.kwargs["content"] == f"{_GERMAN}\n\nthe final answer"
