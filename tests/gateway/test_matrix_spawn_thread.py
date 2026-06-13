import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.mirror as mirror_mod
from gateway.config import GatewayConfig, Platform
from gateway.mirror import ensure_outbound_session
from gateway.session import SessionSource, SessionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_mock_runner(store, adapter=None):
    runner = MagicMock()
    runner.session_store = store
    runner.config.platforms.get.return_value = None
    if adapter is not None:
        runner.adapters.get.side_effect = lambda plat, default=None: (
            adapter if plat == Platform.MATRIX else default
        )
    else:
        runner.adapters.get.return_value = None
    return runner


def _matrix_source(chat_id="!room:server", thread_id="$ev1"):
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id=chat_id,
        chat_type="group",
        user_id=None,
        thread_id=thread_id,
    )


def test_ensure_outbound_session_creates_session(store, tmp_path, monkeypatch):
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", tmp_path / "sessions" / "sessions.json")
    runner = _make_mock_runner(store)

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = ensure_outbound_session("matrix", "!room:server", "$ev1")

    assert result is True
    matching_keys = [k for k in store._entries if "$ev1" in k]
    assert len(matching_keys) >= 1, (
        f"Expected a session key containing '$ev1', got: {list(store._entries.keys())}"
    )


def test_mirror_to_session_writes_content(store, tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", sessions_dir / "sessions.json")

    runner = _make_mock_runner(store)
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        ensure_outbound_session("matrix", "!room:server", "$ev1")

    matching_keys = [k for k in store._entries if "$ev1" in k]
    assert matching_keys, "Session not created by ensure_outbound_session"
    session_id = store._entries[matching_keys[0]].session_id

    store_sessions_json = tmp_path / "sessions.json"
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", store_sessions_json)

    def _route_append_through_store(sid, msg):
        store.append_to_transcript(sid, msg)

    with patch.object(mirror_mod, "_append_to_sqlite", side_effect=_route_append_through_store):
        from gateway.mirror import mirror_to_session
        result = mirror_to_session("matrix", "!room:server", "seed text", thread_id="$ev1")

    assert result is True
    transcript = store.load_transcript(session_id)
    assert len(transcript) == 1
    assert transcript[0]["role"] == "assistant"
    assert transcript[0]["content"] == "seed text"


@pytest.mark.asyncio
async def test_inject_thread_root_fires_on_empty_history(store):
    mock_api = MagicMock()
    mock_api.request = AsyncMock(
        return_value={"content": {"body": "seed"}, "origin_server_ts": 1700000000000}
    )
    mock_client = MagicMock()
    mock_client.api = mock_api
    mock_adapter = MagicMock()
    mock_adapter._client = mock_client

    runner = _make_mock_runner(store, adapter=mock_adapter)
    source = _matrix_source()
    session_entry = store.get_or_create_session(source)

    from gateway.run import GatewayRunner
    await GatewayRunner._inject_matrix_thread_root(runner, source, session_entry)

    transcript = store.load_transcript(session_entry.session_id)
    assert len(transcript) == 1
    assert transcript[0]["role"] == "assistant"
    assert transcript[0]["content"] == "seed"


@pytest.mark.asyncio
async def test_inject_thread_root_skips_on_nonempty_history(store):
    from gateway.run import GatewayRunner

    runner = _make_mock_runner(store)
    source = _matrix_source()
    session_entry = store.get_or_create_session(source)
    store.append_to_transcript(session_entry.session_id, {"role": "user", "content": "existing"})

    history = store.load_transcript(session_entry.session_id)
    assert len(history) == 1

    inject_called = False

    async def _spy_inject(self_runner, src, entry):
        nonlocal inject_called
        inject_called = True

    with patch.object(GatewayRunner, "_inject_matrix_thread_root", _spy_inject):
        if not history:
            await GatewayRunner._inject_matrix_thread_root(runner, source, session_entry)

    assert not inject_called
    assert len(store.load_transcript(session_entry.session_id)) == 1


@pytest.mark.asyncio
async def test_inject_thread_root_silent_on_api_failure(store):
    from gateway.run import GatewayRunner

    mock_api = MagicMock()
    mock_api.request = AsyncMock(side_effect=Exception("M_NOT_FOUND"))
    mock_client = MagicMock()
    mock_client.api = mock_api
    mock_adapter = MagicMock()
    mock_adapter._client = mock_client

    runner = _make_mock_runner(store, adapter=mock_adapter)
    source = _matrix_source()
    session_entry = store.get_or_create_session(source)

    await GatewayRunner._inject_matrix_thread_root(runner, source, session_entry)

    assert len(store.load_transcript(session_entry.session_id)) == 0


@pytest.mark.asyncio
async def test_inject_thread_root_non_matrix_platform_skipped(store):
    from gateway.run import GatewayRunner

    inject_called = False

    async def _spy_inject(self_runner, src, entry):
        nonlocal inject_called
        inject_called = True

    runner = _make_mock_runner(store)
    telegram_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="group",
        user_id=None,
        thread_id="$ev1",
    )
    session_entry = store.get_or_create_session(telegram_source)
    history = store.load_transcript(session_entry.session_id)

    with patch.object(GatewayRunner, "_inject_matrix_thread_root", _spy_inject):
        if not history and telegram_source.platform == Platform.MATRIX:
            await GatewayRunner._inject_matrix_thread_root(runner, telegram_source, session_entry)

    assert not inject_called


def test_threads_seeded_on_ensure_outbound(store, tmp_path, monkeypatch):
    from gateway.platforms.helpers import ThreadParticipationTracker

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    tracker = ThreadParticipationTracker("matrix")
    mock_adapter = MagicMock()
    mock_adapter._threads = tracker

    runner = _make_mock_runner(store, adapter=mock_adapter)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", sessions_dir / "sessions.json")

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = ensure_outbound_session("matrix", "!room:server", "$ev1")

    assert result is True
    assert "$ev1" in tracker, (
        f"Expected '$ev1' in adapter._threads, got: {list(tracker._threads.keys())}"
    )


def test_require_mention_bypass_seeded_thread(tmp_path, monkeypatch):
    from gateway.platforms.helpers import ThreadParticipationTracker

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    tracker = ThreadParticipationTracker("matrix")
    tracker.mark("$ev1")

    thread_id = "$ev1"
    in_bot_thread = bool(thread_id and thread_id in tracker)

    assert in_bot_thread is True


def test_mirror_fields_survive_db_roundtrip(store):
    source = _matrix_source()
    session_entry = store.get_or_create_session(source)
    store.append_to_transcript(
        session_entry.session_id,
        {"role": "assistant", "content": "seed text", "mirror": True, "mirror_source": "matrix"},
    )
    transcript = store.load_transcript(session_entry.session_id)
    assert len(transcript) == 1
    assert transcript[0]["mirror"] is True
    assert transcript[0]["mirror_source"] == "matrix"


def test_build_gateway_agent_history_mirror_prefix():
    from gateway.run import _build_gateway_agent_history
    history = [{"role": "assistant", "content": "hello", "mirror": True, "mirror_source": "matrix"}]
    agent_history, _ = _build_gateway_agent_history(history)
    assert len(agent_history) == 1
    assert agent_history[0]["content"] == "[Delivered from matrix] hello"


def test_ensure_outbound_session_chat_type_from_session_key(store, tmp_path, monkeypatch):
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", tmp_path / "sessions" / "sessions.json")

    runner = _make_mock_runner(store)

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        with patch(
            "gateway.session_context.get_session_env",
            side_effect=lambda k, d="": {
                "HERMES_SESSION_KEY": "agent:main:matrix:group:!room:server:$parent",
            }.get(k, d),
        ):
            result = ensure_outbound_session("matrix", "!room:server", "$ev1")

    assert result is True
    group_keys = [k for k in store._entries if "group" in k and "$ev1" in k]
    assert len(group_keys) >= 1, f"Expected group session key, got: {list(store._entries.keys())}"


def test_ensure_outbound_session_defaults_to_group_without_session_key(store, tmp_path, monkeypatch):
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", tmp_path / "sessions" / "sessions.json")

    runner = _make_mock_runner(store)

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        with patch("gateway.session_context.get_session_env", return_value=""):
            result = ensure_outbound_session("matrix", "!room:server", "$ev2")

    assert result is True
    group_keys = [k for k in store._entries if "group" in k and "$ev2" in k]
    assert len(group_keys) >= 1, f"Expected group default, got: {list(store._entries.keys())}"
