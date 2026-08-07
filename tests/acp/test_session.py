"""Tests for acp_adapter.session — SessionManager and SessionState."""

import contextlib
import io
import json
import time
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from acp_adapter import session as acp_session
from acp_adapter.session import SessionManager, SessionState
from hermes_state import SessionDB


def _mock_agent():
    return MagicMock(name="MockAIAgent")


@pytest.fixture()
def manager():
    """SessionManager with a mock agent factory (avoids needing API keys)."""
    return SessionManager(agent_factory=_mock_agent)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session_returns_state(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert isinstance(state, SessionState)
        assert state.cwd == "/tmp/work"
        assert state.session_id
        assert state.history == []
        assert state.agent is not None



    def test_register_task_cwd_translates_windows_drive_for_wsl_tools(self, monkeypatch):
        captured = {}

        def fake_register_task_env_overrides(task_id, overrides):
            captured["task_id"] = task_id
            captured["overrides"] = overrides

        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        monkeypatch.setattr(
            "tools.terminal_tool.register_task_env_overrides",
            fake_register_task_env_overrides,
        )

        acp_session._register_task_cwd("session-1", r"E:\Projects\AI\paperclip")

        assert captured == {
            "task_id": "session-1",
            "overrides": {"cwd": "/mnt/e/Projects/AI/paperclip"},
        }


    def test_get_session(self, manager):
        state = manager.create_session()
        fetched = manager.get_session(state.session_id)
        assert fetched is state


    def test_make_agent_stamps_session_cwd_for_codex_runtime(self, monkeypatch):
        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "acp_adapter.session.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.session_cwd == "/tmp/project"




# ---------------------------------------------------------------------------
# WSL cwd translation
# ---------------------------------------------------------------------------


class TestWslCwdTranslation:
    def test_translate_acp_cwd_converts_windows_drive_path_when_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd(r"E:\Projects\AI\paperclip") == "/mnt/e/Projects/AI/paperclip"





    def test_fork_session_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        original = manager.create_session(cwd="/tmp/base")

        forked = manager.fork_session(original.session_id, cwd=r"D:\work\project")

        assert forked is not None
        assert forked.cwd == "/mnt/d/work/project"

    def test_update_cwd_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        state = manager.create_session(cwd="/tmp/old")

        updated = manager.update_cwd(state.session_id, cwd=r"C:\Users\foo\project")

        assert updated is not None
        assert updated.cwd == "/mnt/c/Users/foo/project"

# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# list / cleanup / remove
# ---------------------------------------------------------------------------


class TestListAndCleanup:
    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []



    def test_save_session_preserves_existing_messages_on_encode_failure(self, manager):
        """Regression for #13675: a bad message in state.history must not
        clobber the previously-persisted transcript.  replace_messages()
        wraps DELETE + INSERT in a single rolled-back-on-exception txn.
        """
        state = manager.create_session()
        state.history.append({"role": "user", "content": "original"})
        manager.save_session(state.session_id)

        # Now swap history with a message whose tool_calls is non-JSON-serializable.
        # _execute_write rolls back; the previously persisted "original" stays.
        state.history = [
            {"role": "user", "content": "replacement"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"bad": object()}],
            },
        ]
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"
        assert isinstance(messages[0].get("timestamp"), (int, float))




    def test_cleanup_clears_all(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        s1.history.append({"role": "user", "content": "one"})
        s2.history.append({"role": "user", "content": "two"})
        assert len(manager.list_sessions()) == 2
        manager.cleanup()
        assert manager.list_sessions() == []

    def test_remove_session(self, manager):
        state = manager.create_session()
        assert manager.remove_session(state.session_id) is True
        assert manager.get_session(state.session_id) is None
        # Removing again returns False
        assert manager.remove_session(state.session_id) is False


# ---------------------------------------------------------------------------
# persistence — sessions survive process restarts (via SessionDB)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify that sessions are persisted to SessionDB and can be restored."""














    def test_only_restores_acp_sessions(self, manager):
        """get_session should not restore non-ACP sessions from DB."""
        db = manager._get_db()
        # Manually create a CLI session in the DB.
        db.create_session(session_id="cli-session-123", source="cli", model="test")
        # Should not be found via ACP SessionManager.
        assert manager.get_session("cli-session-123") is None

    def test_sessions_searchable_via_fts(self, manager):
        """ACP sessions stored in SessionDB are searchable via FTS5."""
        state = manager.create_session()
        state.history.append({"role": "user", "content": "how do I configure nginx"})
        state.history.append({"role": "assistant", "content": "Here is the nginx config..."})
        manager.save_session(state.session_id)

        db = manager._get_db()
        results = db.search_messages("nginx")
        assert len(results) > 0
        session_ids = {r["session_id"] for r in results}
        assert state.session_id in session_ids


    def test_assistant_reasoning_fields_persisted(self, manager):
        """ACP session restore should preserve assistant reasoning context."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        })
        manager.save_session(state.session_id)

        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        msg = restored.history[0]
        assert isinstance(msg.pop("timestamp", None), (int, float))
        assert restored.history == [{
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        }]


    def test_restore_bare_custom_forwards_persisted_base_url(self, tmp_path, monkeypatch):
        """A stored bare ``custom`` provider must forward its persisted base_url.

        Named ``custom_providers`` entries normalize to the bare ``"custom"``
        billing bucket at runtime, and that bucket is what gets persisted
        (``billing_provider``/``billing_base_url``). On restore, the resolver
        can only match the bare bucket back to the named pool entry when the
        persisted URL is forwarded as ``explicit_base_url`` — otherwise
        ``_get_named_custom_provider("custom")`` finds nothing and the resumed
        session dies with "Could not resolve authentication method"
        (create path works, restore path fails, same config).

        Regression test for the ACP restore path; the TUI/gateway path covers
        the same bucket-skipping via ``_stored_session_runtime_overrides``.
        """
        resolve_calls = []

        def fake_resolve_runtime_provider(requested=None, explicit_base_url=None, **kwargs):
            resolve_calls.append({
                "requested": requested,
                "explicit_base_url": explicit_base_url,
            })
            return {
                # Named custom providers normalize to the bare bucket at runtime.
                "provider": "custom",
                "api_mode": "anthropic_messages",
                "base_url": explicit_base_url or "https://myendpoint.example/v1",
                "api_key": "pooled-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "myendpoint", "default": "test-model"},
            "custom_providers": {
                "myendpoint": {
                    "base_url": "https://myendpoint.example/v1",
                    "api_key": "pooled-key",
                }
            },
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")
            manager.save_session(state.session_id)

            with manager._lock:
                del manager._sessions[state.session_id]

            restored = manager.get_session(state.session_id)

        assert restored is not None
        assert resolve_calls, "resolve_runtime_provider was never called"
        # The restore call is the one carrying the bare bucket; it must
        # forward the persisted URL so the pool lookup can succeed.
        restore_call = resolve_calls[-1]
        assert restore_call["requested"] == "custom"
        assert restore_call["explicit_base_url"] == "https://myendpoint.example/v1"

    def test_restore_bare_custom_resolves_pooled_credentials_end_to_end(self, tmp_path, monkeypatch):
        """Round-trip through the REAL resolver — no resolver mocks.

        Complements the argument-capture test above by exercising the actual
        ``resolve_runtime_provider`` code path: the named entry normalizes to
        the bare ``custom`` bucket at runtime and is persisted that way; on
        restore the resolver must match the bucket back to the named pool
        entry via the persisted base_url. Without forwarding, the restore
        falls through to env/default fallbacks and the resumed session gets a
        wrong or empty api_key — while the create path (same config) works.

        Note: ``_make_agent`` prefers the persisted ``base_url`` over the
        resolved one (``base_url or runtime.get("base_url")``), so the
        credential — not the URL — is the observable that distinguishes a
        successful pool match from a silent fallback.

        The pool storage layer (``load_pool`` → auth.json) and pool-key
        derivation are stubbed at the ``runtime_provider`` namespace — the
        resolver logic itself (bare-bucket handling, direct-alias path,
        credential assembly) runs for real, which is the code this fix
        touches.
        """
        pooled_config = {
            "model": {"provider": "myendpoint", "default": "test-model"},
            "custom_providers": [
                {
                    "name": "myendpoint",
                    "base_url": "https://myendpoint.example/v1",
                }
            ],
        }
        # runtime_provider binds load_config via from-import, so patch the
        # name in its own module namespace as well as the config module.
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: pooled_config)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: pooled_config)
        # Pool storage boundary: a pool holding "pooled-key" exists for the
        # endpoint (in production: seeded by `hermes auth` into auth.json).
        # Both stubs are SELECTIVE — they only honor the exact pool key and
        # base_url of the configured entry. An indiscriminate stub would feed
        # the pooled credential to unrelated fallback paths and mask the bug
        # (the test passed without the fix until the stubs were tightened).
        fake_pool = SimpleNamespace(
            has_credentials=lambda: True,
            select=lambda: SimpleNamespace(runtime_api_key="pooled-key", access_token=""),
        )
        empty_pool = SimpleNamespace(has_credentials=lambda: False, select=lambda: None)

        def fake_pool_key(base_url, provider_name=None):
            if provider_name == "myendpoint":
                return "custom:myendpoint"
            if (base_url or "").strip().rstrip("/") == "https://myendpoint.example/v1":
                return "custom:myendpoint"
            return None

        def fake_load_pool(key):
            return fake_pool if key == "custom:myendpoint" else empty_pool

        monkeypatch.setattr("hermes_cli.runtime_provider.get_custom_provider_pool_key", fake_pool_key)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_pool", fake_load_pool)
        # Env fallbacks must not mask a missing pool match.
        for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_key=kwargs.get("api_key"),
                api_mode=kwargs.get("api_mode"),
            )

        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")
            manager.save_session(state.session_id)

            with manager._lock:
                del manager._sessions[state.session_id]

            restored = manager.get_session(state.session_id)

        assert restored is not None
        # Bare bucket on the resumed session is expected — the credential and
        # endpoint must come from the named pool entry, not from fallbacks.
        assert restored.agent.provider == "custom"
        assert restored.agent.base_url == "https://myendpoint.example/v1"
        assert restored.agent.api_key == "pooled-key"

    def test_acp_agents_route_human_output_to_stderr(self, tmp_path, monkeypatch):
        """ACP agents must keep stdout clean for JSON-RPC stdio transport."""

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            return {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "test-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(model=kwargs.get("model"), _print_fn=None)

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            state.agent._print_fn("ACP noise")

        assert stdout_buf.getvalue() == ""
        assert stderr_buf.getvalue() == "ACP noise\n"
