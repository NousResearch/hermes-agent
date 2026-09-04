"""Multi-turn history injection tests for the A2A platform plugin.

When a caller reuses a ``contextId`` (multi-turn continuation), the adapter
prepends the persisted conversation so the agent sees the full thread. The
original message — not the augmented copy — is what gets audited and
persisted, so injected history never accumulates in the on-disk log.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import protocol, security


def _send_params(text: str, context_id: str) -> dict:
    return {
        "message": {
            "role": "user",
            "parts": [{"text": text, "mediaType": "text/plain"}],
        },
        "contextId": context_id,
    }


class TestFormatHistory:
    def test_empty_when_no_history(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        assert protocol.format_history("ctx-none", limit=20) == ""

    def test_roles_rendered_user_assistant(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        protocol.persist_message("ctx-a", "user", "hello")
        protocol.persist_message("ctx-a", "agent", "hi there")
        assert protocol.format_history("ctx-a", limit=20) == (
            "user: hello\nassistant: hi there\n\n"
        )

    def test_limit_keeps_only_recent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        for i in range(5):
            protocol.persist_message("ctx-b", "user", f"msg-{i}")
        out = protocol.format_history("ctx-b", limit=2)
        assert "msg-0" not in out
        assert "msg-4" in out

    def test_default_limit_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        monkeypatch.setenv("A2A_HISTORY_INJECTION_LIMIT", "3")
        for i in range(5):
            protocol.persist_message("ctx-c", "user", f"msg-{i}")
        out = protocol.format_history("ctx-c")
        assert "msg-0" not in out
        assert "msg-4" in out

    def test_zero_limit_disables(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        protocol.persist_message("ctx-d", "user", "hello")
        assert protocol.format_history("ctx-d", limit=0) == ""


def _bare_adapter():
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    return A2AAdapter(PlatformConfig(enabled=True))


class TestInboundHistoryInjection:
    def _drive(self, adapter, monkeypatch):
        """Run _prepare_task with dispatch captured; return the event text list."""
        seen_events = []

        async def fake_handle(event):
            seen_events.append(event)

        adapter.handle_message = fake_handle
        adapter._message_handler = lambda event: None
        adapter._loop = asyncio.new_event_loop()

        captured_coros = []

        def fake_schedule(coro, loop):
            captured_coros.append(coro)
            return SimpleNamespace(done=lambda: True)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_schedule)
        return seen_events, captured_coros

    def test_new_context_no_injection(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        adapter = _bare_adapter()
        seen, coros = self._drive(adapter, monkeypatch)

        task, pending = adapter._prepare_task(
            _send_params("first question", "ctx-new"), peer="peer-x"
        )
        assert pending is not None
        asyncio.run(coros[0])
        text = seen[0].text
        assert "first question" in text
        assert not text.startswith("user:")

    def test_resume_injects_history_into_agent_text(self, monkeypatch, tmp_path):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        protocol.persist_message("ctx-mt", "user", "first question")
        protocol.persist_message("ctx-mt", "agent", "first answer")

        adapter = _bare_adapter()
        seen, coros = self._drive(adapter, monkeypatch)

        task, pending = adapter._prepare_task(
            _send_params("second question", "ctx-mt"), peer="peer-x"
        )
        assert pending is not None
        asyncio.run(coros[0])
        text = seen[0].text
        assert "first question" in text
        assert "first answer" in text
        assert "second question" in text
        assert text.index("first question") < text.index("second question")

    def test_persist_and_audit_store_original_not_augmented(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        protocol.persist_message("ctx-keep", "user", "first question")
        protocol.persist_message("ctx-keep", "agent", "first answer")

        audited = []
        monkeypatch.setattr(security, "audit", lambda *a, **k: audited.append(a))
        adapter = _bare_adapter()
        seen, coros = self._drive(adapter, monkeypatch)

        task, pending = adapter._prepare_task(
            _send_params("second question", "ctx-keep"), peer="peer-x"
        )
        assert pending is not None
        asyncio.run(coros[0])

        # Agent saw the augmented text (wrap_inbound adds its own prefix)…
        assert "user: first question" in seen[0].text
        assert "second question" in seen[0].text
        # …but the persisted log and audit trail carry only the original.
        recs = protocol.load_conversation("ctx-keep", limit=50)
        assert recs[-1]["role"] == "user"
        assert recs[-1]["text"] == "second question"
        assert any("second question" in a[3] for a in audited)
        assert all("first question" not in a[3] for a in audited)

    def test_resume_does_not_duplicate_injected_prefix(self, monkeypatch, tmp_path):
        """A third turn must not re-inject the already-injected prefix."""
        monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
        adapter = _bare_adapter()
        seen, coros = self._drive(adapter, monkeypatch)

        for i, text in enumerate(["q1", "q2", "q3"]):
            task, pending = adapter._prepare_task(
                _send_params(text, "ctx-loop"), peer="peer-x"
            )
            assert pending is not None
            asyncio.run(coros[-1])

        # After three turns the on-disk log holds exactly the three originals.
        recs = protocol.load_conversation("ctx-loop", limit=50)
        assert [r["text"] for r in recs] == ["q1", "q2", "q3"]
