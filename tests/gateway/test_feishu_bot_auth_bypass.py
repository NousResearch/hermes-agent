"""Regression guard for Feishu bot-sender authorization bypass.

Mirrors tests/gateway/test_discord_bot_auth_bypass.py for Platform.FEISHU.
Without the bypass in gateway/run.py, Feishu bot senders admitted by the
adapter would be rejected at _is_user_authorized with "Unauthorized user"
— same class of bug as Discord #4466.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_feishu_env(monkeypatch):
    for var in (
        "FEISHU_ALLOW_BOTS",
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "FEISHU_PUBLIC_GROUP_QA",
        "FEISHU_PUBLIC_GROUP_QA_CHATS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_feishu_bot_source(open_id: str = "ou_peer"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="PeerBot",
        is_bot=True,
    )


def _make_feishu_human_source(open_id: str = "ou_human"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="Human",
        is_bot=False,
    )


def test_feishu_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is True


def test_feishu_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source()) is True


def test_feishu_bot_NOT_authorized_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "none")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is False


def test_feishu_bot_NOT_authorized_when_allow_bots_unset(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is False


def test_feishu_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must NOT open the gate for humans."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_human_source("ou_stranger")) is False
    assert runner._is_user_authorized(_make_feishu_human_source("ou_human")) is True


def test_feishu_bot_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must not authorize Telegram/Discord bot sources."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")

    telegram_bot = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id="999",
        is_bot=True,
    )
    assert runner._is_user_authorized(telegram_bot) is False


def test_feishu_public_group_qa_allows_only_feishu_groups(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_PUBLIC_GROUP_QA", "true")
    monkeypatch.setenv("FEISHU_PUBLIC_GROUP_QA_CHATS", "*")

    assert runner._is_public_group_qa_allowed(_make_feishu_human_source("ou_stranger")) is True

    dm_source = _make_feishu_human_source("ou_stranger")
    dm_source.chat_type = "dm"
    assert runner._is_public_group_qa_allowed(dm_source) is False

    telegram_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="group",
        user_id="999",
    )
    assert runner._is_public_group_qa_allowed(telegram_source) is False


def test_feishu_public_group_qa_honors_chat_allowlist(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_PUBLIC_GROUP_QA", "1")
    monkeypatch.setenv("FEISHU_PUBLIC_GROUP_QA_CHATS", "oc_allowed, oc_other")

    allowed = _make_feishu_human_source("ou_stranger")
    allowed.chat_id = "oc_allowed"
    denied = _make_feishu_human_source("ou_stranger")
    denied.chat_id = "oc_denied"

    assert runner._is_public_group_qa_allowed(allowed) is True
    assert runner._is_public_group_qa_allowed(denied) is False


def test_public_group_qa_is_not_proxied_and_gets_public_only_context():
    """Regression guard: public QA must remain a local no-private-context lane."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._run_agent)
    assert 'if self._get_proxy_url() and not public_group_qa:' in src
    assert 'combined_ephemeral = (' in src
    assert '[Public Feishu group QA mode]' in src
    assert 'else:\n                combined_ephemeral = context_prompt or ""' in src


def test_public_group_qa_does_not_replay_history_or_prefill_messages():
    """Regression guard: non-allowlisted group users must not receive private history/prefill."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._run_agent)
    assert 'prefill_messages=[] if public_group_qa else self._prefill_messages' in src
    assert 'agent_history = []\n            if not public_group_qa:' in src


def test_public_group_qa_cannot_answer_pending_control_prompts():
    """Regression guard: public QA text must not resolve shared control prompts."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._handle_message)
    assert "Public QA users are not fully authorized" in src
    assert "_clarify_mod.get_pending_for_session(_quick_key)" in src
    assert "_slash_confirm_mod.get_pending(_quick_key)" in src
    assert "has_blocking_approval(_quick_key)" in src
    assert "getattr(self, \"_update_prompt_pending\", {}).get(_quick_key)" in src


def test_public_group_qa_is_stateless_no_cache_no_persistence():
    """Regression guard: public QA must not persist into or reuse shared sessions."""
    import inspect
    from gateway.run import GatewayRunner

    handle_src = inspect.getsource(GatewayRunner._handle_message)
    agent_src = inspect.getsource(GatewayRunner._handle_message_with_agent)
    run_src = inspect.getsource(GatewayRunner._run_agent)

    assert "not getattr(source, \"_public_group_qa\", False)" in handle_src
    assert "history = [] if public_group_qa else" in agent_src
    assert "if public_group_qa:" in agent_src and "return response" in agent_src
    assert "use_agent_cache = not public_group_qa" in run_src
    assert "session_db=None if public_group_qa else self._session_db" in run_src


def test_public_group_qa_blocks_while_agent_running():
    """Regression guard: public QA cannot join or steer an already-running shared session."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._handle_message)
    assert 'or _quick_key in self._running_agents' in src


def test_public_group_qa_bypasses_inbound_context_and_media_enrichment():
    """Regression guard: public QA must not read files or run media tooling pre-agent."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._prepare_inbound_message_text)
    assert 'Public QA is a no-tools/no-private-context lane.' in src
    assert 'return message_text or "群聊公开问答模式暂不处理附件、图片或语音；请直接发送文字问题。"' in src
    assert 'return message_text\n\n        # Prepend channel context' in src
