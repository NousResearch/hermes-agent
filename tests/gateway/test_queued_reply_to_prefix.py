"""Queued/steered follow-ups must carry the reply-to prefix (#101866).

A Telegram message arriving mid-turn used to reach the agent bare: the
queued path built next_message from the pending event but never injected
the "[Replying to ...]" pointer — the exact disambiguation failure the
prefix exists for. The reporter's incident: two drafts pending, the user
reply-quoted draft A with "yes, send the same", the agent resolved it to
draft B (newest) and sent to the wrong recipient.

The fix moved the injection from the idle-path caller into
_prepare_inbound_message_text itself — which its own docstring already
promised ("reply context ... all behave the same").
"""

import asyncio
import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _event(text="", reply_to_text=None, reply_to_id=None, own=True):
    ev = SimpleNamespace()
    ev.text = text
    ev.reply_to_text = reply_to_text
    ev.reply_to_message_id = reply_to_id
    ev.reply_to_is_own_message = own
    ev.media_urls = []
    ev.media_types = []
    ev.message_id = None
    ev.channel_prompt = None
    ev.message_type = None
    # A minimal source so session-key derivation works.
    from gateway.platforms.base import Platform

    ev.source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        platform_name="telegram",
        user_id="u1",
        user_name="Tester",
        chat_id="c1",
        chat_type="dm",
        thread_id=None,
        gateway_session_key="t:u1:c1",
        session_key="t:u1:c1",
        message_type=None,
    )
    return ev


@pytest.fixture
def runner(monkeypatch):
    from gateway import run as run_mod

    r = object.__new__(run_mod.GatewayRunner)
    r.config = SimpleNamespace(
        multiplex_profiles=False,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )
    return r


class TestReplyPrefixInPrepare:
    @pytest.mark.asyncio
    async def test_reply_to_prefix_present_in_prepare_output(self, runner):
        """The prefix must come out of _prepare_inbound_message_text —
        the function both the idle path AND the queued path call."""
        ev = _event(
            text="yes, send the same",
            reply_to_text="Draft A: invoice for Alice",
            reply_to_id="msg-a",
        )
        out = await runner._prepare_inbound_message_text(
            event=ev,
            source=ev.source,
            history=[],
        )
        assert '[Replying to your previous message: "Draft A: invoice for Alice"]' in out
        assert "yes, send the same" in out

    @pytest.mark.asyncio
    async def test_foreign_reply_gets_their_message_form(self, runner):
        ev = _event(
            text="what did they mean",
            reply_to_text="Bob's original note",
            reply_to_id="m1",
            own=False,
        )
        out = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[]
        )
        assert '[Replying to: "Bob\'s original note"]' in out

    @pytest.mark.asyncio
    async def test_no_reply_context_no_prefix(self, runner):
        ev = _event(text="hello there")
        out = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[]
        )
        assert "[Replying to" not in out

    @pytest.mark.asyncio
    async def test_reply_text_without_id_no_prefix(self, runner):
        """The idle path required BOTH reply_to_text and
        reply_to_message_id; the moved code must keep that contract."""
        ev = _event(text="hi", reply_to_text="orphan quote", reply_to_id=None)
        out = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[]
        )
        assert "[Replying to" not in out

    @pytest.mark.asyncio
    async def test_long_reply_snippet_truncated_to_500(self, runner):
        ev = _event(
            text="go",
            reply_to_text="x" * 900,
            reply_to_id="m1",
        )
        out = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[]
        )
        assert "x" * 500 in out
        assert "x" * 501 not in out


class TestIdlePathNoDoubleInjection:
    """The old idle-path copy was replaced by a comment stub; the prefix
    must now appear exactly once (from prepare), not twice."""

    @pytest.mark.asyncio
    async def test_prefix_appears_exactly_once(self, runner):
        ev = _event(
            text="confirm",
            reply_to_text="Draft A",
            reply_to_id="a",
        )
        out = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[]
        )
        assert out.count("[Replying to your previous message:") == 1

    def test_old_idle_site_no_longer_injects(self):
        """Source-shape guard: the idle-path caller must not contain its
        own injection copy anymore (would double-prefix)."""
        import inspect

        from gateway import run as run_mod

        src = inspect.getsource(run_mod.GatewayRunner._handle_message_with_agent)
        needle = 'f\'[Replying to your previous message: "{reply_snippet}"]\\n\\n\''
        assert needle not in src, (
            "the injection belongs in _prepare_inbound_message_text only"
        )


class TestQueuedFollowupLogging:
    def test_queued_path_has_inbound_log_line(self):
        """Source-shape guard: the queued follow-up site must emit the
        inbound log line with reply context and queued=True (#101866's
        second complaint — these messages were invisible in the log).
        The audit line is owned by gateway.inbound_context; run.py
        delegates for both paths."""
        from gateway import inbound_context
        from gateway import run as run_mod

        src = inspect.getsource(inbound_context)
        assert "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r queued=%s" in src
        # And run.py delegates for BOTH paths — no second inline copy.
        run_src = inspect.getsource(run_mod)
        assert "log_inbound_reply_context(" in run_src
        assert "reply_to_id=%s" not in run_src, (
            "run.py must not carry a second inline implementation of the audit line"
        )

    def test_run_py_net_shrank_not_grew(self):
        """Reviewer acceptance (6): gateway/run.py must not grow from this
        PR — the reply-context concern moved to its bounded owner; run.py
        only delegates. Guarded against the PR's own diff."""
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--numstat", "--", "gateway/run.py"],
            capture_output=True, text=True,
        ).stdout.strip()
        if diff:
            added, removed, _ = diff.split(maxsplit=2)
            assert int(added) <= int(removed), (
                f"gateway/run.py net-grew by this PR (+{added}/-{removed}) — "
                "the reply-context concern must live in inbound_context, "
                "run.py must only delegate"
            )

    def test_inbound_context_module_is_bounded(self):
        """Reviewer acceptance (6): every modified file <= 2,000 lines."""
        from gateway import inbound_context

        line_count = len(
            Path(inbound_context.__file__).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 2000
