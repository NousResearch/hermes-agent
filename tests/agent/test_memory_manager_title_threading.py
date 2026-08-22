"""MemoryManager threads session_title only to providers that accept it (#86824)."""

from __future__ import annotations

import pytest


class _TitleProvider:
    name = "title-provider"

    def __init__(self, accepts_title: bool):
        self._accepts_title = accepts_title
        self.received: dict = {}

    def sync_turn(self, user_content, assistant_content, **kwargs):
        self.received = kwargs

    @property
    def __signature__capture(self):
        pass


def _provider_with_sig(accepts_title: bool):
    """Provider whose sync_turn signature advertises session_title or not."""

    class P(_TitleProvider):
        def __init__(self):
            super().__init__(accepts_title)

        if accepts_title:

            def sync_turn(self, user_content, assistant_content, *, session_id="", session_title=""):
                self.received = {"session_id": session_id, "session_title": session_title}

        else:

            def sync_turn(self, user_content, assistant_content, *, session_id=""):
                self.received = {"session_id": session_id}

    return P()


def test_title_passed_only_to_accepting_provider():
    from agent.memory_manager import MemoryManager

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._providers = [_provider_with_sig(True), _provider_with_sig(False)]
    mgr._submit_background = lambda fn, **kw: fn()
    mgr._strip_skill_scaffolding = lambda s: s

    mgr.sync_all(
        "user",
        "assistant",
        session_id="sess-1",
        session_title="API Key Update Verified",
    )

    assert mgr._providers[0].received.get("session_title") == "API Key Update Verified"
    assert "session_title" not in mgr._providers[1].received
