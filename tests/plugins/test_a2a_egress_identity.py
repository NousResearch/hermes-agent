"""A2A egress identity: what a remote peer may receive.

Two mechanisms, both deterministic (no NER, no name-shaped inference):
  1. a FAILED profile dispatch reports a fixed-shape status object — state, error class, ids, byte
     counts — instead of the forwarded child's own output;
  2. ``security.redact_outbound`` scrubs the operator's declared identity literals plus phone/postal
     SHAPES as text leaves for a peer.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

import pytest

from plugins.platforms.a2a import protocol, security


@pytest.fixture(autouse=True)
def _isolate_identity_denylist(monkeypatch):
    """Keep tests independent of whatever this host's env/config happens to hold. Nothing to reset any
    more: there is no captured module global — the list rides on ``A2ASecurityContext`` or is resolved
    from the caller's scope at call time."""
    monkeypatch.delenv("A2A_IDENTITY_DENYLIST", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})


class TestDeclaredIdentityLiterals:
    def test_declared_literals_never_reach_the_peer(self, monkeypatch):
        monkeypatch.setenv("A2A_IDENTITY_DENYLIST", "Caleb,example.com")
        out = security.redact_outbound("Caleb asked about example.com; reply to caleb@example.com")
        assert "Caleb" not in out and "example.com" not in out
        assert out.count("[redacted-identity]") == 2
        assert "[redacted-email]" in out  # the e-mail pass still runs first

    def test_word_shaped_literals_do_not_mangle_longer_words(self, monkeypatch):
        monkeypatch.setenv("A2A_IDENTITY_DENYLIST", "roam")
        assert security.redact_outbound("roaming charges apply") == "roaming charges apply"
        assert security.redact_outbound("from roam") == "from [redacted-identity]"

    def test_literals_come_from_config_when_no_env(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"a2a": {"identity_denylist": ["Caleb"]}})
        assert "Caleb" not in security.redact_outbound("a note from Caleb")

    def test_too_short_literals_are_dropped_rather_than_mangle_prose(self, monkeypatch):
        monkeypatch.setenv("A2A_IDENTITY_DENYLIST", "e,at")
        text = "the answer is great at noon"
        assert security.redact_outbound(text) == text

    def test_what_was_scrubbed_is_never_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_IDENTITY_DENYLIST", "Caleb")
        with caplog.at_level(logging.DEBUG):
            security.redact_outbound("Caleb ships from 10001-1234")
        assert "caleb" not in caplog.text.lower()


class TestIdentityShapes:
    @pytest.mark.parametrize("number", ["(555) 123-4567", "555-123-4567", "555.123.4567", "+1 (555) 123-4567"])
    def test_phone_shapes_are_scrubbed(self, number):
        assert "[redacted-phone]" in security.redact_outbound(f"call {number} now")

    def test_zip_plus_four_and_anchored_zip_are_scrubbed(self):
        out = security.redact_outbound("ships to 10001-1234 / ZIP 10001 / NY 10001")
        assert out.count("[redacted-postal]") == 3

    def test_unanchored_five_digit_runs_survive(self):
        """A bare 5-digit run is a byte count or a row id far more often than an address — the
        structured status object below is only parseable if this holds."""
        assert security.redact_outbound("bytes=12345 id=90210") == "bytes=12345 id=90210"


_MARKED_REPLY_SCRIPT = "#!/usr/bin/env python3\nprint('note alphaperson betaperson')\n"


_CHILD_OUT = "source note: CALEB's address is 10001-1234, call 555-123-4567\n"
_CHILD_ERR = "traceback: for peer eyes\n"
_CHILD_SCRIPT = ("#!/usr/bin/env python3\nimport sys\n"
                 f"sys.stdout.write({_CHILD_OUT!r})\n"
                 f"sys.stderr.write({_CHILD_ERR!r})\n"
                 "sys.exit(1)\n")
_REPLY_SCRIPT = "#!/usr/bin/env python3\nprint('fake reply')\n"


def _capture(monkeypatch, denylist: str):
    """A real captured security context under a given env value — the call the adapter makes at startup."""
    monkeypatch.setenv("A2A_IDENTITY_DENYLIST", denylist)
    return security.A2ASecurityContext.capture()


@pytest.mark.platforms("linux")
class TestIdentityListIsPerProfile:
    """The list is PER-PROFILE state. A process-wide cache let a second profile's adapter swap the list
    under the first one's feet, so a peer could receive the other profile's identity unscrubbed."""

    def test_a_and_b_keep_their_own_literals_in_both_orders(self, monkeypatch):
        a = _capture(monkeypatch, "alphaperson")
        b = _capture(monkeypatch, "betaperson")
        a_again = _capture(monkeypatch, "alphaperson")
        assert a.identity_denylist == ("alphaperson",)
        assert b.identity_denylist == ("betaperson",)
        assert a_again.identity_denylist == ("alphaperson",)

        text = "alphaperson and betaperson"
        # B's capture must not change what A's already-captured context scrubs (the A -> B -> A case).
        assert security.redact_outbound(text, denylist=a.identity_denylist) == "[redacted-identity] and betaperson"
        assert security.redact_outbound(text, denylist=b.identity_denylist) == "alphaperson and [redacted-identity]"
        assert security.redact_outbound(text, denylist=a_again.identity_denylist) == "[redacted-identity] and betaperson"

    def test_client_path_resolves_the_callers_scope_not_a_captured_one(self, monkeypatch):
        """``tools._send_task`` calls ``redact_outbound`` with no list; that live resolve must use the
        caller's scope — never a list another profile's adapter captured earlier."""
        _capture(monkeypatch, "adapterperson")
        monkeypatch.setenv("A2A_IDENTITY_DENYLIST", "callerperson")
        assert security.redact_outbound("callerperson and adapterperson") == "[redacted-identity] and adapterperson"

    def test_each_adapters_egress_uses_its_own_captured_list(self, monkeypatch, tmp_path):
        """End to end on the shipped path: two adapters, two lists, neither bleeds into the other."""
        adapter_a, agent_a = _adapter_with_fake_child(monkeypatch, tmp_path, _MARKED_REPLY_SCRIPT,
                                                      identity="alphaperson")
        reply_a, state_a = adapter_a._forward_to_profile(agent_a, "peer", "ctx-a", "hi", "t-scope-a")
        adapter_b, agent_b = _adapter_with_fake_child(monkeypatch, tmp_path, _MARKED_REPLY_SCRIPT,
                                                      identity="betaperson")
        reply_b, state_b = adapter_b._forward_to_profile(agent_b, "peer", "ctx-b", "hi", "t-scope-b")
        assert (state_a, state_b) == (protocol.STATE_COMPLETED, protocol.STATE_COMPLETED)
        assert reply_a == "note [redacted-identity] betaperson"
        assert reply_b == "note alphaperson [redacted-identity]"
        assert adapter_a._security_context.identity_denylist == ("alphaperson",)
        assert adapter_b._security_context.identity_denylist == ("betaperson",)


def _adapter_with_fake_child(monkeypatch, tmp_path, script, identity: str = ""):
    """A real A2AAdapter whose ``hermes`` on PATH is a script — the same seam the existing
    forward-to-profile test drives, so the assertion covers the shipped call path. ``identity`` is the
    operator denylist env value this adapter captures at construction."""
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    monkeypatch.setenv("A2A_IDENTITY_DENYLIST", identity)

    profile_home = tmp_path / "profile"
    profile_home.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(profile_home / "state.db")
    con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL, title TEXT)")
    con.commit()
    con.close()
    fakebin = tmp_path / "bin"
    fakebin.mkdir(parents=True, exist_ok=True)
    child = fakebin / "hermes"
    child.write_text(script)
    child.chmod(0o755)
    monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr("plugins.platforms.a2a.adapter._profile_home", lambda profile: str(profile_home))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
        "agents": {"dev": {"profile": "dev", "tenant": "dev", "timeout": 5}}}))
    return adapter, adapter._agents["dev"]


@pytest.mark.platforms("linux")
class TestFailedProfileDispatchStatus:
    def test_failure_ships_a_status_object_not_the_child_output(self, monkeypatch, tmp_path):
        adapter, agent = _adapter_with_fake_child(monkeypatch, tmp_path, _CHILD_SCRIPT)
        reply, state = adapter._forward_to_profile(agent, "peer", "ctx-1", "hello", "t-dispatch-1")

        assert state == protocol.STATE_FAILED
        for leak in ("CALEB", "10001-1234", "555-123-4567", "traceback", "[redacted"):
            assert leak not in reply
        assert json.loads(reply) == {
            "type": "hermes.profile_dispatch_status", "state": protocol.STATE_FAILED,
            "error": "profile_exit_nonzero", "task_id": "t-dispatch-1", "profile": "dev",
            "exit_code": 1, "stdout_bytes": len(_CHILD_OUT), "stderr_bytes": len(_CHILD_ERR),
        }

    def test_reply_path_is_unchanged(self, monkeypatch, tmp_path):
        """The completed path is the agent's actual reply, not a status string: unchanged."""
        adapter, agent = _adapter_with_fake_child(monkeypatch, tmp_path, _REPLY_SCRIPT)
        reply, state = adapter._forward_to_profile(agent, "peer", "ctx-2", "hello", "t-reply-1")
        assert (reply, state) == ("fake reply", protocol.STATE_COMPLETED)

    def test_a_raising_dispatch_reports_the_status_object_not_the_exception(self, monkeypatch, tmp_path):
        """The exception branch (subprocess raised before any exit code) ships the same fixed shape, so
        the peer learns nothing about the failure beyond its class."""
        from plugins.platforms.a2a import adapter as adapter_mod
        adapter, agent = _adapter_with_fake_child(monkeypatch, tmp_path, _CHILD_SCRIPT)

        def _boom(*_a, **_k):
            raise OSError("spawn failed for caleb@example.com at ZIP 10001")

        monkeypatch.setattr(adapter_mod.subprocess, "run", _boom)
        reply, state = adapter._forward_to_profile(agent, "peer", "ctx-3", "hello", "t-dispatch-2")
        assert state == protocol.STATE_FAILED
        for leak in ("caleb", "10001", "OSError", "spawn failed", "[redacted"):
            assert leak not in reply
        assert json.loads(reply) == {
            "type": "hermes.profile_dispatch_status", "state": protocol.STATE_FAILED,
            "error": "profile_dispatch_error", "task_id": "t-dispatch-2", "profile": "dev",
            "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0,
        }

    def test_local_dispatch_failure_ships_a_status_object_not_the_exception(self, monkeypatch, tmp_path):
        """The in-process dispatch branch: a raising ``run_coroutine_threadsafe`` must not put its own
        exception text in the task's status."""
        from plugins.platforms.a2a import adapter as adapter_mod
        adapter, agent = _adapter_with_fake_child(monkeypatch, tmp_path, _REPLY_SCRIPT, identity="alphaperson")
        agent["local"] = True
        adapter._loop, adapter._message_handler = object(), object()

        def _boom(coro, *_a, **_k):
            coro.close()  # the coroutine is created before this call; closing it avoids a warning
            raise RuntimeError("loop closed for alphaperson")

        monkeypatch.setattr(adapter_mod.asyncio, "run_coroutine_threadsafe", _boom)
        task, pending = adapter._prepare_task(
            {"message": {"role": "user", "parts": [{"kind": "text", "text": "hello"}]}}, "peer", agent)
        assert pending is None
        assert task["status"]["state"] == protocol.STATE_FAILED
        text = protocol.extract_text(task["status"].get("message"))
        assert "alphaperson" not in text and "loop closed" not in text
        assert json.loads(text) == {
            "type": "hermes.dispatch_status", "state": protocol.STATE_FAILED,
            "error": "local_dispatch_error", "task_id": task["id"], "profile": adapter._active_profile,
            "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0,
        }
