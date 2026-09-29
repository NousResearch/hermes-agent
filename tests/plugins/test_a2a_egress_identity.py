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
def _fresh_identity_denylist(monkeypatch):
    """``redact_outbound`` prefers the list the adapter captured at startup; keep tests independent of
    capture order and of whatever this host's env/config happens to hold."""
    monkeypatch.setattr(security, "_IDENTITY_DENYLIST", None)
    monkeypatch.delenv("A2A_IDENTITY_DENYLIST", raising=False)


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


_CHILD_OUT = "source note: CALEB's address is 10001-1234, call 555-123-4567\n"
_CHILD_ERR = "traceback: for peer eyes\n"
_CHILD_SCRIPT = ("#!/usr/bin/env python3\nimport sys\n"
                 f"sys.stdout.write({_CHILD_OUT!r})\n"
                 f"sys.stderr.write({_CHILD_ERR!r})\n"
                 "sys.exit(1)\n")
_REPLY_SCRIPT = "#!/usr/bin/env python3\nprint('fake reply')\n"


def _adapter_with_fake_child(monkeypatch, tmp_path, script):
    """A real A2AAdapter whose ``hermes`` on PATH is a script — the same seam the existing
    forward-to-profile test drives, so the assertion covers the shipped call path."""
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    con = sqlite3.connect(profile_home / "state.db")
    con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL, title TEXT)")
    con.commit()
    con.close()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
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
