"""Tests for per-platform stateless inbound (``platforms.<p>.stateless_inbound``).

Background: Photon inbound from one chat key shares a single persistent
session (``agent:main:photon:dm:<chat_id>``). Every new message dragged the
full conversation history back in, which let a 50-turn history pattern-match
over a strict SOUL.md router contract. ``stateless_inbound: true`` makes the
gateway force a FRESH session per inbound so no prior turns are loaded.

Three layers are pinned here:

* ``TestStatelessInboundConfigBridge`` — ``platforms.photon.stateless_inbound``
  (and the top-level ``photon:`` form) is bridged into ``PlatformConfig.extra``
  by ``load_gateway_config``; absent => off.
* ``TestInboundIsStatelessGate`` — ``GatewayRunner._inbound_is_stateless``
  reads that config and defaults False for unconfigured platforms.
* ``TestStatelessInboundEndToEnd`` — the real ``SessionStore`` behavioral
  contract: two messages from the SAME chat land in two DIFFERENT sessions and
  the second loads an EMPTY transcript (no visibility of the first). A control
  asserts the legacy (persistent) path still carries history — i.e. the bug the
  flag fixes.
"""

from __future__ import annotations

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_store(tmp_path, config=None):
    store = SessionStore(sessions_dir=tmp_path, config=config or GatewayConfig())
    # Isolate the SQLite transcript store so we exercise per-session_id
    # transcripts without touching the developer's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _photon_source():
    # Mirrors the real key seen in the gateway log: chat=any;-;+177****2727.
    return SessionSource(
        platform=Platform.PHOTON,
        chat_id="any;-;+17725212727",
        chat_type="dm",
        user_id="+17725212727",
    )


def _pbi_turn(n):
    return [
        {"role": "user", "content": f"build the PBI deck #{n}"},
        {"role": "assistant", "content": f"TEXT\nPBI deck #{n} ready"},
    ]


# ---------------------------------------------------------------------------
# Config bridge
# ---------------------------------------------------------------------------
class TestStatelessInboundConfigBridge:
    def test_bridged_from_nested_platforms_block(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platforms:\n"
            "  photon:\n"
            "    stateless_inbound: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert Platform.PHOTON in config.platforms
        assert config.platforms[Platform.PHOTON].extra.get("stateless_inbound") is True

    def test_bridged_from_top_level_block(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "photon:\n"
            "  stateless_inbound: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.PHOTON].extra.get("stateless_inbound") is True

    def test_absent_defaults_off(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platforms:\n"
            "  photon:\n"
            "    require_mention: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        photon = config.platforms.get(Platform.PHOTON)
        # Either no photon block materialized, or it has no stateless flag.
        if photon is not None:
            assert not photon.extra.get("stateless_inbound")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
class TestInboundIsStatelessGate:
    def _runner_with(self, platforms):
        # Bypass GatewayRunner.__init__ (heavy adapter/loop wiring); the gate
        # only reads self.config.platforms.
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms=platforms)
        return runner

    def test_true_when_configured(self):
        runner = self._runner_with(
            {Platform.PHOTON: PlatformConfig(extra={"stateless_inbound": True})}
        )
        assert runner._inbound_is_stateless(_photon_source()) is True

    def test_true_for_stringy_truthy_value(self):
        runner = self._runner_with(
            {Platform.PHOTON: PlatformConfig(extra={"stateless_inbound": "yes"})}
        )
        assert runner._inbound_is_stateless(_photon_source()) is True

    def test_false_by_default(self):
        runner = self._runner_with(
            {Platform.PHOTON: PlatformConfig(extra={})}
        )
        assert runner._inbound_is_stateless(_photon_source()) is False

    def test_false_for_unconfigured_platform(self):
        runner = self._runner_with({})
        assert runner._inbound_is_stateless(_photon_source()) is False

    def test_other_platform_unaffected(self):
        # Photon stateless must not leak into telegram.
        runner = self._runner_with(
            {Platform.PHOTON: PlatformConfig(extra={"stateless_inbound": True})}
        )
        tele = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        assert runner._inbound_is_stateless(tele) is False


# ---------------------------------------------------------------------------
# End-to-end: two messages, same chat, second has no history
# ---------------------------------------------------------------------------
class TestStatelessInboundEndToEnd:
    def test_second_inbound_has_no_visibility_of_first(self, tmp_path):
        """Stateless path: the same chat sends two PBI messages; the second
        lands in a fresh session whose transcript is empty."""
        store = _make_store(tmp_path)
        source = _photon_source()

        # --- Message 1 (forced fresh) -> session A, gets a turn of history.
        e1 = store.get_or_create_session(source, force_new=True)
        store._db.create_session(
            session_id=e1.session_id, source="photon", user_id=source.user_id
        )
        store._db.replace_messages(e1.session_id, _pbi_turn(1))
        assert len(store.load_transcript(e1.session_id)) == 2  # precondition

        # --- Message 2 (forced fresh) -> session B, brand new + empty.
        e2 = store.get_or_create_session(source, force_new=True)
        assert e2.session_id != e1.session_id, (
            "stateless inbound must allocate a new session id per message"
        )
        assert store.load_transcript(e2.session_id) == [], (
            "the second message must have ZERO visibility of the first — "
            "no prior turns may be loaded into the fresh session"
        )

        # The first session's history still exists (searchable), just not loaded.
        assert len(store.load_transcript(e1.session_id)) == 2

    def test_legacy_persistent_path_still_carries_history(self, tmp_path):
        """Control / regression guard: WITHOUT the flag (force_new=False), the
        same chat reuses one session and the second message sees turn 1. This is
        the behavior every other platform keeps — and the exact contamination
        stateless inbound removes."""
        store = _make_store(tmp_path)
        source = _photon_source()

        e1 = store.get_or_create_session(source)
        store._db.create_session(
            session_id=e1.session_id, source="photon", user_id=source.user_id
        )
        store._db.replace_messages(e1.session_id, _pbi_turn(1))

        e2 = store.get_or_create_session(source)
        assert e2.session_id == e1.session_id, (
            "default (non-stateless) inbound reuses the persistent session"
        )
        assert len(store.load_transcript(e2.session_id)) == 2, (
            "default path drags the prior turn forward — the dragged history"
        )


# ---------------------------------------------------------------------------
# Source-level guard: the gate method exists and is referenced at the call site
# ---------------------------------------------------------------------------
def test_handler_uses_stateless_gate():
    """Pin that the inbound handler actually consults the gate, so the wiring
    can't silently regress to an unconditional get_or_create_session."""
    import inspect

    src = inspect.getsource(gateway_run.GatewayRunner._handle_message_with_agent)
    assert "_inbound_is_stateless" in src
    assert "force_new" in src
