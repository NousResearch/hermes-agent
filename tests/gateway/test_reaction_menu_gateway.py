"""Unit tests for the consumer-agnostic reaction-menu gateway primitive
and the present_menu tool dispatcher.  No platform adapter required."""

import json

import pytest

from tools import reaction_menu_gateway as rmg
from tools.reaction_menu_gateway import MenuValidationError, validate_options


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path):
    """Each test gets a clean registry + a throwaway persistence DB."""
    rmg.reset_state()
    rmg.set_db_path(tmp_path / "menus.db")
    yield
    rmg.reset_state()
    rmg.set_db_path(None)


def _opts(*emojis):
    return [
        {"emoji": e, "label": f"label-{e}", "payload": f"payload-{e}"}
        for e in emojis
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_normalizes_and_defaults_terminal_false(self):
        out = validate_options([{"emoji": "📖", "label": "Read", "payload": "go"}])
        assert out == [{"emoji": "📖", "label": "Read", "payload": "go", "terminal": False}]

    def test_terminal_flag_preserved(self):
        out = validate_options([{"emoji": "🛑", "label": "Stop", "payload": "stop", "terminal": True}])
        assert out[0]["terminal"] is True

    def test_silent_is_rejected(self):
        with pytest.raises(MenuValidationError, match="silent"):
            validate_options([{"emoji": "📖", "label": "x", "payload": "y", "silent": True}])

    def test_too_many_options_rejected(self):
        with pytest.raises(MenuValidationError, match="1.5 options"):
            validate_options(_opts("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"))

    def test_empty_options_rejected(self):
        with pytest.raises(MenuValidationError):
            validate_options([])

    def test_duplicate_emoji_rejected(self):
        with pytest.raises(MenuValidationError, match="duplicate"):
            validate_options(_opts("📖", "📖"))

    def test_reserved_reload_emoji_rejected(self):
        with pytest.raises(MenuValidationError, match="reload"):
            validate_options([{"emoji": rmg.RELOAD_EMOJI, "label": "x", "payload": "y"}])

    def test_missing_fields_rejected(self):
        with pytest.raises(MenuValidationError, match="emoji"):
            validate_options([{"label": "x", "payload": "y"}])
        with pytest.raises(MenuValidationError, match="payload"):
            validate_options([{"emoji": "📖", "label": "x"}])


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

class TestMarker:
    def test_build_and_parse_roundtrip(self):
        body = rmg.build_menu_choice_body("read passage 4")
        assert body.startswith(rmg.MENU_CHOICE_MARKER)
        assert rmg.is_menu_choice_text(body)
        assert rmg.parse_menu_choice_payload(body) == "read passage 4"

    def test_plain_text_not_a_choice(self):
        assert not rmg.is_menu_choice_text("hello there")

    def test_recognizes_event_like_object(self):
        evt = type("E", (), {"text": rmg.build_menu_choice_body("p")})()
        assert rmg.is_menu_choice_text(evt)


# ---------------------------------------------------------------------------
# Registry + dedup
# ---------------------------------------------------------------------------

class TestRegistry:
    def _register(self, menu_id="m1", session="s1", message="$evt1", emojis=("📖", "🔁")):
        return rmg.register(
            menu_id=menu_id, session_key=session, platform="matrix",
            chat_id="!room:x", message_id=message, prompt="pick",
            options=validate_options(_opts(*emojis)), context_id="ctx",
        )

    def test_register_and_lookup_by_message(self):
        self._register()
        entry = rmg.get_by_message("$evt1")
        assert entry is not None and entry.menu_id == "m1"
        assert entry.option_for_emoji("📖")["payload"] == "payload-📖"

    def test_mark_resolved_is_idempotent_dedup(self):
        self._register()
        assert rmg.mark_resolved("m1") is True   # first tap wins
        assert rmg.mark_resolved("m1") is False  # double-tap deduped
        assert rmg.get_by_message("$evt1") is None

    def test_many_live_menus_independent(self):
        self._register(menu_id="m1", message="$e1", emojis=("📖", "🔁"))
        self._register(menu_id="m2", message="$e2", emojis=("🍎", "🍌"))
        assert rmg.mark_resolved("m1") is True
        # m2 untouched by m1's resolution — proves no single-live-surface core.
        assert rmg.get_by_message("$e2").menu_id == "m2"
        assert rmg.mark_resolved("m2") is True

    def test_newest_pointer_tracks_latest_and_promotes(self):
        self._register(menu_id="m1", message="$e1")
        self._register(menu_id="m2", message="$e2")
        assert rmg.get_newest_for_session("s1").menu_id == "m2"
        rmg.mark_resolved("m2")
        # Resolving newest promotes the prior live menu.
        assert rmg.get_newest_for_session("s1").menu_id == "m1"

    def test_clear_session(self):
        self._register(menu_id="m1", message="$e1")
        self._register(menu_id="m2", message="$e2")
        assert rmg.clear_session("s1") == 2
        assert rmg.list_for_session("s1") == []
        assert rmg.get_newest_for_session("s1") is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_unresolved_menu_survives_reload(self):
        rmg.register(
            menu_id="m1", session_key="s1", platform="matrix", chat_id="!room:x",
            message_id="$evt1", prompt="pick",
            options=validate_options(_opts("📖", "🔁")),
            context_id="ctx",
            source={"platform": "matrix", "chat_id": "!room:x", "user_id": "@u:x"},
        )
        # Simulate a restart: wipe memory, reload from DB.
        rmg.reset_state()
        assert rmg.get("m1") is None
        restored = rmg.load_persisted(platform="matrix")
        assert len(restored) == 1
        entry = rmg.get_by_message("$evt1")
        assert entry is not None and entry.context_id == "ctx"
        assert entry.source["user_id"] == "@u:x"

    def test_resolved_menu_not_reloaded(self):
        rmg.register(
            menu_id="m1", session_key="s1", platform="matrix", chat_id="!room:x",
            message_id="$evt1", prompt="pick",
            options=validate_options(_opts("📖", "🔁")),
        )
        rmg.mark_resolved("m1")
        rmg.reset_state()
        assert rmg.load_persisted() == []

    def test_platform_filter_on_load(self):
        rmg.register(
            menu_id="m1", session_key="s1", platform="matrix", chat_id="c",
            message_id="$e1", prompt="p", options=validate_options(_opts("📖")),
        )
        rmg.reset_state()
        assert rmg.load_persisted(platform="telegram") == []
        # Still loadable for its own platform.
        rmg.reset_state()
        assert len(rmg.load_persisted(platform="matrix")) == 1


# ---------------------------------------------------------------------------
# present_menu tool dispatcher
# ---------------------------------------------------------------------------

class TestPresentMenuTool:
    def test_silent_rejected_at_tool_layer(self):
        from tools.reaction_menu_tool import present_menu_tool
        out = json.loads(present_menu_tool(
            prompt="pick", options=[{"emoji": "📖", "label": "x", "payload": "y", "silent": True}],
            callback=lambda *a: "ok",
        ))
        assert "error" in out and "silent" in out["error"]

    def test_missing_callback_errors(self):
        from tools.reaction_menu_tool import present_menu_tool
        out = json.loads(present_menu_tool(prompt="pick", options=_opts("📖"), callback=None))
        assert "error" in out and "not available" in out["error"]

    def test_blank_prompt_errors(self):
        from tools.reaction_menu_tool import present_menu_tool
        out = json.loads(present_menu_tool(prompt="  ", options=_opts("📖"), callback=lambda *a: "ok"))
        assert "error" in out

    def test_successful_present_delegates_to_callback(self):
        from tools.reaction_menu_tool import present_menu_tool
        seen = {}

        def cb(prompt, options, context_id):
            seen["prompt"] = prompt
            seen["options"] = options
            seen["context_id"] = context_id
            return "delivered"

        out = json.loads(present_menu_tool(
            prompt="Read next?",
            options=[
                {"emoji": "📖", "label": "Read", "payload": "read next"},
                {"emoji": "🛑", "label": "Stop", "payload": "stop", "terminal": True},
            ],
            context_id="story-1",
            callback=cb,
        ))
        assert out["status"] == "menu_presented"
        assert out["context_id"] == "story-1"
        assert seen["options"][1]["terminal"] is True
        assert {o["emoji"] for o in out["options_offered"]} == {"📖", "🛑"}

    def test_undelivered_menu_errors(self):
        from tools.reaction_menu_tool import present_menu_tool
        out = json.loads(present_menu_tool(
            prompt="pick", options=_opts("📖"), callback=lambda *a: "",
        ))
        assert "error" in out and "deliver" in out["error"]


# ---------------------------------------------------------------------------
# Base-adapter text fallback (numbered list; number resolves; one live menu)
# ---------------------------------------------------------------------------

class TestTextFallback:
    def _fake_self(self, sent):
        import types as _t

        async def _send(chat_id, content, metadata=None):
            sent.append(content)
            return _t.SimpleNamespace(success=True, message_id="$txt1")

        return _t.SimpleNamespace(name="signal", send=_send)

    def test_send_reaction_menu_renders_numbered_list_and_registers(self):
        import asyncio
        from gateway.platforms.base import BasePlatformAdapter

        sent = []
        opts = validate_options([
            {"emoji": "📖", "label": "Read", "payload": "read next"},
            {"emoji": "🛑", "label": "Stop", "payload": "stop"},
        ])
        result = asyncio.run(BasePlatformAdapter.send_reaction_menu(
            self._fake_self(sent), chat_id="c", menu_id="m1", prompt="Continue?",
            options=opts, session_key="s1",
        ))
        assert result.success
        body = sent[0]
        assert "1. 📖 Read" in body and "2. 🛑 Stop" in body
        # Registered as the live menu for the session.
        assert rmg.get_newest_for_session("s1").menu_id == "m1"

    def test_numbered_reply_resolves_to_payload(self):
        """The intercept logic: newest menu + number → payload, deduped."""
        import asyncio
        from gateway.platforms.base import BasePlatformAdapter

        sent = []
        opts = validate_options([
            {"emoji": "📖", "label": "Read", "payload": "read next"},
            {"emoji": "🛑", "label": "Stop", "payload": "stop"},
        ])
        asyncio.run(BasePlatformAdapter.send_reaction_menu(
            self._fake_self(sent), chat_id="c", menu_id="m1", prompt="Continue?",
            options=opts, session_key="s1",
        ))

        menu = rmg.get_newest_for_session("s1")
        choice = menu.options[1]              # user typed "2"
        assert rmg.mark_resolved(menu.menu_id) is True
        assert rmg.build_menu_choice_body(choice["payload"]) == \
            rmg.MENU_CHOICE_MARKER + "\nstop"
        # Deduped: a second "2" finds no live menu.
        assert rmg.get_newest_for_session("s1") is None
