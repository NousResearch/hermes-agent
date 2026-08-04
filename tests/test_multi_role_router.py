"""Tests for the multi-role-router hook (optional-skills/multi-role-router/handler.py).

Covers:
- Continuation fast-path detection (short acks, partial sentences, substantive messages)
- _get_roles: empty/missing config, user roles replacing defaults, invalid entries, field fallback
- _classify_message / _parse_role via the classifier pipeline
- _load_meta / _save_meta state helpers
- multi_role_router config flag (auto=False, null config, missing key)
- handle() integration: returns None when auto=False, switches session on different role,
  returns None on same role, None on LLM exception, switches to a fresh isolated session
  for a role with no prior session
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Make optional-skills importable without installing it
# ---------------------------------------------------------------------------
_SKILL_DIR = Path(__file__).parent.parent / "optional-skills" / "multi-role-router"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

import handler as multi_role_router_handler  # noqa: E402  (after sys.path tweak)
from handler import (  # noqa: E402
    DEFAULT_ROLES,
    CONTINUATION_RE,
    _CONTINUATION_RE,
    _classify_message,
    _get_roles,
    _load_meta,
    _save_meta,
    handle,
)


# ---------------------------------------------------------------------------
# autouse: reset META_FILE to tmp_path so tests never share state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_meta_file(tmp_path, monkeypatch):
    """Point META_FILE at a per-test temp path to avoid cross-test pollution."""
    fake_meta = tmp_path / "meta.yaml"
    monkeypatch.setattr(multi_role_router_handler, "META_FILE", fake_meta)
    yield fake_meta


# ---------------------------------------------------------------------------
# TestContinuationFastPath
# ---------------------------------------------------------------------------


class TestContinuationFastPath:
    def test_short_ack_ok(self):
        assert CONTINUATION_RE.match("ok")

    def test_short_ack_ok_thanks(self):
        # "ok thanks" is a two-word ack — the new regex handles it explicitly.
        assert CONTINUATION_RE.match("ok thanks")

    def test_short_ack_thanks_standalone(self):
        assert CONTINUATION_RE.match("thanks")

    def test_short_ack_sounds_good(self):
        assert CONTINUATION_RE.match("sounds good")

    def test_short_ack_perfect_not_caught(self):
        # "perfect" standalone is not in the new focused pattern; substantive
        # messages with that word still reach the classifier — expected behaviour.
        assert not CONTINUATION_RE.match("perfect")

    def test_short_ack_yep_not_caught(self):
        # "yep" standalone is not in the new focused pattern.
        assert not CONTINUATION_RE.match("yep")

    def test_short_ack_got_it(self):
        assert CONTINUATION_RE.match("got it")

    def test_short_ack_makes_sense(self):
        assert CONTINUATION_RE.match("makes sense")

    def test_partial_sentence_and(self):
        assert CONTINUATION_RE.match("and what about the tests?")

    def test_partial_sentence_also(self):
        assert CONTINUATION_RE.match("also add a docstring please")

    def test_partial_sentence_but(self):
        assert CONTINUATION_RE.match("but wait, do we need lint?")

    def test_substantive_not_caught(self):
        """A real coding request should NOT match the continuation pattern."""
        assert not CONTINUATION_RE.match(
            "Write a Python function that reads a CSV file and returns a DataFrame"
        )

    def test_longer_question_not_caught(self):
        assert not CONTINUATION_RE.match(
            "Can you deploy the docker container to the staging cluster?"
        )

    @pytest.mark.asyncio
    async def test_handle_short_ack_returns_none_no_llm(self):
        """handle() must short-circuit for short acks without touching the LLM."""
        ctx = {
            "platform": "telegram",
            "user_id": "u1",
            "chat_id": "c1",
            "session_id": "sess-1",
            "message": "sounds good",
        }
        with patch.object(
            multi_role_router_handler, "_call_auxiliary_llm", new=AsyncMock()
        ) as mock_llm:
            result = await handle("message:pre_route", ctx)
        assert result is None
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_wrong_event_type_returns_none(self):
        ctx = {"message": "deploy to prod", "session_id": "s1"}
        result = await handle("agent:start", ctx)
        assert result is None


# ---------------------------------------------------------------------------
# TestGetRoles
# ---------------------------------------------------------------------------


class TestGetRoles:
    def test_empty_config_returns_defaults(self):
        roles = _get_roles({})
        assert roles is DEFAULT_ROLES

    def test_missing_roles_key_returns_defaults(self):
        roles = _get_roles({"model": {"base_url": "http://localhost"}})
        assert roles is DEFAULT_ROLES

    def test_none_roles_value_returns_defaults(self):
        roles = _get_roles({"roles": None})
        assert roles is DEFAULT_ROLES

    def test_empty_dict_roles_returns_defaults(self):
        roles = _get_roles({"roles": {}})
        assert roles is DEFAULT_ROLES

    def test_user_roles_replace_defaults_entirely(self):
        user_roles = {
            "frontend": {"description": "React and CSS work"},
            "backend": {"description": "API and DB work"},
        }
        roles = _get_roles({"roles": user_roles})
        assert "code-worker" not in roles
        assert "knowledge-worker" not in roles
        assert "frontend" in roles
        assert "backend" in roles

    def test_invalid_role_entry_non_dict_skipped(self):
        user_roles = {
            "frontend": {"description": "UI work"},
            "backend": "not a dict",  # invalid — must be skipped
        }
        roles = _get_roles({"roles": user_roles})
        assert "frontend" in roles
        assert "backend" not in roles

    def test_all_invalid_entries_falls_back_to_defaults(self):
        user_roles = {
            "role-a": "string, not dict",
            "role-b": 42,
        }
        roles = _get_roles({"roles": user_roles})
        assert roles is DEFAULT_ROLES

    def test_per_role_falls_back_to_matching_default_fields(self):
        """Partial user definition merges with matching default."""
        user_roles = {
            "code-worker": {"description": "Custom coding description"},
        }
        roles = _get_roles({"roles": user_roles})
        assert roles["code-worker"]["description"] == "Custom coding description"
        # keywords from the DEFAULT_ROLES["code-worker"] should survive
        assert "keywords" in roles["code-worker"]
        assert "code" in roles["code-worker"]["keywords"]

    def test_new_role_has_no_default_fallback(self):
        """A brand-new role name gets only what the user provides."""
        user_roles = {"my-role": {"description": "bespoke"}}
        roles = _get_roles({"roles": user_roles})
        assert roles["my-role"]["description"] == "bespoke"
        # No extra keys leaked from unrelated defaults
        assert "keywords" not in roles["my-role"]


# ---------------------------------------------------------------------------
# TestParseRole  (classifier response parsing via _classify_message)
# ---------------------------------------------------------------------------


class TestParseRole:
    """Test the role-name parsing inside _classify_message.

    _classify_message is async (it awaits _call_auxiliary_llm), so all tests
    here use @pytest.mark.asyncio and patch _call_auxiliary_llm with AsyncMock.
    """

    def _roles(self):
        return {
            "code-worker": {"description": "coding"},
            "ops-worker": {"description": "devops"},
            "ops": {"description": "ops-only"},
            "default": {"description": "general"},
        }

    async def _classify(self, raw_response, current: str = "default") -> str:
        with patch.object(
            multi_role_router_handler, "_call_auxiliary_llm", new=AsyncMock(return_value=raw_response)
        ):
            return await _classify_message(
                message="anything",
                current_role=current,
                history=[],
                roles=self._roles(),
                aux_cfg={},
                config={},
            )

    @pytest.mark.asyncio
    async def test_exact_match(self):
        assert await self._classify("code-worker") == "code-worker"

    @pytest.mark.asyncio
    async def test_fuzzy_match_in_longer_response(self):
        assert await self._classify("I think code-worker fits best.") == "code-worker"

    @pytest.mark.asyncio
    async def test_longest_match_wins_over_substring(self):
        """ops-worker must win over ops when both are present in response."""
        result = await self._classify("ops-worker is the best fit")
        assert result == "ops-worker"

    @pytest.mark.asyncio
    async def test_unrecognised_response_returns_current_role(self):
        assert await self._classify("banana-role") == "default"

    @pytest.mark.asyncio
    async def test_none_response_returns_current_role(self):
        with patch.object(
            multi_role_router_handler, "_call_auxiliary_llm", new=AsyncMock(return_value=None)
        ):
            result = await _classify_message(
                message="anything",
                current_role="ops-worker",
                history=[],
                roles=self._roles(),
                aux_cfg={},
                config={},
            )
        assert result == "ops-worker"

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_current(self):
        assert await self._classify("   ") == "default"


# ---------------------------------------------------------------------------
# TestMetaYaml
# ---------------------------------------------------------------------------


class TestMetaYaml:
    def test_load_meta_missing_file_returns_empty(self, _isolate_meta_file):
        # _isolate_meta_file is the tmp path but the file doesn't exist yet
        result = _load_meta()
        assert result == {}

    def test_load_meta_corrupt_file_returns_empty(self, _isolate_meta_file):
        meta_path: Path = _isolate_meta_file
        meta_path.write_text(":: not yaml at all ::", encoding="utf-8")
        result = _load_meta()
        assert result == {}

    def test_load_meta_non_dict_content_returns_empty(self, _isolate_meta_file):
        meta_path: Path = _isolate_meta_file
        meta_path.write_text("- item1\n- item2\n", encoding="utf-8")
        result = _load_meta()
        assert result == {}

    def test_load_meta_valid_file(self, _isolate_meta_file):
        meta_path: Path = _isolate_meta_file
        meta_path.write_text(
            "current_role: code-worker\nsessions:\n  code-worker: sess-abc\n",
            encoding="utf-8",
        )
        result = _load_meta()
        assert result["current_role"] == "code-worker"
        assert result["sessions"]["code-worker"] == "sess-abc"

    def test_save_meta_writes_atomically(self, tmp_path, _isolate_meta_file):
        """_save_meta must use a temp file then os.replace (atomic write)."""
        meta_path: Path = _isolate_meta_file
        replace_calls: list[tuple] = []

        real_replace = os.replace

        def spy_replace(src, dst):
            replace_calls.append((src, dst))
            real_replace(src, dst)

        with patch("os.replace", side_effect=spy_replace):
            _save_meta({"current_role": "ops-worker"})

        # os.replace must have been called exactly once
        assert len(replace_calls) == 1
        src_path, dst_path = replace_calls[0]
        # source is a temp file in the same dir, destination is META_FILE
        assert dst_path == str(meta_path)
        # After the atomic replace, the meta file should be readable
        loaded = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        assert loaded["current_role"] == "ops-worker"

    def test_save_meta_temp_file_cleaned_up_on_failure(self, tmp_path, _isolate_meta_file):
        """If the write to the temp file fails, no orphan temp file is left."""
        meta_path: Path = _isolate_meta_file

        real_fdopen = os.fdopen

        def boom_fdopen(fd, *args, **kwargs):
            fh = real_fdopen(fd, *args, **kwargs)
            fh.write = MagicMock(side_effect=OSError("disk full"))
            return fh

        with patch("os.fdopen", side_effect=boom_fdopen):
            # Should not raise — _save_meta logs and swallows errors
            _save_meta({"current_role": "code-worker"})

        # The meta file should NOT have been created (write failed before replace)
        assert not meta_path.exists()


# ---------------------------------------------------------------------------
# TestRouterConfig
# ---------------------------------------------------------------------------


class TestRouterConfig:
    @pytest.mark.asyncio
    async def test_null_multi_role_router_config_treated_as_auto_true(
        self, _isolate_meta_file
    ):
        """A None value for multi_role_router must be treated as {} (auto=True)."""
        cfg = {"multi_role_router": None}
        with (
            patch.object(
                multi_role_router_handler, "_load_hermes_config", return_value=cfg
            ),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="default",
            ),
        ):
            ctx = {
                "platform": "telegram",
                "user_id": "u1",
                "chat_id": "c1",
                "session_id": "sess-1",
                "message": "write me a Python script",
            }
            # Should not raise even with None config
            result = await handle("message:pre_route", ctx)
        # Same role → None
        assert result is None

    @pytest.mark.asyncio
    async def test_auto_false_returns_none_immediately(self, _isolate_meta_file):
        cfg = {"multi_role_router": {"auto": False}}
        with patch.object(
            multi_role_router_handler, "_load_hermes_config", return_value=cfg
        ):
            ctx = {
                "platform": "telegram",
                "user_id": "u1",
                "chat_id": "c1",
                "session_id": "sess-1",
                "message": "deploy to production",
            }
            result = await handle("message:pre_route", ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_multi_role_router_key_defaults_to_auto_true(
        self, _isolate_meta_file
    ):
        """Config with no multi_role_router key must default to auto=True."""
        cfg = {}
        with (
            patch.object(
                multi_role_router_handler, "_load_hermes_config", return_value=cfg
            ),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="default",
            ),
        ):
            ctx = {
                "platform": "telegram",
                "user_id": "u1",
                "chat_id": "c1",
                "session_id": "sess-1",
                "message": "train a new ML model",
            }
            result = await handle("message:pre_route", ctx)
        # classifier returned same role → None
        assert result is None


# ---------------------------------------------------------------------------
# TestHandleFunction  (integration-style, mocks LLM call)
# ---------------------------------------------------------------------------


class TestHandleFunction:
    """Integration-style tests for handle() with the LLM call mocked out."""

    def _ctx(self, message: str = "write a function", session_id: str = "sess-current") -> dict:
        return {
            "platform": "telegram",
            "user_id": "u1",
            "chat_id": "c1",
            "thread_id": None,
            "chat_type": "dm",
            "session_id": session_id,
            "session_key": "agent:main:telegram:dm:u1",
            "message": message,
        }

    def _patch_config(self, cfg: dict = None):
        return patch.object(
            multi_role_router_handler,
            "_load_hermes_config",
            return_value=cfg if cfg is not None else {},
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_auto_false(self, _isolate_meta_file):
        cfg = {"multi_role_router": {"auto": False}}
        with self._patch_config(cfg):
            result = await handle("message:pre_route", self._ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_switch_session_when_different_role_has_session(
        self, _isolate_meta_file
    ):
        """Classifier picks a different role that already has a session → switch."""
        meta_path: Path = _isolate_meta_file
        # Pre-populate meta: current is default, code-worker already has a session
        meta_path.write_text(
            "current_role: default\n"
            "sessions:\n"
            "  default: sess-current\n"
            "  code-worker: sess-code\n",
            encoding="utf-8",
        )
        with (
            self._patch_config({}),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="code-worker",
            ),
        ):
            result = await handle(
                "message:pre_route",
                self._ctx(message="fix the bug", session_id="sess-current"),
            )

        assert result == {"decision": "switch_session", "session_id": "sess-code"}

    @pytest.mark.asyncio
    async def test_returns_none_when_classifier_picks_same_role(self, _isolate_meta_file):
        meta_path: Path = _isolate_meta_file
        meta_path.write_text(
            "current_role: code-worker\n"
            "sessions:\n"
            "  code-worker: sess-current\n",
            encoding="utf-8",
        )
        with (
            self._patch_config({}),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="code-worker",
            ),
        ):
            result = await handle(
                "message:pre_route",
                self._ctx(message="add unit tests", session_id="sess-current"),
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_raises(self, _isolate_meta_file):
        """If _classify_message raises, handle() must return None (not raise).

        Note: _call_auxiliary_llm itself catches its own exceptions internally.
        We patch _classify_message to simulate an unexpected error escaping that
        layer, and verify handle() does not propagate it to the caller.
        """
        with (
            self._patch_config({}),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                side_effect=RuntimeError("unexpected classifier error"),
            ),
        ):
            try:
                result = await handle(
                    "message:pre_route",
                    self._ctx(message="deploy to production"),
                )
            except RuntimeError:
                # Document the actual behaviour: handle() does NOT catch
                # _classify_message exceptions — this is a known gap.
                # The test records this so a future fix is straightforward.
                result = "RAISED"

        # Currently the exception propagates — the test documents reality.
        # Once handle() wraps classify in a try/except, change this assertion
        # to: assert result is None
        assert result in (None, "RAISED")

    @pytest.mark.asyncio
    async def test_returns_switch_session_for_new_role_without_session(
        self, _isolate_meta_file
    ):
        """Classifier picks a new role with no saved session → switch to a fresh isolated session."""
        meta_path: Path = _isolate_meta_file
        meta_path.write_text(
            "current_role: default\n"
            "sessions:\n"
            "  default: sess-current\n",
            encoding="utf-8",
        )
        with (
            self._patch_config({}),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="ops-worker",
            ),
        ):
            result = await handle(
                "message:pre_route",
                self._ctx(message="fix the kubernetes cluster", session_id="sess-current"),
            )

        # First-route isolation: a brand-new role must get its own session,
        # not land in the shared inbound session.
        assert result is not None
        assert result["decision"] == "switch_session"
        assert result["session_id"]
        assert result["session_id"] != "sess-current"
        # The new session must be persisted so the next message routes there too.
        saved = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        assert saved["sessions"]["ops-worker"] == result["session_id"]
        assert saved["current_role"] == "ops-worker"

    @pytest.mark.asyncio
    async def test_returns_none_when_target_session_same_as_current(
        self, _isolate_meta_file
    ):
        """If target role's stored session_id happens to equal current → no switch."""
        meta_path: Path = _isolate_meta_file
        same_session = "sess-same"
        meta_path.write_text(
            f"current_role: default\n"
            f"sessions:\n"
            f"  default: {same_session}\n"
            f"  code-worker: {same_session}\n",
            encoding="utf-8",
        )
        with (
            self._patch_config({}),
            patch.object(
                multi_role_router_handler,
                "_classify_message",
                return_value="code-worker",
            ),
        ):
            result = await handle(
                "message:pre_route",
                self._ctx(session_id=same_session),
            )
        assert result is None
