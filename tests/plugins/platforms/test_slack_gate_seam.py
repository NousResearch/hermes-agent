"""Seam-identity tests for the SlackGateMixin extraction (adapter.py god-file slice R5-S1).

Verifies the mixin-first class line keeps every C1 gate/member bound through
the MRO with zero test edits, plus aggressive behavior cases for the mention
policy, the DM gate, and the channel allowlist.
"""

import ast
import sys

import pytest

import plugins.platforms.slack.adapter as adapter_mod
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.slack_gate_mixin import SlackGateMixin

GATE_VARS = [
    "SLACK_REQUIRE_MENTION",
    "SLACK_STRICT_MENTION",
    "SLACK_IGNORE_OTHER_USER_MENTIONS",
    "SLACK_THREAD_REQUIRE_MENTION",
    "SLACK_FREE_RESPONSE_CHANNELS",
    "SLACK_DISABLE_DMS",
    "SLACK_ALLOWED_CHANNELS",
    "SLACK_REQUIRE_MENTION_CHANNELS",
    "SLACK_MENTION_PATTERNS",
]


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    """Keep real host gate env vars from leaking into gate assertions."""
    for var in GATE_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in GATE_VARS:
        import os

        os.environ.pop(var, None)


C1_MEMBERS = [
    "_slack_require_mention",
    "_slack_strict_mention",
    "_slack_ignore_other_user_mentions",
    "_slack_thread_require_mention",
    "_slack_message_addressed_to_other_user",
    "_slack_message_mentions_self",
    "_slack_free_response_channels",
    "_slack_disable_dms",
    "_slack_allowed_channels",
    "_slack_require_mention_channels",
    "_slack_mention_patterns",
    "_slack_message_matches_mention_patterns",
]


def _adapter(extra: dict | None = None) -> SlackAdapter:
    adapter = object.__new__(SlackAdapter)
    adapter.config = type(
        "C", (), {"extra": dict(extra or {})}
    )()  # duck-typed config; gates only touch .extra
    return adapter


class TestSeamIdentity:
    """The class line must resolve every C1 member to the mixin (MRO)."""

    def test_adapter_subclasses_mixin(self):
        assert issubclass(SlackAdapter, SlackGateMixin)

    def test_all_12_members_identity_bound(self):
        for name in C1_MEMBERS:
            assert getattr(SlackAdapter, name) is getattr(SlackGateMixin, name), name

    def test_mixin_has_no_module_level_adapter_import(self):
        # Circular-import guard: the mixin must not import adapter at module
        # level (no lazy imports are needed here -- the moved members are
        # pure stdlib config reads).
        import pathlib

        path = pathlib.Path(adapter_mod.__file__).with_name("slack_gate_mixin.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "slack.adapter" in (node.module or ""):
                assert node.col_offset > 0, f"module-level adapter import at {node.lineno}"
        assert True


class TestMentionPolicy:
    """Aggressive: mention gating defaults and explicit parsing."""

    def test_require_mention_default_true(self):
        # Safe default: gating ON when nothing is configured.
        assert _adapter()._slack_require_mention() is True

    def test_require_mention_explicit_false_via_extra(self):
        adapter = _adapter({"require_mention": "false"})
        assert adapter._slack_require_mention() is False

    def test_require_mention_explicit_false_via_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_REQUIRE_MENTION", "0")
        assert _adapter()._slack_require_mention() is False

    def test_require_mention_unrecognized_value_stays_gated(self, monkeypatch):
        monkeypatch.setenv("SLACK_REQUIRE_MENTION", "banana")
        assert _adapter()._slack_require_mention() is True

    def test_message_matches_mention_patterns(self):
        adapter = _adapter({"mention_patterns": ["hey hermes"]})
        assert adapter._slack_message_matches_mention_patterns("hey Hermes, look") is True
        assert adapter._slack_message_matches_mention_patterns("nothing here") is False
        assert adapter._slack_message_matches_mention_patterns("") is False


class TestDMGate:
    """Aggressive: DM disabling via config and env."""

    def test_disable_dms_default_false(self):
        assert _adapter()._slack_disable_dms() is False

    def test_disable_dms_true_via_extra(self):
        assert _adapter({"disable_dms": "true"})._slack_disable_dms() is True

    def test_disable_dms_true_via_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_DISABLE_DMS", "1")
        assert _adapter()._slack_disable_dms() is True

    def test_disable_dms_extra_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_DISABLE_DMS", "true")
        assert _adapter({"disable_dms": "false"})._slack_disable_dms() is False


class TestChannelAllowlist:
    """Aggressive: allowed-channels list/CSV coercion and empty default."""

    def test_allowed_channels_from_list(self):
        adapter = _adapter({"allowed_channels": ["C111", " C222 "]})
        assert adapter._slack_allowed_channels() == {"C111", "C222"}

    def test_allowed_channels_from_csv_string(self):
        adapter = _adapter({"allowed_channels": "C111,C222"})
        assert adapter._slack_allowed_channels() == {"C111", "C222"}

    def test_allowed_channels_empty_means_unrestricted(self):
        assert _adapter()._slack_allowed_channels() == set()

    def test_allowed_channels_via_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "C111, C222")
        assert _adapter()._slack_allowed_channels() == {"C111", "C222"}


class TestImportabilityWithoutSlackSdk:
    """The mixin module must import with the slack SDK blocked."""

    def test_mixin_imports_without_slack_sdk(self, monkeypatch):
        for modname in list(sys.modules):
            if modname == "slack_sdk" or modname.startswith("slack_sdk."):
                monkeypatch.setitem(sys.modules, modname, None)
            if modname == "slack_bolt" or modname.startswith("slack_bolt."):
                monkeypatch.setitem(sys.modules, modname, None)
        import importlib

        module = importlib.import_module("plugins.platforms.slack.slack_gate_mixin")
        assert module.SlackGateMixin is SlackGateMixin
