"""Tests for the context-handover plugin.

Covers:
- Threshold detection (below / at / above)
- Handover document write + content
- Discord presence string formatting and truncation
- Config parsing and defaults
- Compaction disabled when handover fires
- Hook is no-op when agent=None is passed

All heavy dependencies (discord, gateway, live model) are mocked.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to load the plugin modules without full Hermes install
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "context-handover"


def _ensure_ns_package() -> None:
    """Ensure hermes_plugins namespace package exists in sys.modules."""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []  # type: ignore[attr-defined]
        sys.modules["hermes_plugins"] = ns


def _load_handover_module() -> types.ModuleType:
    """Import handover.py directly, without the plugin __init__."""
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.context_handover.handover",
        PLUGIN_DIR / "handover.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = mod  # type: ignore[union-attr]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_plugin_init(monkeypatch=None) -> types.ModuleType:
    """Import the plugin __init__.py with mocked Hermes imports."""
    _ensure_ns_package()

    # Stub out hermes_cli.config so the plugin can import it.
    fake_cli = types.ModuleType("hermes_cli")
    fake_config = types.ModuleType("hermes_cli.config")
    fake_config.cfg_get = lambda cfg, *keys, default=None: default  # type: ignore
    fake_config.load_config = lambda: {}  # type: ignore
    sys.modules.setdefault("hermes_cli", fake_cli)
    sys.modules["hermes_cli.config"] = fake_config

    pkg_name = "hermes_plugins.context_handover"

    # Ensure the handover sub-module is loaded into sys.modules first so
    # the relative import `from . import handover` inside __init__.py resolves.
    hw_name = f"{pkg_name}.handover"
    if hw_name not in sys.modules:
        hw = _load_handover_module()
    else:
        hw = sys.modules[hw_name]

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__package__ = pkg_name
    mod.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    # Register both the package and the sub-module BEFORE exec so the relative
    # import inside __init__.py (`from . import handover`) can find it.
    sys.modules[pkg_name] = mod
    sys.modules[hw_name] = hw
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def handover_mod():
    """The handover.py module, freshly loaded."""
    name = "hermes_plugins.context_handover.handover"
    if name in sys.modules:
        return sys.modules[name]
    return _load_handover_module()


def _make_agent(last_prompt_tokens: int = 0, context_length: int = 100_000) -> MagicMock:
    """Build a minimal mock agent with a context_compressor."""
    agent = MagicMock()
    agent.context_compressor = MagicMock()
    agent.context_compressor.last_prompt_tokens = last_prompt_tokens
    agent.context_compressor.context_length = context_length
    agent.compression_enabled = True
    agent._gateway_session_key = "agent:main:discord:dm:999"
    agent.platform = "discord"
    return agent


# ---------------------------------------------------------------------------
# Unit tests: handover.py — compute_context_pct
# ---------------------------------------------------------------------------

class TestComputeContextPct:
    def test_returns_pct_correctly(self, handover_mod):
        agent = _make_agent(last_prompt_tokens=90_000, context_length=100_000)
        assert handover_mod.compute_context_pct(agent) == pytest.approx(90.0)

    def test_zero_tokens(self, handover_mod):
        agent = _make_agent(last_prompt_tokens=0, context_length=100_000)
        assert handover_mod.compute_context_pct(agent) == pytest.approx(0.0)

    def test_zero_context_length_returns_none(self, handover_mod):
        agent = _make_agent(last_prompt_tokens=50_000, context_length=0)
        assert handover_mod.compute_context_pct(agent) is None

    def test_missing_compressor_returns_none(self, handover_mod):
        agent = MagicMock(spec=[])  # no attributes
        assert handover_mod.compute_context_pct(agent) is None

    def test_above_100_pct(self, handover_mod):
        # Tokens can exceed context_length in edge cases (rough estimates).
        agent = _make_agent(last_prompt_tokens=110_000, context_length=100_000)
        assert handover_mod.compute_context_pct(agent) == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# Unit tests: handover.py — format_presence
# ---------------------------------------------------------------------------

class TestFormatPresence:
    def test_basic_format(self, handover_mod):
        result = handover_mod.format_presence(75.3, "fix the bug")
        assert result == "75% · fix the bug"

    def test_rounds_pct(self, handover_mod):
        result = handover_mod.format_presence(89.9, "task")
        assert result.startswith("90%")

    def test_truncates_long_task(self, handover_mod):
        long_task = "x" * 200
        result = handover_mod.format_presence(50.0, long_task)
        # Should be truncated to _PRESENCE_TASK_MAX chars + ellipsis
        assert "…" in result
        # Total string length: "50% · " (6) + 80 chars + "…" (3 bytes) = manageable
        task_part = result.split("·", 1)[1].strip()
        # task_part has 80 chars of x + "…"
        assert len(task_part) <= handover_mod._PRESENCE_TASK_MAX + 5

    def test_exact_max_length_no_ellipsis(self, handover_mod):
        exact_task = "a" * handover_mod._PRESENCE_TASK_MAX
        result = handover_mod.format_presence(50.0, exact_task)
        assert "…" not in result

    def test_empty_task(self, handover_mod):
        result = handover_mod.format_presence(50.0, "")
        assert "·" in result


# ---------------------------------------------------------------------------
# Unit tests: handover.py — write_handover_doc
# ---------------------------------------------------------------------------

class TestWriteHandoverDoc:
    def test_creates_file(self, tmp_path, handover_mod):
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="implement feature X",
            assistant_response="Done.",
            conversation_history=[
                {"role": "user", "content": "implement feature X"},
                {"role": "assistant", "content": "Done."},
            ],
            pct=91.5,
            model="claude-3-opus",
        )
        assert doc.exists()
        assert doc.suffix == ".md"

    def test_content_contains_task(self, tmp_path, handover_mod):
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="fix the authentication bug",
            assistant_response="I have fixed it.",
            conversation_history=[],
            pct=92.0,
            model="gpt-4",
        )
        content = doc.read_text()
        assert "fix the authentication bug" in content

    def test_content_contains_pct_and_model(self, tmp_path, handover_mod):
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="task",
            assistant_response="response",
            conversation_history=[],
            pct=93.7,
            model="nous-hermes-3",
        )
        content = doc.read_text()
        assert "93.7%" in content
        assert "nous-hermes-3" in content

    def test_content_contains_last_response(self, tmp_path, handover_mod):
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="task",
            assistant_response="The answer is 42.",
            conversation_history=[],
            pct=90.0,
            model="m",
        )
        content = doc.read_text()
        assert "The answer is 42." in content

    def test_long_response_truncated(self, tmp_path, handover_mod):
        long_response = "word " * 1000
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="task",
            assistant_response=long_response,
            conversation_history=[],
            pct=90.0,
            model="m",
        )
        content = doc.read_text()
        # Should contain truncation marker
        assert "…" in content

    def test_creates_notes_dir_if_missing(self, tmp_path, handover_mod):
        new_dir = tmp_path / "does" / "not" / "exist"
        assert not new_dir.exists()
        handover_mod.write_handover_doc(
            notes_dir=new_dir,
            task="t",
            assistant_response="r",
            conversation_history=[],
            pct=90.0,
            model="m",
        )
        assert new_dir.exists()

    def test_list_content_blocks_handled(self, tmp_path, handover_mod):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is the plan?"},
                ],
            }
        ]
        doc = handover_mod.write_handover_doc(
            notes_dir=tmp_path,
            task="task",
            assistant_response="here is the plan",
            conversation_history=history,
            pct=90.0,
            model="m",
        )
        content = doc.read_text()
        assert "what is the plan?" in content


# ---------------------------------------------------------------------------
# Unit tests: handover.py — extract_task
# ---------------------------------------------------------------------------

class TestExtractTask:
    def test_uses_user_message_first(self, handover_mod):
        result = handover_mod._extract_task(
            [{"role": "user", "content": "ignored"}],
            "current user message",
        )
        assert result == "current user message"

    def test_falls_back_to_history(self, handover_mod):
        result = handover_mod._extract_task(
            [{"role": "user", "content": "first user msg"}],
            "",
        )
        assert "first user msg" in result

    def test_empty_returns_unknown(self, handover_mod):
        result = handover_mod._extract_task([], "")
        assert result == "unknown task"

    def test_truncates_long_message(self, handover_mod):
        long = "x" * 500
        result = handover_mod._extract_task([], long)
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# Integration: threshold detection in post_llm_call
# ---------------------------------------------------------------------------

class TestThresholdDetection:
    """Test that the post_llm_call hook fires handover only above threshold."""

    def _make_init_mod(self, monkeypatch):
        # Clear any cached module so we get a fresh one.
        for k in list(sys.modules.keys()):
            if "context_handover" in k:
                del sys.modules[k]
        return _load_plugin_init(monkeypatch)

    def _hw(self, init_mod):
        """Return the handover sub-module bound to the given init module."""
        # __init__.py imports `from . import handover as _handover`.
        # After exec_module the attribute name is _handover.
        return init_mod._handover

    def test_below_threshold_no_handover(self, monkeypatch, tmp_path):
        init = self._make_init_mod(monkeypatch)
        agent = _make_agent(last_prompt_tokens=50_000, context_length=100_000)  # 50%

        written = []
        monkeypatch.setattr(
            self._hw(init), "write_handover_doc",
            lambda **kw: written.append(kw) or (tmp_path / "h.md"),
        )

        init._config = {"threshold": 0.90, "notes_dir": str(tmp_path),
                        "auto_continue": False, "presence": False}
        init._on_post_llm_call(
            agent=agent, session_id="s", task_id="t",
            user_message="hi", assistant_response="hello",
            conversation_history=[], model="m",
        )
        assert len(written) == 0

    def test_at_threshold_triggers_handover(self, monkeypatch, tmp_path):
        init = self._make_init_mod(monkeypatch)
        agent = _make_agent(last_prompt_tokens=90_000, context_length=100_000)  # 90%

        written = []
        hw = self._hw(init)
        monkeypatch.setattr(hw, "write_handover_doc",
                            lambda **kw: written.append(kw) or (tmp_path / "h.md"))
        monkeypatch.setattr(hw, "trigger_handover", lambda **kw: None)

        # Ensure gateway import doesn't blow up — runner returns None.
        fake_gw = types.ModuleType("gateway")
        fake_gw_run = types.ModuleType("gateway.run")
        fake_gw_run._gateway_runner_ref = lambda: None
        sys.modules.setdefault("gateway", fake_gw)
        sys.modules["gateway.run"] = fake_gw_run

        init._config = {"threshold": 0.90, "notes_dir": str(tmp_path),
                        "auto_continue": True, "presence": False}
        init._on_post_llm_call(
            agent=agent, session_id="s", task_id="t",
            user_message="hi", assistant_response="hello",
            conversation_history=[], model="m",
        )
        assert len(written) == 1

    def test_above_threshold_triggers_handover(self, monkeypatch, tmp_path):
        init = self._make_init_mod(monkeypatch)
        agent = _make_agent(last_prompt_tokens=95_000, context_length=100_000)  # 95%

        written = []
        hw = self._hw(init)
        monkeypatch.setattr(hw, "write_handover_doc",
                            lambda **kw: written.append(kw) or (tmp_path / "h.md"))
        monkeypatch.setattr(hw, "trigger_handover", lambda **kw: None)

        fake_gw = types.ModuleType("gateway")
        fake_gw_run = types.ModuleType("gateway.run")
        fake_gw_run._gateway_runner_ref = lambda: None
        sys.modules.setdefault("gateway", fake_gw)
        sys.modules["gateway.run"] = fake_gw_run

        init._config = {"threshold": 0.90, "notes_dir": str(tmp_path),
                        "auto_continue": True, "presence": False}
        init._on_post_llm_call(
            agent=agent, session_id="s", task_id="t",
            user_message="task", assistant_response="result",
            conversation_history=[], model="m",
        )
        assert len(written) == 1

    def test_no_op_when_agent_is_none(self, monkeypatch, tmp_path):
        init = self._make_init_mod(monkeypatch)
        hw = self._hw(init)
        written = []
        monkeypatch.setattr(hw, "write_handover_doc",
                            lambda **kw: written.append(kw) or (tmp_path / "h.md"))
        init._config = {"threshold": 0.90, "notes_dir": str(tmp_path),
                        "auto_continue": False, "presence": False}
        init._on_post_llm_call(agent=None, session_id="s")
        assert len(written) == 0


# ---------------------------------------------------------------------------
# Compaction disabled when handover fires
# ---------------------------------------------------------------------------

class TestCompactionDisabled:
    """When trigger_handover fires, it must set agent.compression_enabled=False."""

    def test_compression_disabled_on_trigger(self, monkeypatch, tmp_path):
        hw = _load_handover_module()
        agent = _make_agent(last_prompt_tokens=95_000, context_length=100_000)
        agent.compression_enabled = True

        runner = MagicMock()
        session_store = MagicMock()
        new_entry = MagicMock()
        new_entry.session_id = "new-session-123"
        session_store.reset_session.return_value = new_entry
        runner.session_store = session_store
        runner.adapters = {}

        hw.trigger_handover(
            agent=agent,
            runner=runner,
            doc_path=tmp_path / "handover.md",
            session_key="agent:main:discord:dm:999",
            auto_continue=True,
        )

        assert agent.compression_enabled is False

    def test_no_reset_when_auto_continue_false(self, monkeypatch, tmp_path):
        hw = _load_handover_module()
        agent = _make_agent()
        runner = MagicMock()

        hw.trigger_handover(
            agent=agent,
            runner=runner,
            doc_path=tmp_path / "handover.md",
            session_key="key",
            auto_continue=False,
        )

        # reset_session must NOT be called when auto_continue is False.
        runner.session_store.reset_session.assert_not_called()


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_defaults_applied_when_config_empty(self, monkeypatch):
        for k in list(sys.modules.keys()):
            if "context_handover" in k:
                del sys.modules[k]
        init = _load_plugin_init(monkeypatch)

        # Verify defaults loaded (hermes_cli.config.load_config returns {} stub).
        cfg = init._load_config()
        assert cfg["threshold"] == pytest.approx(0.90)
        assert cfg["notes_dir"] == "~/notes"
        assert cfg["auto_continue"] is True
        assert cfg["presence"] is True

    def test_register_sets_config(self, monkeypatch):
        for k in list(sys.modules.keys()):
            if "context_handover" in k:
                del sys.modules[k]
        init = _load_plugin_init(monkeypatch)

        ctx = MagicMock()
        init.register(ctx)
        ctx.register_hook.assert_called_once_with("post_llm_call", init._on_post_llm_call)
        assert init._config["threshold"] == pytest.approx(0.90)
