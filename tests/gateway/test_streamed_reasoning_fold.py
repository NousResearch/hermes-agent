"""Streamed-turn reasoning fold (#57693).

With streaming on, the stream consumer commits the final message and the
gateway suppresses the normal send (``already_sent=True``). The normal send
path is the only place the 💭 reasoning block is prepended, so streaming
silently disabled reasoning display for every model/platform. The fix folds
the block into the already-streamed message with one final edit, routed
through the stream consumer's *metadata-aware* edit path so the edit still
carries the routing metadata Slack uses to select the workspace client (a raw
``adapter.edit_message`` would drop it and a non-default workspace would lose
the reasoning).

Coverage, on real code:
  - gateway boundary (``GatewayRunner._run_agent`` with a live
    ``GatewayStreamConsumer``): the streamed message is edited to carry the
    reasoning block above the answer, ``already_sent`` still suppresses the
    normal send, and there is no duplicate delivery;
  - the fold's edit preserves the consumer's routing metadata;
  - best-effort no-ops: no reasoning, no consumer, uncommitted stream,
    ``__no_edit__``, split delivery (#78541), over-limit text, failing edit;
  - ``_format_reasoning_block`` renders the same block the normal path did.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer


# ---------------------------------------------------------------------------
# Boundary-test fakes (same shape as tests/gateway/test_stale_finalize_suppression.py)
# ---------------------------------------------------------------------------


class FinalizeCaptureAdapter(BasePlatformAdapter):
    """Adapter that records every send/edit, including routing metadata."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self._next_id = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        self._next_id += 1
        return f"m-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=self._mint_id())

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "finalize": finalize,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


ANSWER = "27 times 43 is 1161."
REASONING = "27*40 = 1080\n27*3 = 81\n1080 + 81 = 1161"
REASONING_BLOCK = f"💭 **Reasoning:**\n```\n{REASONING}\n```"


class ReasoningStreamAgent:
    """Streams the complete answer and reports the reasoning it used."""

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback(ANSWER)
        return {
            "final_response": ANSWER,
            "last_reasoning": REASONING,
            "response_previewed": False,
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig.from_dict(
            {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1}
        ),
    )
    return runner


async def _run_streaming_turn(monkeypatch, tmp_path, *, show_reasoning: bool, session_id: str):
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "display": {
                    "tool_progress": "off",
                    "interim_assistant_messages": False,
                    "show_reasoning": show_reasoning,
                },
                "streaming": {
                    "enabled": True,
                    "edit_interval": 0.01,
                    "buffer_threshold": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ReasoningStreamAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = FinalizeCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    result = await runner._run_agent(
        message="what is 27 times 43?",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key="agent:main:telegram:group:-1001",
    )
    return adapter, result


# ---------------------------------------------------------------------------
# Gateway-boundary regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streamed_turn_folds_reasoning_and_still_suppresses(monkeypatch, tmp_path):
    """A streamed turn with show_reasoning on ends with the streamed message
    carrying the reasoning block above the answer, the normal send suppressed,
    and no duplicate delivery."""
    adapter, result = await _run_streaming_turn(
        monkeypatch, tmp_path, show_reasoning=True, session_id="sess-57693-fold"
    )

    assert result["final_response"] == ANSWER
    assert result.get("already_sent") is True

    folded = [
        e for e in adapter.edits
        if e["content"] == f"{REASONING_BLOCK}\n\n{ANSWER}" and e["finalize"]
    ]
    assert folded, f"no finalize edit carried the reasoning block; edits: {adapter.edits!r}"
    # The fold edits the message the stream committed (minted by the first
    # send), not a new one.
    assert folded[-1]["message_id"] == "m-1"
    # Exactly one platform message holds the answer (streamed, then edited);
    # the fold never triggers a second full send.
    full_sends = [c for c in adapter.sent if ANSWER in c["content"]]
    assert len(full_sends) <= 1, f"duplicate final delivery: {full_sends!r}"
    assert not any("💭" in c["content"] for c in adapter.sent)


@pytest.mark.asyncio
async def test_streamed_turn_without_show_reasoning_is_unchanged(monkeypatch, tmp_path):
    """Control: with show_reasoning off no edit carries a reasoning block and
    suppression still happens."""
    adapter, result = await _run_streaming_turn(
        monkeypatch, tmp_path, show_reasoning=False, session_id="sess-57693-control"
    )

    assert result.get("already_sent") is True
    assert not any("💭" in e["content"] for e in adapter.edits)
    assert not any("💭" in c["content"] for c in adapter.sent)


# ---------------------------------------------------------------------------
# Fold helper: metadata preservation + best-effort gates (real consumer)
# ---------------------------------------------------------------------------


class RecordingAdapter:
    """Minimal adapter whose edit_message accepts (and records) metadata."""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self):
        self.edits = []

    async def edit_message(self, *, chat_id, message_id, content, finalize=False, metadata=None):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "finalize": finalize,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True, message_id=message_id)


class MetadataBlindAdapter:
    """Adapter whose edit_message cannot accept metadata (no such param)."""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self):
        self.edits = []

    async def edit_message(self, *, chat_id, message_id, content, finalize=False):
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        return SimpleNamespace(success=True, message_id=message_id)


def _runner_with_block(block: str):
    """A GatewayRunner with _format_reasoning_block stubbed to a fixed block,
    isolating the fold wiring from gateway-config resolution."""
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._format_reasoning_block = lambda source, last_reasoning: (
        block if last_reasoning else ""
    )
    return runner


def _consumer(adapter, *, metadata=None, message_id="msg-1"):
    sc = GatewayStreamConsumer(adapter, "chat-42", metadata=metadata)
    sc._message_id = message_id
    return sc


def _fold(runner, sc, **overrides):
    kwargs = dict(
        source=SimpleNamespace(platform=Platform.SLACK),
        stream_consumer=sc,
        final_text="1161",
        last_reasoning="27*43 = 1161",
        session_key="sess-1",
    )
    kwargs.update(overrides)
    return asyncio.run(runner._fold_reasoning_into_streamed_message(**kwargs))


def test_fold_edits_streamed_message_with_reasoning_and_metadata():
    runner = _runner_with_block(REASONING_BLOCK)
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    assert _fold(runner, sc) is True
    assert len(adapter.edits) == 1
    edit = adapter.edits[0]
    assert edit["content"] == f"{REASONING_BLOCK}\n\n1161"   # block above answer
    assert edit["message_id"] == "msg-1"                    # the streamed message
    assert edit["finalize"] is True
    assert edit["metadata"] == {"slack_team_id": "T999"}    # routing preserved


def test_fold_omits_metadata_when_adapter_cannot_accept_it():
    runner = _runner_with_block(REASONING_BLOCK)
    adapter = MetadataBlindAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    assert _fold(runner, sc) is True
    assert adapter.edits == [
        {"chat_id": "chat-42", "message_id": "msg-1", "content": f"{REASONING_BLOCK}\n\n1161"}
    ]


@pytest.mark.parametrize(
    "case",
    ["no_reasoning", "no_consumer", "uncommitted", "no_edit_sentinel", "split_delivery"],
)
def test_fold_noops_when_nothing_editable(case):
    runner = _runner_with_block(REASONING_BLOCK)
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})
    overrides = {}
    if case == "no_reasoning":
        overrides["last_reasoning"] = None
    elif case == "no_consumer":
        overrides["stream_consumer"] = None
    elif case == "uncommitted":
        sc._message_id = None
    elif case == "no_edit_sentinel":
        sc._message_id = "__no_edit__"
    elif case == "split_delivery":
        # message_id is only the LAST chunk; editing it with the complete
        # response would repeat every sealed head chunk (#78541).
        sc._turn_split_delivery = True

    assert _fold(runner, sc, **overrides) is False
    assert adapter.edits == []


def test_fold_skips_when_folded_text_exceeds_chat_limit():
    runner = _runner_with_block(REASONING_BLOCK)
    adapter = RecordingAdapter()
    adapter.MAX_MESSAGE_LENGTH = len(REASONING_BLOCK)  # no room for the answer
    sc = _consumer(adapter)

    assert _fold(runner, sc) is False
    assert adapter.edits == []


def test_fold_is_best_effort_on_edit_failure():
    """A failed edit must not raise; it only loses the reasoning display."""
    runner = _runner_with_block(REASONING_BLOCK)

    class BoomAdapter:
        MAX_MESSAGE_LENGTH = 4096

        async def edit_message(self, *, chat_id, message_id, content, finalize=False, metadata=None):
            raise RuntimeError("workspace client unavailable")

    class FailingAdapter:
        MAX_MESSAGE_LENGTH = 4096

        async def edit_message(self, *, chat_id, message_id, content, finalize=False, metadata=None):
            return SimpleNamespace(success=False, error="edit rejected")

    assert _fold(runner, _consumer(BoomAdapter(), metadata={"slack_team_id": "T999"})) is False
    assert _fold(runner, _consumer(FailingAdapter())) is False


# ---------------------------------------------------------------------------
# _format_reasoning_block: same rendering the normal send path used
# ---------------------------------------------------------------------------


def _runner_for_format(monkeypatch, config: dict):
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: config)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._show_reasoning = False
    return runner


def test_format_block_code_style_escapes_inner_fences(monkeypatch):
    runner = _runner_for_format(monkeypatch, {"display": {"show_reasoning": True}})
    block = runner._format_reasoning_block(
        SimpleNamespace(platform=Platform.TELEGRAM), "use ```py\nx=1\n``` here"
    )
    assert block.startswith("💭 **Reasoning:**\n```\n")
    assert block.endswith("\n```")
    assert block.count("```") == 2  # only the outer fence survives


def test_format_block_collapses_long_reasoning(monkeypatch):
    runner = _runner_for_format(monkeypatch, {"display": {"show_reasoning": True}})
    block = runner._format_reasoning_block(
        SimpleNamespace(platform=Platform.TELEGRAM),
        "\n".join(f"line {i}" for i in range(20)),
    )
    assert "line 14" in block and "line 15" not in block
    assert "_... (5 more lines)_" in block


@pytest.mark.parametrize(
    "style, head, line_prefix",
    [("subtext", "-# 💭 Reasoning", "-# "), ("blockquote", "> 💭 **Reasoning:**", "> ")],
)
def test_format_block_styles(monkeypatch, style, head, line_prefix):
    runner = _runner_for_format(
        monkeypatch, {"display": {"show_reasoning": True, "reasoning_style": style}}
    )
    block = runner._format_reasoning_block(
        SimpleNamespace(platform=Platform.DISCORD), "a\nb"
    )
    assert block == f"{head}\n{line_prefix}a\n{line_prefix}b"


def test_format_block_empty_when_display_off_or_no_reasoning(monkeypatch):
    runner = _runner_for_format(monkeypatch, {"display": {"show_reasoning": False}})
    src = SimpleNamespace(platform=Platform.TELEGRAM)
    assert runner._format_reasoning_block(src, "thinking") == ""
    runner = _runner_for_format(monkeypatch, {"display": {"show_reasoning": True}})
    assert runner._format_reasoning_block(src, "") == ""
    assert runner._format_reasoning_block(src, None) == ""


def test_format_block_mattermost_requires_platform_override(monkeypatch):
    src = SimpleNamespace(platform=Platform.MATTERMOST)
    runner = _runner_for_format(monkeypatch, {"display": {"show_reasoning": True}})
    assert runner._format_reasoning_block(src, "thinking") == ""
    runner = _runner_for_format(
        monkeypatch,
        {"display": {"show_reasoning": True, "platforms": {"mattermost": {"show_reasoning": True}}}},
    )
    assert runner._format_reasoning_block(src, "thinking").startswith("💭 **Reasoning:**")
