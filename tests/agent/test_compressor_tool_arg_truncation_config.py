"""``compression.tool_arg_head_chars`` / ``tool_arg_min_chars`` — configurability for pass 3.

Pass 3 of ``_prune_old_tool_results`` shrinks large tool-call arguments on non-tail assistant
messages. It used to be a fixed 200-char head with a 500-char trigger and no config surface, so a
``write_file`` / ``execute_code`` / heredoc payload was silently reduced to a stub (~81% of a 1.1 KB
file, with no tail preserved) once its message left the protected tail.

These tests pin both ends: the defaults stay byte-identical to the old behaviour, and the new keys
let a user raise the cut or turn it off.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import context_compressor as cc  # noqa: E402
from hermes_cli.config_defaults import DEFAULT_CONFIG  # noqa: E402


def _payload(content_chars: int = 1100):
    """A realistic write_file argument: a content leaf plus small sibling fields."""
    content = "\n".join(f"def handler_{i}(request):\n    return process(request, {i})\n"
                        for i in range(max(1, content_chars // 57)))
    args = json.dumps({"path": "C:/project/app.py", "content": content, "cross_profile": True})
    return args, content


@pytest.fixture(autouse=True)
def _fresh_limits():
    """Every test starts from an uncached read, so a monkeypatched config is honoured."""
    cc._reset_tool_arg_limits_cache()
    yield
    cc._reset_tool_arg_limits_cache()


def _set_compression(monkeypatch, **keys):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {"compression": dict(keys)})


# ---- defaults are unchanged ---------------------------------------------------------------


def test_defaults_match_the_historical_behaviour(monkeypatch):
    _set_compression(monkeypatch)  # empty compression section
    assert cc.get_tool_arg_truncation_limits() == (200, 500)


def test_default_head_still_truncates_a_large_payload(monkeypatch):
    """The old behaviour, pinned: 200 chars kept, no tail preserved, JSON stays valid."""
    _set_compression(monkeypatch)
    args, content = _payload()
    assert len(args) > 500

    out = json.loads(cc._truncate_tool_call_args_json(args))

    assert len(out["content"]) == len("x" * 200 + "...[truncated]")
    assert out["content"].endswith("...[truncated]")
    assert content[-30:] not in out["content"], "tail must not be preserved (unchanged behaviour)"
    assert out["path"] == "C:/project/app.py", "short leaves are untouched"


def test_declared_defaults_in_config_defaults_match():
    """The documented defaults and the code defaults must not drift apart."""

    compression = DEFAULT_CONFIG["compression"]
    assert compression["tool_arg_head_chars"] == cc._TOOL_ARG_HEAD_CHARS_DEFAULT
    assert compression["tool_arg_min_chars"] == cc._TOOL_ARG_MIN_CHARS_DEFAULT
    assert cc._TOOL_ARG_HEAD_CHARS_DEFAULT == 200
    assert cc._TOOL_ARG_MIN_CHARS_DEFAULT == 500


# ---- the new keys ------------------------------------------------------------------------


def test_head_chars_zero_disables_the_shrink(monkeypatch):
    """The off-switch: a payload that would have been cut comes through untouched."""
    _set_compression(monkeypatch, tool_arg_head_chars=0)
    args, _ = _payload()

    assert cc._truncate_tool_call_args_json(args) == args
    assert cc.get_tool_arg_truncation_limits()[0] <= 0


def test_negative_head_chars_also_disables(monkeypatch):
    _set_compression(monkeypatch, tool_arg_head_chars=-1)
    args, _ = _payload()
    assert cc._truncate_tool_call_args_json(args) == args


def test_raised_head_keeps_more_content(monkeypatch):
    _set_compression(monkeypatch, tool_arg_head_chars=5000)
    args, content = _payload()

    out = json.loads(cc._truncate_tool_call_args_json(args))

    assert out["content"] == content, "a 5 KB head should leave this ~1.1 KB payload intact"


def test_min_chars_gate_spares_smaller_blobs(monkeypatch):
    """Raising the trigger means the pass never inspects a payload this size.

    The gate lives in the call site (``_truncate_tool_call_args_at``), not inside the JSON helper —
    so it is asserted there, which is also where it matters.
    """
    args, _ = _payload()
    _set_compression(monkeypatch, tool_arg_min_chars=100_000, tool_arg_head_chars=10)
    messages = [_assistant_msg(args)]

    assert cc.ContextCompressor._truncate_tool_call_args_at(messages, 0) is False
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == args


# ---- the real call site -------------------------------------------------------------------


def _assistant_msg(args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "write_file", "arguments": args}}]}


def test_call_site_returns_false_and_leaves_the_message_alone_when_disabled(monkeypatch):
    _set_compression(monkeypatch, tool_arg_head_chars=0)
    args, _ = _payload()
    messages = [_assistant_msg(args)]

    assert cc.ContextCompressor._truncate_tool_call_args_at(messages, 0) is False
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == args


def test_call_site_truncates_under_the_default(monkeypatch):
    _set_compression(monkeypatch)
    args, _ = _payload()
    messages = [_assistant_msg(args)]

    assert cc.ContextCompressor._truncate_tool_call_args_at(messages, 0) is True
    assert messages[0]["tool_calls"][0]["function"]["arguments"] != args


# ---- the prune count must include pass 3 --------------------------------------------------


def _prune_compressor(**kw):
    """A compressor whose ONLY eligible prune is a large tool-call argument."""
    defaults = dict(
        model="test", quiet_mode=True, threshold_percent=0.50, protect_first_n=2, protect_last_n=4,
        proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return cc.ContextCompressor(**defaults)


def test_arg_only_truncation_is_counted_as_pruning(monkeypatch):
    """An argument-only change is a real prune and must reach the returned count.

    ``prune_tool_results_only`` returns the INPUT list when the count is 0 (its documented no-op
    contract), so an uncounted pass-3 truncation was discarded and the configured shrink silently did
    nothing. Count it, and the caller commits.
    """
    _set_compression(monkeypatch)
    args, _ = _payload()
    c = _prune_compressor()
    messages = [_assistant_msg(args), {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]

    result, count = c._prune_old_tool_results(messages, protect_tail_count=1)

    assert count == 1
    assert result[0]["tool_calls"][0]["function"]["arguments"] != args


def test_prune_tool_results_only_commits_an_arg_only_truncation(monkeypatch):
    """End-to-end: the modified list must come back, not the original one."""
    _set_compression(monkeypatch)
    args, _ = _payload()
    c = _prune_compressor()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "write the file"},
        _assistant_msg(args),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ] + [{"role": "user", "content": f"tail {i}"} for i in range(6)]

    result, count = c.prune_tool_results_only(messages, current_tokens=120_000)

    assert count >= 1
    assert result is not messages, "the truncated list must be committed, not discarded"
    assert result[2]["tool_calls"][0]["function"]["arguments"] != args
    assert len(result[2]["tool_calls"][0]["function"]["arguments"]) < 500


# ---- robustness ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["abc", None, [], {}])
def test_invalid_values_fall_back_to_defaults(monkeypatch, bad):
    _set_compression(monkeypatch, tool_arg_head_chars=bad)
    head, minimum = cc.get_tool_arg_truncation_limits()
    assert head == cc._TOOL_ARG_HEAD_CHARS_DEFAULT or head <= 0
    assert minimum == cc._TOOL_ARG_MIN_CHARS_DEFAULT


def test_config_failure_never_breaks_compaction(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    assert cc.get_tool_arg_truncation_limits() == (200, 500)


def test_non_json_arguments_are_returned_unchanged(monkeypatch):
    _set_compression(monkeypatch)
    assert cc._truncate_tool_call_args_json("not json at all") == "not json at all"
