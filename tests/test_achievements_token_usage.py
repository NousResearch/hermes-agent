"""Achievements dashboard surfaces decoded per-session/lifetime token usage.

The plugin previously read messages but ignored token data; analyze_messages
now sums the decoded `tokens` bucket (with a raw-token_count fallback) and
aggregate_stats rolls it up.
"""
import importlib.util
from pathlib import Path

import pytest

from hermes_token_codec import (
    pack_input_tokens,
    pack_assistant_tokens,
    resolve_message_tokens,
)

_PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "hermes-achievements" / "dashboard" / "plugin_api.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("ach_plugin_api", _PLUGIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flat(role, packed):
    return {"role": role, "content": "x", "tokens": resolve_message_tokens(role, packed)}


def test_analyze_messages_sums_decoded_tokens(mod):
    msgs = [
        _flat("user", pack_input_tokens(1000, 400)),
        _flat("assistant", pack_assistant_tokens(300, 50)),
    ]
    s = mod.analyze_messages("s1", "t", msgs)
    assert s["total_input_tokens"] == 1000
    assert s["total_cache_read_tokens"] == 400
    assert s["total_output_tokens"] == 300
    assert s["total_reasoning_tokens"] == 50
    assert s["total_message_tokens"] == 1000 + 300 + 50


def test_fallback_decodes_raw_token_count(mod):
    # Un-flattened rows (raw packed token_count, no `tokens` key) still decode.
    msgs = [{"role": "user", "content": "q", "token_count": pack_input_tokens(500, 100)}]
    s = mod.analyze_messages("s2", "t", msgs)
    assert s["total_input_tokens"] == 500
    assert s["total_cache_read_tokens"] == 100


def test_aggregate_rolls_up_lifetime_tokens(mod):
    s = mod.analyze_messages("s1", "t", [
        _flat("user", pack_input_tokens(1000, 400)),
        _flat("assistant", pack_assistant_tokens(300, 50)),
    ])
    agg = mod.aggregate_stats([s, s])
    assert agg["total_input_tokens"] == 2000
    assert agg["total_output_tokens"] == 600
    assert agg["total_message_tokens"] == 2 * (1000 + 300 + 50)


def test_no_token_data_is_zero(mod):
    s = mod.analyze_messages("s3", "t", [{"role": "user", "content": "hi"}])
    assert s["total_message_tokens"] == 0
