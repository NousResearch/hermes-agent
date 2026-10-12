"""Tameru's discovered adapter preserves Hermes call options and text-only queries."""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from plugins.context_engine import load_context_engine


def test_compress_forwards_overflow_cooldown_override():
    engine = load_context_engine("tameru")
    assert engine is not None
    messages = [{"role": "user", "content": "Find the deployment error"}]
    with patch.object(ContextCompressor, "compress", return_value=messages) as parent:
        assert engine.compress(messages, bypass_cooldown=True) == messages
    assert parent.call_args.kwargs["bypass_cooldown"] is True


def test_multimodal_user_query_excludes_image_payload_in_both_paths():
    engine = load_context_engine("tameru")
    assert engine is not None
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Find DB-PROD-42"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    with (
        patch("plugins.context_engine.tameru.apply_extractive_tool_prune", return_value=(messages, 0)) as prune,
        patch.object(ContextCompressor, "prune_tool_results_only", return_value=(messages, 0)),
        patch.object(ContextCompressor, "compress", return_value=messages),
    ):
        engine.prune_tool_results_only(messages)
        engine.compress(messages)
    assert [call.args[1] for call in prune.call_args_list] == ["Find DB-PROD-42"] * 2
