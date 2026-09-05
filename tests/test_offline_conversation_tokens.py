"""Budget decisions must measure the serialized conversation, including empty turns."""

import json
import multiprocessing
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import sample_and_compress
from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics


class ChatTokenizer:
    bos_token_id = None
    eos_token_id = None
    unk_token_id = None

    def get_chat_template(self):
        return "test conversation format"

    def encode(self, text):
        return text.split()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is False
        assert all(message["role"] in {"system", "user", "assistant", "tool"} for message in messages)
        # Conversation framing and per-message delimiters count even for empty content.
        return [0] * (2 + sum(4 + len(self.encode(message["content"])) for message in messages))


def test_sampling_and_compression_use_the_formatted_budget(monkeypatch):
    tokenizer = ChatTokenizer()
    monkeypatch.setattr(sample_and_compress, "_TOKENIZER", tokenizer)
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.tokenizer = tokenizer
    compressor.logger = MagicMock()
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]
    expected = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    sharegpt = [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": ""}]

    for trajectory in (sharegpt, messages):
        entry = {"conversations": trajectory}
        assert sample_and_compress._count_tokens_for_entry(entry) == (entry, expected)
        assert compressor.count_trajectory_tokens(trajectory) == expected
        for budget in (expected - 1, expected):
            compressor.config = CompressionConfig(target_max_tokens=budget)
            metrics = TrajectoryMetrics()
            compressor._plan_compression(trajectory, metrics)
            assert metrics.original_tokens == expected
            assert metrics.skipped_under_target is (expected <= budget)
            assert metrics.still_over_limit is (expected > budget)

    compressor.config = CompressionConfig(target_max_tokens=expected)
    metrics = TrajectoryMetrics(original_tokens=expected, original_turns=len(sharegpt))
    compressed = compressor._assemble_compressed(sharegpt, 0, 1, "a longer summary", metrics)
    assert metrics.compressed_tokens == compressor.count_trajectory_tokens(compressed)
    assert metrics.still_over_limit is (metrics.compressed_tokens > expected)

    tokenizer.apply_chat_template = MagicMock(side_effect=ValueError("missing template"))
    with pytest.raises(ValueError, match="missing template"):
        sample_and_compress._count_tokens_for_entry({"conversations": sharegpt})
    with pytest.raises(ValueError, match="missing template"):
        compressor.count_trajectory_tokens(sharegpt)


def test_pipeline_uses_one_target_tokenizer_config_for_both_stages(tmp_path, monkeypatch):
    """Run YAML -> sampling -> JSONL -> compressor, replacing Hub I/O and the process pool."""
    tokenizer = ChatTokenizer()
    loads = []

    def from_pretrained(name, **kwargs):
        loads.append((name, kwargs))
        return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
    ))

    class InlinePool:
        def __init__(self, *, processes, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, function, entries, chunksize):
            return map(function, entries)

    monkeypatch.setattr(multiprocessing, "Pool", InlinePool)
    monkeypatch.setattr(TrajectoryCompressor, "_init_summarizer", lambda self: None)
    monkeypatch.setattr(sample_and_compress, "__file__", str(tmp_path / "scripts" / "sample_and_compress.py"))
    trajectory = [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": ""}]
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]
    expected = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    monkeypatch.setattr(sample_and_compress, "load_dataset_from_hf", lambda name: [
        {"conversations": trajectory[:1]}, {"conversations": trajectory},
    ])
    revision = "a" * 40
    config = tmp_path / "compression.yaml"
    config.write_text(
        f"tokenizer:\n  name: selected-target\n  revision: {revision}\n  trust_remote_code: false\n"
        f"compression:\n  target_max_tokens: {expected}\n",
        encoding="utf-8",
    )
    sample_and_compress.main(
        total_samples=1, output_name="test", datasets="fixture", config=str(config),
        min_tokens=expected, num_proc=1,
    )
    assert loads
    assert all(load == ("selected-target", {"revision": revision, "trust_remote_code": False}) for load in loads)
    output = [json.loads(line) for line in (tmp_path / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(output) == 1
    assert output[0]["conversations"] == trajectory
    assert output[0]["_original_tokens"] == expected
    report = json.loads((tmp_path / "data" / "test_batches" / "compression_metrics.json").read_text(encoding="utf-8"))
    assert report["tokenizer"]["revision"] == revision
    assert report["tokenizer"]["name"] == loads[0][0]
    assert report["summary"]["trajectories_skipped_under_target"] == 1
