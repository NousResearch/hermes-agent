"""Budget decisions must measure the serialized conversation, including empty turns."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import sample_and_compress
from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics


class ChatTokenizer:
    bos_token_id = None
    eos_token_id = None
    unk_token_id = None
    message_overhead = 4

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
        return [0] * (2 + sum(self.message_overhead + len(self.encode(message["content"])) for message in messages))


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


def test_pipeline_uses_one_target_tokenizer_config_for_both_stages(tmp_path, monkeypatch, caplog):
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
    assert not any("--skip_download" in record.message for record in caplog.records)

    # A different template changes the budget, but cannot recover discarded rows.
    raw_path = tmp_path / "data" / "test_raw" / "batch_0.jsonl"
    cached_input = raw_path.read_bytes()
    tokenizer.message_overhead = 8
    revised_count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    revision = "b" * 40
    config.write_text(
        f"tokenizer:\n  name: selected-target\n  revision: {revision}\n  trust_remote_code: false\n"
        f"compression:\n  target_max_tokens: {revised_count}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sample_and_compress, "load_dataset_from_hf", MagicMock(
        side_effect=AssertionError("Cached-input reuse must not download datasets"),
    ))
    loads.clear()
    caplog.clear()
    sample_and_compress.main(
        output_name="test", config=str(config), min_tokens=revised_count, skip_download=True,
    )
    assert raw_path.read_bytes() == cached_input
    assert loads == [("selected-target", {"revision": revision, "trust_remote_code": False})]
    report = json.loads((tmp_path / "data" / "test_batches" / "compression_metrics.json").read_text(encoding="utf-8"))
    assert report["tokenizer"]["revision"] == revision
    assert report["tokens"]["total_before"] == revised_count
    assert report["tokens"]["total_after"] == revised_count
    warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert any("--skip_download" in warning and "_original_tokens" in warning for warning in warnings)


@pytest.mark.parametrize("error_type", [ValueError, OSError, RuntimeError])
def test_tokenizer_resolution_failure_stops_pipeline(tmp_path, monkeypatch, caplog, error_type):
    config = tmp_path / "compression.yaml"
    config.write_text("tokenizer:\n  name: unavailable-target\n", encoding="utf-8")
    monkeypatch.setattr(sample_and_compress, "__file__", str(tmp_path / "scripts" / "sample_and_compress.py"))
    resolver = MagicMock(side_effect=error_type("revision unavailable"))
    monkeypatch.setattr(sample_and_compress, "resolve_tokenizer_revision", resolver)
    sample = MagicMock(side_effect=AssertionError("Must not sample after resolution fails"))
    compress = MagicMock(side_effect=AssertionError("Must not compress after resolution fails"))
    monkeypatch.setattr(sample_and_compress, "sample_from_datasets", sample)
    monkeypatch.setattr(sample_and_compress, "run_compression", compress)

    # Expected configuration/I/O failures are CLI errors; programming errors remain visible.
    expected_error = RuntimeError if error_type is RuntimeError else SystemExit
    with pytest.raises(expected_error) as failure:
        sample_and_compress.main(config=str(config), output_name="failed")

    resolver.assert_called_once_with("unavailable-target", None)
    sample.assert_not_called()
    compress.assert_not_called()
    assert not (tmp_path / "data").exists()
    errors = [record.message for record in caplog.records if record.levelname == "ERROR"]
    if error_type is RuntimeError:
        assert str(failure.value) == "revision unavailable"
        assert not errors
    else:
        assert failure.value.code == 1
        assert any("unavailable-target" in error and "revision unavailable" in error for error in errors)


def test_invalid_tokenizer_cli_reports_error_without_traceback(tmp_path):
    pytest.importorskip("transformers")
    config = tmp_path / "compression.yaml"
    config.write_text("tokenizer:\n  name: invalid/name/extra\n", encoding="utf-8")
    script = Path(sample_and_compress.__file__).resolve()
    output_name = f"invalid-tokenizer-{tmp_path.name}"
    output_dir = script.parent.parent / "data" / f"{output_name}_raw"
    assert not output_dir.exists()
    result = subprocess.run(
        [sys.executable, str(script), f"--config={config}", f"--output_name={output_name}"],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "HERMES_HOME": str(tmp_path / "hermes"),
            "HF_HOME": str(tmp_path / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "PYTHON_DOTENV_DISABLED": "1",
            **{key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ},
        },
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 1
    assert "Cannot resolve tokenizer" in result.stderr
    assert "invalid/name/extra" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()
