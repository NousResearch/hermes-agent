"""Offline usage accounting and installed-helper CLI contract."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SKILL = Path(__file__).resolve().parents[2] / "optional-skills/devops/prompt-cache-tuning"
SCRIPT = SKILL / "scripts/summarize_usage.py"
spec = importlib.util.spec_from_file_location("prompt_cache_usage", SCRIPT)
assert spec is not None and spec.loader is not None
usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usage)


def report(records, shape="chat"):
    return usage.summarize((json.dumps(record) for record in records), shape)


@pytest.mark.parametrize("shape,record", [
    ("chat", {"prompt_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 1500}}),
    ("chat", {"prompt_tokens": 2000, "prompt_cache_hit_tokens": 1500}),
    ("chat", {"input_tokens": 2000, "cache_read_input_tokens": 1500}),
    ("chat", {"prompt_tokens": 2000, "cached_tokens": 1500}),
    ("responses", {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 1500}}),
    ("anthropic", {"input_tokens": 500, "cache_read_input_tokens": 1500,
                   "cache_creation_input_tokens": 0}),
])
def test_wire_shapes_have_the_same_inclusive_denominator(shape, record):
    result = report([{"usage": record}], shape)
    assert result["cache_read_fraction"] == 0.75
    assert result["per_call"][0]["prompt_tokens"] == 2000
    assert result["measured_calls"] == 1


def test_ratio_is_token_weighted_and_unknown_is_not_zero():
    result = report([
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}},
        {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 900}},
        {"prompt_tokens": 4000},
    ])
    assert result["cache_read_fraction"] == 0.9
    assert (result["calls"], result["measured_calls"], result["unmeasured_calls"]) == (3, 2, 1)
    assert result["calls_with_missing_cache_reads"] == 1
    assert result["per_call"][2]["cache_read_tokens"] is None


def test_explicit_zero_wins_over_a_conflicting_alias():
    result = report([{"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0},
                      "prompt_cache_hit_tokens": 90}])
    assert result["cache_read_fraction"] == 0
    assert result["calls_with_reported_cache_reads"] == 0


@pytest.mark.parametrize("record,shape", [
    ({"prompt_tokens": 10}, "chat"),
    ({"input_tokens": 10, "cache_read_input_tokens": 5}, "anthropic"),
    ({"prompt_tokens": 0, "cached_tokens": 0}, "chat"),
    ({"prompt_tokens": None, "cached_tokens": 0}, "chat"),
])
def test_unmeasurable_or_zero_total_has_no_ratio(record, shape):
    assert report([record], shape)["cache_read_fraction"] is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "20"])
def test_invalid_counter_is_rejected(value):
    with pytest.raises(ValueError, match="line 1: token counters"):
        report([{"prompt_tokens": 100, "cached_tokens": value}])


def test_native_cache_writes_are_included_in_total():
    result = report([{"input_tokens": 100, "cache_read_input_tokens": 200,
                      "cache_creation_input_tokens": 700}], "anthropic")
    assert result["cache_read_fraction"] == 0.2
    assert result["per_call"][0]["cache_write_tokens"] == 700


@pytest.mark.parametrize("shape,record", [
    ("chat", {"prompt_tokens": 100, "cached_tokens": 101}),
    ("responses", {"input_tokens": 100, "input_tokens_details": {
        "cached_tokens": 90, "cache_creation_tokens": 20}}),
])
def test_inconsistent_inclusive_buckets_fail(shape, record):
    with pytest.raises(ValueError, match="cache buckets exceed"):
        report([record], shape)


@pytest.mark.parametrize("text", ["", "\n\n", "secret-not-json", "[]", '{"usage": null}'])
def test_invalid_records_fail_without_echoing_contents(text):
    with pytest.raises(ValueError) as exc:
        usage.summarize(text.splitlines(), "chat")
    assert "secret-not-json" not in str(exc.value)


def test_cli_is_read_only_deterministic_and_portable(tmp_path):
    path = tmp_path / "usage sample.jsonl"
    original = '\ufeff{"usage":{"prompt_tokens":100,"cached_tokens":80},"secret":"DO-NOT-ECHO"}\n\n'
    path.write_text(original, encoding="utf-8")
    command = [sys.executable, str(SCRIPT), str(path), "--format", "chat"]
    first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    second = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["cache_read_fraction"] == 0.8
    assert "DO-NOT-ECHO" not in first.stdout + first.stderr
    assert path.read_text(encoding="utf-8") == original


def test_cli_stdin_and_error_are_safe():
    command = [sys.executable, str(SCRIPT), "-", "--format", "responses"]
    success = subprocess.run(command, input='{"input_tokens":100,"input_tokens_details":{"cached_tokens":90}}\n',
                             capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(success.stdout)["cache_read_fraction"] == 0.9
    failure = subprocess.run(command, input="\nSECRET\n", capture_output=True, text=True, encoding="utf-8", check=False)
    assert failure.returncode == 2
    assert "line 2: invalid JSON" in failure.stderr
    assert "SECRET" not in failure.stderr
    assert failure.stdout == ""


def test_missing_file_returns_a_safe_cli_error(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "missing"),
                             "--format", "chat"], capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "cannot read usage file" in result.stderr
