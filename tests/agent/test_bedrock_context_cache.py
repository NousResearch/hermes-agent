"""Bedrock cache provenance and compressor budgets across disk-backed restarts.

AWS documents Grok 4.6's Bedrock context window as 500K, independently of
xAI's direct API catalog: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-6.html
Only provider I/O is stubbed; cache/config readers and compressor are real.
"""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agent import bedrock_adapter as ba
from agent import model_metadata as mm
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mm._BEDROCK_PROBE_FAILURE_CACHE.clear()
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda: "us-east-1")
    yield tmp_path
    mm._BEDROCK_PROBE_FAILURE_CACHE.clear()


@pytest.mark.parametrize("model", ["global.xai.grok-4.6"])  # prefix variants resolve identically
@pytest.mark.parametrize("base_url", ["", "https://bedrock-runtime.us-east-1.amazonaws.com"])
@pytest.mark.parametrize("legacy", [None, 128_000, 700_000])
@pytest.mark.parametrize("probed", [None, 128_000, 800_000])
def test_bedrock_resolution_migrates_ambiguous_cache_and_preserves_probe(
    isolated_home, monkeypatch, model, base_url, legacy, probed,
):
    """Neither an old small nor large scalar proves where it came from.

    A successful probe is authoritative, including below-table limits. Its
    provenance must survive unrelated writes and an actual process restart.
    """
    cache_url = base_url or "bedrock://"
    key = mm._context_cache_key(model, cache_url)
    cache_file = isolated_home / "context_length_cache.yaml"
    if legacy is not None:
        cache_file.write_text(yaml.safe_dump({"context_lengths": {key: legacy}}))
    probe = Mock(return_value=probed)
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    expected = probed if probed is not None else 500_000
    compressor = ContextCompressor(model, provider="bedrock", base_url=base_url, quiet_mode=True)
    assert compressor.context_length == expected
    # Preserve the existing raise-only 75% floor for windows below 512K.
    expected_threshold = int(expected * (0.75 if expected < 512_000 else 0.5))
    assert compressor.threshold_tokens == expected_threshold
    assert mm.get_model_context_length(model, provider="bedrock", base_url=base_url) == expected
    probe.assert_called_once_with(model, "us-east-1")
    assert mm.get_cached_context_length(model, cache_url, bedrock_confirmed=True) == probed
    # Ordinary cache updates must not erase another entry's provenance.
    mm.save_context_length("other-model", "https://other.example/v1", 64_000)
    mm._invalidate_cached_context_length("other-model", "https://other.example/v1")
    if probed is None:
        assert yaml.safe_load(cache_file.read_text())["context_lengths"].get(key) == legacy
    else:
        script = '''
import json, sys
from agent import bedrock_adapter as ba
from agent.context_compressor import ContextCompressor
ba.probe_bedrock_context_length = lambda *a, **k: (_ for _ in ()).throw(AssertionError("reprobed persisted success"))
c = ContextCompressor(sys.argv[1], provider="bedrock", base_url=sys.argv[2], quiet_mode=True)
print(json.dumps([c.context_length, c.threshold_tokens]))
'''
        result = subprocess.run([sys.executable, "-c", script, model, base_url],
                                env=dict(os.environ), text=True, capture_output=True, check=True)
        assert json.loads(result.stdout) == [expected, expected_threshold]


@pytest.mark.parametrize("base_url", ["", "https://bedrock-runtime.us-east-1.amazonaws.com"])
@pytest.mark.parametrize("retry", ["ttl", "invalidate", "restart", "profile", "endpoint"])
def test_failure_memo_retry_scope_and_expiry(isolated_home, monkeypatch, base_url, retry):
    model = "global.xai.grok-4.6"
    cache_url = base_url or "bedrock://"
    probe = Mock(side_effect=[None, 96_000])
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    assert mm.get_model_context_length(model, provider="bedrock", base_url=base_url) == 500_000
    assert mm.get_model_context_length(model, provider="bedrock", base_url=base_url) == 500_000
    assert probe.call_count == 1
    assert not (isolated_home / "context_length_cache.yaml").exists()
    if retry == "ttl":
        expired = time.monotonic() - mm._BEDROCK_PROBE_FAILURE_TTL_SECONDS - 1
        for key in mm._BEDROCK_PROBE_FAILURE_CACHE:
            mm._BEDROCK_PROBE_FAILURE_CACHE[key] = expired
    elif retry == "invalidate":
        mm._invalidate_cached_context_length(model, cache_url)
    elif retry == "restart":
        mm._BEDROCK_PROBE_FAILURE_CACHE.clear()
    elif retry == "profile":
        new_home = isolated_home / "other-profile"
        new_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(new_home))
    else:
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com/other"
    assert mm.get_model_context_length(model, provider="bedrock", base_url=base_url) == 96_000
    assert probe.call_count == 2
    if retry == "ttl":
        assert not mm._BEDROCK_PROBE_FAILURE_CACHE


@pytest.mark.parametrize("base_url", ["", "https://bedrock-runtime.us-east-1.amazonaws.com"])
@pytest.mark.parametrize("override", ["argument", "config"])
def test_explicit_context_override_and_compressor_caps_win(isolated_home, monkeypatch, base_url, override):
    model = "us.xai.grok-4.6"
    probe = Mock(side_effect=AssertionError("explicit override must not probe"))
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    explicit_context = 80_000 if override == "argument" else None
    if override == "config":
        (isolated_home / "config.yaml").write_text(yaml.safe_dump({
            "model_overrides": {"bedrock": {model: {"context_window": 80_000}}},
        }))
    compressor = ContextCompressor(model, provider="bedrock", base_url=base_url,
                                   threshold_tokens_cap=30_000, max_tokens=10_000,
                                   quiet_mode=True, config_context_length=explicit_context)
    assert compressor.context_length == 80_000
    assert compressor.threshold_tokens == 30_000
    assert not (isolated_home / "context_length_cache.yaml").exists()
    probe.assert_not_called()


def test_unknown_model_fallback_and_host_inference(monkeypatch):
    probe = Mock(return_value=None)
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert mm.get_model_context_length("unknown.future-model", base_url=base_url) == ba.BEDROCK_DEFAULT_CONTEXT_LENGTH
    assert mm.get_model_context_length("xai.grok-4.6", base_url=base_url) == 500_000
    assert mm.get_cached_context_length("unknown.future-model", base_url) is None


@pytest.mark.parametrize("base_url", ["", "https://bedrock-runtime.us-east-1.amazonaws.com"])
@pytest.mark.parametrize("writer", ["overflow", "usage"])
def test_provider_confirmed_writers_survive_restart(monkeypatch, base_url, writer):
    from agent.turn_overflow import _adopt_provider_context_limit
    from agent.turn_usage import record_response_usage

    model = "global.xai.grok-4.6"
    compressor = ContextCompressor(model, base_url=base_url, provider="bedrock", quiet_mode=True)
    compressor.context_length = 500_000
    agent = SimpleNamespace(model=model, provider="bedrock", api_mode="bedrock", base_url=base_url,
                            context_compressor=compressor, _buffer_vprint=lambda *a: None,
                            _safe_print=lambda *a: None, log_prefix="", client=None,
                            _session_db=None, verbose_logging=False, quiet_mode=True,
                            session_api_calls=0, session_estimated_cost_usd=0)
    if writer == "overflow":
        assert _adopt_provider_context_limit(SimpleNamespace(agent=agent),
            "maximum context length is 96000 tokens", 500_000) == 96_000
    else:
        compressor.context_length = 96_000
        compressor._context_probed = compressor._context_probe_persistable = True
        for name in ("prompt", "completion", "total", "input", "output", "cache_read", "cache_write", "reasoning"):
            setattr(agent, f"session_{name}_tokens", 0)
        response = SimpleNamespace(usage={"input_tokens": 100, "output_tokens": 5})
        record_response_usage(agent, response, messages=[{"role": "user", "content": "hi"}],
                              api_call_count=1, api_duration=0.1, compression_attempts=0, max_compression_attempts=3)
    assert mm.get_cached_context_length(model, base_url or "bedrock://") == 96_000
    script = '''
import sys
from agent import model_metadata as mm, bedrock_adapter as ba
ba.probe_bedrock_context_length = lambda *a, **k: (_ for _ in ()).throw(AssertionError("lost provider limit"))
assert mm.get_model_context_length(sys.argv[1], base_url=sys.argv[2], provider="bedrock") == 96000
'''
    subprocess.run([sys.executable, "-c", script, model, base_url], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("base_url", ["", "https://bedrock-runtime.us-east-1.amazonaws.com"])
def test_readonly_legacy_cache_does_not_reset_probe_cooldown(isolated_home, monkeypatch, base_url):
    model = "global.xai.grok-4.6"
    key = mm._context_cache_key(model, base_url or "bedrock://")
    cache_file = isolated_home / "context_length_cache.yaml"
    cache_file.write_text(yaml.safe_dump({"context_lengths": {key: 128_000}}))
    probe = Mock(return_value=None)
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    monkeypatch.setattr(mm, "_write_context_cache", Mock(side_effect=OSError("read-only")))
    for _ in range(3):
        assert mm.get_model_context_length(model, provider="bedrock", base_url=base_url) == 500_000
    assert probe.call_count == 1


@pytest.mark.parametrize("rewrite", [None, 96_000, 120_000])
def test_provenance_is_backward_readable_and_generic_writes_clear_it(isolated_home, monkeypatch, rewrite):
    model, base_url = "xai.grok-4.6", "https://bedrock-runtime.us-east-1.amazonaws.com"
    probe = Mock(side_effect=[96_000, 110_000])
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    assert mm.get_model_context_length(model, base_url=base_url) == 96_000
    raw = yaml.safe_load((isolated_home / "context_length_cache.yaml").read_text())
    # Old readers take this value directly into arithmetic. Metadata is additive.
    assert raw["context_lengths"][mm._context_cache_key(model, base_url)] + 1 == 96_001
    if rewrite is not None:
        mm.save_context_length(model, base_url, rewrite)
        assert mm.get_model_context_length(model, base_url=base_url) == 110_000
    else:
        assert mm.get_model_context_length(model, base_url=base_url) == 96_000


@pytest.mark.parametrize("lengths", [True, [128_000], "bad"])
def test_malformed_lengths_do_not_block_provider_persistence(isolated_home, monkeypatch, lengths):
    (isolated_home / "context_length_cache.yaml").write_text(yaml.safe_dump({"context_lengths": lengths}))
    monkeypatch.setattr(ba, "probe_bedrock_context_length", lambda *a: 96_000)
    assert mm.get_model_context_length("xai.grok-4.6", provider="bedrock") == 96_000
    assert mm.get_cached_context_length("xai.grok-4.6", "bedrock://") == 96_000


@pytest.mark.parametrize("marker", [True, "96000", 128_000, [96_000], {"source": "probe"}])
def test_malformed_or_mismatched_provenance_requires_revalidation(isolated_home, monkeypatch, marker):
    model, base_url = "xai.grok-4.6", "bedrock://"
    key = mm._context_cache_key(model, base_url)
    (isolated_home / "context_length_cache.yaml").write_text(yaml.safe_dump({
        "context_lengths": {key: 96_000}, "bedrock_confirmed_v1": {key: marker},
    }))
    probe = Mock(return_value=100_000)
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    assert mm.get_model_context_length(model, provider="bedrock") == 100_000
    probe.assert_called_once()


def test_context_local_profile_memos_do_not_cross_and_expired_rows_are_pruned(isolated_home, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    probe = Mock(side_effect=[None, None, 96_000])
    monkeypatch.setattr(ba, "probe_bedrock_context_length", probe)
    model = "global.xai.grok-4.6"
    assert mm.get_model_context_length(model, provider="bedrock") == 500_000
    # Leave a different model's expired row: lookup must prune it too.
    assert mm.get_model_context_length("unknown.future", provider="bedrock") == ba.BEDROCK_DEFAULT_CONTEXT_LENGTH
    for key in mm._BEDROCK_PROBE_FAILURE_CACHE:
        if "unknown.future" in key:
            mm._BEDROCK_PROBE_FAILURE_CACHE[key] = time.monotonic() - mm._BEDROCK_PROBE_FAILURE_TTL_SECONDS - 1
    token = set_hermes_home_override(isolated_home / "routed-profile")
    try:
        assert mm.get_model_context_length(model, provider="bedrock") == 96_000
    finally:
        reset_hermes_home_override(token)
    assert probe.call_count == 3
    assert all("unknown.future" not in key for key in mm._BEDROCK_PROBE_FAILURE_CACHE)
    assert mm.get_model_context_length(model, provider="bedrock") == 500_000
    assert probe.call_count == 3
