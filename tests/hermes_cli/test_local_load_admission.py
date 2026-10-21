"""A cold load that does not fit in the memory free right now is refused before the request (#142082).

The router autoloads any model a request names. With another program holding RAM/VRAM (a model in
LM Studio), a capacity-priced preset loaded anyway and paged a Windows machine to a halt for minutes.
"""
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import admission, presets
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.local_runtime.presets import PresetEntry

_G = 1 << 30
CAPACITY = HardwareBudget(usable_vram_bytes=6 * _G, total_device_bytes=8 * _G, ram_available_bytes=64 * _G)


def _entry(model_id, path):
    return PresetEntry(model_id=model_id, window=65536, spilled=True, keys={"model": str(path)})


@pytest.fixture
def priced(tmp_path, monkeypatch):
    """Two staged models, priced by size from their file name: big = 22 GiB, small = 5 GiB."""
    sizes = {"big": 22 * _G, "small": 5 * _G}
    monkeypatch.setattr(presets, "resident_footprint",
                        lambda gguf, budget, window, drafting=False: sizes[gguf.stem])
    return {m: _entry(m, tmp_path / f"{m}.gguf") for m in sizes}


def _live(free_vram, free_ram):
    return HardwareBudget(usable_vram_bytes=free_vram * _G, total_device_bytes=8 * _G,
                          ram_available_bytes=free_ram * _G)


def test_cold_load_beyond_free_memory_is_refused(priced):
    refusal = admission.admission_refusal("big", {"big": "unloaded"}, priced, 1, CAPACITY, _live(1, 12))
    assert refusal and "big" in refusal and "22.0 GiB" in refusal


def test_cold_load_that_fits_is_admitted(priced):
    assert admission.admission_refusal("small", {"small": "unloaded"}, priced, 1, CAPACITY, _live(1, 12)) is None


def test_a_resident_model_is_never_refused(priced):
    # No memory left, but the request loads nothing.
    assert admission.admission_refusal("big", {"big": "loaded"}, priced, 1, CAPACITY, _live(0, 0)) is None


def test_eviction_at_the_cap_counts_as_free(priced):
    statuses = {"small": "loaded", "big": "unloaded"}
    # At the cap the router evicts "small" before loading "big": 5 GiB come back.
    assert admission.admission_refusal("big", statuses, priced, 1, CAPACITY, _live(1, 17)) is None
    # Below the cap nothing is evicted, so the same free memory is not enough.
    assert admission.admission_refusal("big", statuses, priced, 2, CAPACITY, _live(1, 17))


def test_unpriceable_load_fails_open(priced):
    assert admission.admission_refusal("unknown", {"unknown": "unloaded"}, priced, 1, CAPACITY, _live(0, 0)) is None


def test_probe_failure_fails_open(monkeypatch):
    from hermes_cli.local_runtime import endpoint

    monkeypatch.setattr(endpoint, "managed_root", lambda: ("http://127.0.0.1:1", "k"))
    monkeypatch.setattr(endpoint, "managed_get_json", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert admission.check_load_admission("big") is None


def test_request_to_the_managed_server_raises_before_sending(monkeypatch):
    from agent import chat_completion_local
    from agent.errors import LocalModelAdmissionError
    from hermes_cli.local_runtime import recovery

    monkeypatch.setattr(recovery, "read_state", lambda: {"base_url": "http://127.0.0.1:18434/v1"})
    monkeypatch.setattr(admission, "check_load_admission", lambda model: f"loading {model} needs more")
    managed = SimpleNamespace(base_url="http://127.0.0.1:18434/v1")
    with pytest.raises(LocalModelAdmissionError, match="loading big"):
        chat_completion_local.refuse_unfitting_local_load(managed, {"model": "big"})
    # An external llama-server is not ours to judge.
    chat_completion_local.refuse_unfitting_local_load(
        SimpleNamespace(base_url="http://127.0.0.1:8080/v1"), {"model": "big"})


def test_refusal_classifies_as_non_retryable_with_fallback():
    from agent.error_classifier import FailoverReason, classify_api_error
    from agent.errors import LocalModelAdmissionError
    from agent.turn_failure_copy import nonretryable_copy

    result = classify_api_error(LocalModelAdmissionError("loading big needs ~22.0 GiB"), provider="llamacpp")
    assert result.reason == FailoverReason.local_memory_insufficient
    assert result.retryable is False and result.should_fallback is True
    copy = nonretryable_copy(result, provider="llamacpp", model="big", summary="loading big needs ~22.0 GiB")
    assert "big" in copy and "22.0 GiB" in copy
