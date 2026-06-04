from types import SimpleNamespace

from gateway.run import (
    _compression_calibration_from_agent,
    _compression_calibration_from_session_entry,
    _hydrate_gateway_compression_calibration,
)


def test_hydrates_gateway_compression_calibration_fields():
    compressor = SimpleNamespace(
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        last_compression_rough_tokens=0,
        last_rough_tokens_when_real_prompt_fit=0,
    )
    agent = SimpleNamespace(context_compressor=compressor)
    entry = SimpleNamespace(
        last_prompt_tokens=91_000,
        last_real_prompt_tokens=90_000,
        last_compression_rough_tokens=139_000,
        last_rough_tokens_when_real_prompt_fit=139_000,
    )

    _hydrate_gateway_compression_calibration(agent, entry)

    assert compressor.last_prompt_tokens == 91_000
    assert compressor.last_real_prompt_tokens == 90_000
    assert compressor.last_compression_rough_tokens == 139_000
    assert compressor.last_rough_tokens_when_real_prompt_fit == 139_000


def test_hydrates_gateway_compression_calibration_from_snapshot():
    compressor = SimpleNamespace(
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        last_compression_rough_tokens=0,
        last_rough_tokens_when_real_prompt_fit=0,
    )
    agent = SimpleNamespace(context_compressor=compressor)
    snapshot = {
        "last_prompt_tokens": 91_000,
        "last_real_prompt_tokens": 90_000,
        "last_compression_rough_tokens": 139_000,
        "last_rough_tokens_when_real_prompt_fit": 139_000,
    }

    _hydrate_gateway_compression_calibration(agent, snapshot)

    assert compressor.last_prompt_tokens == 91_000
    assert compressor.last_real_prompt_tokens == 90_000
    assert compressor.last_compression_rough_tokens == 139_000
    assert compressor.last_rough_tokens_when_real_prompt_fit == 139_000


def test_hydrates_legacy_last_prompt_as_real_fit_signal():
    compressor = SimpleNamespace(
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        last_compression_rough_tokens=0,
        last_rough_tokens_when_real_prompt_fit=0,
    )
    agent = SimpleNamespace(context_compressor=compressor)
    entry = SimpleNamespace(last_prompt_tokens=220_000)

    _hydrate_gateway_compression_calibration(agent, entry)

    assert compressor.last_prompt_tokens == 220_000
    assert compressor.last_real_prompt_tokens == 220_000


def test_extracts_gateway_compression_calibration_from_session_entry():
    entry = SimpleNamespace(
        last_prompt_tokens=98_000,
        last_real_prompt_tokens=97_000,
        last_compression_rough_tokens=140_000,
        last_rough_tokens_when_real_prompt_fit=141_000,
    )

    assert _compression_calibration_from_session_entry(entry) == {
        "last_prompt_tokens": 98_000,
        "last_real_prompt_tokens": 97_000,
        "last_compression_rough_tokens": 140_000,
        "last_rough_tokens_when_real_prompt_fit": 141_000,
    }


def test_extracts_gateway_compression_calibration_fields():
    compressor = SimpleNamespace(
        last_prompt_tokens=98_000,
        last_real_prompt_tokens=98_000,
        last_compression_rough_tokens=140_000,
        last_rough_tokens_when_real_prompt_fit=140_000,
    )
    agent = SimpleNamespace(context_compressor=compressor)

    assert _compression_calibration_from_agent(agent) == {
        "last_prompt_tokens": 98_000,
        "last_real_prompt_tokens": 98_000,
        "last_compression_rough_tokens": 140_000,
        "last_rough_tokens_when_real_prompt_fit": 140_000,
    }
