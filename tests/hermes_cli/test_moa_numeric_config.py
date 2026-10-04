"""Malformed numeric MoA settings must degrade to defaults, not break the CLI or JSON."""

import json
import math

import pytest

from hermes_cli.moa_config import normalize_moa_config


@pytest.mark.parametrize("fanout", [
    {"mode": "every_n", "n": float("inf")},
    {"mode": "every_n", "n": "-inf"},
    "every_n:inf",
    "every_n:nan",
])
def test_nonfinite_fanout_falls_back_to_default_cadence(fanout):
    assert normalize_moa_config({"fanout": fanout})["fanout"] == normalize_moa_config({})["fanout"]
    assert normalize_moa_config({"fanout": {"mode": "every_n", "n": "3.0"}})["fanout"] == "every_n:3"


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), "-inf", "NaN"])
def test_nonfinite_temperatures_do_not_escape_into_json(temperature):
    normalized = normalize_moa_config({
        "reference_temperature": temperature,
        "aggregator_temperature": temperature,
    })
    assert normalized["reference_temperature"] is None
    assert normalized["aggregator_temperature"] is None
    json.dumps(normalized, allow_nan=False)

    finite = normalize_moa_config({"reference_temperature": 0, "aggregator_temperature": "0.75"})
    assert finite["reference_temperature"] == 0 and math.isfinite(finite["reference_temperature"])
    assert finite["aggregator_temperature"] == 0.75
