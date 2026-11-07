"""Regression tests for the OpenAI-compatible ``/models`` catalog parse.

A gateway is free to ship keys whose values are lists or dicts (magpie publishes
``"modalities": {"input": ["text", "image"]}``). The pricing guard used set membership
(``value not in {None, ""}``), which raises TypeError on an unhashable value, so one such key
anywhere in one entry aborted ``_parse_models_payload``; the caller read the resulting empty
mapping as a failed probe and fell back to 256K while the picker discovered no models (#137991).
"""

import pytest


def test_parse_models_payload_survives_list_valued_entry_keys():
    from agent import model_metadata as mm

    payload = {"data": [{
        "id": "magpie-chat",
        "context_length": 1_000_000,
        "modalities": {"input": ["text", "image"]},
        "tags": ["fast"],
        "pricing": {"prompt": 0.6, "completion": 1.2},
    }]}
    catalog = mm._parse_models_payload(payload)
    assert catalog["magpie-chat"]["context_length"] == 1_000_000
    assert float(catalog["magpie-chat"]["pricing"]["prompt"]) == pytest.approx(0.0000006)


def test_extract_pricing_ignores_non_scalar_values():
    from agent import model_metadata as mm

    # Neither a list nor a dict is a rate: both aliases are skipped, nothing raises.
    assert mm._extract_pricing({"id": "m", "input": ["text"], "output": {"nested": 1}}) == {}
    # A scalar under the same key is still read (the guard narrows, it does not disable).
    completion = mm._extract_pricing({"id": "m", "input": ["text"], "output": 0.5})
    assert set(completion) == {"completion"}
    assert float(completion["completion"]) == pytest.approx(0.0000005)


def test_extract_pricing_keeps_zero_rates():
    from agent import model_metadata as mm

    # A free model quotes 0; the old membership test kept it and so must the scalar guard.
    assert mm._extract_pricing({"id": "m", "prompt": 0, "completion": 0}) == {"prompt": 0, "completion": 0}
