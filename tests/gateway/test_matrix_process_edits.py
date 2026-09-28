"""Matrix edit follow-ups are configured independently for each room and adapter."""

import os

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter, _apply_yaml_config


@pytest.mark.parametrize("policy", [True, False, "true", ["!room:example.org"], {"*": True}, {"!room:example.org": "true"}])
def test_process_edits_requires_exact_room_ids_and_booleans(policy):
    config = PlatformConfig(extra={"process_edits": policy})
    with pytest.raises(ValueError, match="matrix.process_edits must map exact room IDs to true or false"):
        MatrixAdapter(config)


def test_yaml_room_opt_in_remains_local_to_the_owning_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_PROCESS_EDITS", "true")
    policies = [
        {"!first:example.org": True, "!disabled:example.org": False},
        {"!second:example.org": True},
    ]
    configs = [PlatformConfig(extra=_apply_yaml_config({}, {"process_edits": policy}) or {}) for policy in policies]
    adapters = [MatrixAdapter(config) for config in configs]

    observed = [sorted(adapter._process_edits) for adapter in [adapters[0], adapters[1], adapters[0]]]
    assert (observed, os.environ["MATRIX_PROCESS_EDITS"]) == (
        [["!first:example.org"], ["!second:example.org"], ["!first:example.org"]], "true",
    )
