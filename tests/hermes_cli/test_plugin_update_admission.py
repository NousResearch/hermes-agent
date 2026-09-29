import json
from pathlib import Path

import pytest

from hermes_cli.plugin_update_admission import PluginHostContractError, _verify_registry


def test_published_host_contract_registry_is_versioned_data():
    path = Path(__file__).parents[2] / "hermes_cli" / "plugin_host_contracts.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": 1,
        "contracts": {"desktop.plugin_routed_session": [1]},
    }


def test_candidate_must_support_the_exact_required_contract_version():
    requirement = [("required-plugin", "example.capability", 1)]
    with pytest.raises(PluginHostContractError, match="example.capability API 1"):
        _verify_registry(
            '{"schema": 1, "contracts": {"example.capability": [2]}}',
            requirement,
        )
    _verify_registry(
        '{"schema": 1, "contracts": {"example.capability": [1, 2]}}',
        requirement,
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema": 1, "contracts": {"example.capability": [true]}}',
        '{"schema": 1, "contracts": {"example.capability": [0]}}',
        '{"schema": 1, "contracts": {"example.capability": [1, 1]}}',
        '{"schema": 1, "contracts": {"": [1]}}',
    ],
)
def test_candidate_registry_rejects_ambiguous_contract_versions(raw):
    with pytest.raises(PluginHostContractError, match="registry is invalid"):
        _verify_registry(raw, [("required-plugin", "example.capability", 1)])
