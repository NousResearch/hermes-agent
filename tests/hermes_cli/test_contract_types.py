"""Behavioral tests for documented shared-contract generation."""

from typing import NotRequired, TypedDict

import ts_type as ts

from hermes_cli.contract_types import DocumentedContractBuilder, OpaqueValue


class DocumentedContract(TypedDict, total=False):
    """A contract-level docstring with a slash: */ must stay safe."""

    name: str
    """A human-readable name."""
    payload: OpaqueValue
    """Data whose schema Hermes intentionally does not promise."""


class MixedRequirednessContract(TypedDict):
    required: str
    optional: NotRequired[int]


def test_documented_contract_builder_renders_pep_257_docs_and_unknown_values():
    generator = ts.TypeDefinitionGenerator()
    generator.add(DocumentedContract, "gateway", "DocumentedContract")

    source = generator.render(builder_cls=DocumentedContractBuilder)["gateway"]

    assert "/** A contract-level docstring with a slash: *\\/ must stay safe. */" in source
    assert "    /** A human-readable name. */\n    \"name\"?: string;" in source
    assert (
        "    /** Data whose schema Hermes intentionally does not promise. */\n"
        '    "payload"?: unknown;'
    ) in source


def test_documented_contract_builder_unwraps_python_optional_field_markers():
    generator = ts.TypeDefinitionGenerator()
    generator.add(MixedRequirednessContract, "gateway", "MixedRequirednessContract")

    source = generator.render(builder_cls=DocumentedContractBuilder)["gateway"]

    assert '"required": string;' in source
    assert '"optional"?: number;' in source
