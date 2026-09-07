"""Cross-repo contract conformance: docs/relay-connector-contract.md ⟷ Python.

The contract doc is the formal interface the connector repo
(NousResearch/gateway-gateway) implements against. The connector's TypeScript
structs are hand-mirrored from the doc, so if the Python source of truth drifts
from the doc, the two repos silently diverge and the handshake / session-keying
breaks only at integration time.

These tests make the doc ⟷ code relationship an enforced invariant:

  * Every ``CapabilityDescriptor`` field (§2 table) is documented with the
    correct required/optional flag, and the doc lists no fields the dataclass
    lacks.
  * Every ``SessionSource`` wire key (what ``to_dict()`` actually serializes)
    is named in the contract doc's §3 discriminator section, and every
    discriminator the doc calls out as a column header exists on the dataclass.

They are invariants, NOT change-detector snapshots: they assert the *relation*
between two artifacts that must move together, not a frozen list of names. Add
a field to the descriptor and the doc, and the test stays green; add it to only
one, and CI fails — which is exactly the lockstep guarantee the plan's
Cross-Repo Coordination Checklist calls for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gateway.relay.descriptor import CapabilityDescriptor
from gateway.session import SessionSource

# Repo root: tests/gateway/relay/ -> repo root is parents[3]
_CONTRACT_DOC = (
    Path(__file__).resolve().parents[3] / "docs" / "relay-connector-contract.md"
)


def _doc_text() -> str:
    assert _CONTRACT_DOC.exists(), (
        f"Contract doc missing at {_CONTRACT_DOC}. It is the formal cross-repo "
        f"interface (Phase 1, Task 1.5) and must ship with the relay adapter."
    )
    return _CONTRACT_DOC.read_text(encoding="utf-8")


def _parse_descriptor_table(text: str) -> dict[str, bool]:
    """Parse §2's markdown table → {field_name: required}.

    Rows look like: ``| `field` | type | yes|no | meaning |``. Returns a map of
    field name to whether the Required column says "yes".
    """
    fields: dict[str, bool] = {}
    # Restrict to the §2 section so §3/§4 tables don't bleed in.
    section = text.split("## 2. CapabilityDescriptor", 1)[-1].split("## 3.", 1)[0]
    row_re = re.compile(r"^\|\s*`([a-z_]+)`\s*\|[^|]*\|\s*(yes|no)\s*\|", re.M)
    for name, required in row_re.findall(section):
        fields[name] = required.strip() == "yes"
    return fields


def test_descriptor_fields_match_contract_doc():
    """§2 table ⟷ CapabilityDescriptor dataclass, names + required/optional."""
    documented = _parse_descriptor_table(_doc_text())
    assert documented, "Failed to parse any descriptor fields from the §2 table."

    dc_fields = CapabilityDescriptor.__dataclass_fields__  # type: ignore[attr-defined]
    # A dataclass field is "required" iff it has no default and no default_factory.
    import dataclasses

    code_required = {
        name
        for name, f in dc_fields.items()
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    code_names = set(dc_fields.keys())
    doc_names = set(documented.keys())

    missing_from_doc = code_names - doc_names
    assert not missing_from_doc, (
        f"CapabilityDescriptor fields missing from the §2 contract-doc table: "
        f"{sorted(missing_from_doc)}. Document them so the connector mirrors them."
    )
    extra_in_doc = doc_names - code_names
    assert not extra_in_doc, (
        f"Contract-doc §2 table documents fields the dataclass does not have: "
        f"{sorted(extra_in_doc)}. Remove them or add them to descriptor.py."
    )

    # Required/optional must agree, so the connector knows which fields it may omit.
    for name, doc_required in documented.items():
        assert doc_required == (name in code_required), (
            f"Field '{name}': contract doc says required={doc_required}, but the "
            f"dataclass says required={name in code_required}. Reconcile them."
        )


def _session_source_wire_keys() -> set[str]:
    """Keys ``SessionSource.to_dict()`` can emit (the actual wire surface).

    Build a maximally-populated source so conditionally-included keys (the
    ``if self.x:`` branches in ``to_dict``) all appear.
    """
    from gateway.config import Platform

    src = SessionSource(
        platform=Platform.DISCORD,
        chat_id="c",
        chat_name="n",
        chat_type="channel",
        user_id="u",
        user_name="un",
        thread_id="t",
        chat_topic="topic",
        user_id_alt="ua",
        chat_id_alt="ca",
        guild_id="g",
        parent_chat_id="p",
        message_id="m",
    )
    return set(src.to_dict().keys())


