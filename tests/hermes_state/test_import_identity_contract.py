"""Import fences must use the canonical IDs the import actually writes."""
import pytest

from hermes_state_lineage import SessionIdOwnedElsewhereError
from hermes_state_runtime import begin_runtime_epoch
from tests.hermes_state import test_mutation_write_ownership as ownership

stores = ownership.stores


@pytest.mark.parametrize("raw_id", [" 123 ", 123])
def test_runtime_import_fences_canonical_payload_id(stores, raw_id):
    local, sibling = stores
    epoch = begin_runtime_epoch(local, instance_id="identity-contract")
    ownership.foreign(sibling, "123")
    before = ownership.snapshot(local), ownership.snapshot(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        ownership.mutate(local, epoch, "new", "import", {"sessions": [
            {"id": "new", "source": "cli", "messages": []},
            {"id": raw_id, "source": "cli", "messages": []},
        ]})
    assert (ownership.snapshot(local), ownership.snapshot(sibling)) == before


@pytest.mark.parametrize("raw_id", [" 123 ", 123])
def test_uncontested_runtime_import_keeps_normalization_contract(stores, raw_id):
    local, sibling = stores
    epoch = begin_runtime_epoch(local, instance_id="identity-control")
    before_foreign = ownership.snapshot(sibling)
    result = ownership.mutate(local, epoch, "new", "import", {"sessions": [
        {"id": "new", "source": "cli", "messages": []},
        {"id": raw_id, "source": "cli", "messages": []},
    ]})
    assert result["imported"] == 2
    assert local.get_session("123") is not None
    assert ownership.snapshot(sibling) == before_foreign


def test_runtime_import_preserves_rejection_of_non_list_collection(stores):
    from hermes_state_runtime import RuntimeStoreError

    local, sibling = stores
    epoch = begin_runtime_epoch(local, instance_id="input-shape-control")
    before = ownership.snapshot(local), ownership.snapshot(sibling)
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        ownership.mutate(local, epoch, "new", "import", {"sessions": (
            {"id": "new", "source": "cli", "messages": []},
        )})
    assert (ownership.snapshot(local), ownership.snapshot(sibling)) == before
