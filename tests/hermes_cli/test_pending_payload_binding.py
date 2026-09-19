"""Reviewed-payload binding and cooperative pending-record consumption."""
import copy

import pytest


@pytest.mark.parametrize("verb", ["approve", "apply", "reject", "deny", "drop"])
def test_reviewed_digest_refuses_replacement_and_accepts_matching_payload(tmp_path, monkeypatch, verb):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MemoryStore()
    store.load_from_disk()
    reviewed = wa.stage_write("memory", {"action": "add", "target": "user", "content": "reviewed"},
                              summary="fixture", origin="foreground")
    digest = wa.payload_sha256(reviewed["payload"])
    replacement = copy.deepcopy(reviewed)
    replacement["payload"]["content"] = "replacement"
    path = tmp_path / "pending" / "memory" / f"{reviewed['id']}.json"
    wa.atomic_json_write(path, replacement)
    before = path.read_bytes()

    result = handle_pending_subcommand("memory", [verb, reviewed["id"]], memory_store=store,
                                       expected_payload_sha256=digest)
    assert result is not None
    assert store.user_entries == []
    assert path.read_bytes() == before
    # The exact-ID token cannot authorize a different envelope (all pending IDs).
    handle_pending_subcommand("memory", [verb, "all"], memory_store=store,
                              expected_payload_sha256=digest)
    assert path.read_bytes() == before
    wa.atomic_json_write(path, reviewed)
    result = handle_pending_subcommand("memory", [verb, reviewed["id"]], memory_store=store,
                                       expected_payload_sha256=digest)
    assert result is not None
    assert not path.exists()
    assert store.user_entries == (["reviewed"] if verb in {"approve", "apply"} else [])


@pytest.mark.parametrize("interference", ["replace", "reject"])
def test_apply_does_not_consume_a_replacement_or_admit_another_consumer(tmp_path, monkeypatch, interference):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MemoryStore()
    store.load_from_disk()
    record = wa.stage_write("memory", {"action": "add", "target": "user", "content": "original"},
                            summary="fixture", origin="foreground")
    path = tmp_path / "pending" / "memory" / f"{record['id']}.json"
    real_apply = commands._apply_one

    def apply_with_interference(subsystem, current, memory_store):
        if interference == "replace":
            replacement = copy.deepcopy(current)
            replacement["payload"]["content"] = "not applied"
            wa.atomic_json_write(path, replacement)
        else:
            # A competing consumer cannot remove a record while it is applying.
            assert wa.discard_pending(subsystem, current["id"]) is False
        return real_apply(subsystem, current, memory_store)

    monkeypatch.setattr(commands, "_apply_one", apply_with_interference)
    result = commands.handle_pending_subcommand("memory", ["approve", record["id"]], memory_store=store)
    assert result is not None
    assert store.user_entries == ["original"]
    if interference == "replace":
        assert "replacement retained" in result
        retained = wa.get_pending("memory", record["id"])
        assert retained is not None
        assert retained["payload"]["content"] == "not applied"
    else:
        assert "Approved 1" in result
        assert not path.exists()
