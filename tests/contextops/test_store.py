from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from contextops.models import ContextPack, Event, StateDelta, Thread
from contextops.store import ContextOpsStore, default_store_root


def test_default_store_root_is_local_contextops_data() -> None:
    assert default_store_root() == Path(".data/contextops")


def test_store_appends_and_reads_jsonl_models(tmp_path: Path) -> None:
    store = ContextOpsStore(root=tmp_path / "contextops")
    event = Event(id="evt-1", source="fixture", text="Operator separated heat from recency.")
    delta = StateDelta(
        id="delta-1",
        kind="belief_shift",
        description="Heat changed due to explicit evidence.",
        evidence_refs=[event.id],
    )

    store.append_jsonl("events", event)
    store.append_jsonl("deltas", delta)

    assert store.read_jsonl("events", Event) == [event]
    assert store.read_jsonl("deltas", StateDelta) == [delta]
    assert (tmp_path / "contextops" / "events.jsonl").read_text().count("\n") == 1


def test_store_writes_yaml_atomically_and_round_trips(tmp_path: Path) -> None:
    store = ContextOpsStore(root=tmp_path / "contextops")
    pack = ContextPack(
        id="pack-1",
        thread_ids=["thread:discord:contextops:msg-42"],
        restore=["Restore unresolved tension."],
        avoid=["Do not treat context pack as transcript."],
    )

    store.write_yaml("packs/pack-1", pack)

    raw = yaml.safe_load((tmp_path / "contextops" / "packs" / "pack-1.yaml").read_text())
    assert raw["id"] == "pack-1"
    assert store.read_yaml("packs/pack-1", ContextPack) == pack
    assert not list((tmp_path / "contextops").glob("*.tmp"))


def test_store_rejects_path_traversal_and_absolute_names(tmp_path: Path) -> None:
    store = ContextOpsStore(root=tmp_path / "contextops")
    thread = Thread(
        id="thread:discord:contextops:msg-42",
        anchor_event_ids=["evt-1"],
        stance="Keep thread distinct from topic.",
    )

    for unsafe_name in ("../escape", "/tmp/escape", "nested/../../escape"):
        try:
            store.append_jsonl(unsafe_name, thread)
        except ValueError as exc:
            assert "store name" in str(exc)
        else:  # pragma: no cover - explicit fail path for readability
            raise AssertionError(f"unsafe name accepted: {unsafe_name}")

    assert not (tmp_path / "escape.jsonl").exists()


def test_store_rejects_forged_invalid_thread_before_append(tmp_path: Path) -> None:
    """model_copy(update=...) skips validators; the write boundary must fail closed."""

    store = ContextOpsStore(root=tmp_path / "contextops")
    valid = Thread(
        id="thread:discord:contextops:msg-42",
        anchor_event_ids=["evt-1"],
        stance="Keep thread distinct from topic.",
    )
    forged = valid.model_copy(update={"id": "pricing"})

    with pytest.raises(ValidationError, match="topic-only"):
        store.append_jsonl("threads", forged)

    assert not (tmp_path / "contextops" / "threads.jsonl").exists()


def test_store_rejects_forged_invalid_state_delta_before_append(tmp_path: Path) -> None:
    store = ContextOpsStore(root=tmp_path / "contextops")
    valid = StateDelta(
        id="delta-1",
        kind="belief_shift",
        description="Heat changed due to explicit evidence.",
        evidence_refs=["evt-1"],
    )
    forged = valid.model_copy(update={"evidence_refs": []})

    with pytest.raises(ValidationError, match="evidence"):
        store.append_jsonl("deltas", forged)

    assert not (tmp_path / "contextops" / "deltas.jsonl").exists()


def test_store_rejects_forged_invalid_pack_before_yaml_write(tmp_path: Path) -> None:
    store = ContextOpsStore(root=tmp_path / "contextops")
    valid = ContextPack(
        id="pack-1",
        thread_ids=["thread:discord:contextops:msg-42"],
        restore=["Restore unresolved tension."],
        avoid=["Do not treat context pack as transcript."],
    )
    forged = valid.model_copy(update={"restore": []})

    with pytest.raises(ValidationError, match="restore"):
        store.write_yaml("packs/pack-1", forged)

    assert not (tmp_path / "contextops" / "packs" / "pack-1.yaml").exists()
