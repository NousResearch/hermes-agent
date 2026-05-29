"""Fixture-backed acceptance harness for the Memory v2 retrieval contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.retrieval import MemoryPacketComposer, RuleBasedMemoryRouter
from plugins.memory.memory_v2.schemas import MemoryItem, SourceRef

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "memory_v2_benchmark.yaml"


def load_benchmark() -> dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def benchmark() -> dict[str, Any]:
    return load_benchmark()


@pytest.fixture()
def seeded_provider(tmp_path: Path, benchmark: dict[str, Any]) -> MemoryV2Provider:
    provider = MemoryV2Provider()
    provider.initialize("acceptance-session", hermes_home=tmp_path, platform="pytest")

    # Seed source metadata and same-profile memories from the benchmark fixture.
    for source in benchmark["seed_memory"].get("sources", []):
        provider.index.index_source_ref(SourceRef.from_dict(source))
    for memory in benchmark["seed_memory"].get("memories", []):
        provider.index.index_memory_item(MemoryItem.from_dict(memory))

    # Seed per-case default-profile setup claims, but never seed other-profile memories.
    for case in benchmark["cases"]:
        if setup_claim := case.get("setup_existing_claim"):
            provider.index.index_memory_item(MemoryItem.from_dict(setup_claim))

    return provider


def packet_for(provider: MemoryV2Provider, query: str) -> dict[str, Any]:
    composer = MemoryPacketComposer(provider.index)
    rendered = composer.render(composer.compose(query))
    if not rendered:
        return {"route": RuleBasedMemoryRouter().route(query).route, "token_budget": 0, "items": [], "warnings": []}
    return yaml.safe_load(rendered)


def approx_tokens(packet: dict[str, Any]) -> int:
    if not packet.get("items"):
        return 0
    return len(yaml.safe_dump(packet, sort_keys=False)) // 4


def item_ids(packet: dict[str, Any]) -> list[str]:
    return [str(item.get("id")) for item in packet.get("items", [])]


def active_item_ids(packet: dict[str, Any]) -> list[str]:
    return [str(item.get("id")) for item in packet.get("items", []) if item.get("status") == "active"]


def packet_source_ids(packet: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for item in packet.get("items", []):
        source_ids.update(str(ref) for ref in item.get("source_refs", []))
    return source_ids


@pytest.mark.parametrize("case", load_benchmark()["cases"], ids=lambda case: case["id"])
def test_memory_v2_acceptance_case_retrieves_expected_packet(
    seeded_provider: MemoryV2Provider,
    benchmark: dict[str, Any],
    case: dict[str, Any],
) -> None:
    packet = packet_for(seeded_provider, case["query"])
    ids = item_ids(packet)

    assert packet["route"] == case["expected_route"]
    assert approx_tokens(packet) <= int(case["max_packet_tokens"])

    for required_id in case.get("required_memory_ids", []):
        assert required_id in ids

    for forbidden_id in case.get("forbidden_memory_ids", []):
        assert forbidden_id not in ids

    for stale_id in case.get("forbidden_memory_ids_as_active", []):
        assert stale_id not in active_item_ids(packet)

    source_ids = packet_source_ids(packet)
    fixture_source_ids = {source["id"] for source in benchmark["seed_memory"].get("sources", [])}
    for source_id in source_ids:
        assert source_id in fixture_source_ids

    for required_source_id in case.get("required_source_ids", []):
        assert required_source_id in source_ids

    if case.get("source_required") and case.get("required_memory_ids"):
        assert source_ids

    packet_text = yaml.safe_dump(packet, sort_keys=False)
    for expected in case.get("expected_answer_contains", []):
        assert str(expected).lower() in packet_text.lower()
    for forbidden in case.get("expected_answer_not_contains", []):
        assert str(forbidden).lower() not in packet_text.lower()


def test_memory_v2_acceptance_contradiction_case_creates_review_candidate(
    seeded_provider: MemoryV2Provider,
    benchmark: dict[str, Any],
) -> None:
    case = next(case for case in benchmark["cases"] if case["id"] == "contradiction_host_os")
    before = seeded_provider.index.search("Hermes runtime host_environment WSL", route="contradiction_check", limit=5)
    assert any(result["id"] == "env_host_wsl" and result["status"] == "active" for result in before)

    seeded_provider.sync_turn(case["query"], "Queued for memory review.", session_id="acceptance-session")

    candidates = seeded_provider.store.list_candidates()
    assert candidates, "contradiction write should create a pending review candidate"
    latest = candidates[-1].to_dict()
    assert latest["gate_decision"] == "pending"
    assert "macOS" in latest["claim"]
    assert latest["type"] == "environment"
    assert "contradict" in latest["promotion_reason"].lower() or "conflict" in latest["promotion_reason"].lower()

    after = seeded_provider.index.search("Hermes runtime host_environment WSL", route="contradiction_check", limit=5)
    assert any(result["id"] == "env_host_wsl" and result["status"] == "active" for result in after)


def test_memory_v2_acceptance_profile_isolation_does_not_seed_other_profile_memory(
    seeded_provider: MemoryV2Provider,
    benchmark: dict[str, Any],
) -> None:
    case = next(case for case in benchmark["cases"] if case["id"] == "profile_isolation")
    other = case["setup_other_profile_memory"]

    payload = json.loads(seeded_provider.handle_tool_call("memory_v2_search", {"query": other["summary"], "limit": 10}))

    assert payload["success"] is True
    assert other["id"] not in [result["id"] for result in payload["results"]]
