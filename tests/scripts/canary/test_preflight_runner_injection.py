from __future__ import annotations

from scripts.canary import host_preflight, network_preflight


def test_network_collector_propagates_the_same_injected_runner(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def runner(argv):
        calls.append(tuple(argv))
        if argv[1:4] == ("sql", "instances", "describe"):
            return {"ipAddresses": []}
        return []

    nested: list[object] = []
    monkeypatch.setattr(
        network_preflight,
        "collect_foundation",
        lambda *, run_json: nested.append(run_json) or {},
    )
    monkeypatch.setattr(network_preflight, "evaluate_foundation", lambda _value: {})

    network_preflight.collect(run_json=runner)

    assert nested == [runner]
    assert len(calls) == 4
    assert all(call[0] == "gcloud" for call in calls)


def test_host_collector_propagates_the_same_injected_runner(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def runner(argv):
        calls.append(tuple(argv))
        return [] if argv[1:4] == ("compute", "instances", "list") else {}

    nested: list[object] = []
    monkeypatch.setattr(
        host_preflight,
        "collect_network",
        lambda *, run_json: nested.append(run_json) or {},
    )
    monkeypatch.setattr(host_preflight, "evaluate_network", lambda _value: {})

    host_preflight.collect(run_json=runner)

    assert nested == [runner]
    assert calls == [
        (
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project=adventico-ai-platform",
            "--format=json",
        ),
        (
            "gcloud",
            "compute",
            "images",
            "describe",
            "debian-12-bookworm-v20260609",
            "--project=debian-cloud",
            "--format=json",
        ),
    ]
