"""Regression for least-used selection across fresh runtime pool loads."""

from __future__ import annotations

import json

import pytest

from agent.credential_pool import load_pool


@pytest.mark.parametrize("strategy", ["least_used", "round_robin", "fill_first"])
def test_strategy_is_preserved_across_fresh_pool_loads(tmp_path, monkeypatch, strategy):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy", lambda _provider: strategy
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda _provider, _entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_env",
        lambda _provider, _entries: (False, set()),
    )

    rows = [
        {
            "id": f"key-{index}",
            "label": f"key-{index}",
            "auth_type": "api_key",
            "priority": index,
            "source": "manual",
            "access_token": f"token-{index}",
            "request_count": 0,
        }
        for index in range(4)
    ]
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"anthropic": rows}}),
        encoding="utf-8",
    )

    selected = []
    for _ in range(8):
        entry = load_pool("anthropic").select()
        assert entry is not None
        selected.append(entry.label)

    persisted_rows = json.loads((home / "auth.json").read_text(encoding="utf-8"))[
        "credential_pool"
    ]["anthropic"]
    if strategy == "fill_first":
        assert selected == ["key-0"] * 8
        assert [row["request_count"] for row in persisted_rows] == [0] * 4
    else:
        assert selected == [f"key-{index}" for _ in range(2) for index in range(4)]
        assert [row["request_count"] for row in persisted_rows] == [2] * 4
