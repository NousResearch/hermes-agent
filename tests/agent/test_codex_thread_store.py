from agent.codex_thread_store import get_codex_thread_id, save_codex_thread_id


def test_codex_thread_mapping_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.codex_thread_store.get_hermes_home", lambda: str(tmp_path)
    )

    assert get_codex_thread_id("hermes-1") is None
    save_codex_thread_id("hermes-1", "codex-1", cwd="/tmp/project")

    assert get_codex_thread_id("hermes-1") == "codex-1"
