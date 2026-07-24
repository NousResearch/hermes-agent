from gateway.run import _build_runtime_model_note


def test_runtime_model_note_declares_current_model_and_provider():
    note = _build_runtime_model_note(
        "Qwen3.5-397B-A17B",
        {
            "provider": "custom",
            "base_url": "http://10.128.58.20:8000/v1",
        },
    )

    assert "model=Qwen3.5-397B-A17B" in note
    assert "provider=custom" in note
    assert "base_url=http://10.128.58.20:8000/v1" in note
    assert "historical, not current" in note


def test_runtime_model_note_omits_empty_model():
    assert _build_runtime_model_note("", {"provider": "custom"}) == ""
