from hermes_cli.context_window import (
    entry_context_length,
    model_config_context_length,
    parse_context_window_cap,
    scoped_model_config_context_length,
)


def test_max_context_length_alias_when_context_length_absent():
    cap, key, raw = model_config_context_length({"max_context_length": "262144"})
    assert cap == 262144
    assert key == "max_context_length"
    assert raw == "262144"


def test_context_length_precedes_max_context_length():
    cap, key, _raw = model_config_context_length(
        {"context_length": 131072, "max_context_length": 262144}
    )
    assert cap == 131072
    assert key == "context_length"


def test_auto_clears_explicit_cap():
    assert parse_context_window_cap("auto") is None
    assert entry_context_length({"max_context_length": "auto"}) is None


def test_scoped_model_config_does_not_leak_to_other_route():
    cfg = {"model": {"default": "glm-5.2", "provider": "zai", "max_context_length": 262144}}
    assert scoped_model_config_context_length(cfg, model="glm-5.2", provider="zai") == 262144
    assert scoped_model_config_context_length(cfg, model="gpt-5.5", provider="openai-codex") is None
