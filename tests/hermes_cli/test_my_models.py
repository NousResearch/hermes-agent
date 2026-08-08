from hermes_cli.my_models import MY_MODELS, display_name, picker_models, resolve


def test_qwen_38_max_is_available_through_openrouter():
    entry = ("Qwen 3.8 Max", "openrouter", "qwen/qwen3.8-max")

    assert entry in MY_MODELS

    picker_id = "my-model::openrouter::qwen/qwen3.8-max"
    assert picker_id in picker_models()
    assert display_name(picker_id) == "Qwen 3.8 Max"
    assert resolve(picker_id) == ("openrouter", "qwen/qwen3.8-max")
