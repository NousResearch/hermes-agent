from toolsets import get_toolset, resolve_toolset


def test_prompt_only_toolset_is_registered_without_model_callable_tools():
    toolset = get_toolset("prompt_only", include_registry=False)

    assert toolset is not None
    assert toolset["includes"] == []
    assert resolve_toolset("prompt_only", include_registry=False) == []
