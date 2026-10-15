"""GPU budget detection follows the llama.cpp backend that will allocate model layers."""

from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware


@pytest.mark.parametrize("backend,expected", [
    ("cuda", True),
    ("hip", False),
    ("vulkan", False),
])
def test_nvidia_detection_uses_active_serving_engine(backend, expected, monkeypatch):
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "serving_engine", lambda: SimpleNamespace(backend=backend))

    assert hardware._nvidia_device_selected() is expected


@pytest.mark.parametrize("configured,resolved,expected", [
    ("auto", "cuda", True),
    ("hip", "hip", False),
    ("vulkan", "vulkan", False),
])
def test_nvidia_detection_resolves_configured_backend_when_idle(
    configured, resolved, expected, monkeypatch,
):
    from hermes_cli import config
    from hermes_cli.local_runtime import binaries, bootstrap

    monkeypatch.setattr(bootstrap, "serving_engine", lambda: None)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "local_runtime": {"backend": configured},
    })
    calls = []

    def resolve_backend(backend):
        calls.append(backend)
        return resolved

    monkeypatch.setattr(binaries, "resolve_backend", resolve_backend)

    assert hardware._nvidia_device_selected() is expected
    assert calls == [configured]
