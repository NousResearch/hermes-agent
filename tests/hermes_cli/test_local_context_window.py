"""Configured context-window contracts for the managed local runtime."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 131_072),
        (32_768, 32_768),
        (0, 131_072),
        (-1, 131_072),
        (1_000_000, 131_072),
    ],
)
def test_bootstrap_context_window_cap_is_bounded_and_fail_open(
        tmp_path, monkeypatch, configured, expected):
    """A positive configured cap wins; absent/invalid values retain automatic policy."""
    from hermes_cli.local_runtime import bootstrap, hardware, presets
    from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile

    model = tmp_path / "models" / "configured-window.gguf"
    model.parent.mkdir()
    model.touch()
    profile = ModelProfile(
        name="configured-window",
        weights_bytes=1 << 30,
        embd_table_bytes=0,
        n_ctx_train=131_072,
        layers=[(LayerKind.FULL, 256)] * 4,
    )
    monkeypatch.setattr(
        presets, "read_gguf_header",
        lambda path: SimpleNamespace(path=path, sampling_defaults={}),
    )
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header: profile)
    monkeypatch.setattr(
        hardware,
        "probe_budget",
        lambda **kwargs: HardwareBudget(64 << 30, 64 << 30, 64 << 30),
    )
    section = {} if configured is None else {"context_window": configured}

    preset_path = tmp_path / "presets.ini"
    assert bootstrap._generate_presets(model.parent, preset_path, section) == preset_path
    assert presets.read_preset_decisions(preset_path)["configured-window"].window == expected
