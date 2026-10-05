"""``hermes model`` → Local models runs the desktop's Local Models setup in the terminal."""

from __future__ import annotations

import pytest


def _picker_keys(config: dict) -> list[str]:
    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    return [key for key, _, _ in _build_provider_picker_rows(config, "", {}, {})[0]]


def test_local_models_row_follows_the_engine_pin_and_exclusions():
    from hermes_cli.local_runtime import binaries

    assert ("llamacpp" in _picker_keys({})) == (binaries.unavailable_reason("cpu") is None)
    assert "llamacpp" not in _picker_keys({"model_catalog": {"excluded_providers": ["llama.cpp"]}})


@pytest.mark.parametrize("outcome", ["done", "error"])
def test_picked_model_runs_the_quickstart_job_and_reports_how_it_ended(monkeypatch, capsys, outcome):
    """The CLI starts the same quickstart job the desktop's button does and reports that job's
    terminal state; only a finished job claims the default changed."""
    import hermes_cli.main_provider_setup as provider_setup
    import hermes_cli.setup as setup
    from hermes_cli.model_setup_flows_local import _model_flow_local
    from hermes_cli.web_routers import local_models as lm

    row = {"id": "tiny", "display_name": "Tiny", "fits": True, "needs_engine": False, "recommended": True,
           "downloaded": False, "size_label": "1.0 GB", "fit_summary": "runs fully on your GPU"}
    monkeypatch.setattr(lm, "local_models_hardware", lambda: {"gpu_name": "GPU", "vram_label": "8.0 GB"})
    monkeypatch.setattr(lm, "local_models_status", lambda: {
        "runtime_installed": True, "tag": "b1", "runtime_backend": "cpu", "models": [], "active_model_id": None})
    monkeypatch.setattr(lm, "local_models_catalog", lambda: {"models": [row]})
    started = []

    def quickstart(body):
        started.append(body.model_id)
        job = lm._job("quickstart", "Tiny", model_id=body.model_id)

        def run():
            lm._step(job, "downloading", "Downloading Tiny")
            if outcome == "error":
                raise RuntimeError("disk full")
            lm._finish(job, "Tiny is ready — new chats use it")

        lm._spawn_job(job, "test-quickstart", run, resumable=True)
        return {"job_id": job["job_id"]}

    monkeypatch.setattr(lm, "local_models_quickstart", quickstart)
    monkeypatch.setattr(provider_setup, "_prompt_provider_choice", lambda labels, default=0, title="": default)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda question, default=True: True)

    _model_flow_local({}, "")

    out = capsys.readouterr().out
    assert started == ["tiny"]
    if outcome == "done":
        assert "✓ Tiny is ready" in out and "Default model set to: Tiny" in out
    else:
        assert "✗ disk full" in out and "Default model set to" not in out
