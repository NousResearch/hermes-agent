"""The status route carries the planner's launch refusal per downloaded model, path stripped."""
from types import SimpleNamespace

from hermes_cli.local_runtime import presets
from hermes_cli.web_routers import local_models as lm


def test_launch_refusals_strip_the_gguf_path(monkeypatch):
    monkeypatch.setattr(presets, "read_preset_decisions", lambda: {
        "Big-27B": SimpleNamespace(refusal="/m/Big-27B.gguf: needs ~9.2 GiB but only ~1.3 GiB of memory is free now"),
        "Small-4B": SimpleNamespace(refusal=None),
    })
    assert lm._launch_refusals() == {"Big-27B": "needs ~9.2 GiB but only ~1.3 GiB of memory is free now"}


def test_staged_row_reports_refusal_or_none(tmp_path):
    gguf = tmp_path / "Big-27B.gguf"
    gguf.write_bytes(b"x" * 10)
    row = lm._staged_row(gguf, {lm._model_id_for(gguf): "needs more"})
    assert row["launch_refusal"] == "needs more"
    assert lm._staged_row(gguf)["launch_refusal"] is None
