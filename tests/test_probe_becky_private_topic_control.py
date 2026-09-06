import importlib.util
import io
from pathlib import Path


def _probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_becky_private_topic_control.py"
    )
    spec = importlib.util.spec_from_file_location("becky_topic_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_requires_exact_disposable_ack(monkeypatch) -> None:
    module = _probe_module()

    monkeypatch.setattr(module.sys, "stdin", io.StringIO("DISPOSABLE\n"))
    assert module._read_disposable_ack() is True

    monkeypatch.setattr(module.sys, "stdin", io.StringIO(" DISPOSABLE\n"))
    assert module._read_disposable_ack() is False


def test_probe_requires_a_single_empty_confirmation_line(monkeypatch) -> None:
    module = _probe_module()

    monkeypatch.setattr(module.sys, "stdin", io.StringIO("\n"))
    assert module._read_visual_confirmation() is True

    monkeypatch.setattr(module.sys, "stdin", io.StringIO("continue\n"))
    assert module._read_visual_confirmation() is False

    monkeypatch.setattr(module.sys, "stdin", io.StringIO(""))
    assert module._read_visual_confirmation() is False
