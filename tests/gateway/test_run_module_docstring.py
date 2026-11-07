"""Docstring for gateway.run should advertise the supported CLI start command."""

from pathlib import Path


def test_gateway_run_docstring_uses_hermes_gateway_run():
    source = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = source.read_text(encoding="utf-8")
    docstring = text.split('"""', 2)[1]
    assert "hermes gateway run" in docstring
    assert "python cli.py --gateway" not in docstring
