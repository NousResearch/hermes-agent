"""The gateway start command advertised to users must be the installed ``hermes`` entrypoint,
not the legacy ``python cli.py --gateway`` invocation, which fails in a normal install where no
local ``cli.py`` exists (see #98819)."""

from pathlib import Path


def test_gateway_run_docstring_advertises_hermes_gateway_run():
    """``gateway/run.py`` module docstring is rendered in help text; it must point at the
    installed entrypoint and mention service mode."""
    source = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = source.read_text(encoding="utf-8")
    docstring = text.split('"""', 2)[1]
    assert "hermes gateway run" in docstring
    assert "python cli.py --gateway" not in docstring


def test_gateway_status_advertises_hermes_gateway_run():
    """``_show_gateway_status`` prints a "To start the gateway" block; it must advertise the
    installed ``hermes gateway run`` command (and the ``hermes gateway start`` service variant),
    not the legacy ``python cli.py --gateway`` invocation."""
    source = Path(__file__).resolve().parents[2] / "hermes_cli" / "cli_info_mixin.py"
    text = source.read_text(encoding="utf-8")
    marker = "To start the gateway:"
    idx = text.index(marker)
    block = text[idx:idx + 200]
    assert "hermes gateway run" in block
    assert "hermes gateway start" in block
    assert "python cli.py --gateway" not in block
