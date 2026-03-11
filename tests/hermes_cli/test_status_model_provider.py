"""Basic tests for hermes_cli.status model/provider display."""

from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_includes_model_and_provider(capsys):
    """Status output should include Model and Provider lines."""
    args = SimpleNamespace(all=False, deep=False)

    show_status(args)

    out = capsys.readouterr().out
    assert "Model:" in out
    assert "Provider:" in out

