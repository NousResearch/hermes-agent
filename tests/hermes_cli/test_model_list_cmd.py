"""``hermes model --list`` prints a provider catalog without a TTY."""
import json
import os
from pathlib import Path

import pytest

from hermes_cli.main import main


def _run(monkeypatch, capsys, argv):
    monkeypatch.setattr("sys.argv", ["hermes", *argv])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit) as exc:
        main()
    out = capsys.readouterr()
    return exc.value.code, out.out, out.err


def test_model_list_prints_catalog_and_marks_configured_default(monkeypatch, capsys):
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        "model:\n  provider: gemini\n  default: gemini-2.5-flash\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.models.provider_model_ids",
                        lambda provider, force_refresh=False: ["gemini-3-flash-preview", "gemini-2.5-flash"])

    code, out, _ = _run(monkeypatch, capsys, ["model", "--list", "--json"])
    assert code == 0
    assert json.loads(out) == {"schema_version": 1, "provider": "gemini", "current": "gemini-2.5-flash",
                               "models": ["gemini-3-flash-preview", "gemini-2.5-flash"]}

    code, out, _ = _run(monkeypatch, capsys, ["model", "--list"])
    assert code == 0
    assert "  * gemini-2.5-flash" in out and "    gemini-3-flash-preview" in out


def test_model_list_unknown_provider_fails_on_stderr(monkeypatch, capsys):
    code, out, err = _run(monkeypatch, capsys, ["model", "--list", "--provider", "no-such-provider"])
    assert code == 1
    assert out == "" and "Unknown provider 'no-such-provider'" in err
