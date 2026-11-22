"""Local version diagnostics never initiate remote update checks."""
import ast
from pathlib import Path

import pytest

from hermes_cli import _startup_fast


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_fast_version_never_calls_remote_update_probe(monkeypatch, capsys, flag):
    import hermes_cli.source_check as source_check

    def forbidden(*args, **kwargs):
        raise AssertionError("version triggered unapproved update probe")

    monkeypatch.setattr(source_check, "check_for_updates", forbidden)
    monkeypatch.setattr(_startup_fast, "container_mode_may_be_active", lambda: False)
    assert _startup_fast.try_fast_version([flag]) is True
    output = capsys.readouterr().out
    assert "Install directory:" in output
    assert "Python:" in output
    assert "Update available" not in output


def test_cli_termux_and_repl_version_callers_are_explicitly_offline():
    """Inspect actual source callsites without importing a mixed-version CLI."""
    root = Path(__file__).resolve().parents[2] / "hermes_cli"
    for filename, expected_calls in (("main.py", 3), ("cli_loops_mixin.py", 1)):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "_print_version_info"]
        assert len(calls) == expected_calls
        for call in calls:
            flags = {arg.arg: arg.value for arg in call.keywords}
            assert isinstance(flags.get("check_updates"), ast.Constant)
            assert flags["check_updates"].value is False

    main = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    printers = [node for node in ast.walk(main)
                if isinstance(node, ast.FunctionDef) and node.name == "_print_version_info"]
    assert len(printers) == 1
    assert len(printers[0].args.kw_defaults) == 1
    assert printers[0].args.kw_defaults[0].value is False


def test_explicit_update_status_is_not_deleted(monkeypatch, capsys):
    import hermes_cli.source_check as source_check

    calls = []
    monkeypatch.setattr(source_check, "check_for_updates",
                        lambda **kw: (calls.append(kw) or {"behind": 0}))
    _startup_fast.print_fast_version_info(check_updates=True)
    assert calls == [{"passive": True}]
    assert "Up to date" in capsys.readouterr().out
