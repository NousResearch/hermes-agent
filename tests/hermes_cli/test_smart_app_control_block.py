"""Windows Smart App Control detection in the dashboard deps check (#63796).

When the policy blocks the embedded Python runtime's ``_ssl`` DLL, the
fastapi/uvicorn import fails with the ``DLL load failed ... _ssl`` signature.
The dashboard must say so instead of the generic missing-deps repair guidance
— repair can never lift a policy block, and users looped on it.
"""
from __future__ import annotations

import builtins

import pytest

from hermes_cli import main
from hermes_cli.main_dep_hints import smart_app_control_block_message

SAC_ERROR = ImportError(
    "DLL load failed while importing _ssl: "
    "An Application Control policy has blocked this file."
)


def _raise_for(names, error):
    """``__import__`` stand-in that raises *error* for the given module names."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in names:
            raise error
        return real_import(name, *args, **kwargs)

    return fake_import


def test_smart_app_control_block_message_matches_the_dll_ssl_signature():
    message = smart_app_control_block_message(SAC_ERROR)
    assert message is not None
    assert "Smart App Control" in message
    assert "embedded Python runtime" in message
    assert "_ssl" in message
    assert "repair" in message  # must say the repair loop cannot fix this
    assert "https://aka.ms/smartappcontrol" in message


def test_smart_app_control_block_message_ignores_other_errors():
    assert smart_app_control_block_message(ImportError("No module named 'fastapi'")) is None
    assert smart_app_control_block_message(ImportError("cannot import name '_ssl' from 'ssl'")) is None


def test_dashboard_deps_check_names_the_policy_block(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "__import__", _raise_for({"fastapi"}, SAC_ERROR))
    with pytest.raises(SystemExit) as exc:
        main._require_dashboard_web_deps()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Smart App Control" in output
    assert "embedded Python runtime" in output
    assert "hermes pm install" not in output  # the repair hint must not appear


def test_dashboard_deps_check_keeps_repair_guidance_for_plain_missing_deps(
        monkeypatch, capsys):
    monkeypatch.setattr(
        builtins, "__import__", _raise_for({"fastapi"}, ImportError("No module named 'fastapi'")))
    with pytest.raises(SystemExit) as exc:
        main._require_dashboard_web_deps()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "hermes pm install" in output
    assert "Smart App Control" not in output
