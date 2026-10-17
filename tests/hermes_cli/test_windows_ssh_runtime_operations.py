"""``inspect`` advertises the launcher-ownership capability the Desktop client gates on.

The capability set is what a Windows remote's helper reports to Electron; the pre-existing fields
(``path``, ``version``, ``supported``) keep their exact shape either way.
"""

from __future__ import annotations

import os
import types

from hermes_cli import windows_ssh_runtime as rt

CAPABILITIES = ["launcher-ownership"]


# --- inspect -------------------------------------------------------------------------------------


def test_inspect_reports_capabilities(monkeypatch, tmp_path):
    exe = tmp_path / "hermes.exe"
    exe.write_text("", encoding="utf-8")

    def fake_run(argv, **_kwargs):
        text = "hermes 9.9.9" if "--version" in argv else "usage: --ssh-session-token-file --ssh-owner-nonce"
        return types.SimpleNamespace(returncode=0, stdout=text, stderr="")

    monkeypatch.setattr(rt, "subprocess", types.SimpleNamespace(run=fake_run))
    result = rt.inspect_hermes(str(exe))
    assert result["capabilities"] == CAPABILITIES
    # The pre-existing fields keep their exact shape.
    assert set(result) == {"path", "version", "supported", "capabilities"}
    assert result["path"] == os.path.abspath(str(exe))
    assert result["version"] == "hermes 9.9.9"
    assert result["supported"] is True


def test_inspect_capabilities_are_stable_without_support(monkeypatch, tmp_path):
    exe = tmp_path / "hermes.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(rt, "subprocess", types.SimpleNamespace(
        run=lambda argv, **_kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="boom")))
    result = rt.inspect_hermes(str(exe))
    assert result["capabilities"] == CAPABILITIES
    assert result["supported"] is False
    assert result["version"] == ""
