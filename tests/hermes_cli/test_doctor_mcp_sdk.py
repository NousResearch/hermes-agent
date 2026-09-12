"""Regression tests for issue #109026.

A base PyPI/pipx/Homebrew install ships without the ``mcp`` extra. When the SDK is
missing, ``tools/mcp_tool_transport.py`` raised an ImportError pointing at
``hermes setup`` — the configuration wizard, which never installs a package — so
users looped on a recovery command that cannot change the outcome.

Behavior contracts pinned here:

1. The transport ImportError must name ``hermes doctor --fix`` (a command that
   actually installs the extra), not ``hermes setup``.
2. ``hermes doctor`` must surface the missing SDK as a failed check with a
   doctor-based fix hint (not a bare optional-package warning).
3. ``hermes doctor --fix`` installs ``hermes-agent[mcp]`` into the running
   interpreter and re-verifies; a pip failure degrades to a manual hint.
"""

import asyncio

import pytest

from hermes_cli import doctor_platform


# =========================================================================
# 1. doctor check: missing SDK fails with a doctor-based fix hint
# =========================================================================


def test_sdk_present_ok(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: True)
    f = doctor_platform._check_mcp_sdk(False)
    assert not f.issues and not f.manual_issues


def test_sdk_missing_without_fix_records_issue(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    f = doctor_platform._check_mcp_sdk(False)
    assert len(f.issues) == 1
    assert "hermes doctor --fix" in f.issues[0]
    assert "hermes-agent[mcp]" in f.issues[0]
    out = capsys.readouterr().out
    assert "hermes setup" not in out


# =========================================================================
# 2. doctor --fix installs the extra and re-verifies
# =========================================================================


class _RunResult:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def test_fix_installs_extra_and_verifies(monkeypatch, capsys):
    availability = iter([False, True])  # missing at check time, present after the install
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: next(availability))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _RunResult()

    monkeypatch.setattr(doctor_platform.subprocess, "run", fake_run)
    f = doctor_platform._check_mcp_sdk(True)
    assert not f.issues and not f.manual_issues
    assert captured["cmd"][:4] == [doctor_platform.sys.executable, "-m", "pip", "install"]
    assert "hermes-agent[mcp]" in captured["cmd"]
    assert "installed" in capsys.readouterr().out


def test_fix_pip_failure_degrades_to_manual_hint(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    monkeypatch.setattr(doctor_platform.subprocess, "run", lambda cmd, **kw: _RunResult(1, "no network"))
    f = doctor_platform._check_mcp_sdk(True)
    assert not f.issues
    assert len(f.manual_issues) == 1
    assert "hermes-agent[mcp]" in f.manual_issues[0]


def test_fix_install_succeeds_but_sdk_still_missing(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    monkeypatch.setattr(doctor_platform.subprocess, "run", lambda cmd, **kw: _RunResult())
    f = doctor_platform._check_mcp_sdk(True)
    assert len(f.manual_issues) == 1


# =========================================================================
# 3. transport error names doctor --fix, not the setup wizard
# =========================================================================


def test_stdio_sdk_missing_error_names_doctor_fix(monkeypatch):
    import tools.mcp_tool as mcp_tool
    from tools.mcp_tool_transport import MCPServerTransportMixin

    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: False)

    class _Server(MCPServerTransportMixin):
        name = "demo"

    with pytest.raises(ImportError) as exc_info:
        asyncio.run(_Server()._run_stdio({"command": ["/bin/true"]}))
    assert "hermes doctor --fix" in str(exc_info.value)
    assert "hermes setup" not in str(exc_info.value)
