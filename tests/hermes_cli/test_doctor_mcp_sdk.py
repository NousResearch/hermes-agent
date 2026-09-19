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
4. Availability probes the real runtime import contract (not just the top-level
   package) in a clean interpreter, so broken/partial installs are reported.
5. ``doctor --fix`` resolves an install path the interpreter's owner accepts:
   pip-less ``uv tool`` venvs fall back to ``uv pip --python``, and layouts with
   neither pip nor uv degrade to a manual hint instead of a failed pip run.
"""

import asyncio
import os
import sys
import venv as venv_mod

import pytest

from hermes_cli import doctor_platform

_PIP_INSTALL_CMD = [sys.executable, "-m", "pip", "install", "hermes-agent[mcp]"]


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
    monkeypatch.setattr(doctor_platform, "_mcp_install_cmd", lambda: _PIP_INSTALL_CMD)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _RunResult()

    monkeypatch.setattr(doctor_platform.subprocess, "run", fake_run)
    f = doctor_platform._check_mcp_sdk(True)
    assert not f.issues and not f.manual_issues
    assert captured["cmd"] == _PIP_INSTALL_CMD
    assert "installed" in capsys.readouterr().out


def test_fix_pip_failure_degrades_to_manual_hint(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    monkeypatch.setattr(doctor_platform, "_mcp_install_cmd", lambda: _PIP_INSTALL_CMD)
    monkeypatch.setattr(doctor_platform.subprocess, "run", lambda cmd, **kw: _RunResult(1, "no network"))
    f = doctor_platform._check_mcp_sdk(True)
    assert not f.issues
    assert len(f.manual_issues) == 1
    assert "hermes-agent[mcp]" in f.manual_issues[0]


def test_fix_install_succeeds_but_sdk_still_missing(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    monkeypatch.setattr(doctor_platform, "_mcp_install_cmd", lambda: _PIP_INSTALL_CMD)
    monkeypatch.setattr(doctor_platform.subprocess, "run", lambda cmd, **kw: _RunResult())
    f = doctor_platform._check_mcp_sdk(True)
    assert len(f.manual_issues) == 1


def test_fix_without_pip_or_uv_degrades_to_manual_hint(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform, "_mcp_sdk_available", lambda: False)
    monkeypatch.setattr(doctor_platform, "_mcp_install_cmd", lambda: None)
    ran = []
    monkeypatch.setattr(doctor_platform.subprocess, "run",
                        lambda cmd, **kw: ran.append(cmd) or _RunResult())
    f = doctor_platform._check_mcp_sdk(True)
    assert ran == []  # no install attempt without an owner-compatible path
    assert not f.issues
    assert len(f.manual_issues) == 1
    assert "uv tool install" in f.manual_issues[0]


# =========================================================================
# 3. availability probes the real runtime import contract
# =========================================================================


def test_probe_covers_the_runtime_import_surface():
    probe = doctor_platform._MCP_SDK_PROBE
    assert "ClientSession" in probe
    assert "StdioServerParameters" in probe
    assert "stdio_client" in probe


def test_broken_top_level_mcp_package_is_not_available(tmp_path, monkeypatch):
    # A partial install: `mcp` resolves, but the runtime symbols do not. The probe
    # runs in a clean interpreter, so PYTHONPATH shadowing is honored end to end.
    (tmp_path / "mcp").mkdir()
    (tmp_path / "mcp" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert doctor_platform._mcp_sdk_available() is False


# =========================================================================
# 4. install-path resolution: pip-less venvs fall back to uv, else manual
# =========================================================================


@pytest.mark.skipif(os.name == "nt", reason="posix venv layout")
def test_install_cmd_skips_pipless_interpreter_without_uv(tmp_path, monkeypatch):
    venv_mod.EnvBuilder(with_pip=False, symlinks=True).create(str(tmp_path / "venv"))
    py = str(tmp_path / "venv" / "bin" / "python")
    monkeypatch.setattr(doctor_platform.shutil, "which", lambda name: None)
    assert doctor_platform._mcp_install_cmd(py) is None


@pytest.mark.skipif(os.name == "nt", reason="posix venv layout")
def test_install_cmd_falls_back_to_uv_for_pipless_interpreter(tmp_path, monkeypatch):
    venv_mod.EnvBuilder(with_pip=False, symlinks=True).create(str(tmp_path / "venv"))
    py = str(tmp_path / "venv" / "bin" / "python")
    monkeypatch.setattr(doctor_platform.shutil, "which",
                        lambda name: "/usr/local/bin/uv" if name == "uv" else None)
    assert doctor_platform._mcp_install_cmd(py) == [
        "/usr/local/bin/uv", "pip", "install", "--python", py, "hermes-agent[mcp]"]


# =========================================================================
# 5. transport error names doctor --fix, not the setup wizard
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
