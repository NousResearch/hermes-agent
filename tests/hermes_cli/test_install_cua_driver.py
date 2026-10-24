"""CUA runtime validation, explicit PM setup, and native host integration."""

import codecs
import json
from types import SimpleNamespace
from unittest.mock import patch
from xml.sax.saxutils import escape

import pytest


def _runtime_manifest(version="0.20.0", *, omit=()):
    required = {
        "mcp": {"--socket", "--grant"},
        "serve": {"--socket", "--permission-mode", "--capability-manifest",
                  "--approve-capability-manifest", "--embedded"},
        "stop": {"--socket"},
    }
    return {
        "binary_version": version,
        "mcp_invocation": {"command": "cua-driver", "args": ["mcp"]},
        "subcommands": [
            {"name": command, "args": [{"name": arg} for arg in sorted(args - set(omit))]}
            for command, args in required.items()
        ],
    }


@pytest.mark.parametrize("version,omit,ready", [
    ("0.20.0", (), True),
    ("0.19.4", (), False),
    ("bad-version", (), False),
    ("0.20.0", ("--approve-capability-manifest",), False),
])
def test_runtime_contract(version, omit, ready, tmp_path):
    from hermes_cli import tools_config_cua as cua

    result = SimpleNamespace(returncode=0, stderr="",
                             stdout=json.dumps(_runtime_manifest(version, omit=omit)))
    binary = str(tmp_path / "cua-driver")
    with patch("subprocess.run", return_value=result):
        state = cua._cua_driver_contract_status(binary)
    assert state["ready"] is ready
    assert state["binary"] == binary
    assert bool(state["reason"]) is not ready
    if omit:
        assert "serve --approve-capability-manifest" in state["reason"]


@pytest.mark.parametrize("upgrade", [False, True])
def test_pm_failure_is_reported_without_vendor_fallback(monkeypatch, capsys, upgrade):
    import pm
    from hermes_cli import tools_config_cua as cua

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua, "_resolved_cua_driver_cmd", lambda: None)
    with patch.object(pm, "ensure", side_effect=pm.InstallError("cua-driver", "offline")) as ensure, \
         patch.object(cua.subprocess, "Popen", side_effect=AssertionError("vendor installer")):
        assert not cua.install_cua_driver(upgrade=upgrade)
    ensure.assert_called_once_with("cua-driver", explicit=True)
    assert "offline" in capsys.readouterr().out


@pytest.mark.parametrize("upgrade", [False, True])
def test_broken_override_never_acquires_standard_driver(tmp_path, monkeypatch, upgrade):
    import pm
    from hermes_cli import tools_config_cua as cua

    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(tmp_path / "missing-driver"))
    with patch.object(pm, "ensure", side_effect=AssertionError("override was replaced")):
        assert not cua.install_cua_driver(upgrade=upgrade)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("old_target", [False, True])
def test_autostart_uses_selected_binary_and_verifies_registration(tmp_path, monkeypatch, old_target):
    """``computer_use.autostart`` is opt-in (#97389): the default is on-demand,
    so exercising the registration path here requires arming the opt-in first."""
    from hermes_cli import tools_config_cua as cua

    binary = str(tmp_path / "User's driver directory" / "cua-driver.exe")
    selected = str(tmp_path / "old-cua-driver.exe") if old_target else binary
    calls = []

    def run(cmd, **kwargs):
        nonlocal selected
        calls.append(cmd)
        if cmd[0] == "schtasks.exe":
            xml = ('<?xml version="1.0" encoding="UTF-16"?>'
                   '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
                   f'<Actions><Exec><Command>{escape(selected)}</Command></Exec></Actions></Task>')
            return SimpleNamespace(returncode=0, stdout=xml.encode("utf-16"), stderr=b"")
        assert "-FilePath $exe" in cmd[-1]
        assert "-ArgumentList @('autostart','enable')" in cmd[-1]
        assert cua._ps_single_quote(binary) in cmd[-1]
        selected = binary
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cua, "_cua_autostart_opt_in", lambda: True)
    monkeypatch.setattr(cua.subprocess, "run", run)
    monkeypatch.setattr(cua.shutil, "which", lambda command: command)
    assert cua._repair_cua_driver_autostart_windows(binary, verbose=False)
    assert len([cmd for cmd in calls if cmd[0] != "schtasks.exe"]) == int(old_target)
    assert all("/XML" in cmd for cmd in calls if cmd[0] == "schtasks.exe")


@pytest.mark.platforms("windows")
def test_autostart_does_not_claim_success_without_registered_task(monkeypatch):
    from hermes_cli import tools_config_cua as cua

    monkeypatch.setattr(cua, "_cua_autostart_opt_in", lambda: True)
    monkeypatch.setattr(cua, "_cua_driver_autostart_registered_windows", lambda binary=None: False)
    monkeypatch.setattr(cua.shutil, "which", lambda command: command)
    monkeypatch.setattr(cua, "_run_text", lambda *a, **kw: SimpleNamespace(returncode=0))
    assert not cua._repair_cua_driver_autostart_windows("cua-driver.exe", verbose=False)


@pytest.mark.platforms("windows")
def test_repair_without_opt_in_spawns_no_powershell(monkeypatch):
    """``computer_use.autostart`` defaults to on-demand (#97389): without the
    opt-in the repair degrades to a no-op — returns True (nothing to repair)
    and spawns no PowerShell at all."""
    from hermes_cli import tools_config_cua as cua

    monkeypatch.setattr(cua, "_cua_driver_autostart_registered_windows", lambda binary=None: False)
    monkeypatch.setattr(cua.shutil, "which", lambda command: AssertionError("resolved a binary"))
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch.object(cua.subprocess, "run", side_effect=AssertionError("spawned PowerShell")), \
         patch.object(cua, "_run_text", side_effect=AssertionError("spawned PowerShell")):
        assert cua._repair_cua_driver_autostart_windows("cua-driver.exe", verbose=False)


@pytest.mark.parametrize("config", [
    {},
    {"computer_use": {}},
    {"computer_use": {"autostart": False}},
])
def test_autostart_opt_in_defaults_to_on_demand(config):
    from hermes_cli import tools_config_cua as cua

    with patch("hermes_cli.config.load_config", return_value=config):
        assert cua._cua_autostart_opt_in() is False


def test_autostart_opt_in_explicit_true_registers():
    from hermes_cli import tools_config_cua as cua

    with patch("hermes_cli.config.load_config",
               return_value={"computer_use": {"autostart": True}}):
        assert cua._cua_autostart_opt_in() is True


def test_autostart_opt_in_unreadable_config_fails_closed():
    from hermes_cli import tools_config_cua as cua

    with patch("hermes_cli.config.load_config", side_effect=OSError("unreadable")):
        assert cua._cua_autostart_opt_in() is False


def test_autostart_registration_ps_command_quotes_paths_with_spaces():
    """Start-Process structured ``-FilePath`` / ``-ArgumentList``: older
    install.ps1 builds interpolated the binary path into a command string,
    which split at the first space."""
    from hermes_cli import tools_config_cua as cua

    path = "C:\\Program Files\\cua driver\\cua-driver.exe"
    ps = cua._cua_autostart_registration_ps_command(path)
    assert "Start-Process -FilePath $exe" in ps
    assert "@('autostart','enable')" in ps
    assert "-Verb RunAs -Wait -PassThru" in ps
    # The path is single-quoted (never interpolated bare into the command
    # string), so spaces cannot split it.
    assert f"$exe = '{path}'" in ps


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("opt_in,expected", [(False, True), (True, False)])
def test_install_ready_task_requirement_follows_the_opt_in(monkeypatch, opt_in, expected):
    """With ``computer_use.autostart`` off (#97389) a missing cua-driver-serve
    logon task is not a repair condition; opted in, it still is."""
    from hermes_cli import tools_config_cua as cua

    monkeypatch.setattr(cua, "_cua_driver_contract_status",
                        lambda *a: {"ready": True, "version": "0.20.0",
                                    "binary": "/x/cua-driver", "reason": ""})
    monkeypatch.setattr(cua, "_cua_driver_autostart_registered_windows",
                        lambda binary=None: False)
    monkeypatch.setattr(cua, "_cua_autostart_opt_in", lambda: opt_in)
    assert cua._cua_driver_install_ready() is expected


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("registered", [False, True])
def test_setup_preserves_host_registration_failure(monkeypatch, registered):
    import pm
    from hermes_cli import tools_config_cua as cua

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(pm, "ensure", lambda *a, **kw: None)
    monkeypatch.setattr(cua, "_resolved_cua_driver_cmd", lambda: "cua-driver.exe")
    monkeypatch.setattr(cua, "_cua_driver_contract_status", lambda *a: {"ready": True})
    with patch.object(cua, "_repair_cua_driver_autostart_windows", return_value=registered) as repair:
        assert cua.install_cua_driver(show_installer_progress=False) is registered
    repair.assert_called_once_with("cua-driver.exe", verbose=False)


@pytest.mark.platforms("macos")
def test_setup_refuses_bare_binary_without_required_signed_app(monkeypatch, tmp_path):
    from hermes_cli import tools_config_cua as cua

    driver = tmp_path / "cua-driver"
    driver.write_text("#!/bin/sh\nexit 0\n")
    driver.chmod(0o755)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    monkeypatch.setattr(cua, "_cua_driver_contract_status", lambda *a: {"ready": True})
    assert not cua.install_cua_driver(show_installer_progress=False)


@pytest.mark.platforms("macos")
def test_setup_registers_only_the_validated_selected_app(monkeypatch, tmp_path):
    from hermes_cli import tools_config_cua as cua
    from tools.computer_use import cua_backend_daemon as daemon

    app = tmp_path / "CuaDriver.app"
    driver = app / "Contents" / "MacOS" / "cua-driver"
    driver.parent.mkdir(parents=True)
    driver.write_text("#!/bin/sh\nexit 0\n")
    driver.chmod(0o755)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    monkeypatch.setattr(cua, "_cua_driver_contract_status", lambda *a: {"ready": True})
    with patch.object(daemon, "_validate_cua_driver_app_signature") as validate, \
         patch.object(cua, "_run_text", return_value=SimpleNamespace(returncode=0)) as register:
        assert cua.install_cua_driver(show_installer_progress=False)
    validate.assert_called_once_with(str(app))
    assert register.call_args.args[0][-2:] == ["-f", str(app)]


@pytest.mark.platforms("macos")
def test_setup_keeps_permission_guidance(capsys):
    from hermes_cli.tools_config_cua import _print_cua_platform_notes

    _print_cua_platform_notes(False, False, fresh_install=True)
    output = capsys.readouterr().out
    assert "Accessibility" in output
    assert "Screen Recording" in output


@pytest.mark.parametrize("raw,expected", [
    ("cua-driver 0.20.0", "cua-driver 0.20.0"),
    ("banner\nsecond line", "banner"),
    ("\n\n  cua-driver 0.20.0  ", "cua-driver 0.20.0"),
    ("x" * 500, "x" * 120),
    ("", ""),
    ("   \n \n", ""),
])
def test_version_summary(raw, expected):
    from hermes_cli.tools_config_cua import _cua_version_summary

    assert _cua_version_summary(raw) == expected


# --- Windows autostart readiness check (#123774) ---------------------------------
# The registered task is a powershell launcher; the driver path lives in Exec/Arguments,
# not Exec/Command. schtasks also declares encoding="UTF-16" regardless of what it writes —
# ASCII/console-code-page bytes on one fleet, genuine UTF-16-with-BOM on another — so the
# readiness check must decode (BOM decisive, else the gateway codec) before parsing, and
# match both leaves.

_CUA_BINARY = r"C:\Users\me\AppData\Local\hermes\tools\cua-driver-0.21.0-win32-x64\cua-driver.exe"


def _wrapped_task_xml(binary=_CUA_BINARY, declaration='<?xml version="1.0" encoding="UTF-16"?>'):
    args = (f"-NoProfile -WindowStyle Hidden -NonInteractive -Command "
            f"\"Start-Process -FilePath '{escape(binary)}' -ArgumentList @('autostart','enable')\"")
    return (f"{declaration}\n"
            f"<Task><Actions Context=\"Author\">"
            f"<Exec><Command>powershell.exe</Command><Arguments>{escape(args)}</Arguments></Exec>"
            f"</Actions></Task>")


def test_autostart_match_finds_binary_in_shell_wrapped_arguments():
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    # The powershell launcher: Command is powershell.exe, driver path in Arguments.
    assert _task_xml_targets_cua_binary(_wrapped_task_xml(), _CUA_BINARY) is True


def test_autostart_match_finds_binary_as_direct_command():
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    direct = (f"<Task><Actions><Exec><Command>{escape(_CUA_BINARY)}</Command>"
              f"<Arguments>serve</Arguments></Exec></Actions></Task>")
    assert _task_xml_targets_cua_binary(direct, _CUA_BINARY) is True


def test_autostart_match_is_case_and_separator_insensitive():
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    # schtasks may echo the path with different case / forward slashes than resolved.
    other_case = _CUA_BINARY.replace("cua-driver.exe", "CUA-DRIVER.EXE").replace("\\", "/")
    assert _task_xml_targets_cua_binary(_wrapped_task_xml(), other_case) is True


def test_autostart_match_rejects_a_task_for_a_different_binary():
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    stale = _CUA_BINARY.replace("0.21.0", "0.20.0")
    assert _task_xml_targets_cua_binary(_wrapped_task_xml(binary=stale), _CUA_BINARY) is False


def test_autostart_match_survives_utf16_declaration_over_ascii_bytes():
    # The field repro: declaration says UTF-16, bytes are ASCII/UTF-8. Decoding the raw
    # bytes with the gateway codec then parsing the str must not ParseError (Layer 1).
    from hermes_cli.gateway_windows import _decode_schtasks_output
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    raw = _wrapped_task_xml().encode("utf-8")  # no UTF-16 BOM, UTF-16 declaration
    assert _task_xml_targets_cua_binary(_decode_schtasks_output(raw), _CUA_BINARY) is True


@pytest.mark.parametrize(
    "raw",
    [
        _wrapped_task_xml().encode("utf-16"),  # bare "utf-16": LE payload + BOM, what schtasks emits
        codecs.BOM_UTF16_LE + _wrapped_task_xml().encode("utf-16-le"),
        codecs.BOM_UTF16_BE + _wrapped_task_xml().encode("utf-16-be"),
    ],
    ids=["utf16-native", "utf16-le-bom", "utf16-be-bom"],
)
def test_autostart_match_survives_genuine_utf16_bom_payload(raw):
    # The other field shape (#123774): schtasks emits real UTF-16-with-BOM bytes. Feeding those to
    # the single-byte console codec yields NUL-laden mojibake that ParseErrors; the BOM must be
    # decisive so the readiness check decodes and matches instead of reporting "not registered".
    from hermes_cli.tools_config_cua import _decode_task_xml, _task_xml_targets_cua_binary

    assert raw[:2] in (b"\xff\xfe", b"\xfe\xff")  # a real BOM, unlike the ASCII-bytes repro above
    assert _task_xml_targets_cua_binary(_decode_task_xml(raw), _CUA_BINARY) is True


def test_decode_task_xml_delegates_when_there_is_no_bom():
    # No BOM -> the console-code-page-over-ASCII shape: delegate to the gateway codec, don't
    # mis-read it as UTF-16. A str passes through untouched (ElementTree ignores its declaration).
    from hermes_cli.tools_config_cua import _decode_task_xml

    assert _decode_task_xml(_wrapped_task_xml().encode("utf-8")).lstrip().startswith("<?xml")
    assert _decode_task_xml("<Task/>") == "<Task/>"


def test_autostart_match_returns_false_on_malformed_xml():
    from hermes_cli.tools_config_cua import _task_xml_targets_cua_binary

    assert _task_xml_targets_cua_binary("not xml <<<", _CUA_BINARY) is False
    assert _task_xml_targets_cua_binary(_wrapped_task_xml(), "") is False
