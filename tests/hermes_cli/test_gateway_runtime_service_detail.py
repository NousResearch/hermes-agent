"""A refused service binding names the check that failed (#146763).

``service_identity_unverified`` covered every failing check: a schtasks or whoami query that
exited non-zero, an unexpected whoami shape, and each rule the task XML or launcher broke. The
reason code stays the stable contract; ``detail`` carries the check and the child's own output,
which Desktop prints after the code.
"""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


def _installed_vbs(home, configured, *, extra_env="", extra_line=""):
    script = home / "gateway.vbs"
    script.write_text("\n".join([
        "' Hermes Agent Gateway", "Option Explicit", "Dim sh, env, existing_pp",
        'Set sh = CreateObject("WScript.Shell")', 'Set env = sh.Environment("PROCESS")',
        f'env.Item("HERMES_HOME") = "{configured}"',
        'env.Item("HERMES_SUPERVISED_CHILD") = "1"',
        *([f'env.Item("{extra_env}") = "1"'] if extra_env else []),
        *([extra_line] if extra_line else []),
        'env.Item("VIRTUAL_ENV") = "C:\\\\Hermes"',
        'existing_pp = env.Item("PYTHONPATH")', 'If Len(existing_pp) > 0 Then',
        '  env.Item("PYTHONPATH") = "C:\\\\Hermes;" & existing_pp', 'Else',
        '  env.Item("PYTHONPATH") = "C:\\\\Hermes"', 'End If',
        f'sh.CurrentDirectory = "{home}"',
        f'sh.Run "{sys.executable} -m hermes_cli.main gateway run", 0, False',
    ]) + "\n", encoding="utf-8")
    return script


def _windows_refusal(tmp_path, monkeypatch, *, xml_rc=0, whoami=None, launcher_env="", run_level="",
                     launcher_home=None, launcher_line="", launcher_edit=None,
                     xml_error="ERROR: Access is denied.\r\n"):
    from hermes_cli import gateway_runtime_service as service
    from hermes_cli.gateway_windows import _build_scheduled_task_xml

    home = tmp_path / "profile"
    home.mkdir()
    script = _installed_vbs(home, str(launcher_home or home), extra_env=launcher_env, extra_line=launcher_line)
    if launcher_edit:
        script.write_text(script.read_text(encoding="utf-8").replace(*launcher_edit), encoding="utf-8")
    name = "Hermes_Gateway_" + service.service_suffix(home)
    account, sid = "DESKTOP-1\\owner", "S-1-5-21-123"
    xml = _build_scheduled_task_xml(name, script, account)
    if run_level:
        xml = xml.replace("<RunLevel>LeastPrivilege</RunLevel>", f"<RunLevel>{run_level}</RunLevel>")
    answers = {
        "whoami.exe": (0, (whoami if whoami is not None else f'"{account}","{sid}"\r\n'), ""),
        "/XML": (xml_rc, xml if xml_rc == 0 else "", xml_error if xml_rc else ""),
        "/FO": (0, f'"\\{name}","N/A","Ready"\r\n', ""),
    }

    def run(argv, *_args, **_kwargs):
        code, out, err = next(answer for key, answer in answers.items() if key in argv)
        return SimpleNamespace(returncode=code, stdout=out.encode(), stderr=err.encode())

    monkeypatch.setattr(service, "_run", run)
    with pytest.raises(service.RuntimeStartError) as refused:
        service._windows(home, time.monotonic() + 5)
    return refused.value


def test_a_failing_task_query_names_the_command_and_its_error(tmp_path, monkeypatch):
    refused = _windows_refusal(tmp_path, monkeypatch, xml_rc=1)
    assert refused.reason == "service_identity_unverified"
    assert "/XML exited 1" in refused.detail and "Access is denied" in refused.detail


def test_an_unexpected_whoami_answer_is_named(tmp_path, monkeypatch):
    refused = _windows_refusal(tmp_path, monkeypatch, whoami="\r\n")
    assert refused.reason == "service_identity_unverified"
    assert "whoami" in refused.detail and "expected one account and SID" in refused.detail


@pytest.mark.parametrize("case,needles", [
    ("elevated", ["RunLevel", "HighestAvailable"]),
    ("launcher_env", ["PYTHONSTARTUP"]),
    ("foreign_home", ["another Hermes home", "elsewhere"]),
])
def test_a_task_that_breaks_a_rule_names_the_rule(case, needles, tmp_path, monkeypatch):
    refused = _windows_refusal(
        tmp_path, monkeypatch,
        run_level="HighestAvailable" if case == "elevated" else "",
        launcher_env="PYTHONSTARTUP" if case == "launcher_env" else "",
        launcher_home=tmp_path / "elsewhere" / ".hermes" if case == "foreign_home" else None,
    )
    assert refused.reason == "service_identity_unverified"
    assert all(needle in refused.detail for needle in needles), refused.detail
    assert "\n" not in refused.detail


_OPAQUE = "q8Zr2LmX0vN4pT7wK1yB9cD3fG6hJ5sA"


@pytest.mark.parametrize("line,kept", [
    (f'env.Item("S3_SECRET") = "{_OPAQUE}"', 'env.Item("S3_SECRET") = '),
    (f'env.Item("MY_TOKEN")="{_OPAQUE}"', 'env.Item("MY_TOKEN")='),
    (f'sh.Run "cmd /c set KEY={_OPAQUE}", 0, False', "sh.Run "),
    (f"' rotate this: {_OPAQUE}", "' ***"),
    (f"REM backup key {_OPAQUE}", "REM ***"),
    (f'env.Item("K") = 1 \' {_OPAQUE}', "env.Item(\"K\") = 1 \\' ***"),  # repr escapes the ' when both quote kinds appear
    (f'env.Item("K1") = "it\'s {_OPAQUE}"', 'env.Item("K1") = "***"'),  # a quoted ' is no comment
])
def test_a_rejected_launcher_line_names_the_line_not_its_values(line, kept, tmp_path, monkeypatch):
    # A customised launcher may assign a credential; the detail reaches `gateway ensure` output
    # and Desktop, where users paste it into reports.
    refused = _windows_refusal(tmp_path, monkeypatch, launcher_line=line)
    assert refused.reason == "service_identity_unverified"
    assert "launcher line 8 is not the installer's" in refused.detail, refused.detail
    assert _OPAQUE not in refused.detail and _OPAQUE[:8] not in refused.detail, refused.detail
    assert kept in refused.detail, refused.detail


@pytest.mark.parametrize("edit,number,needle", [
    (("0, False\n", '0, False\nWScript.Echo "done"\n'), 17, "WScript.Echo"),
    (("End If\n", "End Sub\n"), 14, "End Sub"),
    (("Else\n", "ElseIf\n"), 12, "ElseIf"),
])
def test_a_broken_launcher_tail_names_the_line_that_broke(edit, number, needle, tmp_path, monkeypatch):
    refused = _windows_refusal(tmp_path, monkeypatch, launcher_edit=edit)
    assert f"launcher line {number} is not the installer's" in refused.detail, refused.detail
    assert needle in refused.detail and "existing_pp" not in refused.detail, refused.detail


def test_child_output_in_a_refusal_is_redacted(tmp_path, monkeypatch):
    refused = _windows_refusal(tmp_path, monkeypatch, xml_rc=1,
                               xml_error=f"HERMES_API_KEY={_OPAQUE} rejected\r\n")
    assert "/XML exited 1" in refused.detail and "HERMES_API_KEY=" in refused.detail, refused.detail
    assert _OPAQUE not in refused.detail, refused.detail


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("case,needles", [
    ("dynamic_user", ["DynamicUser"]),
    ("wrong_command", ["echo"]),
    ("environment_unreadable", ["show-environment exited 3", "Failed to connect to bus"]),
])
def test_ensure_reports_which_unit_check_failed(case, needles, tmp_path, monkeypatch):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli.gateway_runtime_service import service_suffix

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "custom root"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    unit = tmp_path / ".config/systemd/user" / f"hermes-gateway-{service_suffix(home)}.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n", encoding="utf-8")
    command = "/bin/echo hermes_cli.main gateway run" if case == "wrong_command" else (
        f"{sys.executable} -m hermes_cli.main gateway run")
    props = dict(LoadState="loaded", ActiveState="inactive", SubState="dead", UnitFileState="enabled",
                 User=str(os.getuid()), DynamicUser="yes" if case == "dynamic_user" else "no",
                 Environment=f'"HERMES_HOME={home}" "HERMES_SUPERVISED_CHILD=1"',
                 EnvironmentFiles="", PassEnvironment="", UnsetEnvironment="", PAMName="",
                 RootDirectory="", RootImage="", ExecStartPre="", ExecCondition="", DropInPaths="",
                 ExecStart=f"{{ path={command.split()[0]} ; argv[]={command} ; ignore_errors=no ; }}")
    (tmp_path / "effective.json").write_text(json.dumps({
        "props": props, "env_fails": case == "environment_unreadable"}), encoding="utf-8")
    supervisor = tmp_path / "inert-supervisor"
    supervisor.write_text(
        "#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\n"
        + f"d=json.loads((Path({str(tmp_path)!r})/'effective.json').read_text())\n"
        + "if 'show-environment' in sys.argv:\n"
        + " if d['env_fails']: sys.stderr.write('Failed to connect to bus: No medium found\\n'); sys.exit(3)\n"
        + "elif 'show' in sys.argv:\n"
        + " if '--user' in sys.argv: print('\\n'.join(k+'='+v for k,v in d['props'].items()))\n"
        + " else: print('LoadState=not-found')\n"
        + "elif 'start' not in sys.argv: sys.exit(91)\n", encoding="utf-8")
    supervisor.chmod(0o700)
    monkeypatch.setattr(gw, "_systemctl_cmd", lambda system=False: [str(supervisor)] + ([] if system else ["--user"]))

    result = runtime.ensure_gateway_runtime(home, timeout=3.0)

    assert result.reason_code == "service_identity_unverified"
    assert all(needle in result.detail for needle in needles), result.detail
