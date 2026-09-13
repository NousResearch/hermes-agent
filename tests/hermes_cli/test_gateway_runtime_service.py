"""Installed identity proofs at inert native supervisor executable boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize("case,reason", [
    ("custom", "deadline"), ("named", "deadline"), ("alias", "deadline"),
    ("default", "deadline"),
    ("wrong_home", "profile_mismatch"), ("wrong_user", "service_account_mismatch"),
    ("missing_identity", "service_identity_unverified"),
    ("omitted_empty_exec_lists", "deadline"),
    ("env_file", "service_identity_unverified"), ("dynamic_user", "service_identity_unverified"),
    ("wrong_command", "service_identity_unverified"), ("command_profile", "profile_mismatch"),
    ("dropin_home", "profile_mismatch"), ("manager_home", "profile_mismatch"),
])
def test_ensure_checks_effective_service_binding_before_start(tmp_path, monkeypatch, case, reason):
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli.gateway_runtime_service import service_suffix

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / (".hermes" if case == "default" else "custom root")
    if case == "named":
        home = home / "profiles" / "worker"
    home.mkdir(mode=0o700, parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    suffix = service_suffix(home)
    unit = tmp_path / ".config/systemd/user" / f"hermes-gateway{'-' + suffix if suffix else ''}.service"
    unit.parent.mkdir(parents=True)
    unit.write_text(f'[Service]\nExecStart={sys.executable} -m hermes_cli.main gateway run\nEnvironment="HERMES_HOME={home}"\n', encoding="utf-8")
    original = unit.read_bytes()
    configured = home
    if case in {"wrong_home", "dropin_home"}:
        configured = tmp_path / "unrequested"
    if case == "wrong_home":
        unit.write_text(f'[Service]\nEnvironment="HERMES_HOME={configured}"\n', encoding="utf-8")
        original = unit.read_bytes()
    if case == "dropin_home":
        override = unit.parent / (unit.name + '.d') / 'override.conf'
        override.parent.mkdir()
        override.write_text(f'[Service]\nEnvironment="HERMES_HOME={configured}"\n', encoding="utf-8")
    if case == "alias":
        configured = tmp_path / "alias"
        configured.symlink_to(home, target_is_directory=True)
    command = f'{sys.executable} -m hermes_cli.main gateway run'
    if case == "named":
        command += " --profile worker"
    if case == "command_profile":
        command += " --profile stranger"
    if case == "wrong_command":
        command = '/bin/echo hermes_cli.main gateway run'
    props = dict(LoadState="loaded", ActiveState="inactive", SubState="dead", UnitFileState="enabled",
                 User=str(os.getuid()), DynamicUser="no", Environment=f'"HERMES_HOME={configured}" "HERMES_SUPERVISED_CHILD=1"',
                 EnvironmentFiles="", PassEnvironment="", UnsetEnvironment="", PAMName="",
                 RootDirectory="", RootImage="", ExecStartPre="", ExecCondition="",
                 ExecStart=f'{{ path={command.split()[0]} ; argv[]={command} ; ignore_errors=no ; }}',
                 DropInPaths=str(unit.parent / (unit.name + '.d') / 'override.conf') if case == "dropin_home" else "")
    manager_env = ""
    if case == "wrong_user":
        props['User'] = str(os.getuid() + 1)
    if case == "missing_identity":
        props = {k: props[k] for k in ("LoadState", "ActiveState", "SubState", "UnitFileState")}
    if case == "omitted_empty_exec_lists":
        # systemd >= 25x prints nothing for empty exec-command lists and
        # EnvironmentFiles even under --all; absence there means empty, not unknown.
        for key in ("ExecStartPre", "ExecCondition", "EnvironmentFiles"):
            del props[key]
    if case == "env_file":
        env_file = tmp_path / "override.env"
        env_file.write_text(f'HERMES_HOME={tmp_path / "other"}\n', encoding="utf-8")
        props['EnvironmentFiles'] = f'{env_file} (ignore_errors=no)'
    if case == "dynamic_user":
        props['DynamicUser'] = 'yes'
    if case == "manager_home":
        props['Environment'] = 'HERMES_SUPERVISED_CHILD=1'
        manager_env = f'HERMES_HOME={tmp_path / "other"}\n'
    if case == "default":
        # An older default install may rely on the user manager's HOME.
        props['Environment'] = 'HERMES_SUPERVISED_CHILD=1'
        manager_env = f'HOME={tmp_path}\n'
    definition = tmp_path / "effective.json"
    definition.write_text(json.dumps({'props': props, 'system': False, 'env': manager_env}), encoding="utf-8")
    calls = tmp_path / 'calls.jsonl'
    executable = tmp_path / 'inert-supervisor'
    executable.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
        + f'p=Path({str(tmp_path)!r})\nd=json.loads((p/"effective.json").read_text())\n'
        + 'with (p/"calls.jsonl").open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        + 'if "show-environment" in sys.argv: print(d["env"])\n'
        + 'elif "show" in sys.argv:\n'
        + ' if ("--user" not in sys.argv) == d["system"]: print("\\n".join(k+"="+v for k,v in d["props"].items()))\n'
        + ' else: print("LoadState=not-found")\n'
        + 'elif "start" not in sys.argv: sys.exit(91)\n', encoding='utf-8')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw, "_systemctl_cmd", lambda system=False: [str(executable)] + ([] if system else ["--user"]))
    result = runtime.ensure_gateway_runtime(home, timeout=0.5)
    assert result.reason_code == reason
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    starts = [cmd for cmd in commands if 'start' in cmd]
    assert len(starts) == (1 if reason == 'deadline' else 0)
    assert all('--no-ask-password' in cmd for cmd in starts)
    assert unit.read_bytes() == original
    assert not (home / 'logs').exists()


@pytest.mark.parametrize("configured,command,reason", [
    ("requested", "gateway", None), ("other", "gateway", "profile_mismatch"),
    ("requested", "echo", "service_identity_unverified"),
    (None, "gateway", "service_identity_unverified"),
    ("requested", "unsupervised", "service_identity_unverified"),
    ("requested", "wrong_account", "service_account_mismatch"),
])
def test_loaded_launchd_identity_is_independent_of_disk(configured, command, reason, tmp_path):
    from hermes_cli.gateway_runtime_service_identity import verify_launchd_loaded
    home = tmp_path / "requested"
    argv = [sys.executable, "-m", "hermes_cli.main", "gateway", "run"]
    if command == "echo":
        argv = ["/bin/echo", "hermes_cli.main", "gateway", "run"]
    environment = f"HERMES_HOME => {tmp_path / configured}" if configured else ""
    output = "gui/123/ai.hermes.gateway = {\n\tstate = not running\n\tprogram = " + argv[0] + "\n\targuments = {\n" + "\n".join("\t\t" + a for a in argv) + "\n\t}\n\tenvironment = {\n\t\t" + environment + "\n\t}\n}"
    if command != "unsupervised":
        output = output.replace("\n\tenvironment = {\n", "\n\tenvironment = {\n\t\tHERMES_SUPERVISED_CHILD => 1\n")
    if command == "wrong_account":
        output = output.replace("\n\tstate", "\n\tusername = other\n\tstate")
    if reason:
        with pytest.raises(ValueError, match=reason):
            verify_launchd_loaded(output, home, uid=123, username="owner")
    else:
        verify_launchd_loaded(output, home, uid=123, username="owner")


@pytest.mark.parametrize("case,reason", [
    ("bound", None), ("utf16", None), ("wrong_home", "profile_mismatch"),
    ("wrong_user", "service_account_mismatch"), ("missing_user", "service_identity_unverified"),
    ("extra_action", "service_identity_unverified"), ("disabled", "service_disabled"),
    ("unknown_script", "service_identity_unverified"), ("malformed_xml", "service_identity_unverified"),
])
def test_task_xml_binds_actual_action_and_principal(case, reason, tmp_path):
    from hermes_cli.gateway_runtime_service_identity import verify_windows_task
    from hermes_cli.gateway_windows import _build_scheduled_task_xml
    home = tmp_path / 'requested'
    home.mkdir()
    script = home / 'gateway.vbs'
    configured = str(home if case != 'wrong_home' else tmp_path / 'other')
    # The current installed launcher grammar; no script is executed.
    script.write_text("\n".join([
        "' Hermes Agent Gateway", "Option Explicit", "Dim sh, env, existing_pp",
        'Set sh = CreateObject("WScript.Shell")', 'Set env = sh.Environment("PROCESS")',
        f'env.Item("HERMES_HOME") = "{configured}"',
        'env.Item("HERMES_SUPERVISED_CHILD") = "1"',
        'env.Item("VIRTUAL_ENV") = "C:\\Hermes"',
        'existing_pp = env.Item("PYTHONPATH")', 'If Len(existing_pp) > 0 Then',
        '  env.Item("PYTHONPATH") = "C:\\Hermes;" & existing_pp', 'Else',
        '  env.Item("PYTHONPATH") = "C:\\Hermes"', 'End If',
        f'sh.CurrentDirectory = "{home}"',
        f'sh.Run "{sys.executable} -m hermes_cli.main gateway run", 0, False',
    ]) + "\n", encoding='utf-8')
    user = 'DOMAIN\\owner'
    xml = _build_scheduled_task_xml('Hermes_Gateway', script, user)
    if case == 'wrong_user':
        xml = xml.replace(user, 'DOMAIN\\other')
    if case == 'missing_user':
        xml = xml.replace(f'<UserId>{user}</UserId>', '')
    if case == 'extra_action':
        xml = xml.replace('</Actions>', '<Exec><Command>other.exe</Command></Exec></Actions>')
    if case == 'disabled':
        xml = xml.replace('<Enabled>true</Enabled>', '<Enabled>false</Enabled>')
    if case == 'utf16':
        xml = xml.encode('utf-16')
    if case == 'malformed_xml':
        xml = '<Task>'
    if case == 'unknown_script':
        script.write_text('MsgBox "not a gateway"', encoding='utf-8')
    if reason:
        with pytest.raises(ValueError, match=reason):
            verify_windows_task(xml, home, user, 'S-1-5-21-123')
    else:
        verify_windows_task(xml, home, user, 'S-1-5-21-123')


@pytest.mark.macos_only
@pytest.mark.parametrize("case,reason", [("loaded_good_disk_wrong", None), ("loaded_wrong_disk_good", "profile_mismatch"), ("bootstrap", None)])
def test_native_launchd_checks_loaded_job_before_disk(case, reason, tmp_path, monkeypatch):
    import plistlib
    import pwd
    from types import SimpleNamespace
    from hermes_cli import gateway_runtime_service as service
    home = tmp_path / 'profile'
    home.mkdir(mode=0o700)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_dir=str(tmp_path), pw_name="owner"))
    label = 'ai.hermes.gateway' + ('-' + service.service_suffix(home) if service.service_suffix(home) else '')
    plist = tmp_path / 'Library/LaunchAgents' / (label + '.plist')
    plist.parent.mkdir(parents=True)
    argv = [sys.executable, '-m', 'hermes_cli.main', 'gateway', 'run']
    disk_home = tmp_path / 'wrong' if case == 'loaded_good_disk_wrong' else home
    plist.write_bytes(plistlib.dumps(dict(Label=label, ProgramArguments=argv,
        EnvironmentVariables=dict(HERMES_HOME=str(disk_home), HERMES_SUPERVISED_CHILD='1'))))
    original = plist.read_bytes()
    loaded_home = tmp_path / 'wrong' if case == 'loaded_wrong_disk_good' else home
    output = 'gui/' + str(os.getuid()) + '/' + label + ' = {\n\tstate = not running\n\tprogram = ' + argv[0] + '\n\targuments = {\n' + '\n'.join('\t\t' + arg for arg in argv) + '\n\t}\n\tenvironment = {\n\t\tHERMES_HOME => ' + str(loaded_home) + '\n\t\tHERMES_SUPERVISED_CHILD => 1\n\t}\n}'
    peer = tmp_path / 'supervisor.py'
    peer.write_text('import sys\n' + f'output={output!r}\nbootstrap={case == "bootstrap"!r}\n' +
        'if "print" in sys.argv:\n sys.exit(113) if bootstrap or any(a.startswith("user/") for a in sys.argv) else print(output)\n', encoding='utf-8')
    actual = subprocess.run
    calls = []
    def boundary(argv, **kw):
        assert argv[0] == 'launchctl'
        calls.append(argv)
        return actual([sys.executable, str(peer), *argv[1:]], **kw)
    monkeypatch.setattr(subprocess, 'run', boundary)
    if reason:
        with pytest.raises(service.RuntimeStartError, match=reason):
            service.discover_existing_gateway_service(home, deadline=time.monotonic()+5)
        assert all(a[1] == 'print' for a in calls)
    else:
        found = service.discover_existing_gateway_service(home, deadline=time.monotonic()+5)
        assert found is not None
        service.start_existing_gateway_service(found, deadline=time.monotonic()+5)
        assert calls[-1][1] == ('bootstrap' if case == 'bootstrap' else 'kickstart')
    assert plist.read_bytes() == original


@pytest.mark.windows_only
@pytest.mark.parametrize("wrong", [False, True])
def test_native_task_query_uses_installed_xml_and_vendor_launcher(wrong, tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_service as service
    from hermes_cli.gateway_windows import _build_gateway_vbs_script, _build_scheduled_task_xml
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    script = home / 'gateway.vbs'
    # newline='' mirrors the installer's _atomic_write: the template is already CRLF and
    # Windows text mode would otherwise double it to \r\r\n, which is not the vendor template.
    script.write_text(_build_gateway_vbs_script(sys.executable, str(home), str(home), ''), encoding='utf-8', newline='')
    original = script.read_bytes()
    # whoami is read-only native account evidence; no task-manager mutation.
    identity = subprocess.run(['whoami.exe', '/USER', '/FO', 'CSV', '/NH'], stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
    from hermes_cli.gateway_windows import _schtasks_encoding
    import csv
    account, sid = next(csv.reader(identity.stdout.decode(_schtasks_encoding()).splitlines()))
    suffix = service.service_suffix(home)
    name = 'Hermes_Gateway' + ('_' + suffix if suffix else '')
    xml = _build_scheduled_task_xml(name, script, 'S-1-5-18' if wrong else sid)
    peer = tmp_path / 'supervisor.py'
    row = f'"\\{name}","N/A","Ready"'
    peer.write_text('import sys\n' + f'xml={xml!r}\n' +
        'if "/XML" in sys.argv: sys.stdout.buffer.write(xml.encode("utf-16"))\n'
        + f'elif "/Query" in sys.argv: print({row!r})\n', encoding='utf-8')
    actual = subprocess.run
    calls = []
    def boundary(argv, **kw):
        if argv[0] == 'whoami.exe':
            return actual(argv, **kw)
        assert argv[0] == 'schtasks.exe'
        calls.append(argv)
        return actual([sys.executable, str(peer), *argv[1:]], **kw)
    monkeypatch.setattr(subprocess, 'run', boundary)
    if wrong:
        with pytest.raises(service.RuntimeStartError, match='service_account_mismatch'):
            service.discover_existing_gateway_service(home, deadline=time.monotonic()+5)
        assert not any('/Run' in a for a in calls)
    else:
        found = service.discover_existing_gateway_service(home, deadline=time.monotonic()+5)
        assert found is not None
        service.start_existing_gateway_service(found, deadline=time.monotonic()+5)
        assert sum('/Run' in a for a in calls) == 1
    assert script.read_bytes() == original


@pytest.mark.linux_only
def test_service_definition_reads_reject_fifo_without_waiting(tmp_path):
    from hermes_cli.gateway_runtime_service_identity import read_definition
    regular = tmp_path / 'regular'
    regular.write_bytes(b'installed definition')
    assert read_definition(regular) == regular.read_bytes()
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError, match='service_identity_unverified'):
        read_definition(fifo)


@pytest.mark.parametrize("outer", ["python", "echo"])
def test_gateway_wrapper_requires_actual_python_entrypoint(outer, tmp_path):
    from hermes_cli.gateway_runtime_service_identity import verify_gateway_argv
    argv = [sys.executable if outer == 'python' else '/bin/echo', '-m', 'hermes_cli.stderr_timestamp', '--error-log', str(tmp_path / 'error.log'), '--', sys.executable, '-m', 'hermes_cli.main', 'gateway', 'run']
    if outer == 'echo':
        with pytest.raises(ValueError, match='service_identity_unverified'):
            verify_gateway_argv(argv, tmp_path)
    else:
        verify_gateway_argv(argv, tmp_path)
