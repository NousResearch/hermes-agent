"""Runner-only real Windows SID rejection probe; never changes a gateway/service.

Run on a GitHub-hosted Windows runner, from the checkout after uv sync/npm ci:
  uv run --no-sync python evals/desktop_gateway_ticket_native/windows_sid.py \
      --output native-ticket-results/windows-sid --electron-auto

The only machine-level fixture is a uniquely named, on-demand SYSTEM scheduled
TASK (not a service); it is stopped/deleted/read-back-verified in finally. No
accounts, credentials, installed services or production homes are modified.
Electron is optional; --electron-auto requires and exercises the real native app.
The receipts describe controlled decoys, not real gateway-issued credentials.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
DENIAL = 'named-pipe peer belongs to another user'
SDDL = 'D:P(A;;GA;;;WD)'
FIXTURE_TICKET = 'inert-sid-probe-not-a-real-ticket'


def require_runner():
    if sys.platform != 'win32' or os.name != 'nt':
        raise RuntimeError('requires actual Windows; platform emulation is forbidden')
    if (os.environ.get('GITHUB_ACTIONS') != 'true'
            or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted'):
        raise RuntimeError('refusing outside a disposable GitHub-hosted runner')
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError('runner administrator token required for owned SYSTEM task')


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def wait_json(path, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding='utf-8'))
        time.sleep(.05)
    raise TimeoutError(f'fixture receipt deadline: {Path(path).name}')


def isolated_env(base):
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'SYSTEMDRIVE',
        'GITHUB_ACTIONS', 'RUNNER_ENVIRONMENT'}}
    env.update(HOME=str(base / 'home'), USERPROFILE=str(base / 'home'),
               HERMES_HOME=str(base / 'state'), APPDATA=str(base / 'appdata'),
               LOCALAPPDATA=str(base / 'localappdata'), TEMP=str(base / 'tmp'),
               TMP=str(base / 'tmp'), PYTHONPATH=str(REPO), PYTHONUTF8='1',
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    for key in ('HOME', 'HERMES_HOME', 'APPDATA', 'LOCALAPPDATA', 'TEMP'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


@contextmanager
def environment(env):
    original = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def serve(config_path):
    """Same executable/pipe DACL for ordinary and genuine SYSTEM tokens."""
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    os.environ.clear()
    os.environ.update(config['env'])
    require_runner()
    sys.path.insert(0, str(REPO))
    import _winapi as win
    from gateway.runtime_bootstrap_windows import (
        _api, _complete, _disconnect_pipe, _process_sid, _write,
    )
    # A permissive decoy deliberately does NOT perform peer authorization.
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', ctypes.c_void_p),
                    ('inherit', wintypes.BOOL)]
    adv = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    descriptor = ctypes.c_void_p()
    convert = _api(adv, 'ConvertStringSecurityDescriptorToSecurityDescriptorW',
                   wintypes.BOOL, [wintypes.LPCWSTR, wintypes.DWORD,
                                  ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    if not convert(SDDL, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    attrs = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
    handle = None
    receipt = {'pid': os.getpid(), 'sid': _process_sid(os.getpid()),
               'sddl': SDDL, 'connections': [], 'stopped': False}
    try:
        handle = win.CreateNamedPipe(config['pipe'],
            3 | win.FILE_FLAG_OVERLAPPED | 0x00080000, 8, 1, 65536, 65536,
            2000, ctypes.addressof(attrs))
        atomic_json(config['ready'], receipt)
        deadline = time.monotonic() + 240
        while not Path(config['stop']).exists() and time.monotonic() < deadline:
            connected = False
            data = bytearray()
            record = {}
            try:
                ov = win.ConnectNamedPipe(handle, overlapped=True)
                _complete(ov, time.monotonic() + .5)
                connected = True
                client = wintypes.ULONG()
                get_pid = _api(kernel, 'GetNamedPipeClientProcessId', wintypes.BOOL,
                              [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)])
                if not get_pid(handle, ctypes.byref(client)):
                    raise ctypes.WinError(ctypes.get_last_error())
                record['clientPid'] = client.value
                # Read actual OS bytes, including partial data before disconnect.
                io_deadline = time.monotonic() + 12
                while b'\n' not in data and len(data) < 65536:
                    ov, _ = win.ReadFile(handle, 4096, overlapped=True)
                    count = _complete(ov, io_deadline)
                    data.extend(ov.getbuffer()[:count])
                    if not count:
                        break
                if b'\n' in data:
                    _write(handle, (json.dumps(config['response']) + '\n').encode(), io_deadline)
                    # Wait for the reader to consume/close, not FlushFileBuffers
                    # (which has an unbounded blocking wait).
                    ov, _ = win.ReadFile(handle, 4096, overlapped=True)
                    count = _complete(ov, io_deadline)
                    data.extend(ov.getbuffer()[:count])
            except (OSError, TimeoutError, ConnectionError) as error:
                record['close'] = type(error).__name__
                record['winerror'] = getattr(error, 'winerror', None)
            finally:
                if connected:
                    record['requestBytes'] = len(data)
                    record['request'] = data.decode('utf-8', errors='replace')
                    receipt['connections'].append(record)
                    atomic_json(config['events'], receipt)
                try:
                    _disconnect_pipe(handle)
                except OSError as error:
                    if error.winerror != 233:
                        raise
        receipt['stopped'] = True
    finally:
        if handle is not None:
            win.CloseHandle(handle)
        _api(kernel, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(descriptor)
        atomic_json(config['events'], receipt)


def task_command(*args, check=True):
    result = subprocess.run(['schtasks.exe', *args], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=20, check=check)
    # schtasks XML can be UTF-16 when stdout is redirected; ordinary status
    # output uses the console codepage. Task names/SIDs are ASCII regardless.
    for stream in ('stdout', 'stderr'):
        raw = getattr(result, stream)
        encoding = ('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else
                    'utf-16-le' if b'\x00' in raw[:100] else 'utf-8-sig')
        setattr(result, stream, raw.decode(encoding, errors='replace'))
    return result


def task_xml(python, config):
    ns = 'http://schemas.microsoft.com/windows/2004/02/mit/task'
    ET.register_namespace('', ns)
    def add(parent, tag, text=None, **attrs):
        child = ET.SubElement(parent, '{' + ns + '}' + tag, attrs)
        child.text = text
        return child
    root = ET.Element('{' + ns + '}Task', {'version': '1.2'})
    principals = add(root, 'Principals')
    principal = add(principals, 'Principal', id='System')
    add(principal, 'UserId', 'S-1-5-18')
    add(principal, 'RunLevel', 'HighestAvailable')
    settings = add(root, 'Settings')
    add(settings, 'ExecutionTimeLimit', 'PT5M')
    add(settings, 'Enabled', 'true')
    actions = add(root, 'Actions', Context='System')
    execute = add(actions, 'Exec')
    add(execute, 'Command', python)
    add(execute, 'Arguments', subprocess.list2cmdline([
        '-P', str(Path(__file__).resolve()), '--serve', str(config)]))
    add(execute, 'WorkingDirectory', str(config.parent))
    return ET.tostring(root, encoding='utf-16', xml_declaration=True)


class Decoy:
    def __init__(self, base, label, python, env, system=False):
        from gateway.control_socket import windows_pipe_name
        self.folder = base / label
        self.folder.mkdir()
        self.home = (self.folder / 'profile').resolve()
        self.home.mkdir()
        self.process = None
        self.server_handle = None
        self.task_started = False
        self.process_exit_verified = False
        self.task = 'HermesSidProbe-' + uuid.uuid4().hex if system else None
        self.registered = False
        self.log = None
        self.cleaned = False
        self.endpoint = {'profile_id': str(self.home), 'instance_id': 'sid-probe-' + label,
                         'runtime_protocol': 1, 'authority_epoch': 1,
                         'api_origin': 'http://127.0.0.1:1',
                         'capabilities': ['session-authority-v1'], 'supervisor': 'fixture'}
        self.response = {'protocol': 1, 'id': 1, 'ok': True,
                         'result': {**self.endpoint, 'ticket': FIXTURE_TICKET}}
        self.config = {'pipe': windows_pipe_name(self.home), 'env': env,
                       'response': self.response,
                       **{key: str(self.folder / (key + '.json'))
                          for key in ('ready', 'events', 'stop')}}
        self.config_path = self.folder / 'config.json'
        atomic_json(self.config_path, self.config)
        self.python = python

    def start(self):
        if self.task:
            xml = self.folder / 'task.xml'
            xml.write_bytes(task_xml(self.python, self.config_path))
            # Track the attempted registration too: /Create can time out after
            # the scheduler has committed it. The name is exclusively ours.
            self.registered = True
            task_command('/Create', '/TN', self.task, '/XML', str(xml))
            # Read back the exact task before starting it.
            registered = task_command('/Query', '/TN', self.task, '/XML').stdout
            principal = ET.fromstring(registered).find('.//{*}Principal/{*}UserId')
            if principal is None or principal.text not in ('S-1-5-18', 'SYSTEM'):
                raise AssertionError('task principal was not SYSTEM')
            self.task_started = True
            task_command('/Run', '/TN', self.task)
        else:
            self.log = (self.folder / 'server.log').open('w', encoding='utf-8')
            self.process = subprocess.Popen([
                self.python, '-P', str(Path(__file__).resolve()), '--serve', str(self.config_path)],
                env=self.config['env'], cwd=self.folder, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT)
        identity = wait_json(self.config['ready'])
        from gateway.runtime_bootstrap_windows import _api
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        open_process = _api(kernel, 'OpenProcess', wintypes.HANDLE,
                            [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD])
        self.server_handle = open_process(0x101001, False, identity['pid'])
        if not self.server_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return identity

    def events(self, count):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            path = Path(self.config['events'])
            if path.exists():
                receipt = json.loads(path.read_text(encoding='utf-8'))
                if len(receipt['connections']) >= count:
                    return receipt
            time.sleep(.05)
        raise TimeoutError('decoy connection receipt did not arrive')

    def close(self):
        Path(self.config['stop']).touch()
        termination_error = None
        if self.server_handle:
            # A retained OS handle pins the exact process, avoiding PID reuse
            # during teardown and making the cleanup receipt non-vacuous.
            from gateway.runtime_bootstrap_windows import _api
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            wait = _api(kernel, 'WaitForSingleObject', wintypes.DWORD,
                        [wintypes.HANDLE, wintypes.DWORD])
            terminate = _api(kernel, 'TerminateProcess', wintypes.BOOL,
                             [wintypes.HANDLE, wintypes.UINT])
            try:
                if wait(self.server_handle, 15000) != 0:
                    if not terminate(self.server_handle, 1):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if wait(self.server_handle, 10000) != 0:
                        raise AssertionError('owned server process survived termination')
                self.process_exit_verified = True
            except Exception as error:
                termination_error = error
            finally:
                _api(kernel, 'CloseHandle', wintypes.BOOL,
                     [wintypes.HANDLE])(self.server_handle)
                self.server_handle = None
        if self.process:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.log.close()
        if self.registered:
            task_command('/End', '/TN', self.task, check=False)
            task_command('/Delete', '/TN', self.task, '/F', check=False)
            # A nonzero query alone could be an access error, so enumerate all
            # names successfully and prove this exact unique task is absent.
            listing = task_command('/Query', '/FO', 'CSV', '/NH').stdout
            if self.task in listing:
                raise AssertionError('owned scheduled task survived deletion')
        if termination_error:
            raise termination_error
        if self.task_started and not self.process_exit_verified:
            raise AssertionError('task removed but SYSTEM process exit was not independently verified')
        self.cleaned = True


def raw_exchange(home, request):
    """DACL control only: deliberately omit SID policy; not a product client."""
    import _winapi as win
    from gateway.control_socket import windows_pipe_name
    from gateway.runtime_bootstrap_windows import _read_line, _write
    handle = win.CreateFile(windows_pipe_name(home), 0xC0000000, 0, win.NULL,
                            win.OPEN_EXISTING, win.FILE_FLAG_OVERLAPPED | 0x00100000, win.NULL)
    try:
        deadline = time.monotonic() + 5
        _write(handle, request, deadline)
        return _read_line(handle, deadline, 65536) + b'\n'
    finally:
        win.CloseHandle(handle)


def electron_probe(base, output, python, electron, env, positive, negative):
    """Compile a scratch main importing the actual production ticket bootstrap."""
    node = shutil.which('node')
    if not node:
        raise RuntimeError('node missing')
    if electron == 'auto':
        electron = subprocess.check_output([node, '-e', 'console.log(require("electron"))'],
            cwd=REPO / 'apps/desktop', text=True, timeout=15).strip()
    source = base / 'sid-electron.ts'
    module = json.dumps(str(REPO / 'apps/desktop/electron/local-gateway.ts'))
    helper = json.dumps(str(REPO / 'apps/desktop/electron/local-gateway-python.ts'))
    source.write_text('''import { app } from 'electron'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { configureWindowsGatewayTicketClient, mintLocalGatewayTicket } from MODULE
import { mintGatewayTicketWithPython } from HELPER
const input = JSON.parse(fs.readFileSync(process.env.SID_PROBE_INPUT!, 'utf8'))
const receipt: any = { passed: false, platform: process.platform, electron: process.versions.electron, processType: process.type, checks: [] }
app.setPath('userData', input.userData)
app.disableHardwareAcceleration()
const timer = setTimeout(() => app.exit(2), 60000)
app.whenReady().then(async () => {
  assert.equal(process.platform, 'win32')
  assert.equal(process.type, 'browser')
  configureWindowsGatewayTicketClient((endpoint, purpose) => mintGatewayTicketWithPython(
    { command: input.python, env: { PYTHONPATH: input.repo } }, input.repo, endpoint, purpose))
  for (const purpose of ['interactive', 'native-http'] as const) {
    assert.equal(await mintLocalGatewayTicket(input.positive, purpose), input.ticket)
    await assert.rejects(mintLocalGatewayTicket(input.negative, purpose),
      { message: 'Gateway ticket bootstrap failed' })
    receipt.checks.push({ purpose, sameUser: 'accepted inert fixture response', otherUser: 'rejected' })
  }
  receipt.passed = true
}).catch(error => { receipt.error = String(error.stack || error) }).finally(() => {
  clearTimeout(timer)
  fs.writeFileSync(input.receipt, JSON.stringify(receipt, null, 2))
  app.exit(receipt.passed ? 0 : 1)
})
'''.replace('MODULE', module).replace('HELPER', helper), encoding='utf-8')
    bundle = base / 'sid-electron.cjs'
    build = "require('esbuild').buildSync({entryPoints:[process.argv[1]],outfile:process.argv[2],bundle:true,platform:'node',format:'cjs',external:['electron']})"
    subprocess.run([node, '-e', build, str(source), str(bundle)], cwd=REPO / 'apps/desktop',
                   check=True, timeout=60)
    user_data = base / 'electron-user-data'
    user_data.mkdir()
    input_path = base / 'electron-input.json'
    receipt_path = output / 'electron-sid.json'
    atomic_json(input_path, {'python': python, 'repo': str(REPO),
        'positive': positive.endpoint, 'negative': negative.endpoint,
        'ticket': FIXTURE_TICKET, 'receipt': str(receipt_path), 'userData': str(user_data)})
    with (output / 'electron-sid.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([electron, str(bundle)], cwd=REPO,
            env={**env, 'SID_PROBE_INPUT': str(input_path)}, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, timeout=75)
    receipt = wait_json(receipt_path, 2)
    assert result.returncode == 0 and receipt['passed'], receipt
    return receipt


def probe(output, python, electron=None):
    require_runner()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'passed': False, 'host': sys.platform,
               'boundary': 'real process-token SID / permissive named-pipe decoy'}
    decoys = []
    with tempfile.TemporaryDirectory(prefix='h-sid-') as temporary:
        base = Path(temporary).resolve()
        env = isolated_env(base)
        try:
            with environment(env):
                sys.path.insert(0, str(REPO))
                from gateway.runtime_bootstrap_windows import _process_sid, query_runtime_control
                current_sid = _process_sid(os.getpid())
                assert current_sid != 'S-1-5-18', 'runner already SYSTEM; not a second token'
                for label, system in (('same-user', False), ('system-user', True)):
                    decoy = Decoy(base, label, python, env, system)
                    decoys.append(decoy)
                    identity = decoy.start()
                    # Parent independently queries the real server process token.
                    observed = _process_sid(identity['pid'])
                    assert observed == identity['sid']
                    assert (observed != current_sid) == system
                    if system:
                        assert observed == 'S-1-5-18'
                    receipt[label] = {**identity, 'parentObservedSid': observed}
                positive, negative = decoys
                receipt['client'] = {'pid': os.getpid(), 'sid': current_sid}
                request = b'{"protocol":1,"id":1,"verb":"sid-probe"}\n'
                response = query_runtime_control(positive.home, request, timeout=5)
                assert json.loads(response) == positive.response
                same = positive.events(1)['connections'][0]
                assert same['request'].encode() == request
                # Prove the SAME SYSTEM pipe accepts request/response bytes with
                # identical CreateFile access flags when only SID policy is omitted.
                assert json.loads(raw_exchange(negative.home, request)) == negative.response
                acl = negative.events(1)['connections'][0]
                assert acl['request'].encode() == request
                try:
                    query_runtime_control(negative.home, request, timeout=5)
                except PermissionError as error:
                    assert str(error) == DENIAL, str(error)
                    receipt['pythonDenial'] = str(error)
                else:
                    raise AssertionError('production client accepted another OS user')
                denied = negative.events(2)['connections'][1]
                assert denied['requestBytes'] == 0, denied
                assert denied['clientPid'] == os.getpid(), denied
                receipt['sameUserPositive'] = same
                receipt['crossUserDaclPositive'] = acl
                receipt['wrongSidNegative'] = denied
                if electron:
                    receipt['electron'] = electron_probe(base, output, python, electron,
                                                        env, positive, negative)
                    same_events = positive.events(3)['connections']
                    wrong_events = negative.events(4)['connections']
                    assert len(same_events) == 3 and len(wrong_events) == 4
                    assert all(row['requestBytes'] > 0 for row in same_events)
                    assert all(row['requestBytes'] == 0 for row in wrong_events[1:])
                    for row, purpose in zip(same_events[1:], ('interactive', 'native-http')):
                        assert json.loads(row['request'])['params']['purpose'] == purpose
                else:
                    receipt['electron'] = {'status': 'not_exercised', 'reason': 'pass --electron-auto'}
                receipt['passed'] = True
        except Exception as error:
            receipt['error'] = f'{type(error).__name__}: {error}'
            receipt['passed'] = False
        finally:
            cleanup_errors = []
            for decoy in reversed(decoys):
                try:
                    decoy.close()
                    events = Path(decoy.config['events'])
                    if events.exists():
                        atomic_json(output / (decoy.folder.name + '-events.json'),
                                    json.loads(events.read_text(encoding='utf-8')))
                except Exception as error:
                    cleanup_errors.append(f'{type(error).__name__}: {error}')
            receipt['ownedFixturesCleaned'] = all(d.cleaned for d in decoys)
            if cleanup_errors:
                receipt['cleanupErrors'] = cleanup_errors
                receipt['passed'] = False
            atomic_json(output / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--electron')
    parser.add_argument('--electron-auto', action='store_true')
    parser.add_argument('--serve', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        if os.name != 'nt':
            parser.error('decoy requires Windows')
        serve(args.serve)
        return 0
    if not args.output:
        parser.error('--output is required')
    receipt = probe(args.output, os.path.abspath(args.python),
                    'auto' if args.electron_auto else args.electron)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
