"""Run an actual Electron main against ordinary, disposable gateway daemons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import psutil

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hermes_cli.gateway_runtime_discovery import query_identify  # noqa: E402


def stop_owned(process):
    if process.poll() is not None:
        return
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    for item in reversed(children):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    _, alive = psutil.wait_procs(children, timeout=5)
    for item in alive:
        item.kill()
    psutil.wait_procs(alive, timeout=5)


def ready(process, home):
    deadline = time.monotonic() + 90
    last = None
    while time.monotonic() < deadline and process.poll() is None:
        try:
            endpoint = query_identify(home, timeout=2)
            if endpoint.get('state') == 'ready' and endpoint.get('api_origin'):
                profiles = [item for item in endpoint['served_profiles']
                            if Path(item['home']).resolve() == home.resolve()]
                if len(profiles) != 1:
                    raise ValueError('Daemon did not publish exactly our profile')
                return {**endpoint, 'profile_id': profiles[0]['profile_id']}
        except (OSError, ValueError) as error:
            last = f'{type(error).__name__}: {error}'
        time.sleep(0.2)
    raise RuntimeError(f'Owned daemon not ready: exit={process.poll()}, last={last}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--electron', help='Native Electron binary, not electron/index.js')
    parser.add_argument('--full-app', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Keep the venv launcher path; resolving symlinks would discard the venv.
    python = os.path.abspath(args.python)
    node = shutil.which('node')
    if not node:
        parser.error('node is required')
    electron = args.electron or subprocess.check_output(
        [node, '-e', 'console.log(require("electron"))'], cwd=REPO / 'apps/desktop',
        stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', timeout=15).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'passed': False, 'host': sys.platform}
    processes, logs = [], []
    temporary_directory = tempfile.TemporaryDirectory(prefix='h-ticket-')
    try:
        temporary = temporary_directory.name
        base = Path(temporary).resolve()
        home, state = base / 'home', base / 'state'
        sibling = home / '.hermes' / 'profiles' / 'sibling'
        for directory in (home, state, sibling, base / 'user-data', base / 'tmp'):
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        env = {key: value for key, value in os.environ.items() if key.upper() in {
            'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'SYSTEMDRIVE',
            'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR', 'XAUTHORITY',
            'LD_LIBRARY_PATH', 'LANG', 'TZ'}}
        env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
                   APPDATA=str(home / 'AppData' / 'Roaming'),
                   LOCALAPPDATA=str(home / 'AppData' / 'Local'),
                   XDG_CONFIG_HOME=str(home / '.config'), XDG_CACHE_HOME=str(home / '.cache'),
                   TEMP=str(base / 'tmp'), TMP=str(base / 'tmp'), TMPDIR=str(base / 'tmp'),
                   PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1', PYTHONUTF8='1',
                   PYTHONIOENCODING='utf-8')
        endpoints = []
        for index, profile in enumerate((state, sibling)):
            (profile / 'config.yaml').write_text(
                'gateway:\n  multiplex_profiles: false\nauxiliary:\n  title_generation:\n    enabled: false\n',
                encoding='utf-8')
            log = (output / f'daemon-{index}.log').open('w', encoding='utf-8')
            logs.append(log)
            process = subprocess.Popen([python, '-m', 'gateway.run'], cwd=REPO,
                                       env={**env, 'HERMES_HOME': str(profile)},
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            endpoints.append(ready(process, profile))
        bundle = base / 'electron-main.cjs'
        subprocess.run([node, str(Path(__file__).with_name('build.mjs')), str(bundle)],
                       cwd=REPO, stdin=subprocess.DEVNULL, check=True, timeout=60)
        input_file = base / 'input.json'
        input_file.write_text(json.dumps({
            'repo': str(REPO), 'python': python, 'endpoint': endpoints[0],
            'sibling': endpoints[1], 'userData': str(base / 'user-data'),
            'receipt': str(output / 'receipt.json'),
            'fullReceipt': str(output / 'full-app.json')}), encoding='utf-8')
        electron_log = (output / 'electron.log').open('w', encoding='utf-8')
        logs.append(electron_log)
        # Linux's headless Ozone backend is a real native Electron runtime.
        flags = ['--ozone-platform=headless', '--no-sandbox'] if sys.platform == 'linux' else []
        native = subprocess.Popen([electron, *flags, str(bundle)], cwd=REPO,
                                  env={**env, 'NATIVE_TICKET_INPUT': str(input_file)},
                                  stdin=subprocess.DEVNULL, stdout=electron_log, stderr=subprocess.STDOUT)
        processes.append(native)
        status = native.wait(timeout=120)
        receipt = json.loads((output / 'receipt.json').read_text(encoding='utf-8'))
        if status or not receipt.get('passed'):
            raise RuntimeError(f'Electron probe failed: exit={status}')
        if args.full_app:
            full_log = (output / 'full-app.log').open('w', encoding='utf-8')
            logs.append(full_log)
            full = subprocess.Popen([node, str(Path(__file__).with_name('full-app.mjs'))],
                                    cwd=REPO, env={**env, 'NATIVE_TICKET_INPUT': str(input_file)},
                                    stdin=subprocess.DEVNULL, stdout=full_log, stderr=subprocess.STDOUT)
            processes.append(full)
            full_status = full.wait(timeout=180)
            receipt['fullApp'] = json.loads((output / 'full-app.json').read_text(encoding='utf-8'))
            if full_status or not receipt['fullApp'].get('passed'):
                raise RuntimeError(f'Full Electron app failed: exit={full_status}')
    except Exception as error:
        receipt['passed'] = False
        receipt['harnessError'] = str(error)
    finally:
        for process in reversed(processes):
            stop_owned(process)
        for log in logs:
            log.close()
        temporary_directory.cleanup()
        receipt['ownedProcessesStopped'] = all(process.poll() is not None for process in processes)
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
