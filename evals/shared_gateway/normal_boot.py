"""Live POSIX bootstrap descriptor check, not an execution/readiness handshake proof."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time


def identify(state):
    path = state / 'gateway.sock'
    pointer = state / 'gateway.sock.path'
    if pointer.exists():
        path = Path(pointer.read_text(encoding='utf-8').strip())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(1)
        peer.connect(str(path))
        peer.sendall(b'{"protocol":1,"verb":"identify","id":1}\n')
        data = bytearray()
        while b'\n' not in data and len(data) < 524288:
            chunk = peer.recv(65536)
            if not chunk:
                break
            data.extend(chunk)
    response = json.loads(data.split(b'\n', 1)[0])
    if response.get('ok') is not True or response.get('id') != 1:
        raise ValueError('control request rejected')
    result = response['result']
    # Never retain credentials or arbitrary server payloads in the receipt.
    return {key: result.get(key) for key in (
        'state', 'runtime_protocol', 'authority_epoch', 'api_origin', 'capabilities')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=30)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('this probe requires native POSIX; Windows needs its named-pipe probe')
    if args.deadline <= 0:
        parser.error('--deadline must be positive')
    repo = args.repo.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, stdin=subprocess.DEVNULL, text=True).strip(),
        'surface': 'gateway.run control descriptor', 'passed': False, 'observations': []}
    with tempfile.TemporaryDirectory(prefix='ugw-boot-') as directory:
        root = Path(directory)
        home, state = root / 'home', root / 'state'
        home.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        (state / 'config.yaml').write_text(
            'gateway:\n  multiplex_profiles: false\nauxiliary:\n  title_generation:\n    enabled: false\n',
            encoding='utf-8')
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
        env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
                   PYTHONPATH=str(repo), PYTHONUNBUFFERED='1')
        with args.output.with_suffix('.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [str(args.python.absolute()), '-m', 'gateway.run'], cwd=repo, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                deadline = time.monotonic() + args.deadline
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        snapshot = identify(state)
                        if not receipt['observations'] or receipt['observations'][-1] != snapshot:
                            receipt['observations'].append(snapshot)
                        if (snapshot['state'] == 'ready' and snapshot['runtime_protocol'] == 1
                                and snapshot['api_origin'] and snapshot['authority_epoch']):
                            receipt['passed'] = True
                            break
                    except (OSError, ValueError) as error:
                        receipt['last_probe_error'] = type(error).__name__
                    time.sleep(0.25)
                receipt['alive_at_end'] = process.poll() is None
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)  # windows-footgun: ok — POSIX-only owned process
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only owned process
                        process.wait(timeout=5)
                        receipt['forced_cleanup'] = True
                receipt['exit_code'] = process.returncode
    args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
