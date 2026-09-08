"""Linux live Git/OpenSSH PTY probe; fixture ssh config, no user state."""
import json
import os
from pathlib import Path
import pty
import pwd
import select
import signal
import socket
import subprocess
import sys
import time

BASE = Path(os.environ['AUDIT_DIR'])
REPO = Path(os.environ['AUDIT_REPO'])
BASE.mkdir(parents=True, exist_ok=True)
env = {'PATH': f'{BASE}/bin:/usr/bin:/bin', 'HOME': str(BASE/'home'),
       'HERMES_HOME': str(BASE/'hermes'), 'PYTHONPATH': str(REPO),
       'PYTHONDONTWRITEBYTECODE': '1', 'AUDIT_DIR': str(BASE), 'AUDIT_REPO': str(REPO)}
for name in ('home', 'hermes', 'bin'):
    (BASE/name).mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
from hermes_cli import banner

if len(sys.argv) > 1:
    os.environ.clear()
    os.environ.update(env)
    case = json.loads((BASE/'case.json').read_text())
    os.environ.update(case['env'])
    start = time.monotonic()
    result = banner._check_via_local_git(BASE/'checkout')
    print('PRODUCTION_RETURN '+json.dumps({'result': result, 'elapsed': time.monotonic()-start}), flush=True)
    time.sleep(2)
    sys.exit(0)


def run(args, **kw):
    return subprocess.run(args, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, check=True, timeout=15, **kw).stdout.strip()


def git(*args):
    return run(['git', '-C', str(BASE/'checkout'), *args])


def probe(name, extra_env):
    (BASE/'case.json').write_text(json.dumps({'env': extra_env}))
    child, fd = pty.fork()
    if child == 0:
        python = str(REPO/'.venv/bin/python')
        os.execve(python, [python, '-B', __file__, 'child'], env)
    text = ''
    start = time.monotonic()
    try:
        while time.monotonic()-start < 13:
            ready, _, _ = select.select([fd], [], [], .2)
            if ready:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                text += data.decode(errors='replace')
        descendants = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                pid = int(path.name)
                if os.getpgid(pid) == child and pid != child:
                    cmd = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    if cmd:
                        descendants.append({'pid': pid, 'cmdline': cmd})
            except (OSError, ProcessLookupError):
                pass
        returned = [line.split('PRODUCTION_RETURN ', 1)[1] for line in text.splitlines()
                    if 'PRODUCTION_RETURN ' in line]
        return {'name': name, 'pty': text, 'prompt': 'Are you sure you want' in text,
                'return': json.loads(returned[-1]) if returned else None,
                'live_descendants': descendants}
    finally:
        os.close(fd)
        try:
            os.killpg(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(child, 0)


for key in ('host_key', 'client_key'):
    run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(BASE/key)])
(BASE/'authorized_keys').write_text((BASE/'client_key.pub').read_text())
(BASE/'checkout').mkdir()
git('init', '-b', 'main')
git('-c', 'user.name=Probe', '-c', 'user.email=probe@localhost', 'commit', '--allow-empty', '-m', 'probe')
run(['git', 'clone', '--bare', str(BASE/'checkout'), str(BASE/'remote.git')])
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
user = pwd.getpwuid(os.getuid()).pw_name
config = BASE/'sshd_config'
config.write_text(f'Port {port}\nListenAddress 127.0.0.1\nHostKey {BASE}/host_key\nPidFile {BASE}/sshd.pid\nAuthorizedKeysFile {BASE}/authorized_keys\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\nStrictModes no\nLogLevel VERBOSE\n')
ssh_config = BASE/'ssh_config'
ssh_config.write_text(f'Host *\n User {user}\n UserKnownHostsFile {BASE}/known_hosts\n GlobalKnownHostsFile /dev/null\n IdentityFile {BASE}/client_key\n ConnectTimeout 3\n')
wrapper = BASE/'bin/ssh'
wrapper.write_text(f'#!/bin/sh\nexec /usr/bin/ssh -F {ssh_config} "$@"\n')
wrapper.chmod(0o700)
git('remote', 'add', 'origin', f'ssh://{user}@127.0.0.1:{port}{BASE}/remote.git')
log = (BASE/'sshd.log').open('w')
server = subprocess.Popen(['/usr/sbin/sshd', '-D', '-e', '-f', str(config)], env=env,
                          stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
agent = None
rows = []
try:
    for _ in range(40):
        if server.poll() is not None:
            raise RuntimeError('sshd exited: '+(BASE/'sshd.log').read_text())
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2) as conn:
                assert conn.recv(256).startswith(b'SSH-')
            break
        except OSError:
            time.sleep(.1)
    else:
        raise RuntimeError('sshd readiness timed out')
    rows.append(probe('unofficial_unknown_host_default', {}))
    pub = (BASE/'host_key.pub').read_text().split()
    (BASE/'known_hosts').write_text(f'[127.0.0.1]:{port} {pub[0]} {pub[1]}\n')
    rows.append(probe('trusted_host_file_key', {}))
    agent = subprocess.Popen(['ssh-agent', '-D', '-a', str(BASE/'agent.sock')], env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(40):
        if (BASE/'agent.sock').exists():
            break
        time.sleep(.1)
    env['SSH_AUTH_SOCK'] = str(BASE/'agent.sock')
    run(['ssh-add', str(BASE/'client_key')])
    ssh_config.write_text(ssh_config.read_text().replace(f'IdentityFile {BASE}/client_key', 'IdentityFile none'))
    rows.append(probe('trusted_host_agent_key', {'SSH_AUTH_SOCK': str(BASE/'agent.sock')}))
    (BASE/'known_hosts').unlink()
    rows.append(probe('explicit_interactive_override', {'GIT_SSH_COMMAND': 'ssh -o BatchMode=no'}))
finally:
    for proc in (agent, server):
        if proc is not None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
    log.close()
result = {'platform': sys.platform, 'production': banner.__file__, 'rows': rows,
          'isolation': 'PATH ssh adapter only adds -F fixture config; execs real /usr/bin/ssh; loopback sshd and disposable git repos',
          'server_stopped': server.poll() is not None, 'agent_stopped': agent is None or agent.poll() is not None}
(BASE/'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

