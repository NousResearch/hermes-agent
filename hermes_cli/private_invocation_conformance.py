"""Separate-process/network development conformance entrypoint; no synthetic origin.

Requires an already provisioned disposable native messaging fixture, the real
paired receiver, authenticated readback, and a prebounded owned cgroup. It does
not install dependencies, create credentials, forge events or start the receiver.
Observations never mark the acceptance matrix passed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def _bounded_group(group):
    root = Path(group).resolve()
    if not str(root).startswith('/sys/fs/cgroup/'):
        raise ValueError('owned preprovisioned cgroup required')
    memory = int((root / 'memory.max').read_text())
    tasks = int((root / 'pids.max').read_text())
    quota, period = map(int, (root / 'cpu.max').read_text().split())
    if not (0 < memory <= 450 * 1024 * 1024 and 0 < tasks <= 12 and 0 < quota <= period):
        raise ValueError('fixture exceeds 1 CPU / 450 MiB / 12 task allowance')
    if (root / 'cgroup.procs').read_text().strip():
        raise ValueError('fixture cgroup must initially be empty and independently owned')
    return root


def run(args):
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    required = {'home', 'gatewayPython', 'issuerFactory', 'journal', 'cgroup', 'readbackUrl', 'readbackTlsFactory'}
    if set(fixture) != required:
        raise ValueError('exact real-fixture manifest required')
    home = Path(fixture['home']).resolve()
    if not home.is_dir() or not (home / 'config.yaml').is_file():
        raise ValueError('disposable native gateway home/config missing')
    group = _bounded_group(fixture['cgroup'])
    # The driver joins before spawning; all descendant threads/processes inherit
    # the aggregate bound. No root provisioning or resource increase is performed.
    (group / 'cgroup.procs').write_text(str(os.getpid()), encoding='ascii')
    env = dict(os.environ)
    env['HERMES_HOME'] = str(home)
    commands = [[sys.executable, '-m', 'hermes_cli.private_invocation_transport',
                 '--factory', fixture['issuerFactory']],
                [fixture['gatewayPython'], '-m', 'hermes_cli.main', 'gateway', 'run']]
    children, files = [], []
    output = Path(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        for index, command in enumerate(commands):
            log_path = output / f'process-{index}.private.log'
            log = os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')
            files.append(log)
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        print('Use genuine human native messaging events in the disposable fixture now.', flush=True)
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('real product child exited; inspect private process logs')
            time.sleep(0.5)
        # Read only the actual on-disk issuer journal; never insert a claimed origin.
        with sqlite3.connect(Path(fixture['journal']).resolve().as_uri() + '?mode=ro', uri=True) as db:
            observed = db.execute("SELECT COUNT(*) FROM operations WHERE status='observed'").fetchone()[0]
            unknown = db.execute("SELECT COUNT(*) FROM operations WHERE status='unknown'").fetchone()[0]
        # Independent readback is an actual authenticated network GET, never an
        # in-process application client or a factory returning a claimed result.
        import importlib
        import ssl
        from urllib.parse import urlsplit
        from urllib.request import build_opener, HTTPSHandler, HTTPRedirectHandler, ProxyHandler
        module, name = fixture['readbackTlsFactory'].split(':', 1)
        tls = getattr(importlib.import_module(module), name)()
        url = urlsplit(fixture['readbackUrl'])
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or tls.verify_mode != ssl.CERT_REQUIRED or not tls.check_hostname):
            raise ValueError('authenticated independent HTTPS readback required')

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise ValueError('readback redirect refused')

        opener = build_opener(ProxyHandler({}), HTTPSHandler(context=tls), NoRedirect())
        with opener.open(fixture['readbackUrl'], timeout=15) as response:
            data = response.read(1048577)
            if len(data) > 1048576:
                raise ValueError('readback exceeds limit')
            readback = json.loads(data)
        receipt = {'acceptance': 'unqualified', 'observedOperations': observed,
                   'unknownOperations': unknown, 'independentReadback': readback,
                   'commands': commands, 'fixtureSha256': hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest()}
        (output / 'observations.private.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        # Running one observation is never proof of all seven contract rows.
        return 2
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in files:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=int, default=60, choices=range(1, 121))
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError):
        # Don't echo fixture paths, TLS credentials or opaque references in errors.
        print('Real native/authenticated paired fixture unavailable or failed; no acceptance claim.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
