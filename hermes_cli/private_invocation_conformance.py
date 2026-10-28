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
    if not (0 < memory <= 6 * 1024 * 1024 * 1024 and 0 < tasks <= 256 and 0 < quota <= 4 * period):
        raise ValueError('fixture exceeds admitted 4 CPU / 6 GiB / 256 task allowance')
    current = Path('/sys/fs/cgroup' + next(line.split(':', 2)[2].strip() for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0:')))
    if current.resolve() != root:
        raise ValueError('driver must already be owned by the admitted supervisor cgroup')
    return root


def run(args):
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    required = {'home', 'gatewayPython', 'issuerFactory', 'journal', 'cgroup', 'readbackUrl', 'readbackTlsFactory', 'componentRoot'}
    if set(fixture) != required:
        raise ValueError('exact real-fixture manifest required')
    home = Path(fixture['home']).resolve()
    if not home.is_dir() or not (home / 'config.yaml').is_file():
        raise ValueError('disposable native gateway home/config missing')
    group = _bounded_group(fixture['cgroup'])
    # The driver joins before spawning; all descendant threads/processes inherit
    # the aggregate bound. No root provisioning or resource increase is performed.
    # Membership was independently provisioned and verified before any allocation.
    from tools.environments.local import served_profile_child_env
    # An explicit minimal base prevents launch-profile credentials, selectors and
    # import paths from becoming a second source of fixture authority.
    base = {'PATH': os.defpath, 'LANG': 'C.UTF-8'}
    env = served_profile_child_env(base, target_home=home, inherit_credentials=True)
    issuer_env = served_profile_child_env(base, target_home=home, inherit_credentials=False)
    module_path = os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(Path(fixture['componentRoot']).resolve())])
    env['PYTHONPATH'] = issuer_env['PYTHONPATH'] = module_path
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
            children.append(subprocess.Popen(command, env=issuer_env if index == 0 else env, stdout=log, stderr=subprocess.STDOUT))
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
        sys.path.insert(0, str(Path(fixture['componentRoot']).resolve()))
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
