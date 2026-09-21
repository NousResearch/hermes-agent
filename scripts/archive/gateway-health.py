#!/usr/bin/env python3
"""Deterministic gateway process health check.
Groups gateway processes by effective HERMES_HOME, flags true duplicates only.

Expected state: 1 default + N profile gateways, all with unique HERMES_HOME.
Real duplicate: 2+ PIDs with the same effective HERMES_HOME.

Output: JSON with groups, true_duplicates, missing_default status.
Exit 0 on healthy, 1 on any finding.
"""

import json
import os
import glob
from collections import defaultdict

GATEWAY_CMD_SIGNATURE = ['-m', 'hermes_cli.main', 'gateway', 'run']
DEFAULT_HERMES_HOME = '/home/kensei/.hermes'

def parse_cmdline(pid_dir):
    """Parse /proc/PID/cmdline into a list of args."""
    try:
        raw = open(f'{pid_dir}/cmdline', 'rb').read()
        parts = [p.decode('utf-8', 'replace') for p in raw.split(b'\0') if p]
        return parts
    except (OSError, IOError):
        return []

def parse_environ(pid_dir):
    """Parse /proc/PID/environ into a dict of env vars."""
    env = {}
    try:
        raw = open(f'{pid_dir}/environ', 'rb').read()
        for item in raw.split(b'\0'):
            if b'=' in item:
                k, v = item.split(b'=', 1)
                env[k.decode('utf-8', 'replace')] = v.decode('utf-8', 'replace')
    except (OSError, IOError):
        pass
    return env

def get_gateway_processes():
    """Return list of (pid, hermes_home, cmdline, environ) for gateway processes."""
    gateways = []
    for proc_dir in glob.glob('/proc/[0-9]*'):
        pid = int(proc_dir.rsplit('/', 1)[1])
        cmdline = parse_cmdline(proc_dir)
        if not all(x in cmdline for x in GATEWAY_CMD_SIGNATURE):
            continue
        environ = parse_environ(proc_dir)
        hermes_home = environ.get('HERMES_HOME', '<unset>')
        gateways.append((pid, hermes_home, cmdline, environ))
    return gateways

def check():
    gateways = get_gateway_processes()
    total = len(gateways)

    # Group by effective HERMES_HOME
    by_home = defaultdict(list)
    for pid, home, cmdline, environ in gateways:
        by_home[home].append({
            'pid': pid,
            'hermes_home': home,
            'has_profile_flag': '--profile' in cmdline,
        })

    # Detect true duplicates: >1 PID sharing same HERMES_HOME
    true_duplicates = {}
    expected_groups = {}
    for home, members in sorted(by_home.items()):
        if len(members) > 1:
            true_duplicates[home] = members
        elif home != DEFAULT_HERMES_HOME and home != '<unset>':
            expected_groups[home] = members

    missing_default = DEFAULT_HERMES_HOME not in by_home

    result = {
        'total_gateway_processes': total,
        'default_home': DEFAULT_HERMES_HOME,
        'has_default_gateway': not missing_default,
        'profile_count': sum(1 for home in by_home if home != DEFAULT_HERMES_HOME and home != '<unset>'),
        'groups': {home: [m['pid'] for m in members] for home, members in sorted(by_home.items())},
        'true_duplicate_groups': {home: [m['pid'] for m in members] for home, members in true_duplicates.items()},
        'expected_profile_gateways': list(expected_groups.keys()),
        'healthy': not missing_default and len(true_duplicates) == 0 and total > 0,
    }

    findings = []
    if missing_default:
        findings.append(f'MISSING: no gateway with HERMES_HOME={DEFAULT_HERMES_HOME} (default systemd gateway)')
    if true_duplicates:
        for home, members in true_duplicates.items():
            findings.append(f'DUPLICATE: {len(members)} PIDs share HERMES_HOME={home}: {[m["pid"] for m in members]}')
    if total == 0:
        findings.append('MISSING: zero gateway processes running')

    result['findings'] = findings
    result['healthy'] = len(findings) == 0

    return result


if __name__ == '__main__':
    import sys
    result = check()
    print(json.dumps(result, indent=2))
    # Exit 1 when there's a real finding
    sys.exit(0 if result['healthy'] else 1)
