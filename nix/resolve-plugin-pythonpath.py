#!/usr/bin/env python3
"""
Build-time filter for extraPythonPackages PYTHONPATH.

Drops any extra package whose *entire* dist-info set already exists
in the sealed hermes venv. This prevents shadowing while allowing
plugins with shared transitive deps.

Usage:
  python3 resolve-plugin-pythonpath.py /path/to/hermes-venv /path/to/site-packages \
    /nix/store/...-pkg1 /nix/store/...-pkg2 ...
  
Writes filtered colon-separated search path to $TMPDIR/hermes-plugin-pythonpath
"""

import pathlib
import sys
import re
import os


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    if len(sys.argv) < 4:
        print("Usage: resolve-plugin-pythonpath.py <venv-path> <site-packages-rel> <extra-pkg-paths...>", file=sys.stderr)
        sys.exit(1)

    venv_path = pathlib.Path(sys.argv[1])
    site_packages_rel = sys.argv[2]
    extra_paths = [pathlib.Path(p) for p in sys.argv[3:]]

    # Collect core venv package names
    core = set()
    venv_sp = venv_path / site_packages_rel
    for di in venv_sp.glob('*.dist-info'):
        meta = di / 'METADATA'
        if meta.exists():
            for line in meta.read_text().splitlines():
                if line.startswith('Name:'):
                    core.add(canonical(line.split(':', 1)[1].strip()))
                    break

    kept_paths = []
    for edir in extra_paths:
        sp = edir / site_packages_rel
        if not sp.exists():
            continue
        # Read all dist-info names in this package
        dist_names = set()
        for di in sp.glob('*.dist-info'):
            meta = di / 'METADATA'
            if not meta.exists():
                continue
            for line in meta.read_text().splitlines():
                if line.startswith('Name:'):
                    dist_names.add(canonical(line.split(':', 1)[1].strip()))
                    break
        # If every dist-info in this store path is already in core,
        # the whole package is a pure duplicate -> drop it.
        # If it contributes at least one non-core name, keep the path.
        if dist_names and dist_names.issubset(core):
            print('warning: plugin package "{0}" ({1}) already in sealed venv -- using core copy, omitting from PYTHONPATH'.format(
                next(iter(dist_names)), edir), file=sys.stderr)
        else:
            kept_paths.append(str(sp))

    # Write filtered search path (colon-separated) to a temp file
    filtered_path = ':'.join(kept_paths) if kept_paths else ""
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    with open(os.path.join(tmpdir, 'hermes-plugin-pythonpath'), 'w') as f:
        f.write(filtered_path)
    print('=== Plugin PYTHONPATH resolved: {0} package(s) added, {1} filtered ==='.format(
        len(kept_paths), len(extra_paths) - len(kept_paths)))


if __name__ == '__main__':
    main()