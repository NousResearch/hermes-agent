"""Select repository-owned environments without changing checkout version pins."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def select_environment(repository: Path, workspace: Path, allowed_roots: tuple[Path, ...]) -> Path:
    default = repository / '.venv'
    pin_path = workspace / '.python-version'
    if not pin_path.is_file():
        return default
    pin = pin_path.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9]+\.[0-9]+(?:\.[0-9]+)?', pin):
        raise RuntimeError('worktree Python version pin is invalid')
    candidates = [*sorted(repository.glob('venv-ci-*')), default, repository / 'venv',
                  *sorted(repository.glob('venv-*'))]
    visited: set[Path] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve(strict=True)
        if resolved in visited or not any(resolved.is_relative_to(root) for root in allowed_roots):
            continue
        visited.add(resolved)
        python = resolved / 'bin/python'
        if not python.is_file():
            continue
        try:
            probe = subprocess.run(
                [str(python), '-I', '-c', "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
                cwd=repository,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        actual = probe.stdout.strip()
        if probe.returncode == 0 and (actual == pin or (pin.count('.') == 1 and actual.startswith(pin + '.'))):
            return candidate
    raise RuntimeError(f'no governed project virtualenv matches Python {pin}')
