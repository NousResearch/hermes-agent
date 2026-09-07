"""Resolve repository CI ownership without requiring LunaBot scripts in Hermes."""
from pathlib import Path
import json
import tomllib

HERMES_ENV_CHECK = ('uv', 'sync', '--locked', '--check', '--no-install-project',
                    '--extra', 'all', '--extra', 'dev', '--extra', 'anthropic',
                    '--extra', 'mistral', '--extra', 'fal', '--extra', 'modal',
                    '--extra', 'daytona', '--extra', 'hindsight', '--extra', 'parallel-web')

_HERMES_MANIFEST = Path(__file__).with_name('hermes_native_ci.toml')


def manifest_path(worktree: Path) -> Path:
    legacy = worktree / 'tests/manifests/test_lanes.toml'
    if legacy.is_file():
        return legacy
    try:
        project = tomllib.loads((worktree / 'pyproject.toml').read_text())['project']
        if project.get('name') == 'hermes-agent' and (worktree / 'scripts/run_tests.sh').is_file():
            return _HERMES_MANIFEST
    except (OSError, ValueError, KeyError):
        pass
    return legacy


def is_hermes_contract(payload: bytes) -> bool:
    return tomllib.loads(payload.decode('utf-8')).get('schema') == 'hermes-native-ci-v1'


def hermes_commands(worktree: Path, base_sha: str, head_sha: str, changed: tuple[str, ...]):
    targeted_tests = tuple(
        path for path in changed
        if path.startswith('tests/') and path.endswith('.py')
    )
    targeted_python = tuple(path for path in changed if path.endswith('.py'))
    if targeted_tests:
        test_command = ('bash', 'scripts/run_tests.sh', *targeted_tests)
    elif targeted_python:
        test_command = ('python', '-m', 'compileall', '-q', *targeted_python)
    else:
        test_command = ('git', 'diff', '--check', f'{base_sha}..{head_sha}')
    commands = [
        (('git', 'diff', '--check', f'{base_sha}..{head_sha}'), worktree, {}),
        (('uv', 'lock', '--check'), worktree, {}),
        (HERMES_ENV_CHECK, worktree, {}),
        # Local CI is a receipt worker, not a second hosted full-suite lane.
        # Run only test files changed by this PR; hosted CI remains responsible
        # for the repository-wide matrix.
        (test_command, worktree, {}),
    ]
    root_lock = worktree / 'package-lock.json'
    locked_packages = json.loads(root_lock.read_text()).get('packages', {}) if root_lock.is_file() else {}
    root_installed = False
    shared_changed = any(p.startswith('apps/shared/') or p in {'package.json', 'package-lock.json'} for p in changed)
    for package in ('apps/desktop', 'apps/shared', 'ui-tui', 'web', 'website'):
        if not shared_changed and not any(p == package or p.startswith(package + '/') for p in changed):
            continue
        root = worktree / package
        if not (root / 'package.json').is_file():
            continue
        if package in locked_packages:
            if not root_installed:
                commands.append((('npm', 'ci', '--ignore-scripts'), worktree, {}))
                root_installed = True
        elif (root / 'package-lock.json').is_file():
            commands.append((('npm', 'ci', '--ignore-scripts'), root, {}))
        else:
            raise ValueError(f'Hermes CI package lock missing: {package}')
        scripts = json.loads((root / 'package.json').read_text()).get('scripts', {})
        if 'build:ink' in scripts:
            commands.append((('npm', 'run', 'build:ink'), root, {'CI': 'true'}))
        for name in ('lint', 'typecheck', 'test', 'build'):
            if name in scripts:
                commands.append((('npm', 'run', name), root, {'CI': 'true'}))
    return commands


def hermes_coverage_gap(changed: tuple[str, ...]) -> str | None:
    if any(p.endswith((".rs", ".ps1", ".nix")) or p.startswith(".github/")
           or Path(p).name.startswith("Dockerfile") for p in changed):
        return "additional platform or workflow coverage is required beyond host-native CI"
    return None
