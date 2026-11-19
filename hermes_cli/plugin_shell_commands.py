"""Installer-owned shell launchers for opt-in packaged plugin commands.

No plugin code runs while publishing. Launchers re-enter Hermes's stable CLI,
so plugin updates select the current managed environment, not a captured venv.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import tomllib
from typing import AbstractSet

_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_ENTRY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")
_RESERVED = frozenset({'hermes', 'hermes-acp', 'hermes-agent', 'python', 'python3', 'pip', 'pip3', 'sh', 'bash', 'cmd', 'powershell', 'pwsh'})
_PREFIX = '.hermes-plugin-command-'


def declared_commands(plugin: Path) -> dict[str, str]:
    from pm.plugin_declarations import read_native_manifest, native_manifest_file
    path = native_manifest_file(plugin)
    manifest = read_native_manifest(path) if path else {}
    names = manifest.get('shell_commands', [])
    if not isinstance(names, list) or len(names) > 32 or any(not isinstance(n, str) for n in names):
        raise ValueError('shell_commands must be a list of at most 32 command names')
    if not names:
        return {}
    if len(set(names)) != len(names) or any(not _NAME.fullmatch(n) or n in _RESERVED for n in names):
        raise ValueError('shell_commands contains a duplicate, unsafe or reserved name')
    project = tomllib.loads((plugin / 'pyproject.toml').read_text(encoding='utf-8-sig'))
    scripts = project.get('project', {}).get('scripts', {})
    if not isinstance(scripts, dict) or any(n not in scripts for n in names):
        raise ValueError('Every shell_commands name must exist in project.scripts')
    result = {n: scripts[n] for n in names}
    if any(not isinstance(v, str) or not _ENTRY.fullmatch(v) for v in result.values()):
        raise ValueError('Shell entrypoints must be module:function references, not shell snippets')
    return result


def _marker(out: Path, name: str) -> Path:
    return out / f'{_PREFIX}{name}.json'


def _record(path: Path) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > 16_384:
            return {}
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(plugin: Path, home: Path) -> dict[str, str]:
    return {'plugin': str(plugin.resolve()), 'home': str(home.resolve())}


def _owned(target: Path, record: dict, identity: dict) -> bool:
    return (all(record.get(k) == v for k, v in identity.items())
            and not target.is_symlink() and target.is_file()
            and target.stat().st_size < 2 * 1024 * 1024
            and _digest(target) in (record.get('sha256'), record.get('previous_sha256')))


def _windows_launcher(directory: Path, name: str, script: str, python_executable: Path) -> Path:
    # Use an executable, not %* in a .cmd: batch forwarding can turn an argv
    # containing & into a second command. Use the same native builder as Hermes.
    from hermes_cli._launchers import _load_script_maker, _store_script_maker
    maker_class = _load_script_maker() or _store_script_maker(python_executable)
    if maker_class is None:
        raise RuntimeError('Native Windows launcher builder unavailable; no unsafe .cmd fallback was written')
    class Maker(maker_class):  # pyright: ignore[reportGeneralTypeIssues]
        def _get_script_text(self, entry):
            return script
    maker = Maker(None, str(directory), add_launchers=True)
    maker.executable = str(python_executable)
    maker.variants = {''}
    maker.make(f'{name} = hermes_cli.plugin_shell_commands:run_command', {'interpreter_args': ['-I']})
    target = directory / f'{name}.exe'
    if not target.is_file():
        raise RuntimeError('Native Windows launcher generation failed')
    return target


def _windows_command_script(home: Path, argv: list[str]) -> str:
    # CRT exec does not quote argv. subprocess uses CreateProcess quoting and
    # waits so the native launcher reports the child's actual exit status.
    return ('import os,sys,subprocess\n'
            f'os.environ["HERMES_HOME"] = {str(home.resolve())!r}\n'
            'os.environ.pop("HERMES_PROFILE", None)\n'
            'os.environ.pop("HERMES_ACTIVE_PROFILE", None)\n'
            f'sys.exit(subprocess.run({argv!r} + sys.argv[1:], shell=False, check=False).returncode)\n')


def _private_command_directory(out: Path) -> bool:
    if out.is_symlink():
        return False
    if os.name == 'nt':
        return True  # User-bin ACLs are supplied by the Windows user profile.
    info = out.stat()
    uid = getattr(os, 'getuid', lambda: None)()
    if uid is None or info.st_uid != uid or info.st_mode & 0o002:
        return False
    if not info.st_mode & 0o020:
        return True
    import grp
    import pwd
    try:
        me = pwd.getpwuid(uid).pw_name
        members = set(grp.getgrgid(info.st_gid).gr_mem)
        members.update(p.pw_name for p in pwd.getpwall() if p.pw_gid == info.st_gid)
        return not (members - {me})
    except (KeyError, OSError):
        return False


def export_commands(plugin: Path, home: Path, out: Path, hermes: Path, *, only: set[str] | None = None, python_executable: Path | None = None) -> list[Path]:
    commands = declared_commands(plugin)
    if only is not None:
        commands = {k: v for k, v in commands.items() if k in only}
    if not commands:
        return []
    if not _NAME.fullmatch(plugin.name):
        raise ValueError('Shell commands require a simple installed plugin directory name')
    out.mkdir(parents=True, exist_ok=True)
    if plugin.is_symlink() or not _private_command_directory(out):
        raise OSError('Command directory must be owned by the installing user, not a symlink')
    if os.name == 'nt' and hermes.suffix.lower() != '.exe':
        raise RuntimeError('Bare commands require a native Hermes executable on Windows')
    identity = _identity(plugin, home)
    python_executable = python_executable or Path(sys.executable)
    targets = {}
    # Check every destination before writing any of them. Do not inspect the
    # contents of foreign executables, and never shadow a command elsewhere on PATH.
    for name in commands:
        target = out / (name + '.exe' if os.name == 'nt' else name)
        record = _record(_marker(out, name))
        if target.exists() or target.is_symlink():
            if not _owned(target, record, identity):
                raise FileExistsError(f'{name}: refusing to replace an existing or modified command')
        else:
            existing = shutil.which(name)
            if existing:
                raise FileExistsError(f'{name}: another command already exists on PATH')
        targets[name] = target
    with tempfile.TemporaryDirectory(prefix='.hermes-commands-', dir=out) as temporary:
        staging = Path(temporary)
        for name, target in targets.items():
            old_record = _record(_marker(out, name))
            unchanged = (old_record.get('launcher_version') == 2
                         and old_record.get('hermes') == str(hermes.absolute())
                         and old_record.get('entry') == commands[name]
                         and old_record.get('python') == (str(python_executable) if os.name == 'nt' else None)
                         and _owned(target, old_record, identity)
                         and old_record.get('sha256') == _digest(target))
            if unchanged:
                continue
            argv = [str(hermes.absolute()), 'plugins', 'run-shell', plugin.name, name, '--']
            if os.name == 'nt':
                script = _windows_command_script(home, argv)
                staged = _windows_launcher(staging, name, script, python_executable)
            else:
                script = ('#!/bin/sh\n# Hermes-managed plugin shell command\n'
                          f'HERMES_HOME={shlex.quote(str(home.resolve()))}; export HERMES_HOME\n'
                          'unset HERMES_PROFILE HERMES_ACTIVE_PROFILE\n'
                          'exec ' + shlex.join(argv) + ' "$@"\n')
                staged = staging / name
                staged.write_text(script, encoding='utf-8')
                staged.chmod(0o755)
            # Reserve new names without clobbering a concurrent installer.
            if not target.exists():
                fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o755)
                os.close(fd)
            elif not _owned(target, _record(_marker(out, name)), identity):
                raise FileExistsError(f'{name}: destination changed during installation')
            record = {**identity, 'command': name, 'hermes': str(hermes.absolute()),
                      'launcher_version': 2,
                      'entry': commands[name], 'python': str(python_executable) if os.name == 'nt' else None,
                      'sha256': _digest(staged), 'previous_sha256': _digest(target)}
            marker = staging / (name + '.json')
            marker.write_text(json.dumps(record), encoding='utf-8')
            marker.chmod(0o600)
            # Publish ownership first with both hashes: an interruption on either
            # side of the executable rename remains recoverable without clobbering
            # a foreign replacement.
            os.replace(marker, _marker(out, name))
            os.replace(staged, target)
    return list(targets.values())


def remove_commands(plugin: Path, home: Path, out: Path, *, keep: AbstractSet[str] = frozenset()) -> None:
    identity = _identity(plugin, home)
    for marker in out.glob(_PREFIX + '*.json'):
        name = marker.name[len(_PREFIX):-5]
        if not _NAME.fullmatch(name) or name in keep:
            continue
        record = _record(marker)
        target = out / (name + '.exe' if os.name == 'nt' else name)
        if _owned(target, record, identity):
            target.unlink()
            marker.unlink()


def default_command_directory() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / 'bin' if os.name == 'nt' else Path.home() / '.local' / 'bin'


def remove_installed_commands(plugin: Path) -> None:
    from hermes_constants import get_hermes_home
    remove_commands(plugin, get_hermes_home(), default_command_directory())


def command_location() -> tuple[Path, Path]:
    from hermes_cli._launchers import plugin_command_launcher
    from pm.paths import repo_root
    out = default_command_directory()
    path_dirs = {os.path.normcase(os.path.abspath(p)) for p in os.get_exec_path() if p}
    if os.path.normcase(str(out.absolute())) not in path_dirs:
        raise OSError(f'Hermes user command directory is not on PATH: {out}; no shell configuration was changed')
    hermes = plugin_command_launcher(repo_root())
    if not hermes.is_file():
        raise OSError('Stable Hermes launcher is missing; repair the Hermes installation first')
    return out, hermes


def enabled_for_home(plugin: Path, home: Path) -> bool:
    """Check this profile only, not PM's cross-profile dependency union."""
    from pm.plugins_state import read_home_selection
    from pm.plugin_declarations import native_manifest_file, read_native_manifest
    config = read_home_selection(home) or {}
    manifest_file = native_manifest_file(plugin)
    if manifest_file is None:
        return False
    manifest = read_native_manifest(manifest_file)
    names = {plugin.name, str(manifest.get('name') or plugin.name)}
    plugins = config.get('plugins') or {}
    if names & set(plugins.get('disabled') or ()):
        return False
    return bool(names & set(plugins.get('enabled') or ())) or (config.get('memory') or {}).get('provider') in names


def sync_plugin_commands(plugin: Path, console=None) -> None:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    # Plugins without opt-in do not trigger PATH discovery or provisioning.
    commands = declared_commands(plugin)
    if not commands:
        remove_commands(plugin, home, default_command_directory())
        return
    try:
        out, hermes = command_location()
        if not enabled_for_home(plugin, home):
            remove_commands(plugin, home, out)
            return
        written = export_commands(plugin, home, out, hermes)
        remove_commands(plugin, home, out, keep=set(commands))
        if console:
            from rich.markup import escape
            for path in written:
                console.print(f'[green]✓[/green] Terminal command ready: {escape(path.stem if os.name == "nt" else path.name)}')
    except (OSError, ValueError, RuntimeError) as exc:
        if console is None:
            raise
        from rich.markup import escape
        console.print(f'[red]Terminal command installation failed:[/red] {escape(str(exc))}')


def refresh_published_commands(out: Path, hermes: Path, *, python_executable: Path | None = None) -> list[str]:
    """Refresh only already-owned names after a core runtime update; never widen exposure."""
    groups = {}
    errors = []
    for marker in out.glob(_PREFIX + '*.json'):
        record = _record(marker)
        name = record.get('command')
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            continue
        if record.get('hermes') != str(hermes.absolute()):
            continue  # Another Hermes installation owns this command.
        if not isinstance(record.get('home'), str) or not isinstance(record.get('plugin'), str):
            continue
        home, plugin = Path(record['home']), Path(record['plugin'])
        if not home.is_absolute() or plugin.parent != home / 'plugins' or plugin.is_symlink():
            continue
        groups.setdefault((home, plugin), set()).add(name)
    for (home, plugin), names in groups.items():
        try:
            if plugin.is_dir() and enabled_for_home(plugin, home):
                current = declared_commands(plugin)
                export_commands(plugin, home, out, hermes, only=names, python_executable=python_executable)
                remove_commands(plugin, home, out, keep=set(current) & names)
            else:
                remove_commands(plugin, home, out)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f'{plugin.name}: {exc}')
    return errors


def run_command(args) -> int:
    """Dispatch a package console entrypoint without importing a plugin's register()."""
    from hermes_constants import get_hermes_home
    from pm.plugin_declarations import read_python_declaration
    home = get_hermes_home()
    if not _NAME.fullmatch(args.shell_plugin) or not _NAME.fullmatch(args.shell_command):
        raise ValueError('Invalid plugin or shell command')
    plugin = home / 'plugins' / args.shell_plugin
    if plugin.is_symlink() or not enabled_for_home(plugin, home):
        raise ValueError('Plugin is not installed and enabled for this profile')
    commands = declared_commands(plugin)
    expected = commands.get(args.shell_command)
    if expected is None:
        raise ValueError('Command is not declared by this plugin')
    declaration = read_python_declaration(plugin)
    if declaration.pyproject is None or declaration.external:
        raise ValueError('Shell commands require a Hermes-managed Python package')
    project = tomllib.loads(declaration.pyproject.read_text(encoding='utf-8-sig'))['project']
    distribution = importlib.metadata.distribution(project['name'])
    entries = [ep for ep in distribution.entry_points if ep.group == 'console_scripts'
               and ep.name == args.shell_command and ep.value == expected]
    if len(entries) != 1:
        raise ValueError('Managed package entrypoint is unavailable; re-enable the plugin to prepare dependencies')
    raw = list(args.shell_arguments)
    if raw[:1] == ['--']:
        raw = raw[1:]
    sys.argv = [args.shell_command, *raw]
    # Bind standalone defaults to the same state home used by agent tools.
    os.environ['HERMES_HOME'] = str(home)
    result = entries[0].load()()
    return result if isinstance(result, int) else 0
