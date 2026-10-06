"""runtime.ps1 must decode the launcher's UTF-8 path output on Windows (#124526).

``scripts/desktop-update/runtime.ps1`` captures two machine-readable outputs from
Hermes launchers: ``--print-runtime-command`` (a JSON array whose first element
is the interpreter path) and ``--version`` (whose ``Install directory:`` line
names the source root). Both can sit under a non-ASCII profile
(``C:\\Users\\Balázs``). The launchers print UTF-8 — the real CLI's
``hermes_bootstrap`` reconfigures stdio on import — while Windows PowerShell 5.1
decodes captured native stdout with the console OEM code page, mojibaking the
path (``á`` → ``├í``, ``ł`` → ``┼é``) so the returned command cannot run and the
legacy identity check never matches.

Both call sites now scope ``[Console]::OutputEncoding`` to UTF-8 around the
capture and restore it. These tests drive the REAL script against a REAL
published launcher under non-ASCII paths, across three console code pages, and
assert the caller's encoding is restored. They never read the script's text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parents[3]
HELPER = str(ROOT / 'scripts/desktop-update/runtime.ps1').replace("'", "''")

# Mirrors the production launchers: hermes_bootstrap reconfigures stdio to
# UTF-8 on import before the entry module runs, so both outputs carry raw
# UTF-8 bytes on a pipe. sys.reconfigure replicates that for the fixture's
# plain interpreter (the embedded bootstrap imports the real one).
CLI = """
import io, json, os, sys
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        try: _stream.reconfigure(encoding='utf-8', errors='replace')
        except (OSError, ValueError): pass
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    print(json.dumps([sys.executable] + sys.argv[2:]))
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""

# From #124526 / #128326: profile names that die as OEM mojibake.
PROFILE = 'Balázs Paweł Ñuñez'


def _run_helper(script: str, home: Path, env_extra: dict[str, str], out: Path) -> list[str]:
    """Run *script*, which writes its answer lines to ``$out`` as UTF-8.

    The answer goes through a file, not stdout: under code page 437 or 936 the console cannot
    represent ``ł``, so stdout would lose the very characters under test.
    """
    result = subprocess.run(['powershell', '-NoProfile', '-Command', f"$out = '{out}'; " + script],
                            env={**os.environ, 'HERMES_HOME': str(home), **env_extra},
                            capture_output=True, timeout=120)
    assert result.returncode == 0, (result.stdout + result.stderr).decode('utf-8', 'replace')
    return out.read_text(encoding='utf-8-sig').splitlines()


_WRITE = "[IO.File]::WriteAllLines($out, [string[]]@({lines}), (New-Object Text.UTF8Encoding($false)))"


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('code_page', [936, 437, 65001])
def test_runtime_command_survives_a_non_ascii_profile_path(tmp_path: Path, code_page: int) -> None:
    """Get-HermesRuntimeCommand returns the launcher's exact interpreter path.

    The launcher JSON is ASCII-escaped today, so this is the round-trip
    contract for the boundary every other consumer (hermes_cli.windows_ssh_
    runtime, the Electron updater, the Rust bootstrap) already decodes as
    UTF-8: the scoped capture must return the path byte-exact and restore
    the caller's console encoding, whatever code page the console carries.
    """
    profile = tmp_path / PROFILE
    install = profile / 'hermes-agent'
    # PM's store under the same non-ASCII profile, with a facts.json naming a
    # store interpreter there — the layout a real install resolves and prints.
    store = profile / 'tools'
    python_dir = store / 'cpython-3.14-windows-x86_64-none'
    python_dir.mkdir(parents=True)
    python = python_dir / 'python.exe'
    python.write_bytes(Path(sys.executable).read_bytes())
    (store / 'facts.json').write_text(
        json.dumps({'packages': {'python': {'entry': python_dir.name}}}), encoding='utf-8')
    publish_fixture_launcher(install, CLI)
    home = tmp_path / 'home'
    home.mkdir()
    target = str(install).replace("'", "''")
    prefix = f"[Console]::OutputEncoding = [Text.Encoding]::GetEncoding({code_page}); "
    script = (prefix + f". '{HELPER}'; "
              f"$c = @(Get-HermesRuntimeCommand -InstallRoot '{target}'); "
              + _WRITE.format(lines="$c[0], [Console]::OutputEncoding.CodePage"))
    lines = _run_helper(script, home, {'HERMES_RUNTIME_DIR': str(store)}, tmp_path / 'answer.txt')
    assert lines[0] == str(python), (
        f"interpreter path was mojibaked under code page {code_page}: {lines[0]!r}")
    assert int(lines[1]) == code_page, 'the capture must restore the caller console encoding'


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('code_page', [936, 437, 65001])
def test_legacy_version_identity_check_matches_a_non_ascii_install(tmp_path: Path, code_page: int) -> None:
    """The user-bin fallback resolves by reported install directory, not PATH order.

    Red on base: the fixture CLI prints ``Install directory:`` as UTF-8 (the
    production CLI reconfigures stdio the same way), and Windows PowerShell
    5.1 decodes that capture with the console OEM code page — so under CP437
    a profile like ``Paweł Łącki`` reports mojibaked and the identity check's
    GetFullPath comparison never matches (#124526).
    """
    install = tmp_path / PROFILE / 'hermes-agent'
    launcher = publish_fixture_launcher(install, CLI)
    # Earlier PM installers published only to user-bin: move (not copy) the
    # launcher out of .hermes\bin so Get-HermesRuntimeCommand takes the
    # legacy branch. Its body still reports the checkout it was minted from.
    home = tmp_path / 'profile'
    userbin = home / 'bin'
    userbin.mkdir(parents=True)
    external = userbin / launcher.name
    launcher.rename(external)
    target = str(install).replace("'", "''")
    prefix = f"[Console]::OutputEncoding = [Text.Encoding]::GetEncoding({code_page}); "
    script = (prefix + f". '{HELPER}'; "
              f"try {{ $c = @(Get-HermesRuntimeCommand -InstallRoot '{target}') }} catch {{ exit 1 }}; "
              + _WRITE.format(lines="$c[0]"))
    lines = _run_helper(script, home, {}, tmp_path / 'answer.txt')
    assert lines[0] == str(external), (
        f"legacy identity check lost the non-ASCII path under code page {code_page}")
