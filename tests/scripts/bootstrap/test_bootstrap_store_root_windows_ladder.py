"""The store-root resolver's Windows ladder, driven from any host.

``Get-HermesRoot``/``Get-HermesDefaultHome`` take ``$Sep``, so the Windows answers
are reproducible wherever pwsh runs instead of only on the Windows lane.
Expectations come from pathlib's ``PureWindowsPath`` — the lexical model of a
foreign platform — because no .NET path API is: see the ``$joinLeaf`` comment in
``scripts/bootstrap/store_root.ps1`` for why each of them answers for the host it
runs on, which is the reason this file cannot simply call one.

``platforms("any")``: the arm under test is chosen by ``$Sep``, not by the host.
"""
from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_PS1 = REPO_ROOT / "scripts" / "bootstrap" / "store_root.ps1"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = [
    pytest.mark.skipif(
        _POWERSHELL is None, reason="the PowerShell resolver needs pwsh or powershell"
    ),
    pytest.mark.platforms("any"),
]


def test_default_home_join_keeps_the_root_it_was_given(tmp_path: Path) -> None:
    """The default-home join answers what ``Path(parent) / leaf`` answers.

    Both defects came from treating a parent as "a directory plus a trailing
    separator": a parent of only separators trimmed to '' and lost its anchor
    (``LOCALAPPDATA='/'`` gave the relative ``hermes`` instead of ``\\hermes``),
    and the trim removed ``\\`` on POSIX, retargeting ``HOME='/a\\b\\'`` at ``/a\\b``.
    """
    from pathlib import PureWindowsPath

    win_parents = ["\\", "//", "/", "C:\\Users\\t", "C:\\Users\\t\\", "C:\\",
                   "\\\\server\\share", ""]
    posix_parents = ["/a/b", "/a/b/", "/a\\\\b\\\\", "/", "//", ""]

    # A FILE, not -Command: the harness' quoting would otherwise reach PowerShell
    # as bash-expanded text (a "$joinLeaf" that arrived empty). Dot-sourcing first
    # defines Get-HermesTrimChars, which the block calls; the file is all function
    # definitions, so nothing else runs.
    script = (f'. "{CANONICAL_PS1.as_posix()}"\n'
              f'$src = Get-Content "{CANONICAL_PS1.as_posix()}" -Raw\n'
              "$m = [regex]::Match($src, '(?s)\\$joinLeaf = \\{.*?\\n    \\}')\n"
              "Invoke-Expression $m.Value\n"
              "$out = @()\n"
              + "".join(f"$out += (& $joinLeaf '{p}' 'hermes' '\\')\n" for p in win_parents)
              + "".join(f"$out += (& $joinLeaf '{p}' '.hermes' '/')\n" for p in posix_parents)
              + "$out | ConvertTo-Json -Compress\n")
    probe = tmp_path / "join-probe.ps1"
    probe.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    got = json.loads(result.stdout)

    expected = ([str(PureWindowsPath(p) / "hermes") if p else "hermes" for p in win_parents]
                + [posixpath.join(p, ".hermes") if p else ".hermes" for p in posix_parents])
    for parent, ours, want in zip(win_parents + posix_parents, got, expected):
        assert ours == want, (
            f"parent={parent!r}: the join answered {ours!r}, pathlib answers {want!r} "
            "(a parent of only separators must keep its anchor)"
        )


def test_powershell_resolver_answers_the_windows_ladder_on_any_host() -> None:
    """``Get-HermesRoot -Sep '\\'`` reproduces pm's Windows answer on a POSIX host.

    Pins the whole Windows ladder at once: the drive roots, the ``profiles`` fold
    (one level, from a parent named exactly ``profiles``), containment under the
    default home, and a case-SENSITIVE name compare — ``hermes_constants`` uses
    ``str ==``, not normcase, so ``Profiles`` must not fold.
    """
    from pathlib import PureWindowsPath

    native = "C:\\Users\\tester\\hermes"
    cases = [
        native,                              # the default home itself -> unchanged
        native + "\\profiles\\coder",         # named profile -> the native home
        native + "\\foo",                     # anything under it -> the native home
        "C:\\custom\\profiles\\coder",         # custom parent named profiles -> folds
        "C:\\custom",                         # custom root -> itself
        "C:\\custom\\a\\profiles\\b",           # parent IS 'profiles' -> folds one level
        "C:\\Profiles\\coder",                 # case differs -> no fold
        "C:\\custom\\profilesX\\coder",        # not the name -> no fold
        "C:\\custom\\profiles\\coder\\..",      # trailing '..' is a component
    ]

    def expected(home: str) -> str:
        p = PureWindowsPath(home)
        try:
            p.relative_to(native)
        except ValueError:
            return str(p.parent.parent) if p.parent.name == "profiles" else str(p)
        return native

    script = (
        '. "' + CANONICAL_PS1.as_posix() + '"\n'
        "$RS = [string][char]2\n"
        f'$env:LOCALAPPDATA = \'{native.rsplit(chr(92), 1)[0]}\'\n'
        '$env:HERMES_DATA_DIR_SUFFIX = \'\'\n'
        "$homes = @(" + ",".join("'" + c + "'" for c in cases) + ")\n"
        "foreach ($h in $homes) {\n"
        "  $env:HERMES_HOME = $h\n"
        '  Write-Output ($h + $RS + (Get-HermesRoot -Sep \'\\\'))\n'
        "}\n"
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split("\u0002") for line in result.stdout.splitlines() if "\u0002" in line]
    assert len(rows) == len(cases), f"expected {len(cases)} rows, got {len(rows)}: {result.stdout}"
    for (home, got), want_home in zip(rows, cases):
        assert home == want_home, f"row order: {home!r} != {want_home!r}"
        assert got == expected(home), (
            f"HERMES_HOME={home!r}: the -Sep '\\' ladder answered {got!r}, pathlib/pm "
            f"answer {expected(home)!r}"
        )
