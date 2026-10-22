"""The bootstrap consumers resolve ONE identity for a home and a runtime store.

Each test here drives the statements ``scripts/install.sh``, ``setup-hermes.sh``,
``scripts/install.ps1`` and ``setup-hermes.ps1`` actually run — the resolvers
themselves are covered by ``tests/scripts/bootstrap/`` — and asserts that the
home, the runtime-store override and the uv state directories name the same
directory before and after the checkout ``chdir``, and the one pm resolves.

Fixtures live in :mod:`tests.scripts.bootstrap.store_root_support`.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

import pytest

from tests.scripts.bootstrap.store_root_support import (
    CANONICAL_PS1,
    CANONICAL_SH,
    CANON_PS1_BEGIN,
    CANON_PS1_END,
    CANON_SH_BEGIN,
    CANON_SH_END,
    DATA_DIR_SUFFIX,
    GEN_PS1_BEGIN,
    GEN_PS1_END,
    GEN_SH_BEGIN,
    GEN_SH_END,
    HOME_KINDS,
    REPO_ROOT,
    _POWERSHELL,
    _RESOLVER_MARKERS,
    _child_localappdata,
    _hermes_home,
    _named_user_account,
    _named_user_home,
    _pm_store_root,
    _resolver_source,
    _set_home,
    _staged_store,
)

@pytest.mark.platforms("posix")
@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_install_sh_stages_into_the_root_it_resolved(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    """install.sh must feed the resolver's answer to the store, not HERMES_HOME.

    ``install.sh`` is the one bootstrap with a source guard, so this pins the
    wiring rather than the rule: ``HERMES_ROOT`` and ``ensure_uv``'s store both
    have to follow the resolver.
    """
    bash = real_bash
    _set_home(monkeypatch, tmp_path, home_kind)
    # Same cwd as the child, so a relative ``HERMES_HOME`` resolves against the
    # same directory on both sides of the comparison below.
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    # The recomputation has to use the same home the resolver was given: the
    # shell's resolved (cwd-anchored) form, not the raw expression. A defaulted
    # home is shell-local, so there the resolver is handed nothing at all.
    script = (
        'source "$1" --manifest || exit 1\n'
        'if [ "${_HERMES_HOME_DEFAULTED:-0}" = 1 ]; then '
        'recomputed="$(unset HERMES_HOME; hermes_root_of)"; '
        'else recomputed="$(HERMES_HOME="$_hermes_home_shell" hermes_root_of)"; fi\n'
        'printf "%s\\n%s\\n" "$HERMES_ROOT" "$recomputed"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resolved, recomputed = result.stdout.splitlines()[:2]

    assert resolved == recomputed, "HERMES_ROOT must be the resolver's answer"
    assert _staged_store(resolved) == _pm_store_root()


def test_install_sh_default_home_follows_the_host_os_case(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolver's default home must pick its branch with no ``os`` injected.

    The spliced body consumes ``${os:-}`` but never sets it, so each script's own
    ``uname`` case supplies the premise. Sourcing the real script exercises it;
    extracting the body under a manual ``os=win32`` does not. Without it
    install.sh answers the POSIX default under Git Bash while the PM child
    computes the win32 one."""
    bash = real_bash
    home = tmp_path / "home"
    lad = tmp_path / "lad"
    monkeypatch.setenv("HOME", home.as_posix())
    monkeypatch.setenv("LOCALAPPDATA", lad.as_posix())
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "")
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    script = (
        'source "$1" --manifest >/dev/null || exit 1\n'
        'printf "%s" "$(hermes_default_home)"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if sys.platform == "win32":
        # Git Bash's uname matches MINGW*|MSYS*|CYGWIN* -> os=win32 -> the
        # LOCALAPPDATA branch, forward slashes verbatim from the env value.
        assert result.stdout == f"{lad.as_posix()}/hermes"
    else:
        assert result.stdout == f"{home.as_posix()}/.hermes"


@pytest.mark.platforms("posix")
def test_install_sh_drops_an_inherited_export_when_it_defaults_the_home(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited ``export HERMES_HOME=`` must not leak the finalized default.

    A plain assignment to an already-exported variable keeps the export
    attribute, so the literal-suffix default would reach child Python, which
    expands the suffix as explicit configuration and selects a different root
    than the resolver did. install.sh must unset the inherited export first.
    """
    bash = real_bash
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("ROOT_CONTRACT_LABEL", "expanded")
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "-$ROOT_CONTRACT_LABEL")
    monkeypatch.setenv("HERMES_HOME", "")  # exported-empty, not unset
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    script = (
        'source "$1" --manifest || exit 1\n'
        'printf "%s\\n" "$HERMES_ROOT"\n'
        'printf "%s\\n" "$(env | sed -n "s/^HERMES_HOME=//p" | head -1)"\n'
        'PYTHONPATH="$2" "$3" -c "from hermes_constants import get_default_hermes_root; '
        'print(get_default_hermes_root())"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh), str(REPO_ROOT), sys.executable],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root, child_home, child_root = result.stdout.splitlines()[:3]

    assert root == str(native / ".hermes-$ROOT_CONTRACT_LABEL"), root
    assert child_home == "", (
        "a defaulted home must stay shell-local: an exported literal-suffix "
        "HERMES_HOME makes child Python expand the suffix and pick a different root"
    )
    assert child_root == root, (
        f"child Python resolved {child_root!r}, the bootstrap staged into {root!r}"
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "raw_home,expected",
    [
        ("   ", None),                       # whitespace-only == unset -> default
        ("  /tmp/custom  ", "/tmp/custom"),  # outer whitespace is trimmed
    ],
)
def test_install_sh_trims_outer_whitespace_from_the_home(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str, expected: str | None
) -> None:
    """Outer whitespace is not part of the home: INSTALL_DIR must follow the
    trimmed value the resolver and Python's ``str.strip()`` use, not the raw string.

    A whitespace-only HERMES_HOME is "unset", so INSTALL_DIR must fall under the
    default home, not become a whitespace-prefixed relative path.
    """
    bash = real_bash
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    script = (
        'source "$1" --manifest || exit 1\n'
        'printf "%s\\n%s\\n%s\\n" "$_hermes_home_shell" "$INSTALL_DIR" "${HERMES_HOME-<unset>}"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    hermes_home, install_dir, inherited = result.stdout.splitlines()[:3]

    resolved = expected if expected is not None else str(native / ".hermes")
    assert hermes_home == resolved, hermes_home
    assert install_dir == resolved + "/hermes-agent", install_dir
    if expected is None:
        assert inherited == "<unset>", (
            "a defaulted home must stay shell-local; exporting it makes child Python "
            f"expand the literal suffix to a different root (got {inherited!r})"
        )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("raw,kind", [
    ("~/custom-hermes", "native"),        # tilde must expand to HOME/custom-hermes
    ("$HOME/custom-hermes", "native"),    # $VAR must expand
    ("${HOME}/custom-hermes", "native"),  # ${VAR} must expand
    ("custom-hermes", "cwd"),             # relative must anchor at the invocation cwd
])
def test_install_sh_expands_and_anchors_the_explicit_home(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, kind: str
) -> None:
    """F1/F2: the finalizer must hand INSTALL_DIR/INSTALL_LOG and the exported
    HERMES_HOME the expanded, absolute profile home. A raw ``~``/``$VAR`` lands the
    checkout under a literal directory (F1); a relative home re-resolves after
    bootstrap_pm changes cwd (F2). Both consumers must see the same absolute value."""
    bash = real_bash
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    expected = (
        str(native / "custom-hermes") if kind == "native" else str(tmp_path / "custom-hermes")
    )

    script = (
        'source "$1" --manifest || exit 1\n'
        'printf "%s\\n%s\\n%s\\n" "$HERMES_HOME" "$INSTALL_DIR" "$INSTALL_LOG"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ, "HERMES_HOME": raw}, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    hermes_home, install_dir, install_log = result.stdout.splitlines()[:3]

    assert hermes_home == expected, hermes_home
    assert install_dir == expected + "/hermes-agent", install_dir
    assert install_log == expected + "/logs/install.log", install_log


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("raw,kind", [
    ("runtime-store", "relative"),  # must anchor at the invocation cwd
    ("/abs-store", "absolute"),     # left unchanged
])
def test_install_sh_binds_the_runtime_override_before_the_chdir(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, kind: str
) -> None:
    """A relative HERMES_RUNTIME_DIR is resolved on two sides of the checkout chdir
    (ensure_uv before, PM's child after). The finalizer must export one absolute
    value so both sides bind the same store; an absolute override is the control."""
    bash = real_bash
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    want = str(tmp_path / "runtime-store") if kind == "relative" else raw
    script = (
        'source "$1" --manifest || exit 1\n'
        'printf "%s\\n" "$HERMES_RUNTIME_DIR"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ, "HERMES_RUNTIME_DIR": raw}, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[0] == want, result.stdout

    # PM's child resolves the exported value from the checkout cwd; it must agree
    # with what the bootstrap stages into (resolve() normalizes macOS symlinks).
    checkout = tmp_path / "home" / "hermes-agent"
    checkout.mkdir(parents=True)
    child = subprocess.run(
        [sys.executable, "-c",
         "import os; from pathlib import Path; print(Path(os.environ['HERMES_RUNTIME_DIR']).resolve())"],
        env={**os.environ, "HERMES_RUNTIME_DIR": want}, cwd=str(checkout),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.splitlines()[0] == str(Path(want).resolve())


@pytest.mark.skipif(
    shutil.which("cygpath") is None,
    reason="the drive-relative refuse is gated on cygpath (msys/cygwin)",
)
@pytest.mark.parametrize("bad", ["C:rel-store", "C:"])
def test_install_sh_refuses_drive_relative_runtime_dir(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """A drive-relative override splits the two sides under msys: the shell
    escapes the colon into a U+F03A directory name while native resolve()
    strips it, so each side binds a different store. install.sh must refuse
    before anchoring — the anchored form no longer matches the pattern — and
    keep ':' legal on hosts with no msys namespace split."""
    bash = real_bash
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    result = subprocess.run(
        [bash, "-c", 'source "$1" --manifest\n', "test", str(install_sh)],
        env={**os.environ, "HERMES_RUNTIME_DIR": bad}, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "drive-relative HERMES_RUNTIME_DIR" in result.stderr, result.stderr


_PS1_PATH_HELPER = re.compile(
    r'function Test-HermesFullyQualifiedPath \{.*?\n\}',
    re.DOTALL,
)
_PS1_RUNTIME_ANCHOR = re.compile(
    r'if \(\$env:HERMES_RUNTIME_DIR -and -not \(Test-HermesFullyQualifiedPath \$env:HERMES_RUNTIME_DIR\)\) \{'
    r'.*?\$env:HERMES_RUNTIME_DIR = \[System\.IO\.Path\]::GetFullPath\(\$env:HERMES_RUNTIME_DIR\)'
    r'.*?\n\s*\}',
    re.DOTALL,
)


def _ps1_anchor_source(ps1: Path) -> str:
    """The helper plus the runtime-override guard, verbatim from shipped bytes.

    The guard calls the helper, so extracting only the ``if`` would run an undefined
    command. ``install.ps1`` is piped (irm | iex) and carries the helper inline;
    ``setup-hermes.ps1`` dot-sources the canonical resolver instead of carrying a
    copy, so its helper comes from scripts/bootstrap/store_root.ps1 — the same
    bytes the shipped script would execute."""
    text = ps1.read_text(encoding="utf-8")
    helper = _PS1_PATH_HELPER.search(text)
    if helper is None:
        helper_text = _canonical_ps1_body().group(0)
    else:
        helper_text = helper.group(0)
    anchor = _PS1_RUNTIME_ANCHOR.search(text)
    assert anchor, f"{ps1.name}: the runtime-override anchor block is missing"
    return f"{helper_text}\n{anchor.group(0)}"


def _canonical_ps1_body() -> re.Match[str]:
    """The canonical ps1 resolver body, the twin's helper source when a script
    dot-sources it instead of carrying a copy."""
    begin, end = _RESOLVER_MARKERS[(CANONICAL_PS1, "canonical")]
    return re.search(
        re.escape(begin) + r"\r?\n(.*?)" + re.escape(end),
        CANONICAL_PS1.read_text(encoding="utf-8-sig"), re.DOTALL,
    )


@pytest.mark.skipif(_POWERSHELL is None, reason="running the PowerShell anchor needs pwsh or powershell")
@pytest.mark.platforms("any")
@pytest.mark.parametrize("ps1", [
    REPO_ROOT / "scripts" / "install.ps1",
    REPO_ROOT / "setup-hermes.ps1",
])
def test_powershell_anchors_a_relative_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ps1: Path
) -> None:
    """The ps1 twins must anchor a relative HERMES_RUNTIME_DIR before Push-Location,
    so Get-PmStoreRoot (caller cwd) and PM's child (InstallDir) bind the same store.
    Host-independent: only the relative case is platform-neutral — the absolute and
    drive-relative controls live in the platforms("windows") test below."""
    script = (
        f"$env:HERMES_RUNTIME_DIR = 'runtime-store'\n{_ps1_anchor_source(ps1)}\nWrite-Output $env:HERMES_RUNTIME_DIR\n"

    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, f"{ps1.name}: {result.stdout}{result.stderr}"
    got = result.stdout.strip()
    assert Path(got).resolve() == Path(tmp_path / "runtime-store").resolve(), (
        f"{ps1.name}: runtime-store -> {got!r}, want {tmp_path / 'runtime-store'!r}"
    )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("ps1", [
    REPO_ROOT / "scripts" / "install.ps1",
    REPO_ROOT / "setup-hermes.ps1",
])
@pytest.mark.parametrize("raw", [
    "rel-store",          # ordinary relative: anchored at the invocation cwd
    "C:\\abs\\store",     # drive-rooted: unchanged
    "C:rel-store",        # drive-relative: anchored at C:'s current directory
])
def test_windows_runtime_override_anchor_matches_pm_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ps1: Path, raw: str
) -> None:
    """On Windows the anchor must bind the store identity pm's
    ``Path(override).resolve()`` computes from a different cwd (after
    Push-Location): a drive-relative ``C:foo`` anchors at the drive's current
    directory, never converts to a drive-rooted ``C:\\foo``."""
    script = (
        f"$env:HERMES_RUNTIME_DIR = '{raw}'\n{_ps1_anchor_source(ps1)}\nWrite-Output $env:HERMES_RUNTIME_DIR\n"

    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, f"{ps1.name}: {result.stdout}{result.stderr}"
    anchored = result.stdout.strip()
    # The anchor must produce an ABSOLUTE identity: only then is the exported
    # value cwd-independent, so PM's child (after Push-Location into the
    # checkout) resolves the same store the bootstrap staged into.
    assert Path(anchored).is_absolute(), f"{raw!r}: anchor left {anchored!r} non-absolute"
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    child = subprocess.run(
        [sys.executable, "-c",
         "import os; from pathlib import Path; print(Path(os.environ['HERMES_RUNTIME_DIR']).resolve())"],
        env={**os.environ, "HERMES_RUNTIME_DIR": anchored}, cwd=str(checkout),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    pm = child.stdout.strip()
    assert str(Path(pm)).lower() == str(Path(anchored)).lower(), (
        f"{ps1.name} {raw!r}: bootstrap anchored {anchored!r}, PM from checkout resolved {pm!r}"
    )


#: The block setup-hermes.ps1 uses to bind HERMES_HOME before Push-Location.
#: Delimited by markers in the script itself, never by a regex over its body: the
#: body changed once already (the export-expression fix) and a body regex silently
#: stopped matching, which read as "the block is missing" on the one lane that runs
#: these rows.
_BINDING_BEGIN = "# --- BEGIN explicit-home binding ---"
_BINDING_END = "# --- END explicit-home binding ---"


def _explicit_home_binding(text: str) -> re.Match[str] | None:
    """The marked binding block as a match object (``.group(0)`` is the script text)."""
    return re.search(
        re.escape(_BINDING_BEGIN) + r"\r?\n(.*?)\r?\n" + re.escape(_BINDING_END),
        text, re.DOTALL,
    )


@pytest.mark.platforms("windows")
def test_setup_ps1_binds_a_relative_home_before_push_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative HERMES_HOME must be absolutized at invocation time.

    ``Get-HermesRoot`` deliberately returns a relative home lexically, so
    ``$store``/``Set-UvStatePins`` would stay relative and, after
    ``Push-Location $repo``, name directories relative to the checkout — while
    PM's child re-resolves the same raw home too. Binding one absolute home
    first fixes both."""
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("USERPROFILE", str(native))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8")
    resolver = _resolver_source(CANONICAL_PS1, "canonical")
    bind = _explicit_home_binding(text)
    script = (
        f"$env:HERMES_HOME = 'custom-hermes'\n{resolver}"
        + f"\n{bind.group(0)}\n"
        + "Write-Output $env:HERMES_HOME\nWrite-Output (Get-HermesRoot)\n"
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    bound, root = result.stdout.strip().splitlines()[-2:]
    want = (tmp_path / "custom-hermes").resolve()
    assert Path(bound).is_absolute(), f"bound home is still relative: {bound!r}"
    assert Path(bound).resolve() == want, bound
    assert Path(root).resolve() == want, root


@pytest.mark.platforms("windows")
def test_setup_ps1_folds_a_whitespace_only_home_like_pm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'   ' must reach pm's default home — GetFullPath('') throws in the binding block.

    A whitespace-only HERMES_HOME is "unset" for pm (``get_hermes_home`` strips
    it before deciding), for the sh bootstraps (``hermes_root_of`` trims before
    expanding) and for ``Get-HermesRoot`` itself; the explicit-home binding
    block must fold it the same way instead of dying on an empty GetFullPath.
    """
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("USERPROFILE", str(native))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8")
    resolver = _resolver_source(CANONICAL_PS1, "canonical")
    bind = _explicit_home_binding(text)
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$env:HERMES_HOME = '   '\n"
        + resolver + "\n"
        + bind.group(0) + "\n"
        + "Write-Output (Get-HermesRoot)\n"
        + "Write-Output ('home=[' + $env:HERMES_HOME + ']')\n"
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "LOCALAPPDATA": _child_localappdata(tmp_path)},
        cwd=str(tmp_path),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    root_line, home_line = result.stdout.strip().splitlines()[-2:]
    assert home_line == "home=[]", home_line
    assert _staged_store(root_line) == _pm_store_root()


@pytest.mark.platforms("windows")
def test_setup_ps1_skills_dir_treats_a_whitespace_home_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skills-seeding home read must fold '   ' to the unset default.

    Same crash class as the binding block: ``GetFullPath('')`` throws and the
    setup dies after a full install. Whitespace-only and unset must seed the
    same directory — asserted by running the shipped block twice and comparing.
    """
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8")
    resolver = _resolver_source(CANONICAL_PS1, "canonical")
    skills = re.search(
        r"\$skillsDir = if \(\$env:HERMES_HOME -and \$env:HERMES_HOME\.Trim\(\)\) \{.*?\n\} else \{\r?\n.*?\n\}",
        text, re.DOTALL,
    )
    assert skills, "setup-hermes.ps1: the skillsDir block is missing"
    outputs = []
    for assignment in (
        "$env:HERMES_HOME = '   '",
        "Remove-Item Env:HERMES_HOME -ErrorAction SilentlyContinue",
    ):
        script = (
            "$ErrorActionPreference = 'Stop'\n" + assignment + "\n" + resolver + "\n"
            + skills.group(0) + "\nWrite-Output $skillsDir\n"
        )
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
            cwd=str(tmp_path),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{assignment}: {result.stdout}{result.stderr}"
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1], (
        f"whitespace-only seeded {outputs[0]!r} but unset seeded {outputs[1]!r}"
    )


def test_setup_ps1_skills_dir_seeds_the_platform_default_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset home must seed skills under the platform default home the child
    resolves — suffix and LOCALAPPDATA folded in — not a hardcoded account-home
    literal: otherwise setup seeds one skills tree while skills_sync writes
    another (R16 #34's sibling consumer)."""
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", DATA_DIR_SUFFIX)
    # A fresh interpreter: the session fixture relocates the in-process default
    # home to tmp_path, while the PowerShell child resolves the real platform one.
    want = subprocess.run(
        [sys.executable, "-c",
         "from hermes_constants import get_hermes_home; print(get_hermes_home() / 'skills')"],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert want.returncode == 0, want.stdout + want.stderr
    expected = Path(want.stdout.strip())
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8")
    resolver = _resolver_source(CANONICAL_PS1, "canonical")
    skills = re.search(
        r"\$skillsDir = if \(\$env:HERMES_HOME -and \$env:HERMES_HOME\.Trim\(\)\) \{.*?\n\} else \{\r?\n.*?\n\}",
        text, re.DOTALL,
    )
    assert skills, "setup-hermes.ps1: the skillsDir block is missing"
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "Remove-Item Env:HERMES_HOME -ErrorAction SilentlyContinue\n"
        + resolver + "\n" + skills.group(0) + "\nWrite-Output $skillsDir\n"
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=str(tmp_path),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert Path(result.stdout.strip()) == expected


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "raw_home,expected",
    [
        ("   ", None),                       # whitespace-only == unset -> default
        ("  /tmp/custom  ", "/tmp/custom"),  # outer whitespace is trimmed
    ],
)
def test_setup_sh_folds_outer_whitespace_from_the_home(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str, expected: str | None
) -> None:
    """setup-hermes.sh must fold the home before its path consumers read it.

        Outer whitespace is not part of the home and a whitespace-only HERMES_HOME
        is unset; otherwise bin_dir publishes launchers into a ``   /bin`` junk
        directory and ``HERMES_SKILLS_DIR`` seeds ``   /skills`` while the resolver
        and child Python use the trimmed default."""
    bash = real_bash
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    setup_sh = REPO_ROOT / "setup-hermes.sh"
    body = _resolver_source(CANONICAL_SH, "canonical")
    text = setup_sh.read_text(encoding="utf-8")
    fold = re.search(r"# Trim like the resolver.*?\nfi\n", text, re.DOTALL)
    assert fold, "setup-hermes.sh: the HERMES_HOME fold block is missing"
    result = subprocess.run(
        [bash, "-c",
         f"{body}\n{fold.group(0)}\n"
         'printf "value=%s\\n" "$HERMES_HOME"\n'
         'printf "exported=%s\\n" "$(env | sed -n \'s/^HERMES_HOME=//p\')"'],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value_line, exported_line = result.stdout.strip().splitlines()[-2:]
    resolved = expected if expected is not None else str(native / ".hermes")
    assert value_line == f"value={resolved}", value_line
    if expected is None:
        # A defaulted home stays shell-local (install.sh's rule): an exported
        # literal-suffix path would make child Python expandvars it differently.
        assert exported_line == "exported=", exported_line
    else:
        assert exported_line == f"exported={resolved}", exported_line


def test_setup_sh_expands_the_home_for_its_shell_consumers(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fold exports the home's lexical form (children expand it themselves).

        This script's own consumers, cygpath/mkdir/cp, never expand ``~``, so they
        must read an expanded copy or publish launchers and skills into a literal
        ``~/...`` directory the child never writes to. The expected value comes from
        the host ``os.path`` module, the ladder the child's platform runs (ntpath
        under msys, posixpath otherwise)."""
    bash = real_bash
    native = str(tmp_path / "native-home")
    # Pin BOTH rungs at one home: the resolver takes USERPROFILE under msys and
    # HOME elsewhere, and leaving the other at the host's value would let the
    # script expand ``~`` outside the temporary home.
    monkeypatch.setenv("HOME", native)
    monkeypatch.setenv("USERPROFILE", native)
    monkeypatch.setenv("HERMES_HOME", "~/skills-home")
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    setup_sh = REPO_ROOT / "setup-hermes.sh"
    body = _resolver_source(CANONICAL_SH, "canonical")
    text = setup_sh.read_text(encoding="utf-8")
    fold = re.search(r"# Trim like the resolver.*?\nfi\n", text, re.DOTALL)
    assert fold, "setup-hermes.sh: the HERMES_HOME fold block is missing"
    skills = re.search(r"^HERMES_SKILLS_DIR=.*$", text, re.MULTILINE)
    assert skills, "setup-hermes.sh: the skills consumer is missing"
    # A script FILE, not bash -c: Git Bash's argv conversion mangles multiline
    # -c strings on Windows (the posix-marked -c tests never run there).
    script_path = tmp_path / "extracted.sh"
    script_path.write_text(
        f"{body}\n{fold.group(0)}\n{skills.group(0)}\n"
        'printf "root=%s\\n" "$(hermes_root_of)"\n'
        'printf "skills=%s\\n" "$HERMES_SKILLS_DIR"\n'
        'printf "exported=%s\\n" "$(env | sed -n \'s/^HERMES_HOME=//p\')"\n',
        encoding="utf-8", newline="\n",
    )
    result = subprocess.run(
        [bash, script_path.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root_line, skills_line, exported_line = result.stdout.strip().splitlines()[-3:]
    root, skills = root_line.removeprefix("root="), skills_line.removeprefix("skills=")
    # The consumer reads the resolver's own answer, so the assertion never
    # depends on how each host spells that directory (Git Bash POSIX-izes an
    # inherited Windows HOME, and the msys ladder reads USERPROFILE instead).
    assert Path(skills) == Path(root) / "skills", (
        f"HERMES_SKILLS_DIR={skills!r} must hang off the resolved root {root!r}"
    )
    assert Path(root) == Path(os.path.expanduser("~/skills-home")), (
        f"the resolver answered {root!r}, the host ladder expands to "
        f"{os.path.expanduser('~/skills-home')!r}"
    )
    # The export stays lexical: child Python expands it itself.
    assert exported_line == "exported=~/skills-home", exported_line


_SH_RUNTIME_ANCHOR = re.compile(
    r"# Anchor a relative HERMES_RUNTIME_DIR.*?\nfi\n", re.DOTALL
)


def test_setup_sh_anchors_a_relative_runtime_override(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative HERMES_RUNTIME_DIR must be bound to one absolute native path
    at the invocation cwd: the script cds into the checkout before it reads the
    override, and a native Windows child cannot resolve an msys POSIX path —
    the ps1 twin's rule (R16 #34's sibling consumer)."""
    bash = real_bash
    monkeypatch.setenv("HERMES_RUNTIME_DIR", "rel-store")
    monkeypatch.chdir(tmp_path)
    text = (REPO_ROOT / "setup-hermes.sh").read_text(encoding="utf-8")
    anchor = _SH_RUNTIME_ANCHOR.search(text)
    assert anchor, "setup-hermes.sh: the runtime-override anchor block is missing"
    # Placement is behaviour: after the checkout cd, $PWD is no longer the
    # caller's cwd and the anchor would bind a different store than the ps1 twin.
    assert text.index(anchor.group(0)) < text.index('cd "$SCRIPT_DIR"')
    script_path = tmp_path / "anchor.sh"
    script_path.write_text(
        anchor.group(0)
        + 'printf "value=%s\\n" "$HERMES_RUNTIME_DIR"\n'
        + 'printf "exported=%s\\n" "$(env | sed -n \'s/^HERMES_RUNTIME_DIR=//p\')"\n',
        encoding="utf-8", newline="\n",
    )
    result = subprocess.run(
        [bash, script_path.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value_line, exported_line = result.stdout.strip().splitlines()[-2:]
    want = str(tmp_path / "rel-store").replace("\\", "/")
    assert value_line == f"value={want}", value_line
    assert exported_line == f"exported={want}", exported_line
