"""The store-root resolver's own contract, per language.

The resolver has ONE source per language (``scripts/bootstrap/store_root.{sh,ps1}``);
``scripts/install.sh`` / ``scripts/install.ps1`` carry a generated copy between
markers and the checkout bootstraps source the canonical files. This file pins
that the copies stay byte-identical, that the canonical files are what the
checkout scripts load, and that the shipped bytes answer what pm answers for
every home shape (profiles fold, native-home containment, expansion, aliases,
``..``) — on this host's ladder and, through the separator seam, on the other one.

Fixtures and the shared helpers live in :mod:`tests.scripts.bootstrap.store_root_support`.
"""
from __future__ import annotations

from pathlib import Path
import base64
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

# The bootstrap copies the matrix walks, one entry per live resolver copy:
# (display name, path, which marker pair that file's body sits between).
# setup-hermes.sh / setup-hermes.ps1 are absent on purpose — they SOURCE the
# canonical files, and walking the canonical body once per language covers them.
SH_PAIR = {
    "install.sh": (REPO_ROOT / "scripts" / "install.sh", "generated"),
    "store_root.sh": (CANONICAL_SH, "canonical"),
}
PS1_PAIR = {
    "install.ps1": (REPO_ROOT / "scripts" / "install.ps1", "generated"),
    "store_root.ps1": (CANONICAL_PS1, "canonical"),
}


def _run_resolver_body(
    tmp_path: Path, body: str, tail: str, *, cwd: Path, env: dict[str, str] | None = None
):
    """Run an extracted resolver body from a FILE instead of ``bash -c``.

        The body is ~14 KB of shell, and the autouse ``live_system_guard`` fixture
        shlex-parses every suffix of a command line it inspects, which is quadratic
        in argv length (~2 s per matrix case through ``-c``). Production sources a
        file too, so nothing about the bytes under test changes."""
    script = tmp_path / "resolver-probe.sh"
    script.write_text(f"{body}\n{tail}\n", encoding="utf-8", newline="\n")
    result = subprocess.run(
        ["bash", script.as_posix()], env=env if env is not None else {**os.environ},
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    script.unlink(missing_ok=True)
    return result


def test_spliced_resolver_matches_the_generator() -> None:
    """install.sh's generated resolver copy must equal the canonical file (#101269).

    The copy is derived, not maintained: the generator splices
    scripts/bootstrap/store_root.sh into the piped installer, and this is the
    drift check (the same shape as test_bootstrap_pins_fragment.py). Running
    the generator directly keeps the file on disk and this test honest.
    """
    root = REPO_ROOT
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "gen-bootstrap-resolver.py"), "--check"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkout_bootstraps_source_the_canonical_resolver() -> None:
    """setup-hermes.sh and scripts/_activation.sh carry NO resolver copy.

    Both always run from a checkout, so they source the canonical file; a copy
    pasted into either would be a fourth hand-maintained implementation of the
    fold, which is precisely the drift the generator exists to prevent.
    """
    for name in ("setup-hermes.sh", "scripts/_activation.sh"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "bootstrap/store_root.sh" in text, (
            f"{name}: does not source the canonical store-root resolver"
        )
        assert "# --- BEGIN store-root resolver" not in text, (
            f"{name}: carries an inline resolver copy that must be sourced instead"
        )


def test_setup_ps1_sources_the_canonical_files() -> None:
    """setup-hermes.ps1 carries NO resolver or pins copy (it runs from a checkout)."""
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8-sig")
    assert "bootstrap\\store_root.ps1" in text, "setup-hermes.ps1 must dot-source the canonical resolver"
    assert "bootstrap\\uv_state_pins.ps1" in text, "setup-hermes.ps1 must dot-source the canonical pins"
    assert "# --- BEGIN store-root resolver" not in text, "an inline resolver copy came back"
    assert "# --- BEGIN uv state pins" not in text, "an inline pins copy came back"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_posix_bootstraps_resolve_pms_store_root(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    _set_home(monkeypatch, tmp_path, home_kind)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    for name, (path, mirror) in SH_PAIR.items():
        body = _resolver_source(path, mirror)
        result = _run_resolver_body(tmp_path, body, "hermes_root_of", cwd=tmp_path)
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        assert result.stderr == "", (
            f"{name}: the spliced resolver wrote to stderr — the block must be "
            f"self-contained (every helper it calls is defined inside it): "
            f"{result.stderr}"
        )
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            f"for HERMES_HOME={os.environ.get('HERMES_HOME')!r}"
        )


@pytest.mark.platforms("posix")
def test_posix_resolver_unset_home_uses_the_account_database(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME and HERMES_HOME both unset: the resolver must answer with the account
        home (id plus getent/dscacheutil), not the literal relative ``$HOME/.hermes``.

        A fake account database ahead on PATH pins what the child resolves, and the
        pm expectation is pointed at the same fake home, so both name one directory
        without touching the runner account."""
    account = "hermesprobe"
    fake_home = tmp_path / "acct" / "probehome"
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(parents=True)
    (fakebin / "id").write_text(f"#!/bin/sh\nif [ \"$1\" = \"-un\" ]; then echo {account}; fi\n")
    (fakebin / "getent").write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = passwd ] && [ \"$2\" = {account} ]; then "
        f"echo '{account}:x:4242:4242::{fake_home}:/bin/sh'; fi\n",
        encoding="utf-8",
    )
    # macOS hosts have no getent: the resolver then asks dscacheutil.
    (fakebin / "dscacheutil").write_text(
        "#!/bin/sh\n"
        f"echo 'name: {account}'\necho 'dir: {fake_home}'\n",
        encoding="utf-8",
    )
    for tool in ("id", "getent", "dscacheutil"):
        (fakebin / tool).chmod(0o755)

    for key in ("HOME", "HERMES_HOME", "HERMES_RUNTIME_DIR", "HERMES_DATA_DIR_SUFFIX", "LOCALAPPDATA"):
        monkeypatch.delenv(key, raising=False)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: fake_home / ".hermes")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    expected = _pm_store_root()

    child_env = {"PATH": os.pathsep.join([str(fakebin), "/usr/bin", "/bin"])}
    for name, (path, mirror) in SH_PAIR.items():
        body = _resolver_source(path, mirror)
        result = _run_resolver_body(
            tmp_path, body,
            "printf '%s\\n%s\\n' \"$(hermes_root_of)\" \"$(hermes_default_home)\"",
            cwd=tmp_path, env=child_env,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        root_line, default_line = result.stdout.splitlines()[:2]
        assert _staged_store(root_line) == expected
        assert default_line == str(fake_home / ".hermes")


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_powershell_bootstraps_resolve_pms_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    _set_home(monkeypatch, tmp_path, home_kind)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"probe-{name}.ps1"
        probe.write_text(f"{_resolver_source(path, mirror)}\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            # The child must see the same isolated default the parent-side
            # expectation was computed from: on Windows pm consults the host's
            # LOCALAPPDATA, which _set_home deliberately keeps, so the patched
            # default becomes tmp_path/hermes — pass tmp_path (not the host
            # value) so LOCALAPPDATA\hermes names that same home. POSIX ignores
            # the variable; blank it there to keep pwsh's default deterministic.
            env={**os.environ, "LOCALAPPDATA": _child_localappdata(tmp_path)}, cwd=tmp_path,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            f"for HERMES_HOME={os.environ.get('HERMES_HOME')!r}"
        )


@pytest.mark.platforms("windows")
def test_powershell_resolver_keeps_unc_and_ellipsis_semantics(tmp_path: Path) -> None:
    """``_HermesResolvePath`` keeps anchors a filesystem check cannot reach (#101269).

    A UNC share and a drive root exist as path forms even when nothing is
    reachable, so the resolver must not depend on ``Test-Path`` succeeding to
    keep them whole, and one ``..`` pop must not duplicate the tail component
    (PowerShell's ``0..-1`` range is descending). The comparison is
    case-sensitive, like pathlib on both platforms.
    """
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    cases = {
        # UNC share root: verbatim, one leading separator pair, never two pack.
        r"\\server\share": r"\\server\share",
        r"\\server\share\a": r"\\server\share\a",
        r"\\server\share\a\..": r"\\server\share",
        r"\\server\share\a\b\..": r"\\server\share\a",
        # ``..`` at the share root stays there (pathlib keeps it, like a drive root).
        r"\\server\share\..": r"\\server\share",
        # Drive roots: ``..`` cannot rise above the drive letter either, and the
        # ``C:\a\..`` pop is the one-element chain that must not duplicate.
        "C:\\a\\..": "C:\\",
        "C:\\": "C:\\",
    }
    # Both twins carry the same resolver, and each file's own shipped bytes are
    # what the probe must run (the byte-identity test only proves drift-free
    # copies; it does not prove either one right).
    for name, (path, mirror) in PS1_PAIR.items():
        body = _resolver_source(path, mirror)
        script = (
            "$ErrorActionPreference = 'Stop'\n"
            + body
            + "\n"
            '$cases = @{\n'
            + "\n".join(
                f"  '{key}' = '{expected}'" for key, expected in cases.items()
            )
            + "\n}\n"
            "foreach ($k in $cases.Keys) {\n"
            "  $got = _HermesResolvePath $k '\\'\n"
            "  if (($got.TrimEnd('\\') -cne $cases[$k].TrimEnd('\\'))) {\n"
            "    throw \"$k resolved to $got, expected $($cases[$k])\"\n"
            "  }\n"
            "}\n"
            "'OK'\n"
        )
        probe = tmp_path / f"probe-unc-{name}.ps1"
        probe.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ}, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0 and result.stdout.strip().endswith("OK"), (
            f"{name}: {result.stdout}{result.stderr}"
        )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("drive_relative", [False, True])
def test_powershell_resolver_keeps_drive_relative_home_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drive_relative: bool
) -> None:
    """``C:rest`` (drive-relative) must not be rewritten to ``C:\\rest``.

    Windows anchors the former at the drive's own current directory and the
    latter at the drive root, and pm keeps both lexical forms, so each must
    stage where pm resolves it.
    """
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    drive, _ = os.path.splitdrive(os.getcwd())
    assert drive, "the Windows lane always runs with a drive-qualified cwd"
    leaf = f"hermes-data-{os.getpid()}"
    raw_home = f"{drive}{leaf}" if drive_relative else os.path.join(drive + os.sep, leaf)
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"probe-drive-{name}.ps1"
        probe.write_text(f"{_resolver_source(path, mirror)}\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            f"for HERMES_HOME={raw_home!r}"
        )


@pytest.mark.skipif(_POWERSHELL is None, reason="the PowerShell resolver needs pwsh or powershell")
@pytest.mark.platforms("any")
@pytest.mark.parametrize("raw_home", ["/", "//", "\\"])
def test_powershell_resolver_keeps_a_bare_separator_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str
) -> None:
    """A bare separator root is a root, not the cwd (``/`` must not answer ``.``).

        Trimming a bare root to the empty string folded it to ``.``, so
        ``HERMES_HOME=/`` staged the store at ``<cwd>/tools`` while the shell twin
        and pm answer ``/tools``. The answer's spelling follows the host
        separator (a Windows host answers a backslash for ``/``), which is why
        the assertion checks root-ness rather than the input's characters."""
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"probe-bare-sep-{name}.ps1"
        probe.write_text(f"{_resolver_source(path, mirror)}\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        got = result.stdout.strip()
        # Either spelling of the same root is fine; '.' (the cwd) is not.
        assert got and not got.strip("/\\"), (
            f"{name} answered {got!r} for HERMES_HOME={raw_home!r}; a bare root must "
            "stay a root ('.' would stage the store in the caller's cwd)"
        )


@pytest.mark.platforms("posix")
def test_shell_resolver_keeps_a_trailing_backslash_in_a_file_name(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX: a trailing ``\\`` is a file-name character, so it must survive.

    ``Restore-HermesRootSpelling`` trims ``/`` and ``\\`` on Windows because both
    end a path there. Trimming ``\\`` on POSIX retargets ``/a\\b\\`` at
    ``/a\\b`` — a different directory, and neither pm nor the shell twin does it.
    """
    bash = real_bash
    raw_home = f"{tmp_path}/a\\\\b\\\\"
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = str(_pm_store_root())

    for name, (path, mirror) in SH_PAIR.items():
        result = subprocess.run(
            [bash, "-c", f"{_resolver_source(path, mirror)}\nhermes_root_of"],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        assert str(_staged_store(result.stdout)) == expected, (
            f"{name} answered {result.stdout.strip()!r} for HERMES_HOME={raw_home!r}, "
            f"but pm resolves {expected}"
        )


@pytest.mark.platforms("windows")
def test_powershell_resolver_keeps_a_bare_drive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HERMES_HOME=C:\\`` is the drive root, not the drive-relative ``C:``.

    The trailing-separator trim must not collapse the bare root (``C:\\`` ->
    ``C:``), which would re-anchor the store at the drive's current directory.
    """
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    drive, _ = os.path.splitdrive(os.getcwd())
    assert drive, "the Windows lane always runs with a drive-qualified cwd"
    raw_home = drive + os.sep  # "C:\"
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"probe-bare-root-{name}.ps1"
        probe.write_text(f"{_resolver_source(path, mirror)}\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        assert result.stdout.strip() == raw_home, (
            f"{name} returned {result.stdout.strip()!r} for HERMES_HOME={raw_home!r}"
        )
        assert _staged_store(result.stdout) == expected, (
            f"{name} would stage uv into {_staged_store(result.stdout)}, pm resolves {expected}"
        )


@pytest.mark.platforms("windows")
def test_powershell_resolver_falls_back_to_the_platform_default_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOCALAPPDATA unset: the resolver must still land on pm's default home.

        ``_get_platform_default_hermes_home()`` falls back to
        ``Path.home()/"AppData"/"Local"`` on Windows, not ``~/.hermes``. The
        ``unset`` home kind cannot see this: ``_set_home`` keeps ``LOCALAPPDATA`` on
        Windows because pm consults it, which is why this test clears it itself."""
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("USERPROFILE", str(native))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    expected = _pm_store_root()

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"probe-no-localappdata-{name}.ps1"
        probe.write_text(f"{_resolver_source(path, mirror)}\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            "with LOCALAPPDATA unset"
        )


@pytest.mark.platforms("posix")
def test_sh_resolver_never_answers_an_empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate HERMES_HOME ('.', './') must answer "." (pathlib's lexical
    value), never "": an empty ``$HERMES_ROOT/tools`` becomes the filesystem
    root. The matrix rows realpath-compare, which Path("") joins away — the
    raw answer is the contract here.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)

    for name, (path, mirror) in SH_PAIR.items():
        body = _resolver_source(path, mirror)
        for home in (".", "./"):
            monkeypatch.setenv("HERMES_HOME", home)
            result = _run_resolver_body(tmp_path, body, "hermes_root_of", cwd=tmp_path)
            assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
            assert result.stdout == ".", (
                f"{name} answered {result.stdout!r} for HERMES_HOME={home!r}; "
                "an empty answer would make the store slot the filesystem root"
            )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("raw_home", [
    "%LOCALAPPDATA%\\hermes",   # exact name, the common spelling
    "%localappdata%\\hermes",   # ntpath/Environ folds case; printenv alone would not
])
def test_sh_resolver_expands_percent_syntax_like_ntpath(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str
) -> None:
    """ntpath.expandvars resolves ``%VAR%`` in HERMES_HOME (case-folding on
    Windows); the sh resolver's copy must answer the same root or a
    ``%LOCALAPPDATA%`` home stages the store under a literal ``%`` directory
    pm never reads."""
    bash = real_bash
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    # The mirror functions gate the win32 path on $os, which the real script
    # sets from uname before they first run. Exported as a shell variable, not
    # through the environment: Windows' own OS=Windows_NT collides case-
    # insensitively with an `os` entry in the child's environment block.
    expected = _pm_store_root()
    body = _resolver_source(CANONICAL_SH, "canonical")
    script_path = tmp_path / "pct.sh"
    script_path.write_text(
        "os=win32\n" + body + "\nprintf '%s\\n' \"$(hermes_root_of)\"\n",
        encoding="utf-8", newline="\n",
    )
    result = subprocess.run(
        [bash, script_path.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    staged = _staged_store(result.stdout.splitlines()[0])
    assert staged == expected, (
        f"{raw_home} stages the store into {staged}, but pm resolves {expected}"
    )


@pytest.mark.platforms("windows")
def test_resolvers_fold_a_case_variant_of_the_default_home(real_bash,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case-flipped default home folds to the default root, on every twin.

    pm's containment compares ``Path.relative_to`` (WindowsPath normcases) and
    the PowerShell twin folds with OrdinalIgnoreCase; a case-sensitive shell
    containment would stage the store under an upper-cased spelling of the same
    directory instead of the default lexical form both sides must agree on.
    """
    bash = real_bash
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    default = tmp_path / "hermes"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(default).upper())
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    sh_body = _resolver_source(CANONICAL_SH, "canonical")
    sh_script = tmp_path / "casefold.sh"
    sh_script.write_text(
        "os=win32\n" + sh_body + "\nprintf '%s\\n' \"$(hermes_root_of)\"\n",
        encoding="utf-8", newline="\n",
    )
    sh_result = subprocess.run(
        [bash, sh_script.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert sh_result.returncode == 0, sh_result.stdout + sh_result.stderr
    sh_staged = str(_staged_store(sh_result.stdout.splitlines()[0]))
    # String-level: WindowsPath == folds case, which would hide exactly the
    # fold decision this row exists to pin (pm returns the default's spelling).
    assert sh_staged == str(expected), (
        f"the shell resolver answers {sh_staged} for HERMES_HOME="
        f"{os.environ['HERMES_HOME']!r}, but pm resolves {expected}"
    )

    for name, (path, mirror) in PS1_PAIR.items():
        probe = tmp_path / f"casefold-{path.stem}.ps1"
        probe.write_text(_resolver_source(path, mirror) + "\nGet-HermesRoot\n", encoding="utf-8")
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(probe)],
            env={**os.environ, "LOCALAPPDATA": _child_localappdata(tmp_path)},
            cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = str(_staged_store(result.stdout))
        assert staged == str(expected), (
            f"{name} folds the case-variant home into {staged}, not {expected}"
        )


_TILDE_ESCAPE_CASES = [
    # plain tilde and the HOME rung (ntpath never consults HOME)
    ("~/x", {}),
    ("~", {}),
    ("~/x", {"HOME": r"C:\probe"}),
    # presence, not truthiness, at each rung
    ("~/x", {"USERPROFILE": ""}),
    ("~/x", {"USERPROFILE": r"C:\Users\probe"}),
    ("~/x", {"USERPROFILE": "C:\\Users\\probe\\"}),
    ("~/x", {"HOMEDRIVE": "C:", "HOMEPATH": ""}),
    ("~/x", {"HOMEDRIVE": "C:", "HOMEPATH": "/x"}),
    ("~/x", {"HOMEDRIVE": "", "HOMEPATH": "/x"}),
    ("~/x", {"HOMEPATH": "/x"}),
    ("~/x", {"HOMEDRIVE": "C:\\", "HOMEPATH": "\\x"}),
    ("~/x", {"HOMEDRIVE": "C:\\", "HOMEPATH": "x"}),
    # ~user: ntpath compares the account name CASE-SENSITIVELY
    ("~probe/x", {"USERPROFILE": r"C:\Users\probe", "USERNAME": "probe"}),
    ("~Probe/x", {"USERPROFILE": r"C:\Users\probe", "USERNAME": "probe"}),
    ("~other/x", {"USERPROFILE": r"C:\Users\probe", "USERNAME": "probe"}),
    ("~probe/x", {"USERPROFILE": r"C:\Users\probe"}),
    ("~me/sub", {"USERPROFILE": r"C:\Users\me", "USERNAME": "me"}),
    ("~Probe/x", {"USERPROFILE": "C:\\Users\\probe\\", "USERNAME": "probe"}),
    ("~//x", {"USERPROFILE": r"C:\Users\probe"}),
    ("~\\x", {"USERPROFILE": r"C:\Users\probe"}),
    # expandvars: %VAR%, $VAR, the %% / $$ escapes, quoted spans
    ("%PROBE_A%/x", {"PROBE_A": "V"}),
    ("%%PROBE_A%%/x", {"PROBE_A": "V"}),
    ("%%%PROBE_A%%%/x", {"PROBE_A": "V"}),
    ("100%%", {"PROBE_A": "V"}),
    ("%NOPE%/x", {"PROBE_A": "V"}),
    ("a%NOPE/x", {"PROBE_A": "V"}),
    ("%PROBE_A%", {"PROBE_A": "V"}),
    ("$PROBE_A/x", {"PROBE_A": "V"}),
    ("${PROBE_A}/x", {"PROBE_A": "V"}),
    ("$NOPE/x", {"PROBE_A": "V"}),
    ("${NOPE/x", {"PROBE_A": "V"}),
    ("$$PROBE_A/x", {"PROBE_A": "V"}),
    ("$$$PROBE_A/x", {"PROBE_A": "V"}),
    ("$$", {"PROBE_A": "V"}),
    ("'$PROBE_A'/x", {"PROBE_A": "V"}),
    ("'%PROBE_A%'/x", {"PROBE_A": "V"}),
    ("''$PROBE_A/x", {"PROBE_A": "V"}),
    ("it's $PROBE_A/x", {"PROBE_A": "V"}),
    ("'a'b$PROBE_A", {"PROBE_A": "V"}),
]
_TILDE_ENV_KEYS = ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "USERNAME", "PROBE_A")


def _ntpath_expand(env: dict[str, str], value: str) -> str:
    """The oracle: ntpath.expanduser(ntpath.expandvars(value)) under *env*.

    ntpath is pure Python and answers the same on every host, so the Windows
    ladder is pinnable from the POSIX lane too."""
    import ntpath

    saved = {k: os.environ.pop(k, None) for k in _TILDE_ENV_KEYS}
    os.environ.update(env)
    try:
        return ntpath.expanduser(ntpath.expandvars(value))
    finally:
        for key in _TILDE_ENV_KEYS:
            os.environ.pop(key, None)
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


def _probe_env(env: dict[str, str]) -> dict[str, str]:
    child = {k: v for k, v in os.environ.items() if k not in _TILDE_ENV_KEYS}
    child.update(env)
    return child


def _decode_lines(stdout: str) -> list[str]:
    return [base64.b64decode(line).decode() for line in stdout.splitlines() if line]


def _case_groups() -> list[dict]:
    """The matrix grouped by environment, so one child per env runs every value.

    Present-but-empty variables cannot be created with .NET's
    ``SetEnvironmentVariable`` on Unix (an empty value removes the name), so the
    environment must arrive through the child's own environment block — one
    process per distinct env keeps that faithful without spawning per row."""
    groups: dict[tuple, dict] = {}
    for value, env in _TILDE_ESCAPE_CASES:
        groups.setdefault(tuple(sorted(env.items())), {"env": dict(env), "values": []})
        groups[tuple(sorted(env.items()))]["values"].append(value)
    return list(groups.values())


def _b64_items(values: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return (",\n" + pad).join(f"'{base64.b64encode(v.encode()).decode()}'" for v in values)


_PS1_MATRIX_DRIVER = """{body}
$ErrorActionPreference = 'Stop'
$values = @(
    {values}
)
foreach ($b in $values) {{
    $v = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b))
    $out = Expand-HermesHomeValue $v '\\'
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($out))
}}
"""


@pytest.mark.skipif(_POWERSHELL is None, reason="running the PowerShell resolver needs pwsh or powershell")
@pytest.mark.platforms("any")
def test_powershell_resolver_matches_ntpath(tmp_path: Path) -> None:
    """``Expand-HermesHomeValue`` must answer ntpath's value, byte for byte.

    The Windows ladder is selected explicitly (``$Sep`` = backslash), so it runs
    on every host. The account name stays case-sensitive (ntpath compares
    ``os.environ['USERNAME']`` with ``!=``), and ``%%``/``$$`` fold as ntpath's
    ``_varpattern`` does.
    """
    for group in _case_groups():
        env, values = group["env"], group["values"]
        expected = [_ntpath_expand(env, value) for value in values]
        for name, (path, mirror) in PS1_PAIR.items():
            driver = tmp_path / f"matrix-{path.stem}.ps1"
            driver.write_text(
                _PS1_MATRIX_DRIVER.format(
                    body=_resolver_source(path, mirror), values=_b64_items(values)
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver)],
                env=_probe_env(env), capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=180, check=False,
            )
            assert result.returncode == 0, f"{name}: {result.stderr}{result.stdout}"
            got = _decode_lines(result.stdout)
            assert got == expected, _matrix_diffs(name, values, env, got, expected)


_SH_MATRIX_DRIVER = """{body}
_hermes_os() {{ printf win32; }}
while IFS= read -r _b; do
    _v="$(printf '%s' "$_b" | base64 -d)"
    _hermes_expand "$_v" | base64 | tr -d '\r\n'
    printf '\\n'
done <<'HERMES_EOF'
{values}
HERMES_EOF
"""


@pytest.mark.platforms("any")
def test_sh_resolver_matches_ntpath_on_win32(real_bash, tmp_path: Path) -> None:
    """The sh twin answers ntpath's value under msys, where pm runs win32.

    ``setup-hermes.sh`` sources this file on Git Bash/Cygwin, so the Windows
    ladder is live production code there. ``_hermes_os`` is overridden to
    ``win32`` (its only input is ``uname``), which lets the row run on any lane.
    """
    bash = real_bash
    for group in _case_groups():
        env, values = group["env"], group["values"]
        expected = [_ntpath_expand(env, value) for value in values]
        for name, (path, mirror) in SH_PAIR.items():
            driver = tmp_path / f"matrix-{path.stem}.sh"
            driver.write_text(
                _SH_MATRIX_DRIVER.format(
                    body=_resolver_source(path, mirror),
                    values="\n".join(base64.b64encode(v.encode()).decode() for v in values),
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [bash, driver.as_posix()], env=_probe_env(env),
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
            )
            assert result.returncode == 0, f"{name}: {result.stderr}{result.stdout}"
            got = _decode_lines(result.stdout)
            assert got == expected, _matrix_diffs(name, values, env, got, expected)


def _matrix_diffs(
    name: str, values: list[str], env: dict[str, str], got: list[str], expected: list[str]
) -> str:
    lines = [f"{name}: {len(got)} answers vs {len(expected)} expectations, env={env}"]
    for value, want, have in zip(values, expected, got):
        if want != have:
            lines.append(f"  {value!r}: got {have!r}, ntpath answers {want!r}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Cross-twin guard: the shell resolver and the PowerShell resolver, on ONE host.
#
# Each twin already has an oracle test (ntpath / posixpath). That is transitive
# agreement, not agreement: a shared misreading of the fold can move both to the
# same wrong answer while both stay green, because each test's oracle is computed
# by the language that test speaks.
#
# The invariant the bootstraps actually rely on is narrower and directly
# checkable: on one machine, the twin a user's shell happens to run must resolve
# the same store root as the twin the other shell would. Each twin picks its own
# platform from the host (sh: `uname`, PowerShell: `$Sep`), so no override is
# applied here — that is what makes this a real "same host, same answer" check
# rather than a forced-ladder comparison.
# ---------------------------------------------------------------------------

# Raw HERMES_HOME values whose fold decides the store root. Covers the default
# home, the profile fold, a custom root, raw aliases, a relative value (anchored
# at the cwd), and the unset case.
_CROSS_TWIN_HOMES = (
    "/tmp/x/.hermes",
    "/tmp/x/.hermes/profiles/coder",
    "/tmp/x/.hermes/profiles/coder/",
    "/tmp/custom/profiles//coder",
    "/tmp/custom/profiles/coder/.",
    "/tmp/x/.hermes/foo/bar",
    "/tmp/custom/a/../b/profiles/c",
    "/tmp/custom/profiles",
    "relative-hermes",
)

_SH_ROOT_DRIVER = """
{body}
while IFS= read -r _b; do
    _v="$(printf '%s' "$_b" | base64 -d)"
    HERMES_HOME="$_v" hermes_root_of | base64 | tr -d '\\r\\n'
    printf '\\n'
done <<'HERMES_EOF'
{values}
HERMES_EOF
"""

_PS1_ROOT_DRIVER = """
{body}
$ErrorActionPreference = 'Stop'
$values = @(
    {values}
)
foreach ($b in $values) {{
    $v = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b))
    $env:HERMES_HOME = $v
    $out = Get-HermesRoot
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($out))
}}
"""


def _b64_lines(values) -> str:
    return "\n".join(base64.b64encode(v.encode()).decode() for v in values)


@pytest.mark.skipif(_POWERSHELL is None, reason="cross-twin guard needs pwsh or Windows PowerShell")
@pytest.mark.platforms("any")
def test_the_two_twins_resolve_one_home_identically(real_bash, tmp_path: Path) -> None:
    """On one host, both bootstraps name the same store root for the same home.

    A disagreement means a Git Bash user and a PowerShell user on the same machine
    stage the store in different directories — the split #101269 is about, seen
    from the bootstrap side. Runs wherever bash and PowerShell both exist, which
    includes the Windows lane.
    """
    bash = real_bash
    env = {
        "HOME": str(tmp_path / "native-home"),
        "USERPROFILE": str(tmp_path / "native-home"),
        "LOCALAPPDATA": str(tmp_path / "local-app-data"),
        "HERMES_DATA_DIR_SUFFIX": "",
    }
    for directory in ("native-home", "local-app-data"):
        (tmp_path / directory).mkdir(exist_ok=True)
    probe_env = _probe_env(env)
    encoded = _b64_lines(_CROSS_TWIN_HOMES)

    sh_driver = tmp_path / "cross-twin.sh"
    sh_driver.write_text(
        _SH_ROOT_DRIVER.format(body=_resolver_source(CANONICAL_SH, "canonical"), values=encoded),
        encoding="utf-8",
    )
    sh_result = subprocess.run(
        [bash, sh_driver.as_posix()], env=probe_env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
    )
    assert sh_result.returncode == 0, sh_result.stdout + sh_result.stderr

    ps_driver = tmp_path / "cross-twin.ps1"
    ps_driver.write_text(
        _PS1_ROOT_DRIVER.format(
            body=_resolver_source(CANONICAL_PS1, "canonical"), values=_b64_items(_CROSS_TWIN_HOMES)
        ),
        encoding="utf-8",
    )
    ps_result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(ps_driver)],
        env=probe_env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
    )
    assert ps_result.returncode == 0, ps_result.stdout + ps_result.stderr

    sh_answers = _decode_lines(sh_result.stdout)
    ps_answers = _decode_lines(ps_result.stdout)
    assert len(sh_answers) == len(_CROSS_TWIN_HOMES) == len(ps_answers), (
        f"row count differs: sh={len(sh_answers)} ps1={len(ps_answers)} "
        f"expected={len(_CROSS_TWIN_HOMES)}"
    )
    diffs = [
        f"  {home!r}\n    sh : {sh!r}\n    ps1: {ps!r}"
        for home, sh, ps in zip(_CROSS_TWIN_HOMES, sh_answers, ps_answers)
        if not _same_directory(sh, ps)
    ]
    assert not diffs, (
        "the two bootstraps resolve one HERMES_HOME to different store roots on this "
        "host — whichever shell the user runs, they stage tools where the other does not:\n"
        + "\n".join(diffs)
    )


def _same_directory(a: str, b: str) -> bool:
    """Whether two spellings name one directory, rather than being one string.

    The twins are allowed to SPELL a path differently and are not allowed to name a
    different one. On Windows the shell twin answers ``C:/Users/x/.hermes`` (its
    ``_hermes_norm`` only rewrites ``\\`` to ``/``, and ``hermes_root_of`` prints
    that lexical form) while the PowerShell twin answers ``C:\\Users\\x\\.hermes``.
    A string compare would call those different; ``ntpath.normcase(normpath(...))``
    is the platform's own answer to "same path?", so it is the one used here.

    Off Windows, neither normalisation applies: a backslash is an ordinary file-name
    character in POSIX, so folding one into a separator would compare two different
    directories as equal.
    """
    if sys.platform == "win32":
        import ntpath

        return ntpath.normcase(ntpath.normpath(a)) == ntpath.normcase(ntpath.normpath(b))
    return a == b
