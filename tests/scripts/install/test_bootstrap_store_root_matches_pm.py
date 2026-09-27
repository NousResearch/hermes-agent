"""Every bootstrap resolves the pm store root exactly where pm resolves it (#101269).

``hermes_constants.get_default_hermes_root()`` is the rule. The bootstraps cannot
import it — they run before Python exists, and ``install.sh`` is piped to bash —
so each carries a copy; a drifting copy stages a sha256-verified uv where PM
never looks (and hides the warm uv cache from ``uv sync --offline``).

Two things keep the copies honest, and neither re-derives the rule:

* the copies are byte-identical within a language (a fix applied to one file
  and forgotten in its twin fails here), and
* the shipped bytes are executed over a matrix of homes and compared against
  ``pm.paths.store_root()``, so the expectation always comes from pm.

Byte-identity also covers the PowerShell twins' uv-state pin body
(``Set-UvStatePins``): the store slot and uv's cache/python dirs are one
contract — see ``test_mirrored_uv_state_pin_bodies_are_identical``.

The comparison assumes the checkout carries no ``install-stamp.json``: pm honors
a stamped ``runtimeDir`` that a bootstrap cannot read yet (it runs before Python
exists), so a stamped tree would report that split rather than hide it — which is
why CI and Docker move the stamp in and out around a build instead of leaving it
beside the code. No production code currently writes a stamped ``runtimeDir``; a
future producer must revisit this test together with the bootstrap resolver.

Executing the *extracted* blocks rather than sourcing the files is deliberate:
the setup scripts run top-level with no source guard. The wiring — that they feed
real ``HERMES_HOME`` into the resolver and stage into what it returns — is
covered by ``test_install_sh_uv_state.py``, ``test_install_ps1_uv_no_path_borrow.py``
and ``tests/hermes_cli/test_setup_hermes_script.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# name -> (file, the twin its BEGIN marker names)
SH_PAIR = {
    "install.sh": (REPO_ROOT / "scripts" / "install.sh", "setup-hermes.sh"),
    "setup-hermes.sh": (REPO_ROOT / "setup-hermes.sh", "scripts/install.sh"),
}
PS1_PAIR = {
    "install.ps1": (REPO_ROOT / "scripts" / "install.ps1", "setup-hermes.ps1"),
    "setup-hermes.ps1": (REPO_ROOT / "setup-hermes.ps1", "scripts/install.ps1"),
}

_BEGIN = "# --- BEGIN store-root resolver (mirrored in {}) ---"
_END = "# --- END store-root resolver ---"

# The uv-state pins decide where uv writes its cache and managed python; the
# ps1 twins alone carry them as a marked block (the sh twins export the same
# paths at top level), so only PS1_PAIR participates here.
_PIN_BEGIN = "# --- BEGIN uv state pins (mirrored in {}) ---"
_PIN_END = "# --- END uv state pins ---"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

# ``nested_under_default`` is the case a rule that only looks for a literal
# ``profiles`` parent gets wrong: pm folds anything under the platform default
# home, not only a named profile. ``profiles_segment`` and
# ``ancestor_profile`` guard the opposite error — folding a path that merely
# contains a ``profiles`` segment. ``data_dir_suffix`` guards the default-home
# computation itself: all four resolvers carry their own copy of
# ``_get_platform_default_hermes_home()``, and a copy that forgets the suffix
# folds to a home pm never built. ``case_variant`` is a distinct directory that
# differs only in case — POSIX pathlib never folds it, so a case-insensitive
# comparison would. The ``*_ellipsis*`` rows feed ``..`` through the resolvers:
# a single-component pop must empty the chain, not double it (PowerShell's
# ``0..-1`` is a descending range, so a pop written as ``$out[0..($out.Count - 2)]``
# on a one-element array keeps the element twice); a ``..`` after a missing
# segment must collapse lexically; and ``..`` at the physical root stays there.
# ``symlink_escape`` sends the home through a link out of the default root: the
# resolver must follow it (it IS the home) without folding.
DATA_DIR_SUFFIX = "-suffix-under-test"
HOME_KINDS = (
    "unset",
    "default",
    "profile",
    "profile_trailing_slash",
    "custom_profile",
    "profiles_segment",
    "ancestor_profile",
    "nested_under_default",
    "data_dir_suffix",
    "case_variant",
    "raw_alias",
    "raw_alias_dot",
    "home_var",
    # ``dollar_positional``: expandvars leaves a ``$1``-style name literal (no
    # such exported variable), and a resolver reading the environment with
    # ${!name} instead of a printenv-style table lookup would ALSO resolve it
    # to the function's positional parameter — self-doubling the path.
    "dollar_positional",
    "root_profiles",
    "relative_profile",
    # ``..``-bearing homes: a single pop must empty a one-element chain, a
    # missing-segment `missing/..` pair must collapse lexically, and pops at
    # the physical root must stay there. ``ellipsis_escape`` pops the EXISTING
    # default home: a single-segment pop that keeps its component flips the
    # fold (red on the previous shape, green only with the anchored pop).
    # ``ellipsis_mid_*``: a mid-path ``..`` must not survive into the containment
    # check — the fold flips either way. ``ellipsis_profile_leaf``: a
    # ``profiles/..`` parent is not a profiles parent.
    "native_home_ellipsis_parent",
    "ellipsis_missing_mid",
    "ellipsis_consecutive",
    "ellipsis_root",
    "ellipsis_escape",
    "ellipsis_mid_escape",
    "ellipsis_mid_fold",
    "ellipsis_profile_leaf",
    "symlink_escape",
)


def _resolver_source(path: Path, mirror: str) -> str:
    text = path.read_text(encoding="utf-8")
    # The PowerShell bootstraps are CRLF, the shell ones LF; both must extract.
    match = re.search(
        re.escape(_BEGIN.format(mirror)) + r"\r?\n(.*?)" + re.escape(_END), text, re.DOTALL
    )
    assert match, f"{path.name}: the store-root resolver block is missing"
    return match.group(1)


def _hermes_home(tmp_path: Path, home_kind: str) -> str | Path | None:
    default = tmp_path / "native-home" / ".hermes"
    homes: dict[str, str | Path | None] = {
        "unset": None,
        "default": str(default),
        "profile": str(default / "profiles" / "coder"),
        "profile_trailing_slash": str(default / "profiles" / "coder") + os.sep,
        "custom_profile": str(tmp_path / "custom-root" / "profiles" / "coder"),
        "profiles_segment": str(tmp_path / "profiles" / "alice" / ".hermes"),
        "ancestor_profile": str(tmp_path / "profiles" / "alice" / ".hermes" / "profiles" / "coder"),
        "nested_under_default": str(default / "foo" / "bar"),
        "data_dir_suffix": str(tmp_path / "native-home" / f".hermes{DATA_DIR_SUFFIX}"),
        # A different directory that differs only in case: POSIX pathlib never
        # folds it (string comparison), so a case-insensitive -eq in a resolver
        # would stage uv into the default home instead of this one.
        "case_variant": str(tmp_path / "native-home" / ".HERMES"),
        # Raw environment strings, deliberately NOT built through Path: Path
        # construction collapses repeated separators and trailing-dot segments,
        # which erases exactly the aliases a user can export before the shell
        # ever sees them. The resolver must expand variables and normalize the
        # string like Path does, or the fold picks a slot pm never reads.
        "raw_alias": f"{tmp_path}/custom-root/profiles//coder",
        "raw_alias_dot": f"{tmp_path}/custom-root/profiles/coder/.",
        "home_var": "${HOME}/.hermes/profiles/coder",
        "dollar_positional": f"{tmp_path}/custom/$1/profile",
        "root_profiles": "/profiles/coder",
        "relative_profile": "profiles/coder",
        # ``..`` must reach a resolver VERBATIM, so these are raw strings, not
        # Path arithmetic (Path construction collapses the ``..`` away).
        "native_home_ellipsis_parent": f"{default}/..",
        "ellipsis_missing_mid": f"{tmp_path}/custom/missing/../profile",
        "ellipsis_consecutive": f"{tmp_path}/custom/missing/../../profile",
        "ellipsis_root": f"{tmp_path}/..",
        "ellipsis_escape": f"{default}/profiles/new/../../..",
        "ellipsis_mid_escape": f"{default}/b/../../escape",
        "ellipsis_mid_fold": f"{tmp_path}/other/../native-home/.hermes/x",
        "ellipsis_profile_leaf": f"{tmp_path}/custom-root/profiles/foo/../bar",
        "symlink_escape": f"{tmp_path}/native-home/.hermes/escape-link",
    }
    return homes[home_kind]


def _set_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home_kind: str) -> None:
    """Point both the bootstraps and pm at one home.

    ``USERPROFILE`` rides along because ``Path.home()`` reads it on Windows, and
    ``LOCALAPPDATA`` is cleared only where pm would not consult it — pm's default
    home is platform-conditional, and so is the PowerShell resolver's.
    """
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("USERPROFILE", str(native))
    if home_kind in ("ellipsis_escape", "ellipsis_mid_escape", "ellipsis_mid_fold"):
        # A ``..`` flip only surfaces against a provably-EXISTING default home.
        (native / ".hermes").mkdir(parents=True, exist_ok=True)
    if home_kind == "symlink_escape":
        if sys.platform == "win32":
            # Windows runners do not promise SeCreateSymbolicLinkPrivilege; the
            # link semantics are POSIX-native here and probed per-host.
            pytest.skip("symlink creation needs privileges Windows runners do not promise")
        (native / "custom").mkdir(parents=True, exist_ok=True)
        (native / ".hermes").mkdir(parents=True, exist_ok=True)
        (native / ".hermes" / "escape-link").symlink_to(native / "custom", target_is_directory=True)
    if sys.platform != "win32":
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    if home_kind == "data_dir_suffix":
        monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", DATA_DIR_SUFFIX)
    else:
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    value = _hermes_home(tmp_path, home_kind)
    if value is None:
        monkeypatch.delenv("HERMES_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_HOME", value)


def _child_localappdata(tmp_path: Path) -> str:
    """The LOCALAPPDATA the child must see to reproduce pm's default home.

    On Windows the hermetic fixture redirects the platform default under
    ``tmp_path`` (``isolated_platform_default`` -> ``tmp_path/hermes``), while
    the resolver derives its own default as ``$env:LOCALAPPDATA\\hermes``.
    Handing the child the host value makes those two disagree, so the ``unset``
    row compares the host default against the isolated one and fails. Point it
    at ``tmp_path`` so both sides name the same home. POSIX pwsh ignores the
    variable, so it stays blank there.
    """
    return str(tmp_path) if sys.platform == "win32" else ""


def _pm_store_root() -> Path:
    """pm's own answer — imported, never re-derived.

    A second copy of the folding rule in this file would be precisely the drift
    these tests exist to catch. Realpath'd because pm's non-fold answer keeps
    ".." components (the lexical form), which never string-equal a resolver's
    resolved answer — compare the DIRECTORIES they name.
    """
    import os

    from pm.paths import store_root

    return Path(os.path.realpath(str(store_root())))


def _staged_store(resolved: str) -> Path:
    """The store a resolver's answer stages into, canonicalized like ``_pm_store_root``.

    Both sides must be realpath'd the same way: a relative ``HERMES_HOME`` folds
    to ``.`` (or ``profiles``), so the resolver's answer is relative to the
    caller's cwd and never string-equals pm's absolute realpath otherwise.
    """
    import os

    return Path(os.path.realpath(Path(resolved.strip()) / "tools"))


def test_mirrored_resolver_bodies_are_identical() -> None:
    """A fix applied to one bootstrap must not miss its twin (#101269)."""
    for label, pair in (("POSIX", SH_PAIR), ("PowerShell", PS1_PAIR)):
        bodies = {name: _resolver_source(path, mirror) for name, (path, mirror) in pair.items()}
        names = list(bodies)
        assert len(set(bodies.values())) == 1, (
            f"the {label} store-root resolvers have drifted apart: "
            f"{names[0]} and {names[1]} must carry the same body"
        )


def _pin_source(path: Path, mirror: str) -> str:
    """Extract one bootstrap's uv-state pin block, marker lines excluded.

    Same extraction shape as :func:`_resolver_source` (CRLF included): the
    store-root resolver decides the stage slot, these pins decide where uv
    writes its cache and managed python — the two halves of "a uv PM can serve".
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(
        re.escape(_PIN_BEGIN.format(mirror)) + r"\r?\n(.*?)" + re.escape(_PIN_END),
        text, re.DOTALL,
    )
    assert match, f"{path.name}: the uv state pins block is missing"
    return match.group(1)


def test_mirrored_uv_state_pin_bodies_are_identical() -> None:
    """The ps1 twins must pin uv state with the same body (#101269)."""
    bodies = {name: _pin_source(path, mirror) for name, (path, mirror) in PS1_PAIR.items()}
    names = list(bodies)
    assert len(set(bodies.values())) == 1, (
        f"the PowerShell uv state pins have drifted apart: "
        f"{names[0]} and {names[1]} must carry the same body"
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_posix_bootstraps_resolve_pms_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    bash = shutil.which("bash")
    assert bash, "the shell bootstraps require bash"
    _set_home(monkeypatch, tmp_path, home_kind)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    for name, (path, mirror) in SH_PAIR.items():
        body = _resolver_source(path, mirror)
        result = subprocess.run(
            [bash, "-c", f"{body}\nhermes_root_of"],
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            f"for HERMES_HOME={os.environ.get('HERMES_HOME')!r}"
        )


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
            capture_output=True, text=True, timeout=120,
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

    Native Windows only: a UNC share and an absolute drive root exist as path
    FORMS even when no such share/drive is reachable, and the resolver must not
    depend on ``Test-Path`` succeeding to keep them whole. Two pre-fix
    regressions make the cases load-bearing: a single ``..`` pop doubled the
    tail component (PowerShell's ``0..-1`` range is descending, so a one-element
    ``$out[0..-1]`` keeps the element twice), and a UNC root lost its leading
    separator pair, rebuilding as ``\\server\\share``. ``TrimEnd`` separates
    the assert from the root's trailing-separator convention; the comparison
    is case-sensitive, like pathlib on both platforms.
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
        # Drive roots: ``..`` cannot rise above the drive letter either (the
        # ``C:\a\..`` tail pop is the one-element path that used to double).
        "C:\\a\\..": "C:\\",
        "C:\\": "C:\\",
    }
    # Both twins carry the same resolver, and each file's OWN shipped bytes are
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
            env={**os.environ}, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0 and result.stdout.strip().endswith("OK"), (
            f"{name}: {result.stdout}{result.stderr}"
        )


@pytest.mark.platforms("windows")
def test_powershell_resolver_falls_back_to_the_platform_default_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOCALAPPDATA unset: the resolver must still land on pm's default home.

    ``_get_platform_default_hermes_home()`` falls back to
    ``Path.home()/"AppData"/"Local"`` on Windows, not ``~/.hermes`` — a resolver
    that assumed the POSIX layout would stage uv into a home pm never builds on
    exactly the hosts where ``LOCALAPPDATA`` is missing. The ``unset`` home kind
    cannot see this: ``_set_home`` keeps ``LOCALAPPDATA`` on Windows because pm
    consults it, which is the whole point of clearing it here.
    """
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
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            "with LOCALAPPDATA unset"
        )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_install_sh_stages_into_the_root_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    """install.sh must feed the resolver's answer to the store, not HERMES_HOME.

    ``install.sh`` is the one bootstrap with a source guard, so this pins the
    wiring rather than the rule: ``HERMES_ROOT`` and ``ensure_uv``'s store both
    have to follow the resolver.
    """
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
    _set_home(monkeypatch, tmp_path, home_kind)
    # Same cwd as the child, so a relative ``HERMES_HOME`` resolves against the
    # same directory on both sides of the comparison below.
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    script = (
        'source "$1" --manifest || exit 1\n'
        'printf "%s\\n%s\\n" "$HERMES_ROOT" "$(hermes_root_of)"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resolved, recomputed = result.stdout.splitlines()[:2]

    assert resolved == recomputed, "HERMES_ROOT must be the resolver's answer"
    assert _staged_store(resolved) == _pm_store_root()


@pytest.mark.platforms("posix")
def test_sh_resolver_never_answers_an_empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate HERMES_HOME ('.', './') must answer "." (pathlib's lexical
    value), never "": an empty ``$HERMES_ROOT/tools`` becomes the filesystem
    root. The matrix rows realpath-compare, which Path("") joins away — the
    raw answer is the contract here.
    """
    bash = shutil.which("bash")
    assert bash, "the shell bootstraps require bash"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)

    for name, (path, mirror) in SH_PAIR.items():
        body = _resolver_source(path, mirror)
        for home in (".", "./"):
            monkeypatch.setenv("HERMES_HOME", home)
            result = subprocess.run(
                [bash, "-c", f"{body}\nhermes_root_of"],
                env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
            assert result.stdout == ".", (
                f"{name} answered {result.stdout!r} for HERMES_HOME={home!r}; "
                "an empty answer would make the store slot the filesystem root"
            )
