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

# Each row pins one way a bootstrap resolver can drift from pm: the profiles
# fold (parent vs. anything under the default home), the default-home computation
# (suffix, case), variable/user expansion, raw aliases, and ``..`` across a link.
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
    "whitespace_surround",
    "named_user_tilde",
    "data_dir_suffix",
    "case_variant",
    "raw_alias",
    "raw_alias_dot",
    "home_var",
    # A ``$1``-style name must stay literal: a ${!name} lookup would resolve the
    # function's positional instead.
    "dollar_positional",
    "root_profiles",
    "relative_profile",
    # ``..`` homes: a one-element pop must empty the chain, never double it.
    "native_home_ellipsis_parent",
    "ellipsis_missing_mid",
    "ellipsis_consecutive",
    "ellipsis_root",
    "ellipsis_escape",
    "ellipsis_mid_escape",
    "ellipsis_mid_fold",
    "ellipsis_profile_leaf",
    "symlink_escape",
    "braced_unset",
    "braced_unclosed",
    "braced_empty",
    "ancestor_link",
    "missing_then_link",
    # HERMES_HOME unset with a $-bearing suffix: the suffix is appended LITERALLY
    # by pm and both bootstraps (the embedded variable must not expand).
    "data_dir_suffix_literal",
    # posixpath.expanduser's account-home edges: an EMPTY HOME answers "/" (and
    # "/x" for "~/x"), an UNSET HOME falls back to the account database.
    "home_empty",
    "home_empty_tilde",
    "home_unset_tilde",
)


def _resolver_source(path: Path, mirror: str) -> str:
    text = path.read_text(encoding="utf-8")
    # The PowerShell bootstraps are CRLF, the shell ones LF; both must extract.
    match = re.search(
        re.escape(_BEGIN.format(mirror)) + r"\r?\n(.*?)" + re.escape(_END), text, re.DOTALL
    )
    assert match, f"{path.name}: the store-root resolver block is missing"
    return match.group(1)


def _named_user_account() -> tuple[str, str] | None:
    """An existing POSIX account whose ``~user`` home resolves, or None.

    ``ntpath`` resolves only ``~<USERNAME>`` — the Windows half lives in
    :func:`_named_user_home` — so the passwd lookup itself is POSIX-only."""
    try:
        import pwd
    except ImportError:
        return None
    try:
        account = pwd.getpwuid(os.getuid())
    except (KeyError, OSError):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", account.pw_name):
        return None
    return account.pw_name, account.pw_dir


def _named_user_home() -> str:
    """A raw ``~user/.../profiles/<name>`` home the platform's own ``expanduser``
    resolves, else skip.

    POSIX resolves a passwd account; ``ntpath`` resolves only ``~<USERNAME>`` and
    leaves every other ``~user`` literal, so the Windows row must name the current
    account or it would pin a divergence rather than a match."""
    if sys.platform == "win32":
        user = os.environ.get("USERNAME", "").strip()
        if user:
            return f"~{user}/custom-root/profiles/coder"
        pytest.skip("no USERNAME to build a Windows ~user home from")
    account = _named_user_account()
    if account is None:
        pytest.skip("no POSIX account database entry to resolve a ~user home against")
    return f"~{account[0]}/custom-root/profiles/coder"


def _hermes_home(tmp_path: Path, home_kind: str) -> str | Path | None:
    if home_kind == "named_user_tilde":
        # Lazy: probed only for its own row, so an absent account skips one row.
        return _named_user_home()
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
        # ``_staged_store`` strips outer whitespace, so only a home NESTED under
        # the default discriminates a resolver that never trims.
        "whitespace_surround": f"  {default}/nested  ",
        "data_dir_suffix": str(tmp_path / "native-home" / f".hermes{DATA_DIR_SUFFIX}"),
        # Differs only in case: POSIX pathlib never folds it.
        "case_variant": str(tmp_path / "native-home" / ".HERMES"),
        # Raw strings (not built through Path, which would collapse these away).
        "raw_alias": f"{tmp_path}/custom-root/profiles//coder",
        "raw_alias_dot": f"{tmp_path}/custom-root/profiles/coder/.",
        "home_var": "${HOME}/.hermes/profiles/coder",
        "dollar_positional": f"{tmp_path}/custom/$1/profile",
        "root_profiles": "/profiles/coder",
        "relative_profile": "profiles/coder",
        # Raw strings too: Path construction would collapse the ``..`` away.
        "native_home_ellipsis_parent": f"{default}/..",
        "ellipsis_missing_mid": f"{tmp_path}/custom/missing/../profile",
        "ellipsis_consecutive": f"{tmp_path}/custom/missing/../../profile",
        "ellipsis_root": f"{tmp_path}/..",
        "ellipsis_escape": f"{default}/profiles/new/../../..",
        "ellipsis_mid_escape": f"{default}/b/../../escape",
        "ellipsis_mid_fold": f"{tmp_path}/other/../native-home/.hermes/x",
        "ellipsis_profile_leaf": f"{tmp_path}/custom-root/profiles/foo/../bar",
        "symlink_escape": f"{tmp_path}/native-home/.hermes/escape-link",
        # UNSET/malformed braces stay literal: only a CLOSED brace group expands.
        "braced_unset": f"{tmp_path}/custom/${{UNSET}}/profiles/coder",
        "braced_unclosed": f"{tmp_path}/custom/${{UNSET/profiles/coder",
        "braced_empty": f"{tmp_path}/custom/${{}}/profiles/coder",
        # A link mid-path (and one reached after a missing segment + ``..``).
        "ancestor_link": f"{tmp_path}/native-home/.hermes/escape-link/existing-home",
        "missing_then_link": f"{tmp_path}/native-home/.hermes/missing/../escape-link/new-home",
        # Folds to the platform default; _set_home owns the exact env shape.
        "data_dir_suffix_literal": None,
        "home_empty": None,
        "home_empty_tilde": "~/custom-hermes",
        "home_unset_tilde": "~/custom-hermes",
    }
    return homes[home_kind]


def _set_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home_kind: str) -> None:
    """Point both the bootstraps and pm at one home.

    ``USERPROFILE`` rides along for ``Path.home()`` on Windows; ``LOCALAPPDATA``
    is cleared only where pm would not consult it."""
    if home_kind == "data_dir_suffix_literal":
        # Both pm and the bootstraps must append the suffix VERBATIM: with
        # ROOT_CONTRACT_LABEL exported, any expansion turns the literal '$...'
        # into 'expanded', diverging from _get_platform_default_hermes_home().
        native = tmp_path / "native-home"
        monkeypatch.setenv("HOME", str(native))
        monkeypatch.setenv("USERPROFILE", str(native))
        monkeypatch.setenv("ROOT_CONTRACT_LABEL", "expanded")
        monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "-$ROOT_CONTRACT_LABEL")
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
        # LOCALAPPDATA follows the same handling as a normal Windows row (the
        # child is given tmp_path via _child_localappdata); POSIX ignores it.
        if sys.platform != "win32":
            monkeypatch.delenv("LOCALAPPDATA", raising=False)
        return
    if home_kind in ("home_empty", "home_empty_tilde", "home_unset_tilde"):
        # posixpath.expanduser's account-home edges: an EMPTY HOME is SET and
        # answers "/", an UNSET HOME falls back to the account database. The
        # child gets the same shape so the parity comparison is meaningful.
        if home_kind == "home_unset_tilde":
            monkeypatch.delenv("HOME", raising=False)
            monkeypatch.delenv("USERPROFILE", raising=False)
        else:
            monkeypatch.setenv("HOME", "")
            monkeypatch.setenv("USERPROFILE", "")
        if sys.platform != "win32":
            monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
        value = _hermes_home(tmp_path, home_kind)
        if value is None:
            monkeypatch.delenv("HERMES_HOME", raising=False)
        else:
            monkeypatch.setenv("HERMES_HOME", value)
        return
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
    if home_kind in ("ancestor_link", "missing_then_link"):
        # A link that sits in the MIDDLE of the home path (and, for
        # ``missing_then_link``, is only reached after a missing segment + "..")
        # points OUTSIDE the default home, so a resolver that folds lexically
        # instead of following it lands under the default root. Junction on
        # Windows (no privilege needed) keeps this row native there.
        (native / ".hermes").mkdir(parents=True, exist_ok=True)
        target = native / "custom"
        leaf = "existing-home" if home_kind == "ancestor_link" else "new-home"
        (target / leaf).mkdir(parents=True, exist_ok=True)
        link = native / ".hermes" / "escape-link"
        if sys.platform == "win32":
            subprocess.run(
                [_POWERSHELL or "powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"New-Item -ItemType Junction -Path '{link}' -Target '{target}' | Out-Null"],
                check=True,
            )
        else:
            link.symlink_to(target, target_is_directory=True)
    if home_kind in ("braced_unset", "braced_unclosed"):
        monkeypatch.delenv("UNSET", raising=False)
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


@pytest.mark.platforms("posix")
def test_posix_resolver_unset_home_uses_the_account_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME and HERMES_HOME both unset: the resolver must answer with the ACCOUNT
    home (id + getent/dscacheutil), not the literal relative ``$HOME/.hermes``.

    A fake account database ahead on PATH pins what the child resolves; the pm
    expectation is pointed at the same fake home (conftest isolates the native
    default per-test), so both sides name one directory without touching the
    runner account.
    """
    bash = shutil.which("bash")
    assert bash, "the shell bootstraps require bash"
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
        result = subprocess.run(
            [bash, "-c", f"{body}\nprintf '%s\\n%s\\n' \"$(hermes_root_of)\" \"$(hermes_default_home)\""],
            env=child_env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
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
@pytest.mark.parametrize("drive_relative", [False, True])
def test_powershell_resolver_keeps_drive_relative_home_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drive_relative: bool
) -> None:
    """``C:rest`` (drive-relative) must not be rewritten to ``C:\\rest``.

    Windows anchors the former at the drive's own current directory and the
    latter at the drive root; pm keeps both lexical forms (get_default_hermes_root
    returns a custom home verbatim), so staging for the two must land differently
    and each must agree with pm. The pre-fix norm/resolve pair collapsed both
    to the drive-rooted form.
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
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = _staged_store(result.stdout)
        assert staged == expected, (
            f"{name} would stage uv into {staged}, but pm resolves {expected} "
            f"for HERMES_HOME={raw_home!r}"
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
            env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=120,
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

    # A DEFAULTED home stays shell-local (exporting its literal suffix would
    # diverge from Python's verbatim-suffix default), so the recomputation must
    # run with HERMES_HOME raw again; a user-supplied home is exported as-is.
    script = (
        'raw="${HERMES_HOME-}"\n'  # save the raw configured home before the finalizer normalizes it
        'source "$1" --manifest || exit 1\n'
        'if [ "${_HERMES_HOME_DEFAULTED:-0}" = 1 ]; then '
        'recomputed="$(unset HERMES_HOME; hermes_root_of)"; '
        'else recomputed="$(HERMES_HOME="$raw" hermes_root_of)"; fi\n'
        'printf "%s\\n%s\\n" "$HERMES_ROOT" "$recomputed"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resolved, recomputed = result.stdout.splitlines()[:2]

    assert resolved == recomputed, "HERMES_ROOT must be the resolver's answer"
    assert _staged_store(resolved) == _pm_store_root()


def test_install_sh_default_home_follows_the_host_os_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolver's default home must pick its branch with NO ``os`` injected.

    The mirrored block consumes ``${os:-}`` but cannot set it (the body is
    byte-identical across twins), so the premise lives in each script's own
    block-external ``uname`` case. Extracting the body under a manual
    ``os=win32`` prefix never exercises that premise; sourcing the real script
    does. Without the premise install.sh answers POSIX (``$HOME/.hermes``) on
    a Git Bash host while the PM child computes ``%LOCALAPPDATA%\\hermes``, and
    the halves stage uv where the other never reads.
    """
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
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
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if sys.platform == "win32":
        # Git Bash's uname matches MINGW*|MSYS*|CYGWIN* -> os=win32 -> the
        # LOCALAPPDATA branch, forward slashes verbatim from the env value.
        assert result.stdout == f"{lad.as_posix()}/hermes"
    else:
        assert result.stdout == f"{home.as_posix()}/.hermes"


@pytest.mark.platforms("posix")
def test_install_sh_drops_an_inherited_export_when_it_defaults_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited ``export HERMES_HOME=`` must not leak the finalized default.

    A plain assignment to an already-exported variable keeps the export
    attribute, so the literal-suffix default would reach child Python, which
    expands the suffix as explicit configuration and selects a different root
    than the resolver did. install.sh must unset the inherited export first.
    """
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
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
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
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
def test_install_sh_trims_outer_whitespace_from_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str, expected: str | None
) -> None:
    """Outer whitespace is not part of the home: INSTALL_DIR must follow the
    trimmed value the resolver and Python's ``str.strip()`` use, not the raw string.

    A whitespace-only HERMES_HOME is "unset", so INSTALL_DIR must fall under the
    default home, not become a whitespace-prefixed relative path.
    """
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
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
        'printf "%s\\n%s\\n" "$HERMES_HOME" "$INSTALL_DIR"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(install_sh)],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    hermes_home, install_dir = result.stdout.splitlines()[:2]

    resolved = expected if expected is not None else str(native / ".hermes")
    assert hermes_home == resolved, hermes_home
    assert install_dir == resolved + "/hermes-agent", install_dir


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("raw,kind", [
    ("~/custom-hermes", "native"),        # tilde must expand to HOME/custom-hermes
    ("$HOME/custom-hermes", "native"),    # $VAR must expand
    ("${HOME}/custom-hermes", "native"),  # ${VAR} must expand
    ("custom-hermes", "cwd"),             # relative must anchor at the invocation cwd
])
def test_install_sh_expands_and_anchors_the_explicit_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, kind: str
) -> None:
    """F1/F2: the finalizer must hand INSTALL_DIR/INSTALL_LOG and the exported
    HERMES_HOME the expanded, absolute profile home. A raw ``~``/``$VAR`` lands the
    checkout under a literal directory (F1); a relative home re-resolves after
    bootstrap_pm changes cwd (F2). Both consumers must see the same absolute value."""
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
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
        capture_output=True, text=True, timeout=60,
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
def test_install_sh_binds_the_runtime_override_before_the_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, kind: str
) -> None:
    """A relative HERMES_RUNTIME_DIR is resolved on two sides of the checkout chdir
    (ensure_uv before, PM's child after). The finalizer must export one absolute
    value so both sides bind the same store; an absolute override is the control."""
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
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
        capture_output=True, text=True, timeout=60,
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
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.splitlines()[0] == str(Path(want).resolve())


@pytest.mark.skipif(
    shutil.which("cygpath") is None,
    reason="the drive-relative refuse is gated on cygpath (msys/cygwin)",
)
@pytest.mark.parametrize("bad", ["C:rel-store", "C:"])
def test_install_sh_refuses_drive_relative_runtime_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """A drive-relative override splits the two sides under msys: the shell
    escapes the colon into a U+F03A directory name while native resolve()
    strips it, so each side binds a different store. install.sh must refuse
    before anchoring — the anchored form no longer matches the pattern — and
    keep ':' legal on hosts with no msys namespace split."""
    bash = shutil.which("bash")
    assert bash, "install.sh requires bash"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    install_sh = REPO_ROOT / "scripts" / "install.sh"

    result = subprocess.run(
        [bash, "-c", 'source "$1" --manifest\n', "test", str(install_sh)],
        env={**os.environ, "HERMES_RUNTIME_DIR": bad}, cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
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
    """The helper plus the runtime-override guard, verbatim from *ps1*.

    The guard calls the helper, so extracting only the ``if`` would run an undefined
    command; both are taken from the shipped bytes and executed as written."""
    text = ps1.read_text(encoding="utf-8")
    helper = _PS1_PATH_HELPER.search(text)
    anchor = _PS1_RUNTIME_ANCHOR.search(text)
    assert helper and anchor, f"{ps1.name}: the runtime-override anchor block is missing"
    return f"{helper.group(0)}\n{anchor.group(0)}"


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
        "$env:HERMES_RUNTIME_DIR = 'runtime-store'\n{}\nWrite-Output $env:HERMES_RUNTIME_DIR\n"
        .format(_ps1_anchor_source(ps1))
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
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
    """On Windows the anchor must bind the SAME store identity PM's
    ``Path(override).resolve()`` computes from a different cwd (after
    Push-Location): a drive-relative ``C:foo`` anchors at the drive's current
    directory, never converts to a drive-rooted ``C:\\foo``."""
    script = (
        "$env:HERMES_RUNTIME_DIR = '{}'\n{}\nWrite-Output $env:HERMES_RUNTIME_DIR\n"
        .format(raw, _ps1_anchor_source(ps1))
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
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
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    pm = child.stdout.strip()
    assert str(Path(pm)).lower() == str(Path(anchored)).lower(), (
        f"{ps1.name} {raw!r}: bootstrap anchored {anchored!r}, PM from checkout resolved {pm!r}"
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
    resolver = _resolver_source(REPO_ROOT / "setup-hermes.ps1", "scripts/install.ps1")
    bind = re.search(
        r'if \(\$env:HERMES_HOME\) \{\r?\n    \$e = Expand-HermesHomeValue.*?\$env:HERMES_HOME = \$e\r?\n\}',
        text, re.DOTALL,
    )
    assert bind, "setup-hermes.ps1: the explicit-home binding block is missing"
    script = (
        "$env:HERMES_HOME = 'custom-hermes'\n{}".format(resolver)
        + f"\n{bind.group(0)}\n"
        + "Write-Output $env:HERMES_HOME\nWrite-Output (Get-HermesRoot)\n"
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
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
    resolver = _resolver_source(REPO_ROOT / "setup-hermes.ps1", "scripts/install.ps1")
    bind = re.search(
        r'if \(\$env:HERMES_HOME\) \{\r?\n    \$e = Expand-HermesHomeValue.*?\$env:HERMES_HOME = \$e\r?\n\}',
        text, re.DOTALL,
    )
    assert bind, "setup-hermes.ps1: the explicit-home binding block is missing"
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
        capture_output=True, text=True, errors="replace", timeout=120,
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
    resolver = _resolver_source(REPO_ROOT / "setup-hermes.ps1", "scripts/install.ps1")
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
            capture_output=True, text=True, errors="replace", timeout=120,
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
        cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert want.returncode == 0, want.stdout + want.stderr
    expected = Path(want.stdout.strip())
    text = (REPO_ROOT / "setup-hermes.ps1").read_text(encoding="utf-8")
    resolver = _resolver_source(REPO_ROOT / "setup-hermes.ps1", "scripts/install.ps1")
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
        capture_output=True, text=True, errors="replace", timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert Path(result.stdout.strip()) == expected


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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "raw_home,expected",
    [
        ("   ", None),                       # whitespace-only == unset -> default
        ("  /tmp/custom  ", "/tmp/custom"),  # outer whitespace is trimmed
    ],
)
def test_setup_sh_folds_outer_whitespace_from_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str, expected: str | None
) -> None:
    """setup-hermes.sh must fold the home before its path consumers read it.

    Same contract as install.sh's fold: outer whitespace is not part of the
    home, and a whitespace-only HERMES_HOME is "unset" — otherwise bin_dir
    publishes launchers into a ``   /bin`` junk directory and
    ``HERMES_SKILLS_DIR`` seeds ``   /skills`` while the resolver and child
    Python both use the trimmed default.
    """
    bash = shutil.which("bash")
    assert bash, "setup-hermes.sh requires bash"
    native = tmp_path / "native-home"
    monkeypatch.setenv("HOME", str(native))
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    setup_sh = REPO_ROOT / "setup-hermes.sh"
    body = _resolver_source(setup_sh, "scripts/install.sh")
    text = setup_sh.read_text(encoding="utf-8")
    fold = re.search(r"# Trim like the resolver.*?\nfi\n", text, re.DOTALL)
    assert fold, "setup-hermes.sh: the HERMES_HOME fold block is missing"
    result = subprocess.run(
        [bash, "-c",
         f"{body}\n{fold.group(0)}\n"
         'printf "value=%s\\n" "$HERMES_HOME"\n'
         'printf "exported=%s\\n" "$(env | sed -n \'s/^HERMES_HOME=//p\')"'],
        env={**os.environ}, cwd=tmp_path, capture_output=True, text=True, timeout=60,
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


def test_setup_sh_expands_the_home_for_its_shell_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fold exports the home's LEXICAL form (children run expanduser/
    expandvars themselves), but this script's own string consumers — cygpath,
    mkdir, cp — never expand ``~``: they must read an expanded copy or they
    publish launchers and skills into a literal ``~/...`` junk directory the
    child never writes to."""
    bash = shutil.which("bash")
    assert bash, "setup-hermes.sh requires bash"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("HERMES_HOME", "~/skills-home")
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    setup_sh = REPO_ROOT / "setup-hermes.sh"
    body = _resolver_source(setup_sh, "scripts/install.sh")
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
        'printf "home=%s\\n" "$HOME"\n'
        'printf "skills=%s\\n" "$HERMES_SKILLS_DIR"\n'
        'printf "exported=%s\\n" "$(env | sed -n \'s/^HERMES_HOME=//p\')"\n',
        encoding="utf-8", newline="\n",
    )
    result = subprocess.run(
        [bash, script_path.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    home_line, skills_line, exported_line = result.stdout.strip().splitlines()[-3:]
    # Want built from the script's own $HOME: Cygwin POSIX-izes an inherited
    # Windows HOME at startup, so the Python-side spelling would not compare.
    home = home_line.removeprefix("home=")
    assert skills_line == f"skills={home}/skills-home/skills", skills_line
    assert exported_line == "exported=~/skills-home", exported_line


_SH_RUNTIME_ANCHOR = re.compile(
    r"# Anchor a relative HERMES_RUNTIME_DIR.*?\nfi\n", re.DOTALL
)


def test_setup_sh_anchors_a_relative_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative HERMES_RUNTIME_DIR must be bound to one absolute native path
    at the invocation cwd: the script cds into the checkout before it reads the
    override, and a native Windows child cannot resolve an msys POSIX path —
    the ps1 twin's rule (R16 #34's sibling consumer)."""
    bash = shutil.which("bash")
    assert bash, "setup-hermes.sh requires bash"
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
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value_line, exported_line = result.stdout.strip().splitlines()[-2:]
    want = str(tmp_path / "rel-store").replace("\\", "/")
    assert value_line == f"value={want}", value_line
    assert exported_line == f"exported={want}", exported_line


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("raw_home", [
    "%LOCALAPPDATA%\\hermes",   # exact name, the common spelling
    "%localappdata%\\hermes",   # ntpath/Environ folds case; printenv alone would not
])
def test_sh_resolver_expands_percent_syntax_like_ntpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_home: str
) -> None:
    """ntpath.expandvars resolves ``%VAR%`` in HERMES_HOME (case-folding on
    Windows); the sh resolver's copy must answer the same root or a
    ``%LOCALAPPDATA%`` home stages the store under a literal ``%`` directory
    pm never reads."""
    bash = shutil.which("bash")
    assert bash, "the shell bootstraps require bash"
    monkeypatch.setenv("HERMES_HOME", raw_home)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    # The mirror functions gate the win32 path on $os, which the real script
    # sets from uname before they first run. Exported as a shell variable, not
    # through the environment: Windows' own OS=Windows_NT collides case-
    # insensitively with an `os` entry in the child's environment block.
    expected = _pm_store_root()
    body = _resolver_source(REPO_ROOT / "setup-hermes.sh", "scripts/install.sh")
    script_path = tmp_path / "pct.sh"
    script_path.write_text(
        "os=win32\n" + body + "\nprintf '%s\\n' \"$(hermes_root_of)\"\n",
        encoding="utf-8", newline="\n",
    )
    result = subprocess.run(
        [bash, script_path.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    staged = _staged_store(result.stdout.splitlines()[0])
    assert staged == expected, (
        f"{raw_home} stages the store into {staged}, but pm resolves {expected}"
    )


@pytest.mark.platforms("windows")
def test_resolvers_fold_a_case_variant_of_the_default_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case-flipped default home folds to the default root, on every twin.

    pm's containment compares ``Path.relative_to`` (WindowsPath normcases) and
    the PowerShell twin folds with OrdinalIgnoreCase; a case-sensitive shell
    containment would stage the store under an upper-cased spelling of the same
    directory instead of the default lexical form both sides must agree on.
    """
    bash = shutil.which("bash")
    assert bash, "the shell bootstraps require bash"
    if _POWERSHELL is None:
        pytest.skip("running the PowerShell bootstraps needs pwsh or Windows PowerShell")
    default = tmp_path / "hermes"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(default).upper())
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = _pm_store_root()

    sh_body = _resolver_source(REPO_ROOT / "setup-hermes.sh", "scripts/install.sh")
    sh_script = tmp_path / "casefold.sh"
    sh_script.write_text(
        "os=win32\n" + sh_body + "\nprintf '%s\\n' \"$(hermes_root_of)\"\n",
        encoding="utf-8", newline="\n",
    )
    sh_result = subprocess.run(
        [bash, sh_script.as_posix()],
        env={**os.environ}, cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
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
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"
        staged = str(_staged_store(result.stdout))
        assert staged == str(expected), (
            f"{name} folds the case-variant home into {staged}, not {expected}"
        )


@pytest.mark.skipif(_POWERSHELL is None, reason="running the PowerShell resolver needs pwsh or powershell")
@pytest.mark.platforms("windows")
@pytest.mark.parametrize(("homedrive", "expected"), [
    (None, "RESULT=/x"),
    ("C:", "RESULT=C:\\Users\\probe/x"),
])
def test_powershell_tilde_expansion_skips_a_lone_homepath(
    homedrive: str | None, expected: str
) -> None:
    """A lone HOMEPATH walks past the pair rung instead of dying in Join-Path.

    ``Expand-HermesHomeValue`` follows ntpath's ladder — USERPROFILE, else the
    HOMEDRIVE+HOMEPATH pair, else the ``$HOME`` rung — under the twins'
    ``$ErrorActionPreference = 'Stop'``. With HOMEPATH set but HOMEDRIVE
    missing, the unconditional pair rung bound ``$null`` into ``Join-Path``
    and terminated the expansion, so an explicit ``~`` home killed the
    bootstrap; ntpath skips a pair whose drive is absent. The ``C:`` row keeps
    the pair itself honest.
    """
    env = {k: v for k, v in os.environ.items() if k != "USERPROFILE"}
    if homedrive is None:
        env.pop("HOMEDRIVE", None)
    else:
        env["HOMEDRIVE"] = homedrive
    env["HOMEPATH"] = "\\Users\\probe"

    for name, (path, mirror) in PS1_PAIR.items():
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference='Stop'\n" + _resolver_source(path, mirror) + "\n"
             "'RESULT=' + (Expand-HermesHomeValue '~/x')"],
            env=env, capture_output=True, text=True, errors="replace", timeout=120,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}{result.stdout}"
        assert result.stdout.strip() == expected, (
            f"{name}: Expand-HermesHomeValue '~/x' -> {result.stdout.strip()!r}, want {expected!r}"
        )
