"""Fixtures and constants shared by the two bootstrap store-root suites.

``hermes_constants.get_default_hermes_root()`` is the rule. No bootstrap can
import it — ``install.sh`` is piped to bash and the others run before Python
exists — so each carries a copy: canonical in ``scripts/bootstrap/``, generated
into the two piped installers. A drifting copy stages a sha256-verified uv where
pm never looks and hides the warm uv cache from ``uv sync --offline`` (#101269).

Two checks keep the copies honest, and neither re-derives the rule: the copies
stay byte-identical within a language (``test_store_root_resolver``), and the
shipped bytes are executed over a home matrix compared against
``pm.paths.store_root()``, so the expectation always comes from pm.

Reading and executing the shipped bytes (instead of importing them) is the
deliberate exception to ``tests/AGENTS.md``'s "never read source code in tests":
the setup scripts run top-level with no source guard, so there is nothing to
import, and the bytes are executed, never merely matched.

Both suites assume the checkout carries no ``install-stamp.json``: pm honors a
stamped ``runtimeDir`` a bootstrap cannot read yet, so a stamped tree would
report that split instead of hiding it. No production code writes one today; a
future producer must revisit these tests with the resolver.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# The store-root resolver has ONE source per language: scripts/bootstrap/.
# install.sh / install.ps1 are piped (curl|bash, irm|iex) and carry a GENERATED
# copy between markers; setup-hermes.sh / setup-hermes.ps1 / scripts/_activation.sh
# run from a checkout and source the canonical files, so they carry no copy.
CANONICAL_SH = REPO_ROOT / "scripts" / "bootstrap" / "store_root.sh"
CANONICAL_PS1 = REPO_ROOT / "scripts" / "bootstrap" / "store_root.ps1"

# One marker pair per resolver copy. The canonical files mark their own body
# with the same marker text the generator reads (scripts/gen-bootstrap-
# resolver.py), so both spellings staying identical is pinned by the drift test.
CANON_SH_BEGIN = "# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
CANON_SH_END = "# --- END canonical store-root resolver ---"
GEN_SH_BEGIN = "# --- BEGIN GENERATED: store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
GEN_SH_END = "# --- END GENERATED: store-root resolver ---"
CANON_PS1_BEGIN = "# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
CANON_PS1_END = "# --- END canonical store-root resolver ---"
GEN_PS1_BEGIN = "# --- BEGIN GENERATED: store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
GEN_PS1_END = "# --- END GENERATED: store-root resolver ---"

# (path, kind) -> the marker pair that file's resolver body sits between.
_RESOLVER_MARKERS = {
    (CANONICAL_SH, "canonical"): (CANON_SH_BEGIN, CANON_SH_END),
    (REPO_ROOT / "scripts" / "install.sh", "generated"): (GEN_SH_BEGIN, GEN_SH_END),
    (CANONICAL_PS1, "canonical"): (CANON_PS1_BEGIN, CANON_PS1_END),
    (REPO_ROOT / "scripts" / "install.ps1", "generated"): (GEN_PS1_BEGIN, GEN_PS1_END),
}

# The uv-state pins decide where uv writes its cache and managed python; the
# ps1 twins alone carry them as a marked block (the sh twins export the same
# paths at top level), so only PS1_PAIR participates here.

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


def _resolver_source(path: Path, kind: str) -> str:
    """The resolver body a bootstrap executes, between its own marker pair.

    *kind* names which marker pair the file carries:
    ``canonical``  -- scripts/bootstrap/store_root.{sh,ps1}, the single source;
    ``generated``  -- the spliced copy inside the piped installers.

    setup-hermes.sh / setup-hermes.ps1 / scripts/_activation.sh carry NO copy:
    they source the canonical file, so the canonical body is their body.
    """
    begin, end = _RESOLVER_MARKERS[(path, kind)]
    match = re.search(
        re.escape(begin) + r"\r?\n(.*?)" + re.escape(end), path.read_text(encoding="utf-8-sig"),
        re.DOTALL,
    )
    assert match, f"{path.name}: the {kind} store-root resolver block is missing"
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
