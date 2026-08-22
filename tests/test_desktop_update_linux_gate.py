"""The linux_gate prefix check, exercised against the real posix.sh function.

On a Fedora host /home is a symlink to /var/home. The updater reads the
running desktop's path from /proc/<pid>/exe, which the kernel canonicalises
to the /var/home spelling, while INSTALL_ROOT is spelled /home/... (the
symlink). The raw prefix match then false-gates as "skew" and tells the user
to reinstall an app that is actually fine.

The fix canonicalises both sides with `readlink -m` before the prefix
compare. These tests extract the *real* `linux_gate` from posix.sh (no copy,
no mock) and drive it through the symlink case plus the controls that must
still behave the same way.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

requires_bash = pytest.mark.skipif(
    not (Path("/bin/bash").exists() and Path("/usr/bin/readlink").exists()),
    reason="linux_gate test needs /bin/bash and GNU readlink",
)


def extract_linux_gate(src: str) -> str:
    """Return the text of the `linux_gate() { ... }` function from posix.sh.

    The function's terminator is the first line that is exactly `}`; every
    inner brace (case/esac, `&& { ...; }`, for/done) sits mid-line, so this
    is unambiguous. We assert on marker lines so a malformed extraction
    fails loudly rather than testing a partial function.
    """
    m = re.search(r"^linux_gate\(\) \{", src, re.M)
    assert m, "linux_gate() not found in posix.sh"
    lines = src[m.start():].splitlines(keepends=True)
    end = next(i for i, l in enumerate(lines) if l.strip() == "}")
    fn = "".join(lines[: end + 1])
    # Sanity: we grabbed the whole body, not a fragment.
    assert "GATE=manual" in fn, "extraction missing the GATE=manual terminator"
    assert "GATE=skew" in fn, "extraction missing the GATE=skew branch"
    return fn


def run_gate(fn: str, install_root: str, relaunch_target: str) -> str:
    """Drive the extracted function once in a clean bash and return GATE."""
    driver = f"""
set -u
INSTALL_ROOT="{install_root}"
RELAUNCH_TARGET="{relaunch_target}"
SANDBOX_FALLBACK=0
ELECTRON_DISABLE_SANDBOX=""
RELAUNCH_ARGS=()
GATE="" GATE_MSG=""
{fn}
linux_gate
printf '%s' "$GATE"
"""
    with tempfile.TemporaryDirectory() as td:
        fn_path = Path(td) / "gate.sh"
        fn_path.write_text(driver)
        out = subprocess.run(
            ["/bin/bash", str(fn_path)],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()


def make_tree(root: Path) -> Path:
    """Create <root>/apps/desktop/release/linux-unpacked and return it."""
    unpacked = root / "apps" / "desktop" / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    return unpacked


@requires_bash
def test_clean_match_relays_relaunch(tmp_path):
    """Target inside the real install dir, no sandbox helper -> relaunch."""
    src = POSIX_SH.read_text()
    fn = extract_linux_gate(src)
    base = tmp_path / "install"
    unpacked = make_tree(base)
    target = unpacked / "hermes"
    assert run_gate(fn, str(base), str(target)) == "relaunch"


@requires_bash
def test_fedora_symlink_spelling_is_not_false_skew(tmp_path):
    """The regression: INSTALL_ROOT spelled via a symlink that points at the
    canonical dir, target read from the canonical side. Raw prefix match
    (unpatched) false-gates 'skew'; canonicalised compare gives 'relaunch'."""
    src = POSIX_SH.read_text()
    fn = extract_linux_gate(src)
    canonical = tmp_path / "varhome"
    unpacked = make_tree(canonical)
    # A symlink standing in for /home -> /var/home.
    symlink_root = tmp_path / "home"
    symlink_root.symlink_to(canonical)
    # Target is spelled with the canonical (kernel) side, like /proc/<pid>/exe.
    target = unpacked / "hermes"
    assert run_gate(fn, str(symlink_root), str(target)) == "relaunch"


@requires_bash
def test_genuinely_external_target_still_skew(tmp_path):
    """A target outside the install dir must still gate 'skew' (no over-fix)."""
    src = POSIX_SH.read_text()
    fn = extract_linux_gate(src)
    base = tmp_path / "install"
    make_tree(base)
    elsewhere = tmp_path / "other-app"
    (elsewhere / "apps" / "desktop").mkdir(parents=True)
    target = elsewhere / "apps" / "desktop" / "hermes"
    assert run_gate(fn, str(base), str(target)) == "skew"


@requires_bash
def test_gate_canonicalises_the_patch_is_present():
    """Guard the fix itself: the function must canonicalise both sides.

    If the readlink lines are ever removed, the symlink test above regresses
    to 'skew'; this asserts the mechanism is present so the failure is
    localisable.
    """
    src = POSIX_SH.read_text()
    fn = extract_linux_gate(src)
    assert 'readlink -m' in fn, "linux_gate no longer canonicalises paths"
    # Both sides must be canonicalised, not just one.
    assert fn.count("readlink -m") >= 2, "expected both sides canonicalised"


@requires_bash
def test_empty_relaunch_target_falls_to_skew(tmp_path):
    """An unset/empty RELaunch_TARGET must still gate 'skew'.

    The canonicalisation line is guarded with [ -n "$RELAUNCH_TARGET" ] so
    an empty value skips readlink and the raw empty string falls through
    the case to the skew branch. This pins that behavior so the guard
    cannot be "simplified" away without a visible test failure.
    """
    src = POSIX_SH.read_text()
    fn = extract_linux_gate(src)
    base = tmp_path / "install"
    make_tree(base)
    assert run_gate(fn, str(base), "") == "skew"
