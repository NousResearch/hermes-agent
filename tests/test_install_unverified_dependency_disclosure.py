"""Regression: an install that lost hash verification has to say so at the end.

Only the ``uv sync --locked`` tier checks each package against the SHA256 that
``uv.lock`` records. The ``uv pip install`` tiers below it exist to keep an
install working when the lockfile is stale or unreadable, and they re-resolve
every transitive fresh from PyPI with nothing verifying it.

The downgrade used to be a single ``log_warn`` line in the middle of several
minutes of dependency output, after which the run still finished on an
unqualified "Installation Complete!" banner. A run that silently lost hash
verification was therefore indistinguishable from one that kept it, which is
how #90650 and #82446 both came to be filed against a tier that had not been
applying for some time.

These tests pin the disclosure: the fallback announces itself where it happens,
and the completion banner refuses to sign off clean unless the hash-verified
tier is what actually ran.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_SH = REPO_ROOT / "setup-hermes.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

HASH_VERIFIED = "hash-verified (uv.lock)"
DISCLOSURE = "WITHOUT hash verification"
GUARD = 'if [ "$INSTALLED_DEP_TIER" != "$HASH_VERIFIED_DEP_TIER" ]; then'

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash to execute installer functions"
)

# Everything the extracted shell reaches for that it does not define itself.
# Kept deliberately small: if the disclosure grows a new dependency, these
# tests should fail loudly rather than quietly exercise a different path.
_STUBS = """
GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
INSTALL_DIR=/tmp/hermes-install
UV_CMD=uv
log_info() { echo "$1"; }
log_warn() { echo "$1"; }
"""


def _run(script: str) -> str:
    # shutil.which rather than a bare "bash": on a Windows dev box PATH can
    # resolve bash to the WSL launcher, which accepts -c, exits 0, and drops
    # the arguments passed to shell functions -- every assertion below would
    # then fail against empty output for a reason that has nothing to do with
    # the installer.
    proc = subprocess.run(
        [shutil.which("bash"), "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _extract_sh_function(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"{name}() not found in {path.name}"
    return match.group(0)


def _disclosure_block() -> str:
    """The tier guard out of print_success(), verbatim.

    Only this block is executed rather than the whole banner. print_success()
    prints the command list, which includes `hermes update`, and conftest's
    live-system guard rejects any subprocess whose argv looks like that -- a
    guard worth leaving alone rather than tunnelling around. The block is real
    shipped code either way, and test_the_disclosure_lives_in_the_banner below
    pins the fact that it is reached from print_success().
    """
    text = INSTALL_SH.read_text(encoding="utf-8")
    start = text.index(GUARD)
    end = text.index("\n    fi\n", start) + len("\n    fi\n")
    return text[start:end]


def _banner(tier: str) -> str:
    script = (
        _STUBS
        + 'INSTALLED_DEP_TIER="' + tier + '"\n'
        + 'HASH_VERIFIED_DEP_TIER="' + HASH_VERIFIED + '"\n'
        + _disclosure_block()
    )
    return _run(script)


class TestCompletionBanner:
    def test_a_fallback_tier_is_disclosed(self):
        out = _banner("core only (no extras)")
        assert DISCLOSURE in out
        assert "core only (no extras)" in out
        # The remedy has to be in the message, not just the diagnosis.
        assert "uv sync --extra all --locked" in out

    def test_a_hash_verified_install_says_nothing(self):
        assert _banner(HASH_VERIFIED).strip() == ""

    def test_an_unrecorded_tier_is_disclosed_rather_than_assumed(self):
        # Fail closed. A tier nobody recorded is not evidence of verification,
        # and defaulting the other way is how this went unnoticed to begin with.
        out = _banner("")
        assert DISCLOSURE in out
        assert "unknown" in out

    def test_the_disclosure_lives_in_the_banner(self):
        # Guards the wiring the block test cannot: this has to run on the way
        # out of the installer, not somewhere the user has already scrolled past.
        body = _extract_sh_function(INSTALL_SH, "print_success")
        assert GUARD in body
        assert body.index("Installation Complete") < body.index(GUARD)


class TestDowngradeAnnouncement:
    def test_the_fallback_announces_itself_with_its_reason(self):
        body = _extract_sh_function(INSTALL_SH, "warn_unverified_dependency_resolve")
        out = _run(
            _STUBS
            + body
            + '\nwarn_unverified_dependency_resolve "the uv.lock sync above failed"\n'
        )
        assert "NOT be hash-verified" in out
        assert "the uv.lock sync above failed" in out
        assert "uv sync --extra all --locked" in out

    def test_both_routes_into_the_unverified_tiers_announce(self):
        text = INSTALL_SH.read_text(encoding="utf-8")
        calls = re.findall(r"warn_unverified_dependency_resolve \"", text)
        # One for the failed lockfile sync, one for a checkout with no
        # lockfile at all. Both land in the same unverified resolve.
        assert len(calls) == 2, f"expected both fallback routes to announce, found {len(calls)}"

    def test_the_tiers_record_which_one_ran(self):
        text = INSTALL_SH.read_text(encoding="utf-8")
        assert f'INSTALLED_DEP_TIER="{HASH_VERIFIED}"' in text
        assert 'INSTALLED_DEP_TIER="$name"' in text


class TestSetupHermesScript:
    def test_the_flag_defaults_to_unverified(self):
        text = SETUP_SH.read_text(encoding="utf-8")
        assert "HASH_VERIFIED_DEPS=false" in text
        assert text.index("HASH_VERIFIED_DEPS=false") < text.index("HASH_VERIFIED_DEPS=true")

    def test_only_the_locked_sync_may_set_it(self):
        text = SETUP_SH.read_text(encoding="utf-8")
        assert text.count("HASH_VERIFIED_DEPS=true") == 1
        before = text[: text.index("HASH_VERIFIED_DEPS=true")]
        assert before.rstrip().endswith("--extra all --locked; then"), (
            "the verified flag must be set by the locked sync succeeding, nothing else"
        )

    def test_the_completion_message_is_gated_on_it(self):
        text = SETUP_SH.read_text(encoding="utf-8")
        gate = 'if [ "$HASH_VERIFIED_DEPS" != true ]; then'
        assert gate in text
        assert text.index("Setup complete!") < text.index(gate)


class TestWindowsInstaller:
    def test_the_recorded_tier_is_actually_read(self):
        # $script:InstalledTier was written at both install sites and read
        # nowhere, so the Windows banner had the same blind spot.
        text = INSTALL_PS1.read_text(encoding="utf-8")
        completion = text[text.index("function Write-Completion"):]
        assert "$script:InstalledTier" in completion
        assert DISCLOSURE in completion

    def test_the_windows_disclosure_stays_ascii(self):
        # install.ps1 is ASCII-only (see test_install_ps1_ascii_only.py); this
        # keeps the specific block honest even if that guard is relaxed.
        text = INSTALL_PS1.read_text(encoding="utf-8")
        completion = text[text.index("function Write-Completion"):]
        block = completion[: completion.index("* Your files:")]
        assert block.isascii(), "the disclosure block must not introduce non-ASCII"
